import torch
import dgl
import math
import numpy as np
from torch import nn
from dgl.nn.pytorch.utils import Identity
from geoopt.manifolds.stereographic import PoincareBall
import torch.nn.functional as F
import geoopt

ball = PoincareBall(c=1)

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.num_dst_nodes():])
                V = self.w_v(b.srcdata['h'][b.num_dst_nodes():])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst

class HGATLayer(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, dist=True):
        super(HGATLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(att_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.combined = combined
        self.dist = dist
        self.bias = bias
        self.activation = activation
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        if dim_node_feat + dim_time > 0:
            self._in_src_feats, self._in_dst_feats = dim_time, dim_time

        self._out_feats = self.dim_out // self.num_head

        self.fc_src = MobiusLinear(
            self._in_src_feats, self._out_feats * num_head, bias=bias
        )
        self.fc_dst = MobiusLinear(
            self._in_src_feats, self._out_feats * num_head, bias=bias
        )
        self.fc_edge = MobiusLinear(
            dim_edge_feat, self._out_feats * num_head, bias=bias
        )
        self.attn = nn.Parameter(
            torch.FloatTensor(size=(1, num_head, self._out_feats))
        )

        if residual:
            if self._in_dst_feats != self._out_feats * num_head:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * self._out_feats, bias=bias
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        self.reset_parameters()

        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_edge.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, b, edge_weight=None, get_attention=False):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time = torch.cat([torch.zeros(b.num_dst_nodes(), dtype=torch.float32).cuda(), b.edata['dt']])
            time_feat = self.time_enc(time)
            time_feat = ball.projx(time_feat)
        
        # 1. 节点特征就只考虑本来的和时间特征，src是所有的特征，dst是src[:b.num_dst_nodes()]
        # 2. edge特征用来做attn
        if self.dim_node_feat == 0:
            src_feat = time_feat
        else:
            # src_feat = torch.cat([time_feat, b.srcdata['h']], dim=1)
            src_feat = ball.mobius_add(b.srcdata['hyper'], time_feat)
        h_src = self.feat_drop(src_feat)
        h_dst = self.feat_drop(src_feat)[:b.num_dst_nodes()]

        feat_src = self.fc_src(h_src)
        feat_dst = self.fc_dst(h_dst)

        feat_src_e = ball.logmap0(feat_src).view(
            -1, self.num_head, self._out_feats)
        feat_dst_e = ball.logmap0(feat_dst).view(
            -1, self.num_head, self._out_feats)
        
        def get_dist(edges):
            dist = ball.dist(edges.src['ll'], edges.dst['lr'])
            dist = 1 / (1e-15 + dist)
            return {'dist' : dist}
        
        b.srcdata.update({'ll': feat_src, 'el': feat_src_e})# (num_src_edge, num_heads, out_dim)
        b.dstdata.update({'lr': feat_dst, 'er': feat_dst_e})
        b.apply_edges(dgl.function.u_add_v('el', 'er', 'e'))
        e = (b.edata.pop('e') * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
        if self.dist:
            b.apply_edges(get_dist)
            dist = b.edata.pop('dist').view(-1, 1)
            assert dist.shape[0] == e.shape[0]
            dist = dgl.ops.edge_softmax(b, dist).reshape(-1, 1, 1)
            e = e * dist
        # compute softmax
        e = self.leaky_relu(e)
        b.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(b, e)) # (num_edge, num_heads)
        # message passing
        b.update_all(dgl.function.u_mul_e('el', 'a', 'm'),
                            dgl.function.sum('m', 'ft'))
        rst = b.dstdata['ft'].view(-1, self.dim_out)

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, b.edata['a']
        else:
            #print(ball.expmap0(rst), ball.projx(rst))
            #assert ball.expmap0(rst) == ball.projx(rst)
            return ball.expmap0(rst)

# package.nn.modules.py
def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.
    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.
    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float
    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)
    # else trust input
    return ball

class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input, time_bias=None):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
            time_bias=time_bias
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()


# package.nn.functional.py
def mobius_linear(input, weight, time_bias=None, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if time_bias is not None:
        output = ball.mobius_add(output, time_bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output


class HFusion(torch.nn.Module):
    """Hyperbolic Feature Fusion from Euclidean space"""

    def __init__(self, drop):
        super(HFusion, self).__init__()
        self.att = nn.Parameter(torch.Tensor(1, 1))
        self.drop = drop
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.att, 0)

    def forward(self, x_h, x_e):
        dist = ball.dist(x_h, ball.expmap0(x_e)) * self.att
        x_e = ball.mobius_scalar_mul(dist.view([-1, 1]), ball.expmap0(x_e))
        # x_e = F.dropout(x_e, p=self.drop, training=self.training)
        x_h = ball.mobius_add(x_h, x_e)
        return x_h

# Fusion
class EFusion(torch.nn.Module):
    """Euclidean Feature Fusion from hyperbolic space"""

    def __init__(self, drop):
        super(EFusion, self).__init__()
        self.att = nn.Parameter(torch.Tensor(1, 1))
        self.drop = drop
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.att, 0)

    def forward(self, x_h, x_e):
        dist = (ball.logmap0(x_h) - x_e).pow(2).sum(dim=-1) * self.att
        x_h = dist.view([-1, 1]) * ball.logmap0(x_h)
        # x_h = F.dropout(x_h, p=self.drop, training=self.training)
        x_e = x_e + x_h
        return x_e

class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""
    def __init__(self, r=2, t=1):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1)
        return probs

class LinkDecoder(nn.Module):
    """
    Base model for link prediction task.
    """

    def __init__(self, dim_out):
        super(LinkDecoder, self).__init__()
        self.dc = FermiDiracDecoder()
        self.w_e = nn.Linear(dim_out, 1, bias=False)
        self.w_h = nn.Linear(dim_out, 1, bias=False)
        self.drop_e = 0
        self.drop_h = 0
        self.reset_param()

    def reset_param(self):
        self.w_e.reset_parameters()
        self.w_h.reset_parameters()

    def decode(self, h, mode,neg_samples=1):
        num_edge = h[0].shape[0] // (neg_samples + 2)
        if isinstance(h, tuple):
            if mode == 'pos':
                # pos instancs
                emb_in = h[0][:num_edge, :]
                emb_out = h[0][num_edge:2 * num_edge, :]
                emb_in_e = h[1][:num_edge, :]
                emb_out_e = h[1][num_edge:2 * num_edge, :]
            else:
                # neg instance
                emb_in = h[0][:num_edge, :].tile(neg_samples, 1)
                emb_out = h[0][2 * num_edge:, :]
                emb_in_e = h[1][:num_edge, :].tile(neg_samples, 1)
                emb_out_e = h[1][2 * num_edge:, :]

            "compute hyperbolic dist"
            #sqdist_h = ball.dist(emb_in, emb_out) + 1e-15
            emb_in = ball.logmap0(emb_in)
            emb_out = ball.logmap0(emb_out)
            sqdist_h = torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
            probs_h = self.dc.forward(sqdist_h)
            # probs_h = 1/ (sqdist_h + 1)

            "compute dist in Euclidean"
            sqdist_e = torch.sqrt((emb_in_e - emb_out_e).pow(2).sum(dim=-1) + 1e-15)
            probs_e = self.dc.forward(sqdist_e)
            # probs_e = 1/ (sqdist_e + 1)

            # print(sqdist_e, sqdist_h)
            # print(probs_e, probs_h)
            # print(h[0], h[1])
            # print(emb_in, emb_out)
            # print(emb_in_e,emb_out_e)
            # print('-----weight')
            # print(self.w_h.weight, self.w_h.bias)
            # print(self.w_e.weight, self.w_e.bias)
            # sub
            w_h = torch.sigmoid(self.w_h(emb_in - emb_out).view(-1))
            w_e = torch.sigmoid(self.w_e(emb_in_e - emb_out_e).view(-1))
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
            # print('------w')
            # print(w_h, w_e, w)
            w = F.normalize(w, p=1, dim=-1)

            probs = w[:, 0] * probs_h + w[:, 1] * probs_e

            assert torch.min(probs) >= 0
            assert torch.max(probs) <= 1

        return probs

    def forward(self, h, neg_samples=1):
        pos_scores = self.decode(h, mode='pos', neg_samples=neg_samples)
        neg_scores = self.decode(h, mode='neg', neg_samples=neg_samples)

        return pos_scores, neg_scores

class LinkDecoder_new(nn.Module):
    """
    Base model for link prediction task.
    """

    def __init__(self, dim_out):
        super(LinkDecoder_new, self).__init__()
        self.dc = FermiDiracDecoder()
        self.w_e = nn.Linear(dim_out, 1, bias=False)
        # self.w_h = nn.Linear(dim_out, 1, bias=False)
        self.fc_src = nn.Linear(dim_out, dim_out)
        self.fc_dst = nn.Linear(dim_out, dim_out)
        self.fc_out = nn.Linear(dim_out, 1)
        self.drop_e = 0
        self.drop_h = 0
        self.reset_param()

    def reset_param(self):
        self.w_e.reset_parameters()
        # self.w_h.reset_parameters()

    def decode(self, h, mode,neg_samples=1):
        num_edge = h[0].shape[0] // (neg_samples + 2)
        if isinstance(h, tuple):
            if mode == 'pos':
                # pos instancs
                emb_in = h[0][:num_edge, :]
                emb_out = h[0][num_edge:2 * num_edge, :]
                emb_in_e = self.fc_src(h[1][:num_edge, :])
                emb_out_e = self.fc_dst(h[1][num_edge:2 * num_edge, :])
            else:
                # neg instance
                emb_in = h[0][:num_edge, :].tile(neg_samples, 1)
                emb_out = h[0][2 * num_edge:, :]
                emb_in_e = self.fc_src(h[1][:num_edge, :]).tile(neg_samples, 1)
                emb_out_e = self.fc_dst(h[1][2 * num_edge:, :])

            "compute hyperbolic dist"
            sqdist_h = ball.dist(emb_in, emb_out)
            # emb_in = ball.logmap0(emb_in)
            # emb_out = ball.logmap0(emb_out)
            # sqdist_h = torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
            probs_h = self.dc.forward(sqdist_h)

            "compute dist in Euclidean"
            h_edge = torch.nn.functional.relu(emb_in_e + emb_out_e)
            probs_e = torch.sigmoid(self.fc_out(h_edge))

            # sub
            w_e = torch.sigmoid(self.w_e(emb_in_e + emb_out_e).view(-1))
            w_h = 1 - w_e
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
            # print('------w')
            # print(w_h, w_e, w)
            # w = F.normalize(w, p=1, dim=-1)

            # probs = w[-1, 0] * probs_h + w[-1, 1] * probs_e
            probs = probs_h

            assert torch.min(probs) >= 0
            assert torch.max(probs) <= 1
        # else:
        #     emb_in = h[idx[:, 0], :]
        #     emb_out = h[idx[:, 1], :]
        #     sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        #     assert torch.max(sqdist) >= 0
        #     probs = self.dc.forward(sqdist)

        return probs

    def forward(self, h, neg_samples=1):
        pos_scores = self.decode(h, mode='pos', neg_samples=neg_samples)
        neg_scores = self.decode(h, mode='neg', neg_samples=neg_samples)

        return pos_scores, neg_scores
