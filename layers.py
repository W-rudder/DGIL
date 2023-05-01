import torch
import dgl
import math
import numpy as np
from torch import nn
from dgl.nn.pytorch.utils import Identity
from geoopt.manifolds.stereographic import PoincareBall
import torch.nn.functional as F
import geoopt
import geoopt.manifolds.stereographic.math as pmath

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
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
            return torch.zeros((b.num_dst_nodes(), self.dim_out)).cuda()
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32).cuda())
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out)).cuda()
            K = torch.zeros((b.num_edges(), self.dim_out)).cuda()
            V = torch.zeros((b.num_edges(), self.dim_out)).cuda()
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
                Q = torch.ones((b.num_edges(), self.dim_out)).cuda()
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
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1])).cuda(), V], dim=0)
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
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.)
        return probs

class LinkDecoder(nn.Module):
    """
    Base model for link prediction task.
    """

    def __init__(self, dim_out, c):
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

    def __init__(self, dim_out, c):
        super(LinkDecoder_new, self).__init__()
        self.dc = FermiDiracDecoder()
        self.w_e = nn.Linear(dim_out, 1, bias=False)
        # self.w_h = nn.Linear(dim_out, 1, bias=False)
        self.fc_src = nn.Linear(dim_out, dim_out)
        self.fc_dst = nn.Linear(dim_out, dim_out)
        self.fc_out = nn.Linear(dim_out, 1)
        self.drop_e = 0
        self.drop_h = 0
        self.c = c
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
            sqdist_h = pmath.dist(emb_in, emb_out, k=-self.c) ** 2
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
