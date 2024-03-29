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
import manifolds
manifold = manifolds.Lorentz()

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

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, device, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.device = device
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
            self.linear_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.linear_kv = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_out, dim_out)
            self.w_k = torch.nn.Linear(dim_out, dim_out)
            self.w_v = torch.nn.Linear(dim_out, dim_out)

        self.w_out = torch.nn.Linear(dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out)).to(self.device)
        self_loop = torch.tensor([i for i in range(b.num_dst_nodes())], device=self.device)
        num_dst_nodes = b.num_dst_nodes()
        num_src_nodes = b.num_src_nodes()
        num_edges = len(b.edges()[0])

        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32).to(self.device))
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out)).to(self.device)
            K = torch.zeros((b.num_edges(), self.dim_out)).to(self.device)
            V = torch.zeros((b.num_edges(), self.dim_out)).to(self.device)
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
                Q = torch.ones((b.num_edges(), self.dim_out)).to(self.device)
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
                Q = zero_time_feat
                K = torch.cat([b.edata['f'], time_feat], dim=1)
                # V = torch.cat([b.edata['f'], time_feat], dim=1)
            elif self.dim_edge_feat == 0:
                Q = torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1)
                K = torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1)
                # V = torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1)
            else:
                Q = torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1)
                K = torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1)
                # V = torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1)
            Q_ori = self.linear_q(Q)
            Q = torch.cat([Q_ori[b.edges()[1]], Q_ori])
            K_ori = self.linear_kv(K)
            # print(self.linear_q.weight.weight, self.linear_q.weight.bias)
            # print(self.linear_kv.weight.weight, self.linear_kv.weight.bias)
            K = torch.cat([K_ori, Q_ori], dim=0)
            Q = self.w_q(Q)
            # do not modify
            V = self.w_v(K)
            K = self.w_k(K)
            

            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))

            # b.add_edges(self_loop, self_loop)
            c = dgl.create_block((torch.cat([b.edges()[0], self_loop], dim=-1), torch.cat([b.edges()[1], self_loop], dim=-1)), num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes)

            att = dgl.ops.edge_softmax(c, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)

            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            # b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1])), V], dim=0)
            c.srcdata['v'] = torch.cat([V[num_edges:], V[:num_edges]], dim=0)
            c.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            # rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
            rst = c.dstdata['h']
        else:
            rst = c.dstdata['h']
        # rst = self.w_out(rst)
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
        nn.init.constant_(self.att, 0.5)

    def forward(self, x_h, x_e, project):
        if not project:
            x_h = manifold.expmap0(x_h)
        x_e = manifold.expmap0(x_e)
        dist = manifold.dist(x_h, x_e) * self.att
        x_e = dist.view([-1, 1]) * x_e
        x_h = x_h + x_e
        denom = (-manifold.inner(x_h, x_h, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        x_h = x_h / denom
        # x_e = F.dropout(x_e, p=self.drop, training=self.training)
        if not project:
            assert torch.isnan(x_h).int().sum() <= 0
            assert torch.isnan(manifold.logmap0(x_h)).int().sum() <= 0
            return manifold.logmap0(x_h)
        else:
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

    def forward(self, x_h, x_e, project):
        if not project:
            dist = (x_h - x_e).pow(2).sum(dim=-1) * self.att
            x_h = dist.view([-1, 1]) * x_h
        else:
            assert torch.isnan(manifold.logmap0(x_h)).int().sum() <= 0
            dist = (manifold.logmap0(x_h) - x_e).pow(2).sum(dim=-1) * self.att
            x_h = dist.view([-1, 1]) * manifold.logmap0(x_h)
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
        # probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.)
        probs = 1. / (torch.exp(((dist - self.r) / self.t).clamp_max(50.)) + 1.0)
        return probs

class LinkDecoder(nn.Module):
    """
    Base model for link prediction task.
    """

    def __init__(self, dim_out, c):
        super(LinkDecoder, self).__init__()
        self.dc = FermiDiracDecoder()
        self.w_e = nn.Linear(dim_out, 1, bias=False)
        self.w_h = nn.Linear(1, 1, bias=False)
        self.fc_src = nn.Linear(dim_out, dim_out)
        self.fc_dst = nn.Linear(dim_out, dim_out)
        self.fc_out = nn.Linear(dim_out, 1)
        self.drop_e = 0
        self.drop_h = 0
        self.c = c
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
            # sqdist_h = torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
            probs_h = self.dc.forward(sqdist_h)

            "compute dist in Euclidean"
            h_edge = torch.nn.functional.relu(emb_in_e + emb_out_e)
            probs_e = torch.sigmoid(self.fc_out(h_edge).view(-1))

            # sub
            w_h = torch.sigmoid(self.w_h(sqdist_h.view(-1, 1)).view(-1))
            w_e = torch.sigmoid(self.w_e(h_edge).view(-1))
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

    def decode(self, h, mode, neg_samples=1):
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
            probs_e = torch.sigmoid(self.fc_out(h_edge).view(-1))

            # sub
            w_e = torch.sigmoid(self.w_e(emb_in_e + emb_out_e).view(-1))
            w_h = 1 - w_e
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
            # print('------w')
            # print(w_h, w_e, w)
            # w = F.normalize(w, p=1, dim=-1)

            probs = w[:, 0] * probs_h + w[:, 1] * probs_e
            # probs = probs_h

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

class LinkDecoder_Lorentz(nn.Module):
    """
    Base model for link prediction task.
    """

    def __init__(self, dim_out, c, manifold_name='Lorentz'):
        super(LinkDecoder_Lorentz, self).__init__()
        self.manifold = getattr(manifolds, manifold_name)()
        self.dc = FermiDiracDecoder()
        self.w_e = nn.Linear(dim_out * 2, 1)
        # self.w_e = nn.Linear(dim_out, 1)
        # self.w_h = nn.Linear(dim_out * 2, 1)
        self.fc_src = nn.Linear(dim_out * 2, dim_out)
        self.fc_src_h = nn.Linear(dim_out * 2, dim_out)
        # self.fc_dst = nn.Linear(dim_out, dim_out)
        self.fc_out = nn.Linear(dim_out, 1)
        self.fc_out_h = nn.Linear(dim_out, 1)
        self.drop_e = 0
        self.drop_h = 0
        self.c = torch.tensor([1.0])
        self.loss = torch.nn.BCELoss()
        self.reset_param()

    def reset_param(self):
        self.w_e.reset_parameters()
        # self.w_h.reset_parameters()
        self.fc_src.reset_parameters()
        self.fc_src_h.reset_parameters()
        self.fc_out.reset_parameters()
        self.fc_out_h.reset_parameters()

    def decode(self, h, mode, neg_samples=1):
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
            # sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            # # probs_h = torch.sigmoid(sqdist.view(-1))
            # probs_h = self.dc.forward(sqdist)
            emb_in, emb_out = self.manifold.logmap0(emb_in), self.manifold.logmap0(emb_out)
            h_edge = torch.nn.functional.relu(self.fc_src_h(torch.cat([emb_in, emb_out], dim=-1)))
            probs_h = torch.sigmoid(self.fc_out_h(h_edge).view(-1))

            "compute dist in Euclidean"
            e_edge = torch.nn.functional.relu(self.fc_src(torch.cat([emb_in_e, emb_out_e], dim=-1)))
            probs_e = torch.sigmoid(self.fc_out(e_edge).view(-1))

            # sub
            w_e = torch.sigmoid(self.w_e(torch.cat([emb_in_e, emb_out_e], dim=-1)).view(-1))
            w_h = 1 - w_e
            # w_h = torch.sigmoid(self.w_h(torch.cat([emb_in, emb_out], dim=-1)).view(-1))
            # w_e = torch.tensor([0.5])
            # w_h = torch.tensor([0.5])
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
            # w = F.normalize(w, p=1, dim=-1)

            probs = w[:, 0] * probs_h + w[:, 1] * probs_e
            # probs = probs_h
            assert torch.min(probs) >= 0
            assert torch.max(probs) <= 1

            # embeddings_tan = self.manifold.logmap0(emb_in)
            # u_norm = ((1e-6 + embeddings_tan.pow(2).sum(dim=1)).mean()).sqrt()
            u_norm = 1
        # else:
        #     emb_in = h[idx[:, 0], :]
        #     emb_out = h[idx[:, 1], :]
        #     sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        #     assert torch.max(sqdist) >= 0
        #     probs = self.dc.forward(sqdist)
        # print(sqdist)
        # return -sqdist
        return probs, u_norm

    def forward(self, h, neg_samples=1):
        pos_scores, pos_regular_loss = self.decode(h, mode='pos', neg_samples=neg_samples)
        neg_scores, neg_regular_loss = self.decode(h, mode='neg', neg_samples=neg_samples)
        loss = self.loss(pos_scores, torch.ones_like(pos_scores)) + self.loss(neg_scores, torch.zeros_like(neg_scores))

        return pos_scores, neg_scores, loss