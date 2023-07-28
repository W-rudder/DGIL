"""Hyperbolic layers."""
import math
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from geoopt.manifolds.stereographic import math as pmath
from dgl.nn.pytorch.utils import Identity
from layers import TimeEncode
import manifolds

class HGATLayer(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, c_in, c_out, combined=False, negative_slope=0.2, residual=False, activation=None, bias=True):
        super(HGATLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(att_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = bias
        self.activation = HypAct(c_in, c_out, 'relu')
        self.c_in = c_in
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        self._in_src_feats, self._in_dst_feats = dim_time + dim_node_feat, dim_time + dim_node_feat

        self._out_feats = self.dim_out // self.num_head

        self.fc_src = HypLinear(
            self._in_src_feats, self._out_feats * num_head, c_in, dropout, bias=bias
        )
        self.fc_dst = HypLinear(
            self._in_dst_feats, self._out_feats * num_head, c_in, dropout, bias=bias
        )
        self.fc_edge = HypLinear(
            dim_edge_feat, self._out_feats * num_head, c_in, dropout, bias=bias
        )
        self.attn_l = nn.Linear(self._out_feats, 1, bias=True)
        self.attn_r = nn.Linear(self._out_feats, 1, bias=True)

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
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, b, edge_weight=None, get_attention=False):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out)).cuda()
        if self.dim_time > 0:
            time = torch.cat([torch.zeros(b.num_dst_nodes(), dtype=torch.float32).cuda(), b.edata['dt']])
            time_feat = self.time_enc(time)
        
        # 1. 节点特征就只考虑本来的和时间特征，src是所有的特征，dst是src[:b.num_dst_nodes()]
        # 2. edge特征用来做attn
        # 输入的time是e， node_feat 是h-》node_feat log后拼上时间做attn
        if self.dim_node_feat == 0:
            src_feat = pmath.project(time_feat, k=-self.c_in)
        else:
            src_feat = torch.cat([time_feat, pmath.logmap0(b.srcdata['hyper'], k=-self.c_in)], dim=1)
            src_feat = pmath.project(src_feat, k=-self.c_in)
        h_src = self.feat_drop(src_feat)
        h_dst = self.feat_drop(src_feat)[:b.num_dst_nodes()]

        feat_src = self.fc_src(h_src)
        feat_dst = self.fc_dst(h_dst)

        feat_src_e = pmath.logmap0(feat_src, k=-self.c_in).view(
            -1, self.num_head, self._out_feats)
        feat_dst_e = pmath.logmap0(feat_dst, k=-self.c_in).view(
            -1, self.num_head, self._out_feats)

        el = self.attn_l(feat_src_e)
        er = self.attn_r(feat_dst_e)
        assert er.dim() == 3
        b.srcdata.update({"ft": feat_src_e, "el": el})
        b.dstdata.update({"er": er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        b.apply_edges(dgl.function.u_add_v("el", "er", "e"))
        e = self.leaky_relu(b.edata.pop("e"))
        # compute softmax
        b.edata["a"] = self.attn_drop(dgl.ops.edge_softmax(b, e))
        # message passing
        b.update_all(dgl.function.u_mul_e('ft', 'a', 'm'),
                            dgl.function.sum('m', 'ft'))
        rst = b.dstdata['ft'].view(-1, self.dim_out)
        rst = pmath.project(pmath.expmap0(rst, k=-self.c_in), k=-self.c_in)

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
            return rst

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, in_features, out_features, c, dropout, bias):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = pmath.mobius_matvec(drop_weight, x, k=-self.c)
        res = pmath.project(res, k=-self.c)
        if self.use_bias:
            bias = self.bias.view(1, -1)
            hyp_bias = pmath.expmap0(bias, k=-self.c)
            hyp_bias = pmath.project(hyp_bias, k=-self.c)
            res = pmath.mobius_add(res, hyp_bias, k=-self.c)
            res = pmath.project(res, k=-self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.act = getattr(F, act)

    def forward(self, x):
        xt = self.act(pmath.logmap0(x, k=-self.c_in))
        # xt = self.ball_out.proju(xt)
        return pmath.project(pmath.expmap0(xt, k=-self.c_out), k=-self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class LorentzLayer(nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, c_in, c_out, device, combined=False, negative_slope=0.2, manifold='Lorentz', residual=False, activation=None, bias=True, project=False):
        super(LorentzLayer, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = bias
        self.project = project
        self.device = device
        # self.activation = HypAct(c_in, c_out, 'relu')
        # self.bias = nn.Parameter(torch.zeros(()) + 20)
        self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(self.dim_out))
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if project:
            self.q_feats, self.kv_feats = dim_time + dim_node_feat, dim_time + dim_node_feat + dim_edge_feat
        else:
            self.q_feats, self.kv_feats = dim_time + dim_node_feat + 1, dim_time + dim_node_feat + dim_edge_feat + 1

        self._out_feats = self.dim_out // self.num_head

        self.linear_q = LorentzLinear(manifold, self.q_feats, self.dim_out, self.bias, dropout, nonlin=self.leaky_relu if negative_slope != 0 else None)
        self.linear_kv = LorentzLinear(manifold, self.kv_feats, self.dim_out, self.bias, dropout, nonlin=self.leaky_relu if negative_slope != 0 else None)

        self.w_q = LorentzLinear(manifold, self.dim_out, self.dim_out)
        self.w_k = LorentzLinear(manifold, self.dim_out, self.dim_out)
        self.w_v = LorentzLinear(manifold, self.dim_out, self.dim_out)

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out)).to(self.device)
        self_loop = [i for i in range(b.num_dst_nodes())]
        num_dst_nodes = b.num_dst_nodes()
        num_src_nodes = b.num_src_nodes()
        num_edges = len(b.edges()[0])

        time_feat = self.time_enc(b.edata['dt'])
        zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32)).to(self.device)

        if self.project:
            q_node_feat = b.srcdata['hyper'][:b.num_dst_nodes()]
            Q = self.manifold.expmap0(torch.cat([q_node_feat, zero_time_feat], dim=1))

            k_node_feat = b.srcdata['hyper'][b.num_dst_nodes():]
            if self.dim_edge_feat == 0:
                K = self.manifold.expmap0(torch.cat([k_node_feat, time_feat], dim=1))
            else:
                K = self.manifold.expmap0(torch.cat([k_node_feat, b.edata['f'], time_feat], dim=1))
        else:
            if self.dim_node_feat != 0 and self.dim_edge_feat != 0:
                Q = torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1)
                K = torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1)
            elif self.dim_node_feat != 0 and self.dim_edge_feat == 0:
                Q = torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1)
                K = torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1)
            elif self.dim_node_feat == 0 and self.dim_edge_feat != 0:
                Q = zero_time_feat
                K = torch.cat([b.edata['f'], time_feat], dim=1)
            else:
                Q = zero_time_feat
                K = time_feat
            qo = torch.zeros_like(Q)
            ko = torch.zeros_like(K)
            Q = self.manifold.expmap0(torch.cat([qo[:, 0:1], Q], dim=1))
            K = self.manifold.expmap0(torch.cat([ko[:, 0:1], K], dim=1))

        Q_ori = self.linear_q(Q)
        Q = torch.cat([Q_ori[b.edges()[1]], Q_ori])
        K_ori = self.linear_kv(K)
        # print(self.linear_q.weight.weight, self.linear_q.weight.bias)
        # print(self.linear_kv.weight.weight, self.linear_kv.weight.bias)
        K = torch.cat([K_ori, Q_ori], dim=0)
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(K)

        b.add_edges(self_loop, self_loop)
        c = dgl.create_block((b.edges()[0], b.edges()[1]), num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes)
        b = c
        dist = 2 + 2 * self.manifold.inner(None, Q, K, keepdim=True, dim=-1)
        attn = dist / self.scale
        attn = dgl.ops.edge_softmax(b, attn)
        V = V * attn

        b.srcdata['v'] = torch.cat([V[num_edges:], V[:num_edges]], dim=0)
        b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        rst = b.dstdata['h']
        denom = (-self.manifold.inner(None, rst, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        # print(rst)
        # print(torch.isnan(rst).int().sum(), torch.isnan(denom).int().sum())
        rst = rst / denom
        # for i in range(rst.shape[0]):
        #     # print(int(self.manifold.inner(None, rst[i])) == -1)
        #     if abs(self.manifold.inner(None, rst[i]).item() + 1.0) > 1e-3:
        #         print(self.manifold.inner(None, rst[i]).item())
        #         print(rst[i])
        #         print(self.project)
        #         print(torch.equal(self.manifold.inner(None, rst[i]), torch.tensor([-1.0])))
            # assert torch.equal(self.manifold.inner(None, rst[i]), torch.tensor([-1.0]))
            # assert self.manifold.check_point_on_manifold(rst[i])
        assert torch.isnan(rst).int().sum() <= 0
        
        if self.project:
            # rst = self.manifold.expmap0(rst)
            # for i in range(rst.shape[0]):
            #     assert self.manifold.check_point_on_manifold(rst[i])
            pass
        else:
            rst = self.manifold.logmap0(rst)
        return rst

class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)