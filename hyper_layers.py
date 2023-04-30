"""Hyperbolic layers."""
import math
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from geoopt.manifolds.stereographic import PoincareBall
from dgl.nn.pytorch.utils import Identity
from layers import TimeEncode

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
        self.ball = PoincareBall(c_in)
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
        self.attn_l = nn.Linear(self._in_src_feats, 1, bias=True)
        self.attn_r = nn.Linear(self._in_dst_feats, 1, bias=True)

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
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time = torch.cat([torch.zeros(b.num_dst_nodes(), dtype=torch.float32).cuda(), b.edata['dt']])
            time_feat = self.time_enc(time)
        
        # 1. 节点特征就只考虑本来的和时间特征，src是所有的特征，dst是src[:b.num_dst_nodes()]
        # 2. edge特征用来做attn
        # 输入的time是e， node_feat 是h-》node_feat log后拼上时间做attn
        if self.dim_node_feat == 0:
            src_feat = self.ball.projx(time_feat)
        else:
            src_feat = torch.cat([time_feat, self.ball.logmap0(b.srcdata['hyper'])], dim=1)
            src_feat = self.ball.projx(src_feat)
        h_src = self.feat_drop(src_feat)
        h_dst = self.feat_drop(src_feat)[:b.num_dst_nodes()]

        feat_src = self.fc_src(h_src)
        feat_dst = self.fc_dst(h_dst)

        feat_src_e = self.ball.logmap0(feat_src).view(
            -1, self.num_head, self._out_feats)
        feat_dst_e = self.ball.logmap0(feat_dst).view(
            -1, self.num_head, self._out_feats)

        el = self.attn_l(feat_src_e)
        er = self.attn_r(feat_dst_e)
        assert er.dim() == 3
        b.srcdata.update({"ft": feat_src_e, "el": el})
        b.dstdata.update({"er": er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        b.apply_edges(dgl.u_add_v("el", "er", "e"))
        e = self.leaky_relu(b.edata.pop("e"))
        # compute softmax
        b.edata["a"] = self.attn_drop(dgl.ops.edge_softmax(b, e))
        # message passing
        b.update_all(dgl.function.u_mul_e('ft', 'a', 'm'),
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
            return rst

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.ball = PoincareBall(self.c)
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.ball.mobius_matvec(drop_weight, x)
        assert self.ball.projx(res) == res
        if self.use_bias:
            bias = self.bias.view(1, -1)
            hyp_bias = self.ball.expmap0(bias)
            assert self.ball.projx(hyp_bias) == hyp_bias
            res = self.ball.mobius_add(res, hyp_bias)
            assert self.ball.projx(res) == res
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
        self.ball_in = PoincareBall(c_in)
        self.ball_out = PoincareBall(c_out)
        self.act = getattr(F, act)

    def forward(self, x):
        xt = self.act(self.ball_in.logmap0(x))
        # xt = self.ball_out.proju(xt)
        return self.ball_out.projx(self.ball_out.expmap0(xt, project=False))

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
