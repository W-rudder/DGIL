import torch
import dgl
from memorys import *
from layers import *
from hyper_layers import *
import manifolds
from geoopt import ManifoldParameter

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        self.use_fusion = train_param['use_fusion']
        self.device = train_param['device']
        if 't_c' in gnn_param:
            curvatures = [torch.nn.Parameter(torch.Tensor([1.])) for _ in range(gnn_param['layer'] + 1)]
            self.curvatures = curvatures
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], self.device, combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], self.device, combined=False)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        elif gnn_param['arch'] == 'GIL':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
                self.layers['l0h' + str(h) + 'hyper'] = HGATLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], curvatures[0], curvatures[1], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
                    self.layers['l' + str(l) + 'h' + str(h) + 'hyper'] = HGATLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], curvatures[l], curvatures[l+1], combined=False)
        elif gnn_param['arch'] == 'GIL_Lorentz':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out1'], self.device, combined=combined)
                self.layers['l0h' + str(h) + 'hyper'] = LorentzLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out1'], curvatures[0], curvatures[1], self.device, combined=combined, negative_slope=0)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out1'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out2'], self.device, combined=False)
                    self.layers['l' + str(l) + 'h' + str(h) + 'hyper'] = LorentzLayer(gnn_param['dim_out1'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out2'], curvatures[l], curvatures[l+1], self.device, combined=False, project=True)
        else:
            raise NotImplementedError
        
        if gnn_param['arch'] == 'GIL':
            self.edge_predictor = LinkDecoder_new(gnn_param['dim_out'], curvatures[-1])
            print(curvatures)
        elif  gnn_param['arch'] == 'GIL_Lorentz':
            self.edge_predictor = LinkDecoder_Lorentz(gnn_param['dim_out2'], curvatures[-1])
        else:
            self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])

        if gnn_param['arch'] == 'GIL' or gnn_param['arch'] == 'GIL_Lorentz':
            self.h_fusion = HFusion(train_param['dropout'])
            self.e_fusion = EFusion(train_param['dropout'])
    
    def forward(self, mfgs, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                    rst_h = self.layers['l' + str(l) + 'h' + str(h) + 'hyper'](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                        if self.use_fusion:
                            f_h = self.h_fusion(rst_h, rst, project=False)
                            f_e = self.e_fusion(rst_h, rst, project=False)
                            mfgs[l + 1][h].srcdata['hyper'] = f_h
                            mfgs[l + 1][h].srcdata['h'] = f_e
                        else:
                            mfgs[l + 1][h].srcdata['h'] = rst
                            mfgs[l + 1][h].srcdata['hyper'] = rst_h
                    else:
                        mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                        if self.use_fusion:
                            rst_h = self.h_fusion(rst_h, rst, project=True)
                            rst = self.e_fusion(rst_h, rst, project=True)      
                        out.append((rst_h, rst))
                    else:
                        out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out, neg_samples=neg_samples)

    def get_emb(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                    rst_h = self.layers['l' + str(l) + 'h' + str(h) + 'hyper'](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                        if self.use_fusion:
                            f_h = self.h_fusion(rst_h, rst, project=False)
                            f_e = self.e_fusion(rst_h, rst, project=False)
                            mfgs[l + 1][h].srcdata['hyper'] = f_h
                            mfgs[l + 1][h].srcdata['h'] = f_e
                        else:
                            mfgs[l + 1][h].srcdata['h'] = rst
                            mfgs[l + 1][h].srcdata['hyper'] = rst_h
                    else:
                        mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                        if self.use_fusion:
                            rst_h = self.h_fusion(rst_h, rst, project=True)
                            rst = self.e_fusion(rst_h, rst, project=True)      
                        out.append((rst_h, rst))
                    else:
                        out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return torch.cat(out, dim=-1)

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class, c):
        super(NodeClassificationModel, self).__init__()
        self.dim_in = dim_in // 2
        self.c = c
        self.fc1 = torch.nn.Linear(self.dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

        self.fc_h = HypLinear(self.dim_in, dim_hid, self.c, dropout=0., bias=True)
        self.act = HypAct(self.c, self.c, 'relu')
        self.fc_h_c = torch.nn.Linear(dim_hid, num_class)

        self.w_h = torch.nn.Linear(self.dim_in, 1)
        self.w_e = torch.nn.Linear(self.dim_in, 1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x_h, x_e = x[:, :self.dim_in], x[:, self.dim_in:]
        x_e = self.fc1(x_e)
        x_e_f = torch.nn.functional.relu(x_e)
        x_e = self.fc2(x_e_f)
        prob_e = self.softmax(x_e)

        x_h = self.fc_h(x_h)
        x_h_f = self.act(x_h)
        x_h = pmath.logmap0(x_h_f, k=-self.c)
        prob_h = self.softmax(self.fc_h_c(x_h))

        '''Prob. Assembling'''
        w_h = torch.sigmoid(self.w_h(pmath.logmap0(x_h_f, k=-self.c)))
        w_e = torch.sigmoid(self.w_e(x_e_f))

        w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        probs = w[:, 0].view(-1, 1) * prob_h + w[:, 1].view(-1, 1) * prob_e
        # probs = prob_h

        return probs

class LorentzDecoder(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class, c, use_bias):
        super(LorentzDecoder, self).__init__()
        self.manifold = manifolds.Lorentz()
        self.dim_in = dim_in // 2
        self.num_class = num_class
        self.c = c
        self.use_bias = use_bias
        self.cls = ManifoldParameter(self.manifold.random_normal((self.num_class, dim_hid), std=1./math.sqrt(dim_hid)), manifold=self.manifold)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_class))

        self.fc_h1 = LorentzLinear(self.manifold, self.dim_in, dim_hid)
        self.fc_h2 = LorentzLinear(self.manifold, dim_hid, dim_hid)
        self.fc1 = torch.nn.Linear(self.dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

        self.w_h = torch.nn.Linear(dim_hid, 1)
        self.w_e = torch.nn.Linear(dim_hid, 1)

        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x_h, x_e = x[:, :self.dim_in], x[:, self.dim_in:]
        x_e = self.fc1(x_e)
        x_e_f = torch.nn.functional.relu(x_e)
        x_e = self.fc2(x_e_f)
        prob_e = self.softmax(x_e)

        x_h = self.fc_h1(x_h)
        dist = (2 + 2 * self.manifold.cinner(x_h, self.cls)) + self.bias
        prob_h = self.softmax(dist)

        '''Prob. Assembling'''
        w_h = torch.sigmoid(self.w_h(self.manifold.logmap0(x_h)))
        w_e = torch.sigmoid(self.w_e(x_e_f))
        # w_h = torch.tensor([1.])
        # w_e = torch.tensor([1.])

        w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        probs = w[:, 0].view(-1, 1) * prob_h + w[:, 1].view(-1, 1) * prob_e

        return prob_h, prob_e, probs
    
class GeneralClsModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(GeneralClsModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        self.use_fusion = train_param['use_fusion']
        if 't_c' in gnn_param:
            curvatures = [torch.nn.Parameter(torch.Tensor([1.])) for _ in range(gnn_param['layer'] + 1)]
            self.curvatures = curvatures
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        elif gnn_param['arch'] == 'GIL':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
                self.layers['l0h' + str(h) + 'hyper'] = HGATLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], curvatures[0], curvatures[1], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
                    self.layers['l' + str(l) + 'h' + str(h) + 'hyper'] = HGATLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], curvatures[l], curvatures[l+1], combined=False)
        elif gnn_param['arch'] == 'GIL_Lorentz':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
                self.layers['l0h' + str(h) + 'hyper'] = LorentzLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], curvatures[0], curvatures[1], combined=combined, negative_slope=0)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
                    self.layers['l' + str(l) + 'h' + str(h) + 'hyper'] = LorentzLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], curvatures[l], curvatures[l+1], combined=False, project=True)
        else:
            raise NotImplementedError
        
        if gnn_param['arch'] == 'GIL':
            self.edge_predictor = LinkDecoder_new(gnn_param['dim_out'], curvatures[-1])
            print(curvatures)
        elif  gnn_param['arch'] == 'GIL_Lorentz':
            self.edge_predictor =  LorentzDecoder(gnn_param['dim_out'] * 2, gnn_param['dim_cls'], gnn_param['num_cls'], torch.tensor([1.0]), True)
        else:
            self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])

        if gnn_param['arch'] == 'GIL' or gnn_param['arch'] == 'GIL_Lorentz':
            self.h_fusion = HFusion(train_param['dropout'])
            self.e_fusion = EFusion(train_param['dropout'])
    
    def forward(self, mfgs, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                    rst_h = self.layers['l' + str(l) + 'h' + str(h) + 'hyper'](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                        if self.use_fusion:
                            f_h = self.h_fusion(rst_h, rst, project=False)
                            f_e = self.e_fusion(rst_h, rst, project=False)
                            mfgs[l + 1][h].srcdata['hyper'] = f_h
                            mfgs[l + 1][h].srcdata['h'] = f_e
                        else:
                            mfgs[l + 1][h].srcdata['h'] = rst
                            mfgs[l + 1][h].srcdata['hyper'] = rst_h
                    else:
                        mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    if self.gnn_param['arch'] == 'GIL' or self.gnn_param['arch'] == 'GIL_Lorentz':
                        if self.use_fusion:
                            rst_h = self.h_fusion(rst_h, rst, project=True)
                            rst = self.e_fusion(rst_h, rst, project=True)      
                        out.append((rst_h, rst))
                    else:
                        out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        out = torch.cat(out, dim=-1)
        return self.edge_predictor(out)