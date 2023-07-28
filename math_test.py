from manifolds import Lorentz
from hyper_layers import LorentzLinear
import torch
import dgl

m = Lorentz()
linear = LorentzLinear(m, 4, 6)
x = torch.tensor([[0, 0, 0], [4, 5, 6]])
x_t = torch.tensor([0, 0, 1])
x_h = torch.tensor([[2, 1, 2], [1, 3, 1]])
# print(torch.unique(x))

o = torch.zeros_like(x)
x = torch.cat([o[:, 0:1], x], dim=1)
x_h = torch.cat([o[:, 0:1], x_h], dim=1)

x = m.expmap0(x)
x_h = m.expmap0(x_h)
x_t = m.expmap0(x_t)
print(x, x_h)
print(m.dist0(x_h))
x_ = m.logmap0(x)
x__ = m.logmap0back(x)
print(x_, x__)

# print(torch.unique(x))

# b = dgl.create_block(([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]), num_src_nodes=10, num_dst_nodes=5)
# print(b.edges(), b.num_nodes())
# b.add_edges([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
# c = dgl.create_block((b.edges()[0], b.edges()[1]), num_src_nodes=10, num_dst_nodes=5)
# b = c
# attn = dgl.ops.edge_softmax(b, torch.tensor([[1.], [1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.], [2.]]))
# print(b.edges())
# print(attn)
# V = attn
# V[0] = 0.01
# b.srcdata['v'] = V
# b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
# print(b.dstdata['h'])
# print(torch.tensor([-1.0000]).values)
# y = linear(x)
# print(y)
# print(m.check_point_on_manifold(y))
# y_ = m.logmap0(y)
# print(y_)
# print(m.expmap0(y_))
# import numpy as np
# l = torch.nn.Linear(4, 5)
# x = torch.tensor([1, 0])
# y = np.array([1, 1])
# label = np.array([1, 1])
# print(((np.array(x) == label) | (y == label)).astype(int).tolist())
# print(x)
# y = l(x)
# print(y)

# e = torch.nn.Embedding(10, 3)
# i = torch.LongTensor([1, 3, 3])
# x = e(i)
# y = torch.mean(2 * x)
# y.backward()
# print(e)