from manifolds import Lorentz
from hyper_layers import LorentzLinear
import torch
m = Lorentz()
linear = LorentzLinear(m, 4, 6)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_h = torch.tensor([[1, 3, 2], [4, 6, 5]])

o = torch.zeros_like(x)
x = torch.cat([o[:, 0:1], x], dim=1)
x_h = torch.cat([o[:, 0:1], x_h], dim=1)

x = m.expmap0(x)
x_h = m.expmap0(x_h)
print(x, x_h)
x_ = m.logmap0(x)
x__ = m.logmap0back(x)
print(x_, x__)

y = linear(x)
print(y)
print(m.check_point_on_manifold(y))
y_ = m.logmap0(y)
print(y_)
print(m.expmap0(y_))