import flexflow.torch.fx as fx
import torch.nn as nn
import torch
class Division(nn.Module):
  def __init__(self, scalar):
    super(Division, self).__init__()
    self.scalar = scalar

  def forward(self, x):
    x = x//self.scalar
    return x

model = nn.Sequential(Division(2.0),nn.Flatten(229*229*3,10))
fx.torch_to_flexflow(model, "scalar.ff")
