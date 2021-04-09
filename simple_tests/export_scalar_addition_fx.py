import flexflow.torch.fx as fx
import torch.nn as nn
class Addition(nn.Module):
  def __init__(self, scalar):
    super(Addition, self).__init__()
    self.scalar = scalar

  def forward(self, x):
    x = x+self.scalar
    return x

model = nn.Sequential(Addition(2.0),nn.Flatten(229*229*3,10))
fx.torch_to_flexflow(model, "scalar_addition.ff")
