import flexflow.torch.fx as fx
import torch.nn as nn
import torch

class Permute(nn.Module):
  def __init__(self):
    super(Permute, self).__init__()

  def forward(self, x):
    x.permute(0,2,3,1)
    return x

model = nn.Sequential(Permute(),nn.Flatten(229*229*3,10))
fx.torch_to_flexflow(model, "scalar_multiply.ff")
