import flexflow.torch.fx as fx
import torch.nn as nn
import torch

class Transpose(nn.Module):
  def __init__(self):
    super(Transpose, self).__init__()

  def forward(self, x):
    x = torch.transpose(x,-1,-2)
    return x

model = nn.Sequential(Transpose(),nn.Flatten(229*229*3,10))
fx.torch_to_flexflow(model, "scalar_multiply.ff")
