import flexflow.torch.fx as fx
import torch.nn as nn
import torch

class CustomParam(nn.Module):
  def __init__(self):
    super(CustomParam, self).__init__()
    self.mat = nn.Parameter(torch.zeros(1,1))

  def forward(self, x):
    x = torch.matmul(self.mat,x)
    return x

model = nn.Sequential(CustomParam())
fx.torch_to_flexflow(model, "customParam.ff")
