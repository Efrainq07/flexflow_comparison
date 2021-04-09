import flexflow.torch.fx as fx
import torch.nn as nn
import torch

class Reshape(nn.Module):
  def __init__(self):
    super(Reshape, self).__init__()

  def forward(self, x):
    x = torch.reshape(x,(x.shape[1],x.shape[2],x.shape[0],x.shape[3]))
    return x

model = nn.Sequential(Reshape(),nn.Flatten(229*229*3,10))
fx.torch_to_flexflow(model, "reshape.ff")
