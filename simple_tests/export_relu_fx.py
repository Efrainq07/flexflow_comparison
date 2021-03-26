import flexflow.torch.fx as fx
import torch.nn as nn

model = nn.Sequential(nn.ReLU(),nn.Flatten(229*229*3,10))
fx.torch_to_flexflow(model, "relu.ff")
