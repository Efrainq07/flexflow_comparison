import timm.models.vision_transformer as vit
import flexflow.torch.fx as fx
import torch.nn as nn

model = vit.vit_base_patch32_224_in21k()
model = nn.Sequential(model,nn.Flatten(),nn.Linear(21843,1000))
fx.torch_to_flexflow(model, "vit_base_patch32_224_in21k.ff")
