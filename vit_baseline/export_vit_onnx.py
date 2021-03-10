import torch
import onnx
import timm.models.vision_transformer as vit
from torch.onnx import TrainingMode
import torch.nn as nn

model = vit.vit_base_patch32_224_in21k()
model = nn.Sequential(model,nn.Flatten(),nn.Linear(21843,1000))
input = torch.randn(64, 3, 224, 224)
torch.onnx.export(model, (input), "vit_base_patch32_224_in21k.onnx", export_params=False, training=TrainingMode.TRAINING)
onnx_model = onnx.load("vit_base_patch32_224_in21k.onnx")
