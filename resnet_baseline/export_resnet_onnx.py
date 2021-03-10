import classy_vision.models.regnet as rgn
import torch

model = rgn.RegNetX32gf()
model = nn.Sequential(model,nn.Flatten(),nn.Linear(2520*7*7,1000))
