import torch

state = torch.load("model.pth", map_location="cpu")

for k, v in state.items():
    print(k, v.shape)