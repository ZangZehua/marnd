import torch
from torch import nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        for p in self.parameters():
            p.requires_grad = False
        self.fc2 = nn.Linear(2, 1)

m = M()
for p in m.parameters():
    print(p)