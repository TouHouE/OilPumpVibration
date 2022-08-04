import torch
from torch import nn
from torch.nn import functional as F


class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(*[nn.Conv2d(3, 64, 7, 2, padding=3), nn.GELU(), nn.Conv2d(64, 128, 3, padding=1)])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.fc = nn.Linear(128 * 2 * 2, 2)

    def forward(self, x):
        x = self.head(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        return self.fc(x)