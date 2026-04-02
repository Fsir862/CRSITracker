import torch
import torch.nn as nn


class RHead(nn.Module):
    def __init__(self, in_ch=1, hid1=32, hid2=16, with_stats=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hid1, 3, padding=1)
        self.conv2 = nn.Conv2d(hid1, hid2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.with_stats = with_stats
        self.relu = nn.ReLU(inplace=True)

        in_dim = hid2 + (4 if with_stats else 0)
        self.reg_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

    def forward(self, R, stats=None):
        x = self.relu(self.conv1(R))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)

        if self.with_stats and stats is not None:
            x = torch.cat([x, stats], dim=-1)

        return self.reg_mlp(x)