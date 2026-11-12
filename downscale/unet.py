import torch
import torch.nn.functional as F
from torch import nn


def double_conv(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1, bias=False),
        nn.BatchNorm2d(b),
        nn.ReLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1, bias=False),
        nn.BatchNorm2d(b),
        nn.ReLU(inplace=True),
    )


def up(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class Unet(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        w = [base, base * 2, base * 4, base * 8]
        self.inc = double_conv(in_ch, w[0])
        self.d1 = double_conv(w[0], w[1])
        self.d2 = double_conv(w[1], w[2])
        self.d3 = double_conv(w[2], w[3])
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.Conv2d(w[3], w[2], 3, padding=1)
        self.c3 = double_conv(w[2] * 2, w[2])
        self.u2 = nn.Conv2d(w[2], w[1], 3, padding=1)
        self.c2 = double_conv(w[1] * 2, w[1])
        self.u1 = nn.Conv2d(w[1], w[0], 3, padding=1)
        self.c1 = double_conv(w[0] * 2, w[0])
        self.out = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.d1(self.pool(x0))
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        y = self.c3(torch.cat([self.u3(up(x3)), x2], 1))
        y = self.c2(torch.cat([self.u2(up(y)), x1], 1))
        y = self.c1(torch.cat([self.u1(up(y)), x0], 1))
        return self.out(y)


def pad_to(x, k=8):
    h, w = x.shape[-2:]
    ph = (-h) % k
    pw = (-w) % k
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w
