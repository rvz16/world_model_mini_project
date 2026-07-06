"""CNN encoder / decoder and reward head for the world model.

Encoder/decoder are resolution-adaptive: they use exact halving/doubling conv
blocks (kernel 4, stride 2, padding 1) so any power-of-two image size works.
For a 128x128 input that means 5 down/up blocks (128->64->32->16->8->4).
Kept compact so training is feasible on MPS/CPU.
"""
import math
import torch
import torch.nn as nn


def _num_blocks(image_size: int) -> int:
    n = int(math.log2(image_size // 4))
    assert 4 * (2 ** n) == image_size, "image_size must be a power of two >= 8"
    return n


class ConvEncoder(nn.Module):
    def __init__(self, image_size: int, embed_dim: int = 256, depth: int = 32):
        super().__init__()
        n = _num_blocks(image_size)
        chans = [3]
        layers = []
        for i in range(n):
            out = depth * min(2 ** i, 8)
            layers += [nn.Conv2d(chans[-1], out, 4, 2, 1), nn.ReLU()]
            chans.append(out)
        self.net = nn.Sequential(*layers)
        self.flat_dim = chans[-1] * 4 * 4
        self.fc = nn.Linear(self.flat_dim, embed_dim)

    def forward(self, x):
        h = self.net(x - 0.5)                 # center to [-0.5, 0.5]
        return self.fc(h.reshape(h.shape[0], -1))


class ConvDecoder(nn.Module):
    def __init__(self, feat_dim: int, image_size: int, depth: int = 32):
        super().__init__()
        n = _num_blocks(image_size)
        outs = [depth * min(2 ** i, 8) for i in range(n)]
        rev = outs[::-1]                      # e.g. [256,256,128,64,32]
        self.start_ch = rev[0]
        self.fc = nn.Linear(feat_dim, self.start_ch * 4 * 4)
        layers = []
        prev = rev[0]
        for i in range(1, n):
            layers += [nn.ConvTranspose2d(prev, rev[i], 4, 2, 1), nn.ReLU()]
            prev = rev[i]
        layers += [nn.ConvTranspose2d(prev, 3, 4, 2, 1)]   # final -> 3 channels
        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        h = self.fc(feat).reshape(-1, self.start_ch, 4, 4)
        return self.net(h) + 0.5             # undo encoder centering


class RewardHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)
