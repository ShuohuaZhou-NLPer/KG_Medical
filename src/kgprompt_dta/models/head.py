import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFuse(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Linear(hidden*3, 3)
        self.proj = nn.Linear(hidden*3, hidden)
    def forward(self, zd, zp, zP):
        x = torch.cat([zd, zp, zP], dim=-1)
        g = torch.softmax(self.gate(x), dim=-1)  # (B,3)
        # simple convex combination weights, but we still project concatenation
        out = self.proj(x)
        return out, g

class PredictionHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, z):
        return self.mlp(z)
