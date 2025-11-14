import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        y = self.down(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.up(y)
        return x + y

class StaticPrompt(nn.Module):
    def __init__(self, kg_dim, hidden, m_tokens):
        super().__init__()
        self.m = m_tokens
        self.proj = nn.Sequential(
            nn.Linear(kg_dim*3, hidden*m_tokens),
            nn.GELU(),
            nn.Linear(hidden*m_tokens, hidden*m_tokens)
        )
    def forward(self, ed, r, ep):
        # ed, r, ep: (B, Dk)
        B, Dk = ed.size(0), ed.size(1)
        x = torch.cat([ed, r, ep], dim=-1)  # (B, 3Dk)
        P = self.proj(x).view(B, self.m, -1)  # (B, m, H)
        return P

class DynamicPrompt(nn.Module):
    def __init__(self, in_dim, hidden, m_tokens):
        super().__init__()
        self.m = m_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden*m_tokens)
        )
    def forward(self, zd, zp, ed, r, ep):
        # zd,zp: (B,H) ; ed,r,ep: (B,Dk)
        x = torch.cat([zd, zp, ed, r, ep], dim=-1)
        P = self.mlp(x).view(x.size(0), self.m, -1)  # (B, m, H)
        return P
