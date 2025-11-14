import torch
import torch.nn as nn
import torch.nn.functional as F
from .prompts import Adapter

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden=256, heads=4, dropout=0.1, use_adapters=True, adapter_bottleneck=64):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden*4),
            nn.GELU(),
            nn.Linear(hidden*4, hidden),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.adapter = Adapter(hidden, bottleneck=adapter_bottleneck, dropout=dropout) if use_adapters else nn.Identity()

    def forward(self, Hd, Hp, P=None, need_weights=False):
        """Run one block for a *single* sample (no batch).
        Hd: (Nd, H), Hp: (Lp, H), P: (m, H) or None. We follow Eqs. 7–10.
        Returns: Hd', Hp', (optional attn weights).
        """
        if Hd.ndim != 2 or Hp.ndim != 2:
            raise ValueError('Expected unbatched Hd/Hp')
        X = torch.cat([Hd, Hp], dim=0)  # (Sx, H)
        Q = self.ln1(X).unsqueeze(1)    # (Sx, 1, H) -> MHA expects (S, B, H) with batch_first=False
        if P is not None:
            KV = torch.cat([X, P], dim=0).unsqueeze(1) # (Sx+m,1,H)
        else:
            KV = X.unsqueeze(1)                        # (Sx,1,H)

        out, attn = self.mha(Q, KV, KV, need_weights=need_weights)
        out = out.squeeze(1)                          # (Sx, H)
        X2 = X + out
        X2 = X2 + self.ffn(self.ln2(X2))
        X2 = self.adapter(X2)
        Nd = Hd.size(0)
        Hd2, Hp2 = X2[:Nd, :], X2[Nd:, :]
        return Hd2, Hp2, attn if need_weights else None

class FusionStack(nn.Module):
    def __init__(self, layers=3, hidden=256, heads=4, dropout=0.1, use_adapters=True, adapter_bottleneck=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden, heads, dropout, use_adapters, adapter_bottleneck)
            for _ in range(layers)
        ])
    def forward(self, Hd, Hp, P=None, need_weights=False):
        attn_all = []
        for blk in self.blocks:
            Hd, Hp, attn = blk(Hd, Hp, P, need_weights=need_weights)
            if need_weights and attn is not None:
                attn_all.append(attn.detach().cpu())
        return Hd, Hp, attn_all
