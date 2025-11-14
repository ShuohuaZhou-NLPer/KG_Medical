import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Graph Convolution (GCN) ----------
def add_self_loops(edge_index, num_nodes):
    device = edge_index.device
    loop_index = torch.arange(0, num_nodes, device=device)
    loop_index = torch.stack([loop_index, loop_index], dim=0)
    return torch.cat([edge_index, loop_index], dim=1)

def gcn_norm(edge_index, num_nodes):
    # undirected normalization
    row, col = edge_index
    # add symmetric edges
    edge_index_sym = torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)
    # add self-loops
    edge_index_sym = add_self_loops(edge_index_sym, num_nodes)
    row, col = edge_index_sym
    deg = torch.zeros(num_nodes, device=edge_index_sym.device).index_add_(0, row, torch.ones(row.size(0), device=edge_index_sym.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index_sym, norm

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        ei, norm = gcn_norm(edge_index, num_nodes)
        row, col = ei
        x = self.lin(x)
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.unsqueeze(-1))
        return F.relu(out)

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
    def forward(self, x, batch_index):
        # x: (N, H), batch_index: (N,)
        scores = self.w(x).squeeze(-1)  # (N,)
        scores = torch.softmax(scores, dim=0)
        # aggregate per graph
        B = batch_index.max().item() + 1
        out = torch.zeros(B, x.size(1), device=x.device)
        out.index_add_(0, batch_index, x * scores.unsqueeze(-1))
        return out

class DrugEncoderGCN(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=3, pooling='attn'):
        super().__init__()
        dims = [in_dim] + [hidden]*layers
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i+1]) for i in range(layers)])
        self.pooling = pooling
        if pooling == 'attn':
            self.pool = AttnPool(hidden)
        self.proj = nn.Identity()

    def forward(self, x, edge_index, batch_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        if self.pooling == 'attn':
            z = self.pool(h, batch_index)
        elif self.pooling == 'mean':
            B = batch_index.max().item() + 1
            out = torch.zeros(B, h.size(1), device=h.device)
            ones = torch.ones(h.size(0), device=h.device)
            out.index_add_(0, batch_index, h)
            counts = torch.zeros(B, device=h.device).index_add_(0, batch_index, ones).clamp(min=1.0)
            z = out / counts.unsqueeze(-1)
        else:
            raise ValueError('Unknown pooling')
        return h, z

# ---------- Protein Transformer ----------
class ProteinEncoderTransformer(nn.Module):
    def __init__(self, vocab_size=22, hidden=256, layers=4, heads=4, dropout=0.1, max_len=1024, use_cls=True):
        super().__init__()
        self.use_cls = use_cls
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_len + (1 if use_cls else 0), hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*4, dropout=dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Parameter(torch.zeros(1,1,hidden)) if use_cls else None
        nn.init.normal_(self.cls, mean=0.0, std=0.02) if use_cls else None

    def forward(self, ids, attn_mask):
        B, L = ids.shape
        x = self.emb(ids)  # (B, L, H)
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)  # (B,1,H)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.ones(B, 1, dtype=attn_mask.dtype, device=attn_mask.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos(positions)
        x = self.dropout(x)
        x = self.tr(x, src_key_padding_mask=~attn_mask.bool())
        # pooled
        if self.use_cls:
            z = x[:,0,:]
            tokens = x[:,1:,:]
        else:
            z = (x * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1, keepdim=True).clamp(min=1.0)
            tokens = x
        return tokens, z  # tokens: (B, L', H)
