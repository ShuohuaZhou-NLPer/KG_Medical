import torch
import torch.nn as nn

class TransEEmbedding(nn.Module):
    """Simple TransE-style embeddings.
    Entities and relations are mapped to vectors. 
    We also expose a ranking loss for optional KG pretraining during model learning.
    """
    def __init__(self, num_entities:int, num_relations:int, dim:int=128, margin:float=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.entity_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel_emb.weight, -0.1, 0.1)
        self.margin = margin
        self.dim = dim

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.entity_emb(t_idx)
        score = torch.norm(h + r - t, p=2, dim=-1)
        return score

    def triplet_ranking_loss(self, pos, neg):
        # pos/neg are (B,) scores; lower is better for TransE distance
        return torch.relu(self.margin + pos - neg).mean()
