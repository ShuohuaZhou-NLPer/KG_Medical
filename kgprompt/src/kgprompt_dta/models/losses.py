import torch
import torch.nn.functional as F

def mse_loss(pred, target):
    return F.mse_loss(pred.view_as(target), target)

def kg_prompt_consistency(P, r, proj):
    # P: (B,m,H) -> average then map to relation space
    Pm = P.mean(dim=1)  # (B,H)
    Pr = proj(Pm)       # (B, Dk)
    r = r  # (B, Dk)
    # cosine similarity (maximize), so loss = 1 - cos
    cos = torch.nn.functional.cosine_similarity(Pr, r, dim=-1)
    return (1.0 - cos).mean()

def info_nce(zfuse, r, temperature=0.1):
    # zfuse: (B,H) -> map to relation space r-dim outside this function if desired
    # Here we assume zfuse and r already in same dimension
    z = F.normalize(zfuse, dim=-1)
    r = F.normalize(r, dim=-1)
    sim = z @ r.t()  # (B,B)
    logits = sim / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, labels)
