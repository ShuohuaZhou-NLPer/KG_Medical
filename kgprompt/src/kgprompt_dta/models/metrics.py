import torch
import numpy as np

def mse(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(((y_true - y_pred)**2).mean())

def pearsonr(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    yt = (y_true - y_true.mean())
    yp = (y_pred - y_pred.mean())
    num = (yt * yp).sum()
    den = (torch.sqrt((yt**2).sum()) * torch.sqrt((yp**2).sum()) + 1e-8)
    return float((num/den).item())

def spearmanr(y_true, y_pred):
    # simple torch-based ranking
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rt = torch.argsort(torch.argsort(y_true))
    rp = torch.argsort(torch.argsort(y_pred))
    return pearsonr(rt.float(), rp.float())

def concordance_index(y_true, y_pred):
    # Probability that pairwise order is preserved
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n = y_true.size(0)
    if n < 2:
        return 1.0
    concordant, permissible = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] == y_true[j]:
                continue
            permissible += 1
            concordant += int((y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0)
    return concordant / max(1, permissible)

def r2m(y_true, y_pred):
    # Robust correlation metric (Roy). We compute averaged r_m^2.
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    # r^2
    r = pearsonr(y_true, y_pred)
    r2 = r*r
    # r0^2 : regression through origin
    eps = 1e-8
    k = ( (y_true*y_pred).sum() / ( (y_pred**2).sum() + eps ) )
    y_pred0 = k * y_pred
    r0 = pearsonr(y_true, y_pred0)
    r02 = r0*r0
    rm2 = r2 * (1 - torch.sqrt(torch.tensor(abs(r2 - r02)) + eps)).item()
    # reverse (y as function of x) for symmetry
    k_prime = ( (y_true*y_pred).sum() / ( (y_true**2).sum() + eps ) )
    y_true0 = k_prime * y_true
    r0p = pearsonr(y_true0, y_pred)
    r0p2 = r0p*r0p
    rm2p = r2 * (1 - torch.sqrt(torch.tensor(abs(r2 - r0p2)) + eps)).item()
    return float( (rm2 + rm2p) / 2.0 )
