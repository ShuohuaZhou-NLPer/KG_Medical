import torch
import numpy as np

# --- Amino-acid vocabulary (20 canonical + extras) ---
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_EXTRA = "XBZUOJ"  # rarely used; mapped to unknown
AA2IDX = {a:i+1 for i,a in enumerate(AA)}  # 0 is PAD
AA2IDX['UNK'] = len(AA2IDX)+1

def tokenize_protein(seq, max_len=512):
    ids = [AA2IDX.get(a, AA2IDX['UNK']) for a in seq.strip()]
    ids = ids[:max_len]
    attn_mask = [1]*len(ids)
    # pad
    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids += [0]*pad_len
        attn_mask += [0]*pad_len
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.bool)

# --- Molecular graph featurizer ---
# If RDKit is available, we will use it. Otherwise, we fall back to a simple
# sequence-style chain graph on SMILES characters. This keeps the pipeline runnable
# but you should install RDKit for real experiments.
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

ATOM_LIST = ['C','N','O','S','F','P','Cl','Br','I','H','B','Si','Na','K','Li','Mg','Ca','Fe','Al','Cu','Zn']
def atom_feature(atom):
    sym = atom.GetSymbol()
    onehot = [1 if sym==a else 0 for a in ATOM_LIST]
    degree = atom.GetDegree()
    val = atom.GetTotalValence()
    arom = 1 if atom.GetIsAromatic() else 0
    return np.array(onehot + [degree, val, arom], dtype=np.float32)

def smiles_to_graph(smiles):
    if RDKit_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        atoms = [atom_feature(a) for a in mol.GetAtoms()]
        x = torch.tensor(np.stack(atoms), dtype=torch.float32)
        edges = []
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges.append((u, v)); edges.append((v, u))
        # self-loops
        for i in range(len(atoms)):
            edges.append((i, i))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2,0, dtype=torch.long)
        return x, edge_index
    else:
        # fallback: encode as character chain graph
        s = smiles.strip()
        n = len(s)
        x = torch.eye(max(2, n), dtype=torch.float32)[:n]  # very simple token features
        edges = []
        for i in range(n-1):
            edges.append((i, i+1)); edges.append((i+1, i))
        for i in range(n):
            edges.append((i, i))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2,0, dtype=torch.long)
        return x, edge_index
