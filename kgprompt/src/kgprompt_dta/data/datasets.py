import os, csv, math, random
import torch
from torch.utils.data import Dataset, DataLoader
from .featurizers import tokenize_protein, smiles_to_graph

def read_csv(path):
    rows = []
    with open(path, 'r') as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0: 
                header = [h.strip() for h in row]
                continue
            if not row: continue
            rows.append({k:v for k, v in zip(header, row)})
    return rows

class DTADataset(Dataset):
    def __init__(self, root, drugs_csv, proteins_csv, pairs_csv, kg_triples=None,
                 max_drug_nodes=64, max_protein_len=512):
        self.root = root
        self.drugs = {r['drug_id']: r['smiles'] for r in read_csv(os.path.join(root, drugs_csv) if not os.path.isabs(drugs_csv) else drugs_csv)}
        self.proteins = {r['protein_id']: r['sequence'] for r in read_csv(os.path.join(root, proteins_csv) if not os.path.isabs(proteins_csv) else proteins_csv)}
        self.pairs = read_csv(os.path.join(root, pairs_csv) if not os.path.isabs(pairs_csv) else pairs_csv)
        self.kg_triples = read_csv(os.path.join(root, kg_triples) if kg_triples and not os.path.isabs(kg_triples) else kg_triples) if kg_triples else []
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_len = max_protein_len

        # build relation vocab
        self.rel2idx = {}
        for t in self.kg_triples:
            r = t['relation']
            if r not in self.rel2idx:
                self.rel2idx[r] = len(self.rel2idx)
        # add any from pairs
        for p in self.pairs:
            r = p.get('relation','binds')
            if r not in self.rel2idx:
                self.rel2idx[r] = len(self.rel2idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        d_id, t_id = item['drug_id'], item['protein_id']
        smiles = self.drugs[d_id]
        seq = self.proteins[t_id]
        label = float(item['label'])
        relation = item.get('relation', 'binds')
        r_idx = self.rel2idx.get(relation, 0)

        # drug graph
        x, edge_index = smiles_to_graph(smiles)
        if x.size(0) > self.max_drug_nodes:
            x = x[:self.max_drug_nodes]
            mask_nodes = self.max_drug_nodes
            # filter edges to < max_drug_nodes
            keep = (edge_index[0] < mask_nodes) & (edge_index[1] < mask_nodes)
            edge_index = edge_index[:, keep]
        N = x.size(0)

        # protein tokens
        ids, attn_mask = tokenize_protein(seq, self.max_protein_len)

        return {
            'pair_id': item.get('pair_id', f'pair_{idx}'),
            'drug_id': d_id, 'protein_id': t_id,
            'drug_x': x, 'drug_edge_index': edge_index, 'drug_n': N,
            'prot_ids': ids, 'prot_mask': attn_mask,
            'label': torch.tensor([label], dtype=torch.float32),
            'rel_idx': torch.tensor([r_idx], dtype=torch.long),
        }

def collate_fn(batch):
    # Pad variable-size drug graphs by block-diagonalization
    # Build a global node list and shift edge indices
    total_nodes = sum(b['drug_n'] for b in batch)
    x = torch.zeros(total_nodes, batch[0]['drug_x'].size(1))
    edge_indices = []
    offset = 0
    drug_batch_index = torch.zeros(total_nodes, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b['drug_n']
        x[offset:offset+n] = b['drug_x']
        ei = b['drug_edge_index'].clone()
        ei = ei + offset
        edge_indices.append(ei)
        drug_batch_index[offset:offset+n] = i
        offset += n
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2,0, dtype=torch.long)

    # protein tokens
    prot_ids = torch.stack([b['prot_ids'] for b in batch], dim=0)     # (B, L)
    prot_mask = torch.stack([b['prot_mask'] for b in batch], dim=0)   # (B, L)
    labels = torch.cat([b['label'] for b in batch], dim=0)            # (B, 1)
    rel_idx = torch.cat([b['rel_idx'] for b in batch], dim=0)         # (B,)

    pair_ids = [b['pair_id'] for b in batch]
    drug_nodes_per_graph = [b['drug_n'] for b in batch]
    return {
        'pair_ids': pair_ids,
        'drug_x': x, 'drug_edge_index': edge_index, 'drug_batch_index': drug_batch_index,
        'prot_ids': prot_ids, 'prot_mask': prot_mask,
        'labels': labels, 'rel_idx': rel_idx,
        'drug_nodes_per_graph': drug_nodes_per_graph,
    }
