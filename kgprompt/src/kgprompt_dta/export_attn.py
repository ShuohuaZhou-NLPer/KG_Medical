import os, yaml, json
import torch
from torch.utils.data import DataLoader
from .data.datasets import DTADataset, collate_fn
from .models.full_model import KGPromptDTA
from .data.kg import TransEEmbedding

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pair-id', type=str, required=True, help='Pair ID to export attention for')
    parser.add_argument('--out', type=str, default='attn.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device','cpu') if torch.cuda.is_available() else 'cpu')

    ds = DTADataset(
        root=cfg['data']['root'],
        drugs_csv=cfg['data']['drugs_csv'],
        proteins_csv=cfg['data']['proteins_csv'],
        pairs_csv=cfg['data']['pairs_csv'],
        kg_triples=cfg['data'].get('kg_triples'),
        max_drug_nodes=cfg['data']['max_drug_nodes'],
        max_protein_len=cfg['data']['max_protein_len']
    )

    # pick the requested pair
    idx = next((i for i,p in enumerate(ds.pairs) if p.get('pair_id')==args.pair_id), None)
    if idx is None:
        raise SystemExit(f"Pair ID {args.pair_id} not found.")
    sample = ds[idx]
    batch = collate_fn([{
        'pair_id': sample.get('pair_id', 'pair'),
        'drug_x': sample['drug_x'], 'drug_edge_index': sample['drug_edge_index'], 'drug_n': sample['drug_n'],
        'prot_ids': sample['prot_ids'], 'prot_mask': sample['prot_mask'],
        'label': sample['label'], 'rel_idx': sample['rel_idx']
    }])

    # Model
    drug_in_dim = sample['drug_x'].size(1)
    kg_dim = cfg['model']['kg']['embedding_dim']
    model = KGPromptDTA(
        drug_in_dim=drug_in_dim,
        vocab_size=22,
        hidden=cfg['model']['hidden_dim'],
        heads=cfg['model']['heads'],
        layers=cfg['model']['layers'],
        dropout=cfg['model']['dropout'],
        prompt_tokens=cfg['model']['prompt_tokens'],
        prompt_mode=cfg['model']['prompt_mode'],
        kg_rel_dim=kg_dim,
        use_adapters=cfg['model']['use_adapters'],
        adapter_bottleneck=cfg['model']['adapter_bottleneck'],
        pooling=cfg['model']['pooling'],
        peft=cfg['model']['peft']
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=False)
    model.eval()

    # KG embeddings
    num_entities = len(ds.drugs) + len(ds.proteins)
    num_relations = len(ds.rel2idx)
    transE = TransEEmbedding(num_entities, num_relations, dim=kg_dim).to(device).eval()
    B = 1
    ed_idx = torch.tensor([0], device=device)
    ep_idx = torch.tensor([len(ds.drugs)], device=device)
    r_idx  = batch['rel_idx'].to(device)
    ed = transE.entity_emb(ed_idx)
    ep = transE.entity_emb(ep_idx)
    r  = transE.rel_emb(r_idx)

    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    yhat, zfuse, zd, zp, zP, P, attn_all = model.forward_once(batch, {'ed': ed, 'ep': ep}, r, need_attention=True)

    # Flatten and save
    attn_serializable = []
    for layer, alist in enumerate(attn_all):
        if alist:
            # alist contains one tensor per block (Sx, Sx+m); we have a single item in batch, so take [0]
            mats = []
            for a in alist:
                mats.append(a.squeeze(1).mean(0).cpu().numpy().tolist())  # average over heads
            attn_serializable.append(mats)
    with open(args.out, 'w') as f:
        json.dump({'pair_id': args.pair_id, 'attn': attn_serializable}, f)
    print(f"Saved attention to {args.out}")

if __name__ == '__main__':
    main()
