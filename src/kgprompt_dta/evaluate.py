import os, yaml, json
import torch
from torch.utils.data import DataLoader
from .data.datasets import DTADataset, collate_fn
from .models.full_model import KGPromptDTA
from .models.metrics import mse as mse_metric, pearsonr, spearmanr, concordance_index, r2m
from .data.kg import TransEEmbedding

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
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
    dl = DataLoader(ds, batch_size=cfg['optim']['batch_size'], shuffle=False, collate_fn=collate_fn)

    # Model
    drug_in_dim = ds[0]['drug_x'].size(1)
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

    # Dummy KG embeddings for evaluation (see training note)
    num_entities = len(ds.drugs) + len(ds.proteins)
    num_relations = len(ds.rel2idx)
    transE = TransEEmbedding(num_entities, num_relations, dim=kg_dim).to(device).eval()

    yhats, ys = [], []
    with torch.no_grad():
        for batch in dl:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            B = batch['prot_ids'].size(0)
            ed_idx = torch.arange(B, device=device) % len(ds.drugs)
            ep_idx = torch.arange(B, device=device) % len(ds.proteins) + len(ds.drugs)
            r_idx  = batch['rel_idx'].to(device)
            ed = transE.entity_emb(ed_idx)
            ep = transE.entity_emb(ep_idx)
            r  = transE.rel_emb(r_idx)
            yhat, *_ = model.forward_once(batch, {'ed': ed, 'ep': ep}, r, need_attention=False)
            yhats.append(yhat.detach().cpu())
            ys.append(batch['labels'].cpu())
    yhat = torch.cat(yhats, 0); y = torch.cat(ys, 0)
    results = {
        'mse': mse_metric(y, yhat),
        'ci': concordance_index(y, yhat),
        'pearson': pearsonr(y, yhat),
        'spearman': spearmanr(y, yhat),
        'r2m': r2m(y, yhat)
    }
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
