import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import DrugEncoderGCN, ProteinEncoderTransformer
from .fusion import FusionStack
from .prompts import StaticPrompt, DynamicPrompt, Adapter
from .head import GatedFuse, PredictionHead

class KGPromptDTA(nn.Module):
    def __init__(self, 
                 drug_in_dim=32,  # will be set dynamically after a sample pass
                 vocab_size=22, hidden=256, heads=4, layers=3, dropout=0.1,
                 prompt_tokens=16, prompt_mode='dynamic',
                 kg_rel_dim=128, use_adapters=True, adapter_bottleneck=64,
                 pooling='attn', peft=True):
        super().__init__()
        self.hidden = hidden
        self.peft = peft
        self.pooling = pooling
        # Encoders
        self.drug = DrugEncoderGCN(in_dim=drug_in_dim, hidden=hidden, layers=3, pooling=pooling)
        self.prot = ProteinEncoderTransformer(vocab_size=vocab_size, hidden=hidden, layers=4, heads=heads, dropout=dropout, max_len=2048, use_cls=True)
        # Fusion with prompt injection
        self.fusion = FusionStack(layers=layers, hidden=hidden, heads=heads, dropout=dropout, use_adapters=use_adapters, adapter_bottleneck=adapter_bottleneck)
        # Prompt generators
        self.prompt_mode = prompt_mode
        if prompt_mode == 'static':
            self.prompt_gen = StaticPrompt(kg_dim=kg_rel_dim, hidden=hidden, m_tokens=prompt_tokens)
        elif prompt_mode == 'dynamic':
            self.prompt_gen = DynamicPrompt(in_dim=hidden*2 + kg_rel_dim*3, hidden=hidden, m_tokens=prompt_tokens)
        else:
            self.prompt_gen = None
        # Projection from prompt to relation space for consistency loss
        self.prompt_to_rel = nn.Linear(hidden, kg_rel_dim)
        # Fusion head
        self.fuser = GatedFuse(hidden)
        self.head = PredictionHead(hidden)

    def set_peft(self, enabled: bool):
        self.peft = enabled
        if enabled:
            for m in [self.drug, self.prot, self.fusion]:
                for p in m.parameters(): p.requires_grad = False
            # prompts + adapters + head already selected by optimizer builder
        else:
            for p in self.parameters(): p.requires_grad = True

    def forward_once(self, sample, kg_entities, kg_relations, need_attention=False):
        # sample is a dict for a *batch*. We will process per-batch via masking/loop for fusion.
        x = sample['drug_x']; edge_index = sample['drug_edge_index']; bidx = sample['drug_batch_index']
        prot_ids = sample['prot_ids']; prot_mask = sample['prot_mask']

        # Drug
        Hd_all, zd = self.drug(x, edge_index, bidx)  # Hd_all: (N_total, H); zd: (B,H)
        # Split Hd by graphs
        nodes_per = sample['drug_nodes_per_graph']
        Hd_list = []
        offset = 0
        for n in nodes_per:
            Hd_list.append(Hd_all[offset:offset+n])
            offset += n

        # Protein
        Hp_tokens, zp = self.prot(prot_ids, prot_mask)  # Hp_tokens: (B, Lp, H)
        Hp_list = [Hp_tokens[i, :prot_mask[i].sum().item()+1, :] for i in range(Hp_tokens.size(0))]  # include CLS (first pos)

        # KG embeddings: ed, r, ep are (B, Dk)
        ed = kg_entities['ed']; ep = kg_entities['ep']; r = kg_relations

        # Prompts
        P = None
        if self.prompt_mode == 'static':
            P = self.prompt_gen(ed, r, ep)  # (B, m, H)
        elif self.prompt_mode == 'dynamic':
            P = self.prompt_gen(zd, zp, ed, r, ep)  # (B, m, H)

        # Fusion (process each sample independently to keep code simple)
        Hd_out_list, Hp_out_list, attn_all = [], [], []
        for i in range(len(Hd_list)):
            Hi_d = Hd_list[i]
            Hi_p = Hp_list[i].squeeze(0) if Hp_list[i].ndim==3 else Hp_list[i]
            Pi = P[i] if P is not None else None
            Hd2, Hp2, attn = self.fusion(Hi_d, Hi_p, Pi, need_weights=need_attention)
            Hd_out_list.append(Hd2)
            Hp_out_list.append(Hp2)
            attn_all.append(attn)

        # Pool to get per-sample representations
        # Drug pooling re-use same strategy (attn pool was already used to produce zd).
        zd2 = zd
        zp2 = zp
        # Prompt pooling: simple mean over tokens if exist; else zeros
        if P is not None:
            zP = P.mean(dim=1)
        else:
            zP = torch.zeros_like(zd2)

        # Fuse and predict
        zfuse, gates = self.fuser(zd2, zp2, zP)
        yhat = self.head(zfuse)
        return yhat, zfuse, zd2, zp2, zP, P, attn_all

