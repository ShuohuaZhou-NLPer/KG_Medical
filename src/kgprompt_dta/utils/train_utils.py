import os, time
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass

@dataclass
class EarlyStopper:
    patience: int = 10
    best: float = float('inf')
    steps: int = 0
    did_stop: bool = False
    def step(self, val):
        if val < self.best - 1e-6:
            self.best = val
            self.steps = 0
        else:
            self.steps += 1
            if self.steps >= self.patience:
                self.did_stop = True
        return self.did_stop

def build_optimizers(model, lr_backbone=3e-4, lr_prompt=1e-3, weight_decay=1e-4):
    # Split params into backbone vs prompt/adapters/head by name
    backbone_params, prompt_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if any(k in n for k in ['prompt', 'adapter', 'head', 'kg_emb', 'kg_embedding', 'transE']):
            prompt_params.append(p)
        else:
            backbone_params.append(p)
    optim = AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': prompt_params, 'lr': lr_prompt},
    ], weight_decay=weight_decay)
    return optim

def build_scheduler(optimizer, T_max):
    return CosineAnnealingLR(optimizer, T_max=T_max)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True
