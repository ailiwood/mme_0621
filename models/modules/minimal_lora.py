"""
models/modules/minimal_lora.py — P6H Minimal LoRA (no PEFT dependency)

Self-contained LoRA implementation for RoBERTa-large fine-tuning.
Works with any PyTorch version. No external library needed beyond torch.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, List, Optional, Set
import re


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear. Freezes base weight, trains A @ B low-rank update.

    output = base(x) + (alpha/r) * B @ A @ dropout(x)
    """
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base
        for p in base.parameters():
            p.requires_grad = False

        device = base.weight.device
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora_to_roberta(model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                          target_patterns: List[str] = None) -> nn.Module:
    """Inject LoRA into RoBERTa attention layers.

    Replaces nn.Linear layers matching target_patterns with LoRALinear wrappers.
    Default targets: query, key, value, dense in attention self-attention.
    """
    if target_patterns is None:
        target_patterns = ['query', 'value']  # Default: query+value only

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pat in target_patterns:
                if pat in name and 'attention' in name:
                    # Replace in parent
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    parent = dict(model.named_modules()).get(parent_name)
                    if parent is not None:
                        lora_linear = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                        setattr(parent, attr_name, lora_linear)
                        replaced += 1
                    break
    print(f'LoRA: replaced {replaced} Linear layers with target patterns {target_patterns}')
    return model


def mark_only_lora_as_trainable(model: nn.Module):
    """Freeze all parameters except lora_A, lora_B, and classifier/head."""
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
