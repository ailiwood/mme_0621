"""
models/pooling/attention_pooling.py

P4V 新增模块：带 mask 的注意力池化。

用途：从上下文增强后的单模态序列 H_ctx_m 中学习有效时间步权重，得到
模态摘要向量 h_m，再交给 AWAF 做样本级动态融合。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MaskedAttentionPooling(nn.Module):
    """Masked attention pooling for variable-length sequences.

    Args:
        hidden_dim: 输入特征维度 D。
        dropout: attention scorer dropout。

    Input:
        H: [B, T, D]
        mask: [B, T], 1=有效, 0=padding

    Output:
        pooled: [B, D]
        attn: [B, T]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _ensure_mask(H: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.ones(H.size(0), H.size(1), device=H.device, dtype=torch.long)
        return mask.to(device=H.device).long()

    @staticmethod
    def _fix_all_padding(mask: torch.Tensor) -> torch.Tensor:
        if mask.sum(dim=1).min().item() > 0:
            return mask
        fixed = mask.clone()
        bad = fixed.sum(dim=1) == 0
        fixed[bad, 0] = 1
        return fixed

    def forward(
        self,
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if H.dim() != 3:
            raise ValueError(f"H must be [B,T,D], got {tuple(H.shape)}")
        mask = self._fix_all_padding(self._ensure_mask(H, mask))

        scores = self.scorer(H).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(~mask.bool(), -1e4)
        attn = torch.softmax(scores, dim=1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-8)

        pooled = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # [B,D]
        if return_weights:
            return pooled, attn
        return pooled
