"""
models/baselines/lmf_lite.py

LMF-lite: 基于 Low-rank Multimodal Fusion 核心思想的轻量重写。
用于本项目统一框架对照实验，不是官方复现。

Reference: Liu et al., Efficient Low-rank Multimodal Fusion, ACL 2018.
"""
import torch
import torch.nn as nn
from .base_baseline import BaseBaseline


class LMFLite(BaseBaseline):
    """Low-rank Multimodal Fusion 轻量复现。"""

    def __init__(self, config: dict):
        super().__init__(config)
        H = config.get('hidden_dim', 64)
        self.rank = config.get('rank', 8)
        self.fusion_dim = config.get('fusion_dim', 64)
        DROP = config.get('dropout', 0.2)
        n = self.n_modalities

        # Low-rank factors: each modality → rank × fusion_dim
        self.factors = nn.ModuleList()
        for i in range(n):
            self.factors.append(nn.Linear(H, self.rank * self.fusion_dim))

        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(self.fusion_dim // 2, 1),
        )

    def forward(self, batch):
        hs = []
        if self._use_text:
            hs.append(self._encode_text(batch))
        if self._use_audio:
            hs.append(self._encode_audio(batch))
        if self._use_vision:
            hs.append(self._encode_vision(batch))

        # Low-rank fusion: element-wise multiply factorized representations
        fused = None
        for i, h in enumerate(hs):
            f = self.factors[i](h)  # [B, rank * fusion_dim]
            f = f.view(-1, self.rank, self.fusion_dim)  # [B, rank, fusion_dim]
            if fused is None:
                fused = f
            else:
                fused = fused * f  # element-wise multiply

        # Sum over rank dimension
        fused = fused.sum(dim=1)  # [B, fusion_dim]

        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
