"""
models/baselines/tfn_lite.py

TFN-lite: 基于 Tensor Fusion Network 核心思想的轻量重写。
用于本项目统一框架对照实验，不是官方复现。

Reference: Zadeh et al., Tensor Fusion Network for Multimodal Sentiment Analysis, EMNLP 2017.
"""
import torch
import torch.nn as nn
from .base_baseline import BaseBaseline


class TFNLite(BaseBaseline):
    """Tensor Fusion Network 轻量复现。"""

    def __init__(self, config: dict):
        super().__init__(config)
        H = config.get('hidden_dim', 32)
        DROP = config.get('dropout', 0.2)

        # Fusion: concat [h_t+1, h_a+1, h_v+1] → outer product
        n = self.n_modalities
        self.fusion_dim = (H + 1) ** n
        cap = config.get('fusion_cap', 512)
        self.fusion_dim = min(self.fusion_dim, cap)

        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(self.fusion_dim // 2, 1),
        )

    def forward(self, batch):
        features = []
        if self._use_text:
            ht = self._encode_text(batch)  # [B, H]
            features.append(torch.cat([ht, torch.ones(ht.size(0), 1, device=ht.device)], dim=-1))
        if self._use_audio:
            ha = self._encode_audio(batch)
            features.append(torch.cat([ha, torch.ones(ha.size(0), 1, device=ha.device)], dim=-1))
        if self._use_vision:
            hv = self._encode_vision(batch)
            features.append(torch.cat([hv, torch.ones(hv.size(0), 1, device=hv.device)], dim=-1))

        if len(features) == 1:
            fused = features[0]
        else:
            fused = features[0]
            for f in features[1:]:
                fused = torch.einsum('bi,bj->bij', fused, f).flatten(1)
            if fused.size(1) > self.fusion_dim:
                fused = fused[:, :self.fusion_dim]

        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
