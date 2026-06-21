"""
models/baselines/self_mm_lite.py

SelfMM-lite: 基于 Self-MM 单模态辅助监督思想的轻量重写。
本阶段使用真实标签辅助监督作为简化实现，不是官方复现。

Reference: Yu et al., Learning Modality-Specific Representations with Self-Supervised
Multi-Task Learning for Multimodal Sentiment Analysis, AAAI 2021.
"""
import torch
import torch.nn as nn
from .base_baseline import BaseBaseline


class SelfMMLite(BaseBaseline):
    """Self-MM 单模态辅助监督 轻量复现。"""

    def __init__(self, config: dict):
        super().__init__(config)
        H = config.get('hidden_dim', 64)
        DROP = config.get('dropout', 0.2)

        # Fusion head
        fusion_in = H * self.n_modalities
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(fusion_in // 2, 1),
        )

        # Unimodal auxiliary heads
        if self._use_text:
            self.text_aux_head = nn.Linear(H, 1)
        if self._use_audio:
            self.audio_aux_head = nn.Linear(H, 1)
        if self._use_vision:
            self.vision_aux_head = nn.Linear(H, 1)

    def forward(self, batch):
        hs = []
        aux_preds = {}
        label = batch['label'].squeeze(-1)  # [B]

        if self._use_text:
            ht = self._encode_text(batch)
            hs.append(ht)
            aux_preds['text'] = self.text_aux_head(ht).squeeze(-1)

        if self._use_audio:
            ha = self._encode_audio(batch)
            hs.append(ha)
            aux_preds['audio'] = self.audio_aux_head(ha).squeeze(-1)

        if self._use_vision:
            hv = self._encode_vision(batch)
            hs.append(hv)
            aux_preds['vision'] = self.vision_aux_head(hv).squeeze(-1)

        # Fusion prediction
        fused = torch.cat(hs, dim=-1)
        reg = self.fusion_head(fused).squeeze(-1)

        # Loss terms
        loss_main = nn.L1Loss()(reg, label)
        loss_aux = sum(nn.L1Loss()(p, label) for p in aux_preds.values())
        loss_total = loss_main + 0.3 * loss_aux

        return {
            'reg': reg,
            'loss_terms': {
                'loss_main': loss_main,
                'loss_aux': loss_aux,
                'loss_total': loss_total,
            }
        }
