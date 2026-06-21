"""MISA-lite: Modality-Invariant/Specific 轻量重写，不是官方复现。
Reference: Hazarika et al., MISA, ACM MM 2020."""
import torch, torch.nn as nn, torch.nn.functional as F
from .base_baseline import BaseBaseline

class MISALite(BaseBaseline):
    def __init__(self, config):
        super().__init__(config)
        H = config.get('hidden_dim', 128)
        DROP = config.get('dropout', 0.2)
        fusion_in = H * self.n_modalities
        self.head = nn.Sequential(nn.Linear(fusion_in, fusion_in//2), nn.ReLU(), nn.Dropout(DROP), nn.Linear(fusion_in//2, 1))
    def forward(self, batch):
        hs = []
        if self._use_text: hs.append(self._encode_text(batch))
        if self._use_audio: hs.append(self._encode_audio(batch))
        if self._use_vision: hs.append(self._encode_vision(batch))
        fused = torch.cat(hs, dim=-1)
        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
