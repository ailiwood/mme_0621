"""MLCL-lite: Multi-Level Contrastive Learning 轻量重写，不是官方复现。
Reference: Zhuang et al., MLCL, IEEE TMM 2025."""
import torch, torch.nn as nn, torch.nn.functional as F
from .base_baseline import BaseBaseline

class MLCLLite(BaseBaseline):
    def __init__(self, config):
        super().__init__(config)
        H = config.get('hidden_dim', 128)
        DROP = config.get('dropout', 0.2)
        fusion_in = H * self.n_modalities
        self.head = nn.Sequential(nn.Linear(fusion_in, fusion_in//2), nn.ReLU(), nn.Dropout(DROP), nn.Linear(fusion_in//2, 1))
        # Unimodal heads for contrastive learning
        if self._use_text: self.text_head = nn.Linear(H, 1)
        if self._use_audio: self.audio_head = nn.Linear(H, 1)
    def forward(self, batch):
        hs, aux = [], {}
        label = batch['label'].squeeze(-1)
        if self._use_text:
            ht = self._encode_text(batch); hs.append(ht)
            aux['text'] = self.text_head(ht).squeeze(-1)
        if self._use_audio:
            ha = self._encode_audio(batch); hs.append(ha)
            aux['audio'] = self.audio_head(ha).squeeze(-1)
        if self._use_vision: hs.append(self._encode_vision(batch))
        fused = torch.cat(hs, dim=-1)
        reg = self.head(fused).squeeze(-1)
        loss = nn.L1Loss()(reg, label) + 0.2 * sum(nn.L1Loss()(p, label) for p in aux.values())
        return {'reg': reg, 'loss_terms': {'loss_total': loss}}
