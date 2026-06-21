"""DLF-lite: Disentangled-Language-Focused 轻量重写，不是官方复现。
Reference: Wang et al., DLF, AAAI 2025."""
import torch, torch.nn as nn, torch.nn.functional as F
from .base_baseline import BaseBaseline

class DLFLite(BaseBaseline):
    def __init__(self, config):
        super().__init__(config)
        H = config.get('hidden_dim', 128)
        DROP = config.get('dropout', 0.2)
        fusion_in = H * self.n_modalities
        # Text-focused: text as primary, audio/vision gated
        if self._use_audio:
            self.audio_gate = nn.Sequential(nn.Linear(H*2, H), nn.Sigmoid())
        if self._use_vision:
            self.vision_gate = nn.Sequential(nn.Linear(H*2, H), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(fusion_in, fusion_in//2), nn.ReLU(), nn.Dropout(DROP), nn.Linear(fusion_in//2, 1))
    def forward(self, batch):
        hs = []
        if self._use_text:
            ht = self._encode_text(batch); hs.append(ht)
        if self._use_audio:
            ha = self._encode_audio(batch)
            g = self.audio_gate(torch.cat([ht, ha], dim=-1))
            hs.append(ha * g)
        if self._use_vision:
            hv = self._encode_vision(batch)
            g = self.vision_gate(torch.cat([ht, hv], dim=-1))
            hs.append(hv * g)
        fused = torch.cat(hs, dim=-1)
        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
