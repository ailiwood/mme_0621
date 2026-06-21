"""
models/baselines/mult_lite.py

MulT-lite: 基于 cross-modal Transformer 核心思想的 text-centric 轻量重写。
不是官方复现。本阶段仅 text→audio 和 text→vision 两个方向。

Reference: Tsai et al., Multimodal Transformer for Unaligned Multimodal Language Sequences, ACL 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_baseline import BaseBaseline


class MulTLite(BaseBaseline):
    """Cross-modal Transformer (text-centric) 轻量复现。"""

    def __init__(self, config: dict):
        super().__init__(config)
        H = config.get('hidden_dim', 64)
        DROP = config.get('dropout', 0.2)
        NHEAD = config.get('nhead', 4)

        # Cross-modal attention: text query attends audio/vision
        if self._use_audio:
            self.cross_ta = nn.MultiheadAttention(H, NHEAD, dropout=DROP, batch_first=True)
            self.norm_ta = nn.LayerNorm(H)
        if self._use_vision:
            self.cross_tv = nn.MultiheadAttention(H, NHEAD, dropout=DROP, batch_first=True)
            self.norm_tv = nn.LayerNorm(H)

        # Sequence pooling: last-step or mean
        self.seq_pool = config.get('seq_pool', 'last')

        # Fusion head
        fusion_in = H * self.n_modalities
        self.head = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(fusion_in // 2, 1),
        )

    def _encode_text_seq(self, batch):
        """Text → full sequence [B, T, H] (not pooled)"""
        if self.use_pretrained_text:
            feat = batch.get('roberta_cls', batch.get('text_feature'))
            if feat is not None:
                feat = self.text_proj(feat)
                return feat.unsqueeze(1).expand(-1, 10, -1)
        emb = self.text_embed(batch['input_ids'])
        out, _ = self.text_gru(emb)
        return self.text_proj(out)

    def _encode_audio_seq(self, batch):
        """Audio → GRU → full sequence [B, T, H]"""
        a = batch['audio']
        ha = self.audio_proj(a)
        out, _ = self.audio_gru(ha)
        return self.audio_out(out)

    def _encode_vision_seq(self, batch):
        """Vision → GRU → full sequence [B, T, H]"""
        v = batch['vision']
        hv = self.vision_proj(v)
        out, _ = self.vision_gru(hv)
        return self.vision_out(out)

    def _pool_seq(self, x):
        if self.seq_pool == 'last':
            return x[:, -1, :]
        return x.mean(dim=1)

    def forward(self, batch):
        h_text = self._encode_text_seq(batch)  # [B, T_t, H]
        ht_pooled = self._pool_seq(h_text)

        outputs = [ht_pooled]

        # Text → Audio cross-attention
        if self._use_audio:
            h_audio = self._encode_audio_seq(batch)  # [B, T_a, H]
            ha_enhanced, _ = self.cross_ta(h_text, h_audio, h_audio)
            ha_enhanced = self.norm_ta(ha_enhanced + h_text)
            outputs.append(self._pool_seq(ha_enhanced))

        # Text → Vision cross-attention
        if self._use_vision:
            h_vision = self._encode_vision_seq(batch)  # [B, T_v, H]
            hv_enhanced, _ = self.cross_tv(h_text, h_vision, h_vision)
            hv_enhanced = self.norm_tv(hv_enhanced + h_text)
            outputs.append(self._pool_seq(hv_enhanced))

        fused = torch.cat(outputs, dim=-1)
        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
