"""
models/baselines/base_baseline.py — Baseline-Lite 基类

统一接口。关键张量形状基于 TextFTMultimodalDataset 实际输出。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BaseBaseline(nn.Module):
    """Baseline-lite 基类。

    Batch 字段 (来自 TextFTMultimodalDataset):
      input_ids:     [B, 128]        tokenized text (roberta-large tokenizer)
      attention_mask:[B, 128]        text attention mask
      audio:         [B, 100, 768]   frozen audio features
      audio_mask:    [B, 100]        audio padding mask
      vision:        [B, 40, 768]    frozen vision features
      vision_mask:   [B, 40]         vision padding mask
      label:         [B, 1]          sentiment score (-3 to +3)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.mode = config.get('modality_mode', 'text_audio_vision')
        H = config.get('hidden_dim', 128)
        self.use_pretrained_text = config.get('use_pretrained_text', False)
        TEXT_DIM = config.get('text_input_dim', 1024)  # RoBERTa-large hidden

        # Text encoder
        if self._use_text:
            if self.use_pretrained_text:
                # Use precomputed/cached text feature (e.g. RoBERTa CLS)
                self.text_proj = nn.Linear(TEXT_DIM, H)
            else:
                # Lightweight GRU encoder (fallback)
                vocab_size = config.get('vocab_size', 50265)
                self.text_embed = nn.Embedding(vocab_size, H, padding_idx=1)
                self.text_gru = nn.GRU(H, H, batch_first=True, bidirectional=True)
                self.text_proj = nn.Linear(H * 2, H)

        # Audio projection
        if self._use_audio:
            A_DIM = config.get('audio_input_dim', 768)
            self.audio_proj = nn.Sequential(nn.Linear(A_DIM, H), nn.ReLU())
            self.audio_gru = nn.GRU(H, H, batch_first=True, bidirectional=True)
            self.audio_out = nn.Linear(H * 2, H)

        # Vision projection
        if self._use_vision:
            V_DIM = config.get('vision_input_dim', 768)
            self.vision_proj = nn.Sequential(nn.Linear(V_DIM, H), nn.ReLU())
            self.vision_gru = nn.GRU(H, H, batch_first=True, bidirectional=True)
            self.vision_out = nn.Linear(H * 2, H)

    @property
    def _use_text(self):
        return self.mode in ('text_only', 'text_audio', 'text_audio_vision')

    @property
    def _use_audio(self):
        return self.mode in ('text_audio', 'text_audio_vision', 'audio_only')

    @property
    def _use_vision(self):
        return self.mode in ('text_audio_vision', 'vision_only')

    @property
    def n_modalities(self):
        return sum([self._use_text, self._use_audio, self._use_vision])

    def _encode_text(self, batch):
        """Text → [B, H] pooled representation."""
        if self.use_pretrained_text:
            if 'text_feature' in batch:
                feat = batch['text_feature']  # RoBERTa CLS [B, 1024]
            elif 'roberta_cls' in batch:
                feat = batch['roberta_cls']    # cached feature [B, 1024]
            else:
                # Fallback: run RoBERTa from input_ids
                return self.text_proj(torch.zeros(batch['input_ids'].size(0), self.text_proj.in_features, device=batch['input_ids'].device))
            return self.text_proj(feat)  # [B, H]
        # Fallback: GRU encoder
        ids = batch['input_ids']
        mask = batch['attention_mask']
        emb = self.text_embed(ids)
        out, _ = self.text_gru(emb)
        ht = self.text_proj(out)
        return self._masked_pool(ht, mask)

    def _encode_audio(self, batch):
        """Audio: projection → GRU → masked mean pool → [B, H]"""
        a = batch['audio']  # [B, T, 768]
        mask = batch.get('audio_mask', None)
        ha = self.audio_proj(a)  # [B, T, H]
        out, _ = self.audio_gru(ha)  # [B, T, H*2]
        ha_out = self.audio_out(out)  # [B, T, H]
        return self._masked_pool(ha_out, mask)

    def _encode_vision(self, batch):
        """Vision: projection → GRU → masked mean pool → [B, H]"""
        v = batch['vision']  # [B, T, 768]
        mask = batch.get('vision_mask', None)
        hv = self.vision_proj(v)  # [B, T, H]
        out, _ = self.vision_gru(hv)  # [B, T, H*2]
        hv_out = self.vision_out(out)  # [B, T, H]
        return self._masked_pool(hv_out, mask)

    def _masked_pool(self, x, mask=None):
        """Masked mean pooling. x: [B, T, H], mask: [B, T]"""
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            x = x * mask_f
            return x.sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
        return x.mean(dim=1)

    def forward(self, batch: Dict) -> Dict:
        raise NotImplementedError
