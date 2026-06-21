"""
models/interaction/cross_modal_transformer.py

P4V 新增模块：序列级跨模态 Transformer。

设计目标：
1. 接收三模态 sLSTM 输出序列 H_t / H_a / H_v；
2. 在 token/frame 级别进行跨模态交互；
3. 使用 key_padding_mask 正确屏蔽 padding；
4. 输出按模态切分后的上下文增强序列；
5. 不替代 AWAF，AWAF 仍负责样本级动态融合与解释。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class CrossModalTransformerEncoder(nn.Module):
    """三模态序列级跨模态交互编码器。

    输入:
        H_t: [B, Tt, D]
        H_a: [B, Ta, D]
        H_v: [B, Tv, D]
        text_mask/audio_mask/vision_mask: [B, T], 1=有效, 0=padding

    输出:
        {
            'H_t': [B, Tt, D],
            'H_a': [B, Ta, D],
            'H_v': [B, Tv, D],
            'joint': [B, Tt+Ta+Tv, D],
            'joint_mask': [B, Tt+Ta+Tv]
        }
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.2,
        use_modality_embedding: bool = True,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.use_modality_embedding = use_modality_embedding

        if use_modality_embedding:
            # 0=text, 1=audio, 2=vision
            self.modality_embedding = nn.Embedding(3, hidden_dim)
        else:
            self.modality_embedding = None

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _ensure_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """生成或规范化 mask，返回 long/bool 兼容的 [B,T] 0/1 mask。"""
        if mask is None:
            return torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.long)
        if mask.dim() != 2:
            raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")
        return mask.to(device=x.device).long()

    @staticmethod
    def _fix_all_padding(mask: torch.Tensor) -> torch.Tensor:
        """防止某个样本整段全 padding 导致 Transformer attention 全 -inf。

        如果某个样本 mask 全 0，则将第 0 个位置设为有效。正常数据不应发生此情况；
        该逻辑只作为安全兜底。
        """
        if mask.sum(dim=1).min().item() > 0:
            return mask
        fixed = mask.clone()
        bad = fixed.sum(dim=1) == 0
        fixed[bad, 0] = 1
        return fixed

    def _add_modality_embedding(self, H: torch.Tensor, modality_id: int) -> torch.Tensor:
        if self.modality_embedding is None:
            return H
        B, T, _ = H.shape
        ids = torch.full((B, T), modality_id, device=H.device, dtype=torch.long)
        return H + self.modality_embedding(ids)

    def forward(
        self,
        H_t: torch.Tensor,
        H_a: torch.Tensor,
        H_v: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if H_t.dim() != 3 or H_a.dim() != 3 or H_v.dim() != 3:
            raise ValueError("H_t/H_a/H_v must be [B,T,D].")
        if not (H_t.size(0) == H_a.size(0) == H_v.size(0)):
            raise ValueError("Batch size mismatch across modalities.")
        if not (H_t.size(-1) == H_a.size(-1) == H_v.size(-1) == self.hidden_dim):
            raise ValueError("Hidden dimension mismatch across modalities.")

        B, Tt, _ = H_t.shape
        Ta = H_a.size(1)
        Tv = H_v.size(1)

        mt = self._fix_all_padding(self._ensure_mask(H_t, text_mask))
        ma = self._fix_all_padding(self._ensure_mask(H_a, audio_mask))
        mv = self._fix_all_padding(self._ensure_mask(H_v, vision_mask))

        Ht = self._add_modality_embedding(H_t, 0)
        Ha = self._add_modality_embedding(H_a, 1)
        Hv = self._add_modality_embedding(H_v, 2)

        joint = torch.cat([Ht, Ha, Hv], dim=1)  # [B, T_all, D]
        joint_mask = torch.cat([mt, ma, mv], dim=1)  # [B, T_all]
        joint_mask = self._fix_all_padding(joint_mask)

        # TransformerEncoder 的 key_padding_mask: True 表示需要 mask 掉的位置。
        key_padding_mask = ~joint_mask.bool()
        joint_ctx = self.encoder(joint, src_key_padding_mask=key_padding_mask)
        joint_ctx = self.output_norm(joint_ctx)

        H_ctx_t = joint_ctx[:, :Tt, :]
        H_ctx_a = joint_ctx[:, Tt:Tt + Ta, :]
        H_ctx_v = joint_ctx[:, Tt + Ta:Tt + Ta + Tv, :]

        return {
            "H_t": H_ctx_t,
            "H_a": H_ctx_a,
            "H_v": H_ctx_v,
            "joint": joint_ctx,
            "joint_mask": joint_mask,
            "text_mask": mt,
            "audio_mask": ma,
            "vision_mask": mv,
        }
