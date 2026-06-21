
"""
models/modules/text_confidence_residual.py

Text-Confidence Conditioned Residual Head.

用途：
1. text_base 作为主预测；
2. residual 只学习 text_base 的错误；
3. gate 由 text confidence 控制；
4. 避免 P6H/P6H-R 中 gate 自由关闭、delta 无效的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextConfidenceResidualConfig:
    hidden_dim: int = 256
    gate_hidden_dim: int = 128
    delta_hidden_dim: int = 128
    dropout: float = 0.1

    max_delta: float = 1.0
    gate_floor: float = 0.2

    detach_text_for_residual: bool = True
    use_av_interaction: bool = True


class TextConfidenceResidualHead(nn.Module):
    """
    Text-confidence conditioned residual.

    输入：
        h_t: [B,H] text hidden
        h_a: [B,H] audio hidden
        h_v: [B,H] vision hidden
        z_av: [B,H] AV/AWAF fused hidden
        reg_text_base: [B,1]
        cls_text_base: [B,1]
        label: [B,1] optional

    输出：
        reg_final: [B,1]
        delta: [B,1]
        gate: [B,1]
        text_confidence: [B,1]
        target_delta: [B,1] optional
    """

    def __init__(self, cfg: TextConfidenceResidualConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        # text confidence features: prob margin + prediction strength
        # residual input = text hidden + AV hidden + interaction + scalar confidence features
        in_dim = H * 4 + 3
        if not cfg.use_av_interaction:
            in_dim = H * 2 + 3

        self.input_norm = nn.LayerNorm(in_dim)

        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.delta_hidden_dim),
            nn.LayerNorm(cfg.delta_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.delta_hidden_dim, 1),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.gate_hidden_dim),
            nn.LayerNorm(cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )

        # 让 gate 初始不要完全关闭
        nn.init.constant_(self.gate_mlp[-1].bias, 0.0)

    def compute_text_confidence(
        self,
        reg_text_base: torch.Tensor,
        cls_text_base: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算文本置信度。"""
        prob = torch.sigmoid(cls_text_base)
        margin = torch.abs(prob - 0.5) * 2.0
        strength = torch.clamp(torch.abs(reg_text_base) / 3.0, 0.0, 1.0)
        confidence = 0.5 * margin + 0.5 * strength
        uncertainty = 1.0 - confidence

        return {
            "text_prob": prob,
            "text_margin": margin,
            "text_strength": strength,
            "text_confidence": confidence,
            "text_uncertainty": uncertainty,
        }

    def forward(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
        z_av: torch.Tensor,
        reg_text_base: torch.Tensor,
        cls_text_base: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        conf = self.compute_text_confidence(reg_text_base, cls_text_base)
        text_conf = conf["text_confidence"]
        text_unc = conf["text_uncertainty"]

        if cfg.detach_text_for_residual:
            h_t_in = h_t.detach()
            reg_base_in = reg_text_base.detach()
            cls_base_in = cls_text_base.detach()
        else:
            h_t_in = h_t
            reg_base_in = reg_text_base
            cls_base_in = cls_text_base

        if cfg.use_av_interaction:
            av_inter = h_a * h_v
            x = torch.cat(
                [
                    h_t_in,
                    z_av,
                    av_inter,
                    h_t_in * z_av,
                    text_conf,
                    text_unc,
                    torch.clamp(torch.abs(reg_base_in) / 3.0, 0.0, 1.0),
                ],
                dim=-1,
            )
        else:
            x = torch.cat(
                [
                    h_t_in,
                    z_av,
                    text_conf,
                    text_unc,
                    torch.clamp(torch.abs(reg_base_in) / 3.0, 0.0, 1.0),
                ],
                dim=-1,
            )

        x = self.input_norm(x)

        raw_delta = self.delta_mlp(x)
        delta = cfg.max_delta * torch.tanh(raw_delta)

        raw_gate = self.gate_mlp(x)
        gate = cfg.gate_floor + (1.0 - cfg.gate_floor) * torch.sigmoid(raw_gate)

        reg_final = reg_text_base + gate * delta

        out = {
            "reg_final": reg_final,
            "delta": delta,
            "raw_delta": raw_delta,
            "gate": gate,
            "raw_gate": raw_gate,
            "text_confidence": text_conf,
            "text_uncertainty": text_unc,
            "text_prob": conf["text_prob"],
            "text_margin": conf["text_margin"],
            "text_strength": conf["text_strength"],
        }

        if label is not None:
            target_delta = label - reg_text_base.detach()
            out["target_delta"] = target_delta
            out["delta_loss"] = F.smooth_l1_loss(delta, target_delta)

        return out
