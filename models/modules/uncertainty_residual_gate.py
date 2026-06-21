"""
models/modules/uncertainty_residual_gate.py

P5E: Uncertainty-Guided Residual Gate (UGR)

目的：避免 residual correction 无条件修改所有样本。
核心先验：
  - |reg_text_base| 越大，文本越自信，residual gate 应越小；
  - |reg_text_base| 越接近 0，文本越不确定，residual gate 应越大；
  - AWAF entropy / delta magnitude 可作为辅助不确定性信号。

输出：
  gate_reg, gate_cls ∈ [0, 1]
  gate_prior: 基于文本置信度的可解释先验门控
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGuidedResidualGate(nn.Module):
    """不确定性感知残差门控。

    Args:
        hidden_dim: h_text_base / z_residual 的维度。
        gate_hidden_dim: 门控 MLP 隐层维度。
        dropout: MLP dropout。
        init_bias: 最后一层 bias，负值使初始 gate 较小。
        margin_init: 文本置信边界，|reg_text_base| 小于该边界时更倾向使用 residual。
        prior_temperature_init: gate_prior 的温度，越小越陡。
        blend_learned_and_prior: 若为 True，最终 gate = learned_gate * gate_prior；
                                 若为 False，直接使用 learned_gate。
    """

    def __init__(
        self,
        hidden_dim: int,
        gate_hidden_dim: int = 128,
        dropout: float = 0.1,
        init_bias: float = -1.5,
        margin_init: float = 0.75,
        prior_temperature_init: float = 0.35,
        use_awaf_entropy: bool = True,
        use_delta_magnitude: bool = True,
        blend_learned_and_prior: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_awaf_entropy = use_awaf_entropy
        self.use_delta_magnitude = use_delta_magnitude
        self.blend_learned_and_prior = blend_learned_and_prior
        self.eps = eps

        input_dim = 2 * hidden_dim + 3  # h_text, z_res, |reg|, |cls|, gate_prior
        if use_awaf_entropy:
            input_dim += 1
        if use_delta_magnitude:
            input_dim += 1

        self.margin_raw = nn.Parameter(torch.tensor(float(margin_init)))
        self.temperature_raw = nn.Parameter(torch.tensor(float(prior_temperature_init)))

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim // 2),
            nn.LayerNorm(gate_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim // 2, 2),
        )
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(init_bias)

    @property
    def margin(self) -> torch.Tensor:
        return F.softplus(self.margin_raw).clamp(min=0.05, max=3.0)

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw).clamp(min=0.05, max=3.0)

    def _gate_prior(self, reg_text_base: torch.Tensor) -> torch.Tensor:
        # |reg| 越小，prior 越大；|reg| 超过 margin 后 prior 变小。
        abs_reg = reg_text_base.abs()
        return torch.sigmoid((self.margin - abs_reg) / self.temperature)

    def forward(
        self,
        h_text_base: torch.Tensor,
        z_residual: torch.Tensor,
        reg_text_base: torch.Tensor,
        cls_text_base: torch.Tensor,
        awaf_weights: Optional[torch.Tensor] = None,
        delta_reg: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gate_prior = self._gate_prior(reg_text_base)
        features = [
            h_text_base,
            z_residual,
            reg_text_base.abs(),
            cls_text_base.abs(),
            gate_prior,
        ]

        if self.use_awaf_entropy and awaf_weights is not None:
            w = awaf_weights.clamp(min=self.eps)
            entropy = -(w * torch.log(w)).sum(-1, keepdim=True)
            features.append(entropy)

        if self.use_delta_magnitude and delta_reg is not None:
            features.append(delta_reg.abs())

        feat = torch.cat(features, dim=-1)
        raw = self.gate_net(feat)
        learned_gate_reg = torch.sigmoid(raw[:, 0:1])
        learned_gate_cls = torch.sigmoid(raw[:, 1:2])

        if self.blend_learned_and_prior:
            gate_reg = learned_gate_reg * gate_prior
            gate_cls = learned_gate_cls * gate_prior
        else:
            gate_reg = learned_gate_reg
            gate_cls = learned_gate_cls

        return {
            "gate_reg": gate_reg,
            "gate_cls": gate_cls,
            "gate_prior": gate_prior,
            "learned_gate_reg": learned_gate_reg,
            "learned_gate_cls": learned_gate_cls,
            "gate_margin": self.margin.detach(),
            "gate_temperature": self.temperature.detach(),
        }


def gate_stats(gate: torch.Tensor, name: str = "gate") -> Dict[str, float]:
    return {
        f"{name}_mean": float(gate.mean().detach().cpu()),
        f"{name}_std": float(gate.std().detach().cpu()),
        f"{name}_min": float(gate.min().detach().cpu()),
        f"{name}_max": float(gate.max().detach().cpu()),
        f"{name}_lt_001": float((gate < 0.01).float().mean().detach().cpu()),
        f"{name}_gt_099": float((gate > 0.99).float().mean().detach().cpu()),
    }
