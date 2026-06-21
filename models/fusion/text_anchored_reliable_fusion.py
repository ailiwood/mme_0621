"""
P6AG: Text-Anchored Reliable Fusion

Hypothesis: Text is the strong primary modality on MOSI. Audio and vision
are conditional auxiliary modalities. Fusion must preserve the text prediction
anchor and only allow reliable audio/vision corrections through per-sample
reliability gates.

Architecture:
  y_text = TextHead(h_t)                    -- text anchor prediction
  g_ta = h_t * h_a                          -- text-audio interaction
  g_tv = h_t * h_v                          -- text-vision interaction
  delta_a = MLP_a([h_t, h_a, g_ta])         -- audio correction
  delta_v = MLP_v([h_t, h_v, g_tv])         -- vision correction
  r_a = sigmoid(GateMLP_a([h_t, h_a, g_ta]) + b_a)  -- audio reliability gate
  r_v = sigmoid(GateMLP_v([h_t, h_v, g_tv]) + b_v)  -- vision reliability gate
  y_hat = clamp(y_text + alpha_a*r_a*delta_a + alpha_v*r_v*delta_v, -3, 3)

Key properties:
  - Text prediction always contributes (anchor)
  - Audio/vision only add reliability-gated residuals
  - Gates initialized near 0 (negative bias) → model starts text-only
  - Gates learn per-sample which modality corrections to trust
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TextAnchoredReliableFusion(nn.Module):
    """
    Text-anchored reliable fusion for small-dataset multimodal learning.

    Args:
        hidden_dim: Dimension of h_t, h_a, h_v (all must be same)
        correction_hidden_dim: Hidden dim for correction MLPs
        gate_hidden_dim: Hidden dim for reliability gate MLPs
        dropout: Dropout rate
        gate_init_bias: Initial bias for reliability gates (negative → start near 0)
        alpha_init: Initial value for alpha scaling factors
        clamp_min, clamp_max: Output clamping range
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        correction_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.1,
        gate_init_bias: float = -2.0,
        alpha_init: float = 1.0,
        clamp_min: float = -3.0,
        clamp_max: float = 3.0,
        # P6AK ablation flags
        no_audio_correction: bool = False,
        no_vision_correction: bool = False,
        no_reliability_gate: bool = False,
        no_interaction: bool = False,
    ):
        super().__init__()
        H = hidden_dim
        CH = correction_hidden_dim
        GH = gate_hidden_dim

        # === Text prediction head (anchor) ===
        self.text_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.LayerNorm(H // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 2, 1),
        )

        # === Audio correction MLP ===
        # Input: [h_t, h_a, g_ta] = 3*H
        self.audio_correction = nn.Sequential(
            nn.Linear(3 * H, CH),
            nn.LayerNorm(CH),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(CH, CH // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(CH // 2, 1),
        )

        # === Vision correction MLP ===
        self.vision_correction = nn.Sequential(
            nn.Linear(3 * H, CH),
            nn.LayerNorm(CH),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(CH, CH // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(CH // 2, 1),
        )

        # === Audio reliability gate ===
        self.audio_gate = nn.Sequential(
            nn.Linear(3 * H, GH),
            nn.LayerNorm(GH),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(GH, 1),
        )

        # === Vision reliability gate ===
        self.vision_gate = nn.Sequential(
            nn.Linear(3 * H, GH),
            nn.LayerNorm(GH),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(GH, 1),
        )

        # === Learnable scaling factors (v2: bounded via sigmoid) ===
        self.raw_alpha_a = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 → 0.5*max_alpha
        self.raw_alpha_v = nn.Parameter(torch.tensor(0.0))
        self.max_alpha = 1.0

        # === Gate bias (initialized negative → gates start near 0) ===
        # P6AG v2: Learnable gate biases (initialized negative for safe start)
        self.gate_bias_a = nn.Parameter(torch.tensor(gate_init_bias))
        self.gate_bias_v = nn.Parameter(torch.tensor(gate_init_bias))

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # P6AK ablation flags
        self.no_audio_correction = no_audio_correction
        self.no_vision_correction = no_vision_correction
        self.no_reliability_gate = no_reliability_gate
        self.no_interaction = no_interaction

    def forward(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_t: Text representation [B, H]
            h_a: Audio representation [B, H]
            h_v: Vision representation [B, H]

        Returns dict with:
            y_text: Text anchor prediction [B, 1]
            delta_a: Audio correction [B, 1]
            delta_v: Vision correction [B, 1]
            r_a: Audio reliability gate [B, 1]
            r_v: Vision reliability gate [B, 1]
            y_hat: Final prediction [B, 1]
        """
        B = h_t.shape[0]

        # --- Text anchor prediction ---
        y_text = self.text_head(h_t)  # [B, 1]

        # --- Interactions (text-anchored only) ---
        if self.no_interaction:
            g_ta = torch.zeros_like(h_t)
            g_tv = torch.zeros_like(h_t)
        else:
            g_ta = h_t * h_a  # [B, H]
            g_tv = h_t * h_v  # [B, H]

        # --- Audio correction ---
        audio_input = torch.cat([h_t, h_a, g_ta], dim=-1)  # [B, 3H]
        delta_a = self.audio_correction(audio_input)  # [B, 1]

        # --- Vision correction ---
        vision_input = torch.cat([h_t, h_v, g_tv], dim=-1)  # [B, 3H]
        delta_v = self.vision_correction(vision_input)  # [B, 1]

        # --- Reliability gates ---
        if self.no_reliability_gate:
            r_a = torch.ones_like(y_text)
            r_v = torch.ones_like(y_text)
        else:
            r_a_logit = self.audio_gate(audio_input) + self.gate_bias_a  # [B, 1]
            r_v_logit = self.vision_gate(vision_input) + self.gate_bias_v  # [B, 1]
            r_a = torch.sigmoid(r_a_logit)  # [B, 1]
            r_v = torch.sigmoid(r_v_logit)  # [B, 1]

        # --- Final prediction (v2: bounded alpha) ---
        alpha_a = self.max_alpha * torch.sigmoid(self.raw_alpha_a)
        alpha_v = self.max_alpha * torch.sigmoid(self.raw_alpha_v)

        # P6AK ablation: zero out corrections
        if self.no_audio_correction:
            contrib_a = torch.zeros_like(y_text)
        else:
            contrib_a = alpha_a * r_a * delta_a

        if self.no_vision_correction:
            contrib_v = torch.zeros_like(y_text)
        else:
            contrib_v = alpha_v * r_v * delta_v

        y_hat = y_text + contrib_a + contrib_v
        y_hat = torch.clamp(y_hat, self.clamp_min, self.clamp_max)

        return {
            'y_text': y_text,
            'delta_a': delta_a,
            'delta_v': delta_v,
            'r_a': r_a,
            'r_v': r_v,
            'y_hat': y_hat,
        }
