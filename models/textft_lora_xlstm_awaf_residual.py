"""
models/textft_lora_xlstm_awaf_residual.py — P6I TextFT LoRA xLSTM AWAF Residual 主模型

P6I 新增:
  - mode: text_only / audio_only / vision_only / av_only /
          text_audio_residual / text_vision_residual / text_av_residual /
          text_confidence_residual
  - TextConfidenceResidualHead: text-confidence conditioned residual
  - text_base 梯度保护 (detach for residual branch)
  - 三阶段训练支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from .modules.minimal_lora import apply_lora_to_roberta, mark_only_lora_as_trainable
from .encoders.slstm import SLSTMEncoder
from .pooling.attention_pooling import MaskedAttentionPooling
from .fusion.awaf import AdaptiveWeightedAttentionFusion
from .fusion.text_anchored_reliable_fusion import TextAnchoredReliableFusion
from .modules.uncertainty_residual_gate import UncertaintyGuidedResidualGate
from .modules.text_confidence_residual import (
    TextConfidenceResidualHead,
    TextConfidenceResidualConfig,
)


@dataclass
class TextFTLoRAConfig:
    """P6I 主模型配置。"""
    # --- Mode (P6I) ---
    mode: str = "text_av_residual"  # text_only|audio_only|vision_only|av_only|
                                     # text_audio_residual|text_vision_residual|
                                     # text_av_residual|text_confidence_residual

    # --- Text ---
    text_model_name: str = "roberta-large"
    text_hidden_dim: int = 1024
    text_mlp_hidden: int = 512
    text_dropout: float = 0.1

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ('query', 'value')

    # --- Common ---
    hidden_dim: int = 256
    audio_input_dim: int = 768
    vision_input_dim: int = 768

    # --- sLSTM ---
    slstm_num_layers: int = 1
    slstm_dropout: float = 0.2
    slstm_bidirectional: bool = False

    # --- AWAF ---
    awaf_fusion_mode: str = "awaf"
    tau_init: float = 3.0
    awaf_dropout: float = 0.1
    use_modality_dropout: bool = True
    modality_dropout_prob: float = 0.1
    use_modal_layernorm: bool = True
    awaf_uniform_mix: float = 0.0
    lambda_awaf_entropy: float = 0.01

    # --- Gate (old UGR, kept for backward compat) ---
    use_uncertainty_gate: bool = False   # P6I: default off, use text_confidence_residual instead
    gate_hidden_dim: int = 128
    gate_dropout: float = 0.1
    gate_init_bias: float = 2.0

    # --- Delta (old, kept for backward compat) ---
    use_delta_experts: bool = False      # P6I: default off
    use_bounded_delta: bool = True
    max_delta: float = 1.0
    delta_scale_init: float = 0.2

    # --- Text-Confidence Residual (P6I) ---
    use_text_conf_residual: bool = False
    tcr_gate_hidden_dim: int = 128
    tcr_delta_hidden_dim: int = 128
    tcr_dropout: float = 0.1
    tcr_max_delta: float = 1.0
    tcr_gate_floor: float = 0.2
    tcr_detach_text_for_residual: bool = True
    delta_loss_weight: float = 1.0

    # --- P6V Ablation ---
    fusion_type: str = "awaf"             # awaf | mean | concat | gated | fixed
    awaf_context: bool = True             # AWAF跨模态上下文增强
    awaf_interaction: bool = True         # AWAF二阶Hadamard交互项
    temporal_encoder: str = "slstm"       # slstm | gru | lstm | none
    fixed_fusion_weights: Tuple[float, float, float] = (0.5, 0.5, 0.0)  # text, audio, vision

    # --- Training hints ---
    device: str = "cuda"
    freeze_text_base: bool = False       # P6I: freeze text params for residual-only stage

    @property
    def needs_text(self) -> bool:
        return self.mode not in ('audio_only', 'vision_only', 'av_only')

    @property
    def needs_audio_branch(self) -> bool:
        # P6AB: canonical_text_vision_awaf_slstm has NO audio branch
        if self.mode == 'canonical_text_vision_awaf_slstm':
            return False
        # P6AG: text_anchored modes always need audio
        if 'text_anchored' in self.mode:
            return True
        return self.mode not in ('text_only', 'vision_only', 'text_vision_residual')

    @property
    def needs_vision_branch(self) -> bool:
        # P6AB: canonical_text_audio_awaf_slstm uses dummy vision, no real vision branch needed
        if self.mode == 'canonical_text_audio_awaf_slstm':
            return False
        # P6AG: text_anchored modes always need vision
        if 'text_anchored' in self.mode:
            return True
        return self.mode not in ('text_only', 'audio_only', 'text_audio_residual')

    @property
    def needs_awaf(self) -> bool:
        # P6AG: text_anchored mode uses its own fusion, not AWAF
        if 'text_anchored' in self.mode:
            return False
        if self.mode in ('canonical_text_audio_awaf_slstm', 'canonical_text_audio_vision_awaf_slstm',
                         'canonical_text_vision_awaf_slstm'):
            return True
        return self.mode in ('text_av_residual', 'text_confidence_residual', 'av_only', 'text_audio_residual')

    @property
    def needs_text_conf_residual(self) -> bool:
        return self.mode == 'text_confidence_residual'

    @property
    def is_text_only(self) -> bool:
        return self.mode == 'text_only'


class TextFTLoRAXLSTMAWAFResidual(nn.Module):
    """P6I 多模态主模型 (支持 8 种训练模式)。"""

    def __init__(self, config: TextFTLoRAConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim
        D = config.text_hidden_dim
        DEVICE = config.device

        # ============================================================
        # 1. RoBERTa + Minimal LoRA (仅 text 模式需要)
        # ============================================================
        if config.needs_text:
            from transformers import AutoModel
            self.roberta = AutoModel.from_pretrained(config.text_model_name)
            self.roberta = apply_lora_to_roberta(
                self.roberta, r=config.lora_r, alpha=config.lora_alpha,
                dropout=config.lora_dropout, target_patterns=list(config.lora_targets),
            )
            mark_only_lora_as_trainable(self.roberta)

            self.text_mlp = nn.Sequential(
                nn.Linear(D, config.text_mlp_hidden), nn.LayerNorm(config.text_mlp_hidden),
                nn.GELU(), nn.Dropout(config.text_dropout),
                nn.Linear(config.text_mlp_hidden, H), nn.LayerNorm(H),
                nn.GELU(), nn.Dropout(config.text_dropout),
            )
            self.reg_head_text = nn.Linear(H, 1)
            self.cls_head_text = nn.Linear(H, 1)

        # ============================================================
        # 2. Audio branch
        # ============================================================
        self._temporal_encoder_type = getattr(config, 'temporal_encoder', 'slstm')
        if config.needs_audio_branch:
            self.audio_proj = nn.Sequential(
                nn.Linear(config.audio_input_dim, H), nn.LayerNorm(H),
                nn.GELU(), nn.Dropout(config.slstm_dropout),
            )
            if self._temporal_encoder_type == 'slstm':
                self.audio_temporal = SLSTMEncoder(
                    H, H, config.slstm_num_layers, dropout=config.slstm_dropout,
                    bidirectional=config.slstm_bidirectional, pooling='masked_mean',
                )
            elif self._temporal_encoder_type == 'gru':
                self.audio_temporal = nn.GRU(H, H, num_layers=1, batch_first=True, bidirectional=False)
                self.audio_temporal_proj = nn.Linear(H, H)
            elif self._temporal_encoder_type == 'lstm':
                self.audio_temporal = nn.LSTM(H, H, num_layers=1, batch_first=True, bidirectional=False)
                self.audio_temporal_proj = nn.Linear(H, H)
            else:  # none
                self.audio_temporal = None
            self.audio_pool = MaskedAttentionPooling(H, dropout=config.slstm_dropout)

        # ============================================================
        # 3. Vision branch
        # ============================================================
        if config.needs_vision_branch:
            self.vision_proj = nn.Sequential(
                nn.Linear(config.vision_input_dim, H), nn.LayerNorm(H),
                nn.GELU(), nn.Dropout(config.slstm_dropout),
            )
            if self._temporal_encoder_type == 'slstm':
                self.vision_temporal = SLSTMEncoder(
                    H, H, config.slstm_num_layers, dropout=config.slstm_dropout,
                    bidirectional=config.slstm_bidirectional, pooling='masked_mean',
                )
            elif self._temporal_encoder_type == 'gru':
                self.vision_temporal = nn.GRU(H, H, num_layers=1, batch_first=True, bidirectional=False)
                self.vision_temporal_proj = nn.Linear(H, H)
            elif self._temporal_encoder_type == 'lstm':
                self.vision_temporal = nn.LSTM(H, H, num_layers=1, batch_first=True, bidirectional=False)
                self.vision_temporal_proj = nn.Linear(H, H)
            else:
                self.vision_temporal = None
            self.vision_pool = MaskedAttentionPooling(H, dropout=config.slstm_dropout)

        # ============================================================
        # 4. AV-only reg head (for audio_only/vision_only/av_only)
        # ============================================================
        if not config.needs_text:
            self.av_reg_head = nn.Sequential(
                nn.Linear(H, H // 2), nn.LayerNorm(H // 2), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(H // 2, 1),
            )

        # ============================================================
        # 5. AWAF (supports all fusion_type variants via fusion_mode)
        # ============================================================
        # P6W-C: Canonical mode creates dedicated 2-modality AWAF-style fusion
        self._is_canonical = config.mode in ('canonical_text_audio_awaf_slstm', 'canonical_text_audio_vision_awaf_slstm',
                                              'canonical_text_vision_awaf_slstm')
        # P6AB: resolve fusion_mode from config (shared between canonical and residual paths)
        ft = getattr(config, 'fusion_type', 'awaf')
        if ft == 'awaf':
            if not getattr(config, 'awaf_context', True):
                fm = 'awaf_no_context'
            elif not getattr(config, 'awaf_interaction', True):
                fm = 'awaf_no_interaction'
            else:
                fm = 'awaf'
        elif ft == 'awaf_no_context':
            fm = 'awaf_no_context'
        elif ft == 'awaf_no_interaction':
            fm = 'awaf_no_interaction'
        elif ft == 'mean':
            fm = 'mean'
        elif ft == 'concat':
            fm = 'concat'
        elif ft == 'gated':
            fm = 'gated'
        elif ft == 'fixed':
            fm = 'fixed'
        else:
            fm = getattr(config, 'awaf_fusion_mode', 'awaf')

        if self._is_canonical:
            self.canonical_fusion = AdaptiveWeightedAttentionFusion(
                hidden_dim=H, fusion_mode=fm,
                tau_init=config.tau_init, dropout=config.awaf_dropout,
                use_modality_dropout=config.use_modality_dropout,
                modality_dropout_prob=config.modality_dropout_prob,
                use_modal_layernorm=config.use_modal_layernorm,
                return_diagnostics=True,
            )
            self.canonical_head = nn.Sequential(
                nn.Linear(H, H // 2), nn.ReLU(),
                nn.Linear(H // 2, 1),
            )

        # P6AG: Text-Anchored Reliable Fusion (for MOSI small-dataset TAV)
        self._is_text_anchored = 'text_anchored' in config.mode
        if self._is_text_anchored:
            # P6AK: resolve ablation flags from mode string
            _no_audio_corr = 'no_audio_corr' in config.mode
            _no_vision_corr = 'no_vision_corr' in config.mode
            _no_gate = 'no_gate' in config.mode
            _no_interaction = 'no_interaction' in config.mode
            self.text_anchored_fusion = TextAnchoredReliableFusion(
                hidden_dim=H,
                correction_hidden_dim=128,
                gate_hidden_dim=64,
                dropout=config.awaf_dropout,
                gate_init_bias=-2.0,
                no_audio_correction=_no_audio_corr,
                no_vision_correction=_no_vision_corr,
                no_reliability_gate=_no_gate,
                no_interaction=_no_interaction,
            )

        # ============================================================
        if config.needs_awaf:
            # P6AB: reuse fm resolved above (with awaf_context/awaf_interaction support)
            self.awaf = AdaptiveWeightedAttentionFusion(
                hidden_dim=H, fusion_mode=fm,
                tau_init=config.tau_init, dropout=config.awaf_dropout,
                use_modality_dropout=config.use_modality_dropout,
                modality_dropout_prob=config.modality_dropout_prob,
                use_modal_layernorm=config.use_modal_layernorm,
                awaf_uniform_mix=config.awaf_uniform_mix,
                return_diagnostics=True,
            )
            # P6V: fusion correction head (maps AWAF output Z to scalar for residual)
            if config.mode == 'text_audio_residual':
                self.fusion_correction = nn.Sequential(
                    nn.Linear(H, H // 2), nn.ReLU(),
                    nn.Linear(H // 2, 1),
                )

        # ============================================================
        # 6. Old UGR Gate (backward compat)
        # ============================================================
        self.use_gate = config.use_uncertainty_gate
        if self.use_gate:
            self.gate = UncertaintyGuidedResidualGate(
                hidden_dim=H, gate_hidden_dim=config.gate_hidden_dim,
                dropout=config.gate_dropout, init_bias=config.gate_init_bias,
            )

        # ============================================================
        # 7. Old Delta experts (backward compat)
        # ============================================================
        self.use_delta = config.use_delta_experts
        if self.use_delta:
            def _make_de():
                return nn.Sequential(
                    nn.Linear(H, H // 2), nn.LayerNorm(H // 2), nn.GELU(),
                    nn.Dropout(0.2), nn.Linear(H // 2, 1),
                )
            self.delta_reg_t = _make_de(); self.delta_reg_a = _make_de()
            self.delta_reg_v = _make_de()
            self.delta_scale_reg = nn.Parameter(torch.tensor(config.delta_scale_init))
            self.delta_scale_cls = nn.Parameter(torch.tensor(config.delta_scale_init))

        # ============================================================
        # 8. Text-Confidence Residual Head (P6I)
        # ============================================================
        self.use_tcr = config.use_text_conf_residual
        if self.use_tcr:
            tcr_cfg = TextConfidenceResidualConfig(
                hidden_dim=H,
                gate_hidden_dim=config.tcr_gate_hidden_dim,
                delta_hidden_dim=config.tcr_delta_hidden_dim,
                dropout=config.tcr_dropout,
                max_delta=config.tcr_max_delta,
                gate_floor=config.tcr_gate_floor,
                detach_text_for_residual=config.tcr_detach_text_for_residual,
                use_av_interaction=True,
            )
            self.text_conf_residual = TextConfidenceResidualHead(tcr_cfg)

        # Freeze text if requested
        if config.freeze_text_base and config.needs_text:
            self._set_text_trainable(False)

    def _set_text_trainable(self, trainable: bool):
        """冻结/解冻 text branch 参数。"""
        for p in self.roberta.parameters():
            p.requires_grad = trainable
        for p in self.text_mlp.parameters():
            p.requires_grad = trainable
        for p in self.reg_head_text.parameters():
            p.requires_grad = trainable
        for p in self.cls_head_text.parameters():
            p.requires_grad = trainable

    def _compute_text(self, ids, am):
        """Compute text features and predictions."""
        ro = self.roberta(input_ids=ids, attention_mask=am)
        ht = ro.last_hidden_state[:, 0, :]  # [B, 1024]
        ht = self.text_mlp(ht)              # [B, H]
        rtb = self.reg_head_text(ht)        # [B, 1]
        ctb = self.cls_head_text(ht)        # [B, 1]
        return ht, rtb, ctb

    def _apply_temporal(self, x, mask, temporal_module, has_proj=False):
        """Apply temporal encoder (sLSTM/GRU/LSTM/none) to projected features."""
        if temporal_module is None:  # none mode
            return x.mean(dim=1)  # [B, T, H] -> [B, H]
        if self._temporal_encoder_type == 'slstm':
            out = temporal_module(x, mask)
            return out['H'].mean(dim=1)  # mean pool over time
        else:  # gru or lstm
            out, _ = temporal_module(x)
            # Use last step or mean
            pooled = out.mean(dim=1)
            if has_proj:
                pooled = getattr(self, f'{temporal_module.__class__.__name__.lower()}_proj', lambda z: z)(pooled)
            return pooled

    def _compute_audio(self, a, am_a):
        ha = self.audio_proj(a)
        if self.audio_temporal is None:  # no temporal
            hap, _ = self.audio_pool(ha, am_a)
        elif self._temporal_encoder_type == 'slstm':
            hao = self.audio_temporal(ha, am_a)
            hap, _ = self.audio_pool(hao['H'], am_a)
        else:  # gru or lstm
            out, _ = self.audio_temporal(ha)
            hap, _ = self.audio_pool(out, am_a)
        return hap

    def _compute_vision(self, v, vm_v):
        hv = self.vision_proj(v)
        if self.vision_temporal is None:
            hvp, _ = self.vision_pool(hv, vm_v)
        elif self._temporal_encoder_type == 'slstm':
            hvo = self.vision_temporal(hv, vm_v)
            hvp, _ = self.vision_pool(hvo['H'], vm_v)
        else:
            out, _ = self.vision_temporal(hv)
            hvp, _ = self.vision_pool(out, vm_v)
        return hvp

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mode = self.config.mode
        DEVICE = self.config.device
        lbl = batch.get('label')
        if lbl is not None:
            lbl = lbl.to(DEVICE)

        # === Text branch ===
        ht, rtb, ctb = None, None, None
        if self.config.needs_text:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            ht, rtb, ctb = self._compute_text(ids, am)

        # === Audio branch ===
        hap = None
        if self.config.needs_audio_branch:
            a = batch['audio'].to(DEVICE)
            am_a = batch['audio_mask'].to(DEVICE)
            hap = self._compute_audio(a, am_a)

        # === Vision branch ===
        hvp = None
        if self.config.needs_vision_branch and mode != 'text_audio_residual' or mode in ('canonical_text_audio_vision_awaf_slstm', 'canonical_text_vision_awaf_slstm'):
            v = batch.get('vision', torch.zeros(1, 1, 768, device=DEVICE))
            vm_v = batch.get('vision_mask', torch.zeros(1, 1, device=DEVICE))
            if v is not None and vm_v is not None and vm_v.sum() > 0:
                v = v.to(DEVICE); vm_v = vm_v.to(DEVICE)
                hvp = self._compute_vision(v, vm_v)

        # === Mode-specific forward ===
        if mode == 'canonical_text_audio_vision_awaf_slstm':
            return self._forward_canonical_tav_awaf(ht, hap, hvp, lbl)
        elif mode == 'canonical_text_audio_awaf_slstm':
            return self._forward_canonical_ta_awaf(ht, hap, lbl)
        elif mode == 'canonical_text_vision_awaf_slstm':
            return self._forward_canonical_tv_awaf(ht, hvp, lbl)
        elif mode == 'text_only':
            return self._forward_text_only(ht, rtb, ctb, lbl)
        elif mode == 'audio_only':
            return self._forward_audio_only(hap, lbl)
        elif mode == 'vision_only':
            return self._forward_vision_only(hvp, lbl)
        elif mode == 'av_only':
            return self._forward_av_only(ht, hap, hvp, lbl)
        elif mode == 'text_audio_residual':
            return self._forward_text_x_residual(ht, rtb, ctb, hap, None, 'audio', lbl)
        elif mode == 'text_vision_residual':
            return self._forward_text_x_residual(ht, rtb, ctb, None, hvp, 'vision', lbl)
        elif mode == 'text_av_residual':
            return self._forward_text_av_residual(ht, rtb, ctb, hap, hvp, lbl)
        elif mode == 'text_confidence_residual':
            return self._forward_text_conf_residual(ht, rtb, ctb, hap, hvp, lbl)
        elif 'text_anchored' in mode:
            return self._forward_mosi_text_anchored_tav(ht, hap, hvp, lbl)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ================================================================
    # Mode-specific forward implementations
    # ================================================================
    def _forward_canonical_ta_awaf(self, ht, hap, lbl):
        """P6W-C Canonical: text+audio → AWAF → head. NO bypass, NO gate, NO delta."""
        dummy_v = torch.zeros_like(hap)
        aw = self.canonical_fusion(ht, hap, dummy_v)
        z = aw['Z']; w = aw['weights']
        reg = self.canonical_head(z)
        return {'reg': reg, 'z_fused': z, 'awaf_weights': w,
                'reg_text_base': reg, 'cls_text_base': reg}

    def _forward_canonical_tav_awaf(self, ht, hap, hvp, lbl):
        """P6AA TAV Canonical: text+audio+vision → AWAF → head. NO bypass."""
        aw = self.canonical_fusion(ht, hap, hvp)
        z = aw['Z']; w = aw['weights']
        reg = self.canonical_head(z)
        return {'reg': reg, 'z_fused': z, 'awaf_weights': w,
                'reg_text_base': reg, 'cls_text_base': reg}

    def _forward_canonical_tv_awaf(self, ht, hvp, lbl):
        """P6AB TV Canonical: text+vision → AWAF → head. Dummy audio, NO bypass."""
        dummy_a = torch.zeros_like(hvp)
        aw = self.canonical_fusion(ht, dummy_a, hvp)
        z = aw['Z']; w = aw['weights']
        reg = self.canonical_head(z)
        return {'reg': reg, 'z_fused': z, 'awaf_weights': w,
                'reg_text_base': reg, 'cls_text_base': reg}

    def _forward_mosi_text_anchored_tav(self, ht, hap, hvp, lbl):
        """P6AG MOSI Text-Anchored TAV: text anchor + reliability-gated audio/vision corrections."""
        # Handle None for counterfactual/testing
        if hap is None:
            hap = torch.zeros_like(ht)
        if hvp is None:
            hvp = torch.zeros_like(ht)
        out = self.text_anchored_fusion(ht, hap, hvp)
        return {
            'reg': out['y_hat'],
            'reg_text_base': out['y_text'],
            'cls_text_base': out['y_text'],
            'y_text': out['y_text'],
            'delta_a': out['delta_a'],
            'delta_v': out['delta_v'],
            'r_a': out['r_a'],
            'r_v': out['r_v'],
        }

    def _forward_text_only(self, ht, rtb, ctb, lbl):
        return {
            'reg': rtb, 'cls': ctb,
            'reg_text_base': rtb, 'cls_text_base': ctb,
        }

    def _forward_audio_only(self, hap, lbl):
        reg = self.av_reg_head(hap)
        return {'reg': reg, 'reg_text_base': reg}

    def _forward_vision_only(self, hvp, lbl):
        reg = self.av_reg_head(hvp)
        return {'reg': reg, 'reg_text_base': reg}

    def _forward_av_only(self, ht, hap, hvp, lbl):
        # AWAF with dummy text (zeros) or just audio+vision
        aw = self.awaf(
            torch.zeros_like(hap) if ht is None else ht.detach() if not self.config.needs_text else ht,
            hap, hvp)
        z = aw['Z']
        reg = self.av_reg_head(z)
        return {'reg': reg, 'reg_text_base': reg, 'awaf_weights': aw['weights']}

    def _forward_text_x_residual(self, ht, rtb, ctb, hap, hvp, which, lbl):
        """Text + single modality residual (P6V: AWAF fusion with correction head)."""

        # === P6V: Use AWAF fusion correction when available ===
        if hasattr(self, 'fusion_correction') and self.awaf is not None:
            # AWAF fuses text+audio → Z, then project to scalar correction
            if which == 'audio':
                dummy_v = torch.zeros_like(hap)
                aw = self.awaf(ht, hap, dummy_v)
            elif which == 'vision':
                dummy_a = torch.zeros_like(hvp)
                aw = self.awaf(ht, dummy_a, hvp)
            else:
                aw = self.awaf(ht, hap, hvp)
            z = aw['Z']  # [B, H]
            fcorr = self.fusion_correction(z)  # [B, 1]
            reg = rtb + fcorr
            return {
                'reg': reg, 'reg_text_base': rtb, 'cls_text_base': ctb,
                'awaf_weights': aw.get('weights', None),
                'gate_reg': torch.ones_like(rtb),
                'delta_reg': fcorr, 'effective_delta_reg': fcorr,
                'delta_scale_reg': torch.tensor(1.0, device=rtb.device),
            }

        # === Legacy path (no AWAF) ===
        if self.use_delta:
            if which == 'audio':
                dr = self.delta_reg_a(hap)
            elif which == 'vision':
                dr = self.delta_reg_v(hvp)
            else:
                dr = (self.delta_reg_a(hap) + self.delta_reg_v(hvp)) / 2.0
            bdr = self.config.max_delta * torch.tanh(dr) if self.config.use_bounded_delta else dr
            dsr = self.delta_scale_reg
        else:
            bdr = torch.zeros_like(rtb)
            dsr = torch.tensor(0.0, device=rtb.device)
            dr = torch.zeros_like(rtb)  # No delta → zero delta regression

        # Gate
        if self.use_gate:
            dummy_aw = torch.zeros(ht.size(0), 3, device=ht.device)  # avoid dim mismatch
            go = self.gate(ht, z, rtb, ctb, dummy_aw, dr if self.use_delta else None)
            gr = go['gate_reg']
        else:
            gr = torch.ones_like(rtb)

        edr = gr * dsr * bdr
        reg = rtb + edr

        return {
            'reg': reg, 'reg_text_base': rtb, 'cls_text_base': ctb,
            'gate_reg': gr, 'delta_reg': dr, 'effective_delta_reg': edr,
            'delta_scale_reg': dsr,
        }

    def _forward_text_av_residual(self, ht, rtb, ctb, hap, hvp, lbl):
        """Text + AV residual (old gate+delta path with AWAF)."""
        aw = self.awaf(ht, hap, hvp)
        z = aw['Z']; w = aw['weights']

        if self.use_delta:
            dr_t = self.delta_reg_t(ht); dr_a = self.delta_reg_a(hap); dr_v = self.delta_reg_v(hvp)
            dr = w[:, 0:1] * dr_t + w[:, 1:2] * dr_a + w[:, 2:3] * dr_v
            bdr = self.config.max_delta * torch.tanh(dr) if self.config.use_bounded_delta else dr
            dsr = self.delta_scale_reg
        else:
            bdr = torch.zeros_like(rtb)
            dsr = torch.tensor(0.0, device=rtb.device)
            dr = bdr

        if self.use_gate:
            go = self.gate(ht, z, rtb, ctb, w, dr)
            gr = go['gate_reg']
        else:
            gr = torch.ones_like(rtb)

        edr = gr * dsr * bdr
        reg = rtb + edr

        return {
            'reg': reg, 'reg_text_base': rtb, 'cls_text_base': ctb,
            'awaf_weights': w, 'awaf_Z': z,
            'awaf_diagnostics': aw.get('diagnostics', {}),
            'gate_reg': gr, 'delta_reg': dr, 'effective_delta_reg': edr,
            'delta_scale_reg': dsr,
        }

    def _forward_text_conf_residual(self, ht, rtb, ctb, hap, hvp, lbl):
        """Text-confidence conditioned residual (P6I new architecture)."""
        # AWAF for AV fusion
        aw = self.awaf(ht, hap, hvp)
        z_av = aw['Z']; w = aw['weights']

        # Text-confidence residual
        tcr_out = self.text_conf_residual(
            h_t=ht, h_a=hap, h_v=hvp, z_av=z_av,
            reg_text_base=rtb, cls_text_base=ctb, label=lbl,
        )

        return {
            'reg': tcr_out['reg_final'],
            'reg_text_base': rtb,
            'cls_text_base': ctb,
            'awaf_weights': w, 'awaf_Z': z_av,
            'awaf_diagnostics': aw.get('diagnostics', {}),
            'delta': tcr_out['delta'],
            'gate': tcr_out['gate'],
            'text_confidence': tcr_out['text_confidence'],
            'text_uncertainty': tcr_out['text_uncertainty'],
            'target_delta': tcr_out.get('target_delta'),
            'delta_loss': tcr_out.get('delta_loss'),
            'effective_delta_reg': tcr_out['gate'] * tcr_out['delta'],
        }

    # ================================================================
    # Helpers
    # ================================================================
    def collect_trainable_params(self, stage: str = 'all') -> List[nn.Parameter]:
        """收集可训练参数，支持分阶段。

        Args:
            stage: 'all' | 'text_only' | 'residual_only'
        """
        params = []

        if stage in ('all', 'text_only'):
            if self.config.needs_text:
                params += list(self.roberta.parameters())
                params += list(self.text_mlp.parameters())
                params += list(self.reg_head_text.parameters())
                params += list(self.cls_head_text.parameters())

        if stage in ('all', 'residual_only'):
            if self.config.needs_audio_branch:
                for m in [self.audio_proj, self.audio_temporal, self.audio_pool]:
                    params += list(m.parameters())
            if self.config.needs_vision_branch:
                for m in [self.vision_proj, self.vision_temporal, self.vision_pool]:
                    params += list(m.parameters())
            if self.config.needs_awaf:
                params += list(self.awaf.parameters())
            if self.use_delta:
                for m in [self.delta_reg_t, self.delta_reg_a, self.delta_reg_v]:
                    params += list(m.parameters())
                params += [self.delta_scale_reg, self.delta_scale_cls]
            if self.use_gate:
                params += list(self.gate.parameters())
            if self.use_tcr:
                params += list(self.text_conf_residual.parameters())
            if not self.config.needs_text:
                params += list(self.av_reg_head.parameters())

        return params

    def count_trainable(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_M': total / 1e6, 'trainable_M': trainable / 1e6}
