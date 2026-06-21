# Final Model Architecture

**Date**: 2026-06-21
**Release**: p6am-final-freeze

## Overview

This project implements two multimodal sentiment analysis architectures:

| Dataset | Architecture | Fusion | Status |
|---------|-------------|--------|--------|
| **MOSEI** | Canonical AWAF sLSTM | Adaptive Weighted Attention Fusion | Primary 3-modal evidence |
| **MOSI** | Text-Anchored Reliable Fusion | Text anchor + gated corrections | Final TAV candidate |

The architecture choice is **dataset-size-dependent**: MOSEI (13,239 train) supports equal-opportunity softmax fusion; MOSI (1,284 train) requires a text-preserving architecture.

---

## MOSEI: Canonical AWAF sLSTM

```
Text:  Raw → RoBERTa-large + LoRA(r=16) → TextMLP → h_t
Audio: COVAREP 74d → Linear(74,256) → sLSTM(1 layer) → MaskedAttentionPool → h_a
Vision: OpenFace2 713d → Linear(713,256) → sLSTM(1 layer) → MaskedAttentionPool → h_v

Fusion: AdaptiveWeightedAttentionFusion (AWAF)
  Context: Cross-modal attention enhancement
  Interaction: g_ta=h_t⊙h_a, g_tv=h_t⊙h_v, g_av=h_a⊙h_v
  Scoring: MLP([h_t,h_a,h_v,g_ta,g_tv,g_av]) → softmax → [w_t,w_a,w_v]
  Output: z = w_t·h_t + w_a·h_a + w_v·h_v

Prediction: Linear(256,128) → ReLU → Linear(128,1)
```

**Key files**: `models/fusion/awaf.py`, `models/encoders/slstm.py`, `models/textft_lora_xlstm_awaf_residual.py` (mode: `canonical_text_audio_vision_awaf_slstm`)

## MOSI: Text-Anchored Reliable Fusion

```
Text:  Raw → RoBERTa-large + LoRA(r=16) → TextMLP → h_t
Audio: data2vec 768d → Linear(768,256) → sLSTM(1 layer) → MaskedAttentionPool → h_a
Vision: CLIP-L14 1024d → Linear(1024,256) → sLSTM(1 layer) → MaskedAttentionPool → h_v

Fusion: TextAnchoredReliableFusion
  y_text = TextHead(h_t)                              # Text anchor
  g_ta = h_t ⊙ h_a, g_tv = h_t ⊙ h_v                  # Interactions
  delta_a = MLP_a([h_t, h_a, g_ta])                    # Audio correction
  delta_v = MLP_v([h_t, h_v, g_tv])                    # Vision correction
  r_a = sigmoid(GateMLP_a([h_t, h_a, g_ta]) + b_a)    # Audio reliability gate
  r_v = sigmoid(GateMLP_v([h_t, h_v, g_tv]) + b_v)    # Vision reliability gate
  y_hat = clamp(y_text + α_a·r_a·delta_a + α_v·r_v·delta_v, -3, 3)

Initialization: P6K text+audio checkpoint (seed 42)
Gate bias: -2.0 (gates start near 0 → text-only behavior)
```

**Key files**: `models/fusion/text_anchored_reliable_fusion.py`, `models/textft_lora_xlstm_awaf_residual.py` (mode: `canonical_mosi_text_anchored_tav`)

## Shared Components

| Component | File | Used By |
|-----------|------|---------|
| RoBERTa + LoRA | `models/modules/minimal_lora.py` | Both |
| sLSTM Encoder | `models/encoders/slstm.py` | Both |
| MaskedAttentionPool | `models/pooling/attention_pooling.py` | Both |
| TextFTMultimodalDataset | `data/textft_multimodal_dataset.py` | Both |
| Metrics | `utils/metrics.py` | Both |

## Feature Versions

| Dataset | Audio | Vision | Text |
|---------|-------|--------|------|
| MOSEI | COVAREP 74d, 100 frames | OpenFace2 713d, 50 frames | Raw → RoBERTa-large |
| MOSI | data2vec 768d, 100 frames | CLIP-L14 1024d, 32 frames | Raw → RoBERTa-large |

**These are different feature sources and MUST be documented as such.**
