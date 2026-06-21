# Final Results Ledger

**Date**: 2026-06-21
**Release**: p6am-final-freeze

## MOSI — Final Results

### Main Model (3-seed)

| Seed | ACC2 | F1 | MAE | Corr | ACC7 | Config |
|------|------|-----|-----|------|------|--------|
| 42 | 87.65% | 85.03 | 0.668 | 0.837 | 48.10 | `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s42.yaml` |
| 2024 | 87.50% | 85.51 | 0.699 | 0.816 | 45.48 | `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s2024.yaml` |
| 3407 | 88.26% | 86.03 | 0.644 | 0.852 | 48.54 | `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s3407.yaml` |
| **Mean** | **87.80%** | **85.52** | **0.670** | **0.835** | **47.37** | — |

**Status**: `final_candidate` — 3-seed freeze gate passed. Paper-ready.
**Architecture**: Text-Anchored Reliable Fusion (`canonical_mosi_text_anchored_tav`)
**Init**: P6K text+audio checkpoint (seed 42)
**Features**: `mosi_tav_v1` (data2vec 768d + CLIP-L14 1024d)

### Ablation (3-seed mean)

| Variant | ACC2 | Δ vs F0 | Status |
|---------|------|---------|--------|
| A1 Text-only | **88.11%** | +0.31 | Paper-ready |
| F0 Full TAV | 87.80% | — | Paper-ready |
| A4 No gate | 87.75% | -0.05 | Paper-ready |
| A5 No interaction | 87.70% | -0.10 | Paper-ready |
| A2 No audio corr | 87.55% | -0.25 | Paper-ready |
| A3 No vision corr | 87.55% | -0.25 | Paper-ready |

**Limits**: All ablation effects are within seed variation (std 0.33pp). Audio/vision/gate/interaction contributions are marginal on MOSI.

### Historical Reference

| Model | ACC2 | Status |
|-------|------|--------|
| P6K T+A conservative | 88.72% | `historical_reference` — different features (v3_T40), T+A only, not TAV |

---

## MOSEI — Final Results

### Main Model (TAV Canonical AWAF sLSTM)

| Metric | Value | Status |
|--------|-------|--------|
| ACC2_Non0 | 87.98% | `aligned_or_mixed_tav_reference` |
| F1_Non0 | 90.49% | — |
| MAE | 0.533 | — |
| Corr | 0.785 | — |
| ACC7 | 53.92% | — |

**Config**: `configs/experiments/p6aa_tav/mosei/control_awaf_slstm_s42.yaml`
**Status note**: P6AA used 20,680 samples (not strict 18,571 complete-case). Reclassified as `aligned_or_mixed_tav_reference`. Strict cohort (18,571) built but not re-trained.

### Ablation (4-epoch, 7 variants)

| Variant | ACC2 | Δ vs F0 | Significant? |
|---------|------|---------|--------------|
| F3 Global static | **79.30%** | +0.98 | ✅ F3 > F0 |
| F4 Fixed mean | **79.30%** | +0.98 | ✅ F4 > F0 |
| F2 T+V (no audio) | 78.72% | +0.40 | ❌ |
| F0 TAV AWAF sLSTM | 78.32% | — | Control |
| E1 LSTM | 77.78% | -0.54 | ❌ |
| F1 T+A (no vision) | 76.65% | -1.67 | ✅ F0 > F1 |
| F5 No interaction | 75.96% | -2.36 | ✅ F0 > F5 |

**Key findings**:
- Vision: +1.67pp (significant, 95% CI [+0.61, +2.73])
- Hadamard interaction: +2.37pp (largest effect, significant)
- Global static > dynamic AWAF at 4 epochs (+0.97pp, significant)
- sLSTM vs LSTM: +0.54pp (not significant)

---

## Experiment Status Classification

| Status | Definition |
|--------|------------|
| `final_candidate` | Frozen, multi-seed, paper-ready |
| `paper_ready` | Single seed or ablation, evidence complete |
| `aligned_or_mixed_tav_reference` | Valid TAV result but not strict complete-case |
| `historical_reference` | Valid result from different config/features, not TAV |
| `not_comparable` | Different cohort/protocol prevents direct comparison |
