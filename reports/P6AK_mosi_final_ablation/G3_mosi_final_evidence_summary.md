# P6AK-G3: MOSI Final Evidence Summary

**Date**: 2026-06-21
**Status**: All G0/G1/G2 complete. Evidence frozen.

## 1. P6AJ Artifact Truth (G0)

- All 3 P6AJ checkpoints verified consistent with result.json (diff=0.0000pp)
- Reliable fusion diagnostics exported: r_a, r_v, delta_a, delta_v, contributions per sample
- Counterfactuals verified: audio-zero, vision-zero, audio/vision permutation all change predictions
- P6AJ checkpoints are valid TAV models

## 2. Final Model (F0)

```text
Name: canonical_mosi_text_anchored_tav
Architecture: Text-Anchored Reliable Fusion
Init: P6K text+audio checkpoint
Features: mosi_tav_v1 (data2vec 768d audio, CLIP-L14 1024d vision)
Params: 4.16M trainable
```

### 3-Seed Results

| Seed | ACC2 | F1 | MAE | Corr | ACC7 |
|------|------|-----|-----|------|------|
| 42 | 87.65% | 85.03 | 0.668 | 0.837 | 48.10 |
| 2024 | 87.50% | 85.51 | 0.699 | 0.816 | 45.48 |
| 3407 | 88.26% | 86.03 | 0.644 | 0.852 | 48.54 |
| **Mean** | **87.80%** | **85.52** | **0.670** | **0.835** | **47.37** |

## 3. Ablation Conclusions (G2)

| Comparison | Δ ACC2 | Conclusion |
|-----------|--------|------------|
| TAV vs Text-only | -0.31pp | Text backbone dominates; TAV adds no net benefit |
| Audio correction | 0.25pp | Marginal positive contribution |
| Vision correction | 0.25pp | Marginal positive contribution |
| Reliability gate | 0.05pp | Negligible effect (corrections already small) |
| Hadamard interaction | 0.10pp | Negligible effect |

## 4. Architecture Comparison (MOSEI vs MOSI)

| Property | MOSEI | MOSI |
|----------|-------|------|
| Train samples | 13,239 | 1,284 |
| Architecture | Canonical AWAF | Text-Anchored Reliable Fusion |
| Vision contribution | +1.67pp (significant) | +0.25pp (marginal) |
| Hadamard interaction | +2.37pp (critical) | +0.10pp (negligible) |
| Dynamic AWAF | -0.97pp (global static better) | N/A (text-anchored) |
| Collapse risk | Low (large dataset) | High (canonical AWAF collapses) |

## 5. Dataset-Size Hypothesis (Validated)

Canonical AWAF requires sufficient training data to learn meaningful modality weights. With 1,284 MOSI samples, softmax blending destroys the strong text signal. Text-anchored fusion resolves this by preserving the text anchor, but auxiliary modalities add minimal value. With 13,239 MOSEI samples, AWAF can learn effective cross-modal interactions.

## 6. Deliverables Checklist

- [x] G0: Artifact truth audit (3 seeds verified)
- [x] G1: Switch integrity audit (all 6 variants pass)
- [x] G2: Ablation training (15/15 runs complete)
- [x] G3: Evidence summary (this file)
- [x] Paper claims (allowed + prohibited)
- [ ] Cleanup plan
- [ ] Final handoff
