# P6AK-G2: MOSI Final Ablation Results

**Date**: 2026-06-21
**Protocol**: 35 epochs max, early stop patience 8, P6K text+audio init, 3 seeds (42, 2024, 3407)

## Ablation Matrix (3-seed mean ± std)

| Variant | ACC2 | F1 | MAE | Corr | ACC7 | Δ vs F0 |
|---------|------|-----|-----|------|------|---------|
| **F0** Full TAV | 87.80% | 85.52 | 0.670 | 0.835 | 47.37 | — |
| **A1** Text-only | **88.11%** | 85.80 | 0.622 | 0.854 | 47.81 | **+0.31** |
| A2 No audio correction | 87.55% | 85.33 | 0.705 | 0.822 | 43.78 | -0.25 |
| A3 No vision correction | 87.55% | 85.33 | 0.703 | 0.822 | 44.17 | -0.25 |
| A4 No reliability gate | 87.75% | 85.42 | 0.709 | 0.825 | 45.92 | -0.05 |
| A5 No interaction | 87.70% | 85.45 | 0.672 | 0.833 | 47.38 | -0.10 |

## Per-Seed Breakdown

| Variant | s42 | s2024 | s3407 | Std |
|---------|-----|-------|-------|-----|
| F0 Full TAV | 87.65 | 87.50 | 88.26 | 0.33 |
| A1 Text-only | 88.57 | 87.35 | 88.41 | 0.55 |
| A2 No audio | 87.35 | 87.35 | 87.96 | 0.29 |
| A3 No vision | 87.35 | 87.35 | 87.96 | 0.29 |
| A4 No gate | 87.04 | 87.80 | 88.41 | 0.56 |
| A5 No interaction | 87.80 | 87.35 | 87.96 | 0.26 |

## Key Comparisons

| Comparison | Effect | Interpretation |
|-----------|--------|----------------|
| F0 vs A1 (TAV vs text-only) | -0.31pp | TAV adds no net benefit over text-only on MOSI |
| F0 vs A2 (audio contribution) | 0.25pp | Audio correction provides marginal benefit |
| F0 vs A3 (vision contribution) | 0.25pp | Vision correction provides marginal benefit |
| F0 vs A4 (gate contribution) | 0.05pp | Reliability gate has negligible effect |
| F0 vs A5 (interaction contribution) | 0.10pp | Hadamard interaction has negligible effect |

## Scientific Conclusions

1. **Text backbone dominates**: At 88.11%, text-only is the strongest variant. The P6K-initialized RoBERTa+LoRA text encoder provides excellent sentiment prediction on MOSI.

2. **Text-anchored fusion is safe**: Unlike canonical AWAF (which collapsed to 42-44%), the text-anchored architecture never degrades below text-only by more than 0.3pp. The architecture achieves its design goal of preserving the text baseline.

3. **Audio/vision contributions are marginal**: At 0.25pp each, audio and vision corrections provide small benefits that are within seed variation. The CLIP-L14 vision and data2vec audio features do not systematically improve over the strong text baseline on MOSI's 1,284 training samples.

4. **Reliability gates and interactions are not critical on MOSI**: With corrections already near-zero, the gates (r_a=0.11-0.17, r_v=0.07-0.15) and Hadamard interactions have minimal impact. This is consistent with the finding that auxiliary modality contributions are small.

5. **Cross-seed stability**: All variants show stable performance across seeds (std 0.26-0.56pp), confirming the training protocol is robust.
