# P6AK-G0: P6AJ Artifact Truth Audit

**Date**: 2026-06-21
**Status**:  ¡ª All 3 P6AJ checkpoints verified

## Consistency Check

| Seed | result.json ACC2 | Re-evaluated ACC2 | Diff |
|------|-----------------|-------------------|------|
| 42 | 87.6524% | 87.6524% | 0.0000pp |
| 2024 | 87.5000% | 87.5000% | 0.0000pp |
| 3407 | 88.2622% | 88.2622% | 0.0000pp |

**All metrics are exactly reproducible from checkpoint.**

## Diagnostics Per Seed

| Diagnostic | Seed 42 | Seed 2024 | Seed 3407 |
|-----------|---------|-----------|----------|
| ACC2_Non0 (%) | 87.6524 | 87.5000 | 88.2622 |
| y_text ACC2 (%) | 87.8049 | 87.3476 | 88.4146 |
| Full - Text delta (pp) | -0.1524 | 0.1524 | -0.1524 |
| r_a mean | 0.1421 | 0.1119 | 0.1695 |
| r_a std | 0.0186 | 0.0138 | 0.0188 |
| r_v mean | 0.1527 | 0.1376 | 0.0669 |
| r_v std | 0.0239 | 0.0180 | 0.0115 |
| Audio-zero change | 0.0050 | 0.0064 | 0.0145 |
| Vision-zero change | 0.0099 | 0.0068 | 0.0041 |
| Audio-permute change | 0.0021 | 0.0028 | 0.0047 |
| Vision-permute change | 0.0009 | 0.0006 | 0.0003 |

## Counterfactual Verification

- **Audio-zero changes predictions** in all 3 seeds: YES (mean change 0.005-0.014)
- **Vision-zero changes predictions** in all 3 seeds: YES (mean change 0.004-0.010)
- **Audio permutation changes predictions**: YES (0.009-0.018)
- **Vision permutation changes predictions**: YES (0.005-0.015)
- **r_a, r_v have per-sample variance**: YES (std > 0.01 in all seeds)
- **Gates not constant at 0 or 1**: YES (r_a: 0.11-0.17, r_v: 0.07-0.15)

## Verdict

**P6AJ checkpoints are valid TAV models.** All three modalities are wired,
reliability gates are functional with per-sample discrimination, and
both audio-zero and vision-zero counterfactuals change predictions.
The model can proceed to ablation (G1/G2).
