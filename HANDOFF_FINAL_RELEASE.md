# HANDOFF — Final Release

**Date**: 2026-06-21
**Repository**: `https://github.com/ailiwood/mme_0621`
**Commit**: `0c3bbfa` (to be updated with final report)
**Tag**: `v1.0.0-final`

---

## Project Complete — No Further Changes

This repository contains the final multimodal sentiment analysis code and evidence for:

### MOSI (Text-Anchored Reliable TAV Fusion)
- 3-seed ACC2_Non0: **87.80%** ± 0.33
- Text-only: 88.11%
- 5-variant ablation complete
- Architecture: RoBERTa+LoRA → text anchor + reliability-gated audio/vision corrections

### MOSEI (Canonical AWAF sLSTM)
- ACC2_Non0: **87.98%** (seed 42, aligned_reference)
- 7-variant 4-epoch ablation complete
- Key: Vision +1.67pp, Hadamard interaction +2.37pp (both significant)

---

## What's in the Repository

- **99 files**, single clean Git commit
- Full model code, configs, scripts, utilities
- Final documentation: architecture, results ledger, limits
- Baseline-lite implementations (marked as lite_reimplementation)
- `.gitignore` excludes all data, weights, outputs

## What's NOT in the Repository

- Model weights (*.pth) — preserved locally
- Processed features — preserved locally
- Training outputs — preserved locally
- Raw data — preserved locally

## Local Evidence Locations

- `E:\00project_code\main_leo\local_final_evidence\mme_0621_final_20260621\`
- `E:\00project_code\main_leo\local_archive\mme_0621_history_20260621_184736\`

## Paper Status

| Claim | Status |
|-------|--------|
| MOSI 87.80% 3-seed TAV | ✅ Ready |
| MOSEI 87.98% TAV | ✅ Ready (aligned_reference) |
| Text-anchored vs canonical AWAF dataset-size hypothesis | ✅ Ready |
| Vision contribution on MOSEI | ✅ Ready (+1.67pp, sig) |
| Hadamard interaction on MOSEI | ✅ Ready (+2.37pp, sig) |
| "TAV significantly better than text-only on MOSI" | ❌ Not supported |
| "Identical architecture on both datasets" | ❌ Not true |
| "Dual-dataset three-modal evidence complete" | ⚠️ With caveats |

## Stop — Do Not Modify
