# P6AM Final Freeze Report

**Date**: 2026-06-21
**Branch**: `release/p6am-final-freeze`
**Tag**: `p6am-final-freeze`

---

## Freeze Status

| Item | Status |
|------|--------|
| MOSI final model (3-seed) | ✅ Frozen |
| MOSI ablation (5 variants × 3 seeds) | ✅ Frozen |
| MOSEI main model + ablation | ✅ Frozen |
| Code frozen | ✅ |
| Configs frozen | ✅ |
| Cleanup executed | ✅ |
| Documentation complete | ✅ |
| **Ready for Git release** | ✅ |

## Final Results Summary

### MOSI (3-seed)
- F0 Full TAV: **87.80%** ± 0.33 ACC2
- A1 Text-only: **88.11%** ± 0.55 ACC2
- Architecture: Text-Anchored Reliable Fusion
- Init: P6K text+audio checkpoint

### MOSEI (seed 42)
- TAV Canonical AWAF: **87.98%** ACC2
- Vision contribution: +1.67pp (significant)
- Hadamard interaction: +2.37pp (significant)
- Status: `aligned_or_mixed_tav_reference`

## Files NOT in Git

- `data/processed/` — processed features
- `outputs/` — training outputs, checkpoints, predictions
- `*.pth, *.pt, *.ckpt` — model weights
- `*.npz, *.npy, *.pkl` — data files
- `artifacts/` — local archive

## Files Committed to Git

- `models/` — model code
- `configs/` — experiment configurations
- `scripts/` — training and evaluation scripts
- `utils/` — metrics and utilities
- `data/` — dataset loading code (not data files)
- `docs/` — documentation
- `reports/final/` — final reports and manifests
- `env/` — environment setup
- `README.md`, `.gitignore`, `.gitattributes`

## Cleanup Executed

- Removed: `__pycache__/` directories (9)
- Archived: P6Z temp files → `artifacts/archive_local_only/`
- Removed: empty output directories (18)

## Release Commands

```bash
git add -A
git commit -m "release: final MOSI/MOSEI multimodal evidence and reproducibility package

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git tag -a p6am-final-freeze -m "Final frozen code and experiment evidence"
git push origin release/p6am-final-freeze
git push origin p6am-final-freeze
```
