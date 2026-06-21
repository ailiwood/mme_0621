# Multimodal Sentiment Analysis — Canonical TAV + Text-Anchored Reliable Fusion

**Final Release**: p6am-final-freeze (2026-06-21)

## Overview

This repository contains the code, configurations, and experiment documentation for multimodal sentiment analysis on CMU-MOSEI and CMU-MOSI datasets. Two fusion architectures are implemented:

| Dataset | Architecture | ACC2 | Status |
|---------|-------------|------|--------|
| MOSEI | Canonical AWAF sLSTM | 87.98% | Primary 3-modal evidence |
| MOSI | Text-Anchored Reliable Fusion | 87.80% (3-seed mean) | Final TAV candidate |

## Directory Structure

```
├── configs/          # Experiment configurations
│   ├── final/        # Frozen final configs (to be organized)
│   ├── baselines/    # Baseline-lite configs
│   └── references/   # Historical reference configs
├── data/             # Data loading and dataset code
├── docs/             # Architecture, results, reproducibility docs
├── env/              # Environment setup
├── models/           # Model components
│   ├── fusion/       # AWAF + Text-Anchored Reliable Fusion
│   ├── encoders/     # sLSTM, pooling
│   ├── baselines/    # Baseline model implementations
│   └── modules/      # LoRA, gates
├── reports/          # Experiment reports and results
│   └── final/        # Final frozen reports, tables, figures
├── scripts/          # Training and evaluation scripts
├── utils/            # Metrics and utilities
└── tests/            # Test scripts
```

## Installation

```bash
conda env create -f env/environment_mme_xlstm_stable.yml
conda activate mme_xlstm_stable
```

Requirements: Python 3.10, PyTorch 2.11+ with CUDA 12.8, RTX 5070 Ti or compatible GPU.

## Data Preparation

Data is NOT included in this repository. Download CMU-MOSEI and CMU-MOSI from official sources.

Feature extraction uses:
- MOSEI: COVAREP (audio) + OpenFace2 (vision)
- MOSI: data2vec (audio) + CLIP-L14 (vision)

See `data/README_data.md` for details.

## Training

### MOSI (Text-Anchored Reliable Fusion)
```bash
python scripts/train_textft_lora_mainline.py \
  --config configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s42.yaml \
  --init_checkpoint outputs/P6K/text_audio_conservative_s42_s42_20260619_031645/best_model.pth
```

### MOSEI (Canonical AWAF sLSTM)
```bash
python scripts/train_textft_lora_mainline.py \
  --config configs/experiments/p6aa_tav/mosei/control_awaf_slstm_s42.yaml
```

## Results

See `docs/FINAL_RESULTS_LEDGER.md` for complete results.

### MOSI (3-seed mean)
- ACC2_Non0: 87.80% ± 0.33
- F1_Non0: 85.52%
- Corr: 0.835

### MOSEI (seed 42)
- ACC2_Non0: 87.98%
- F1_Non0: 90.49%
- Corr: 0.785

## Reproducibility

See `docs/REPRODUCIBILITY.md` for:
- Exact environment (conda env export)
- Feature versions and sources
- Training protocol
- Known limitations

## Baseline Note

Baseline implementations (TFN, LMF, MulT, SelfMM, MISA, MLCL, DLF) are `lite_reimplementations` — NOT official reproductions. They are provided for controlled comparison under identical data and training conditions.

## Model Weights and Data

Model checkpoints, raw data, and processed features are NOT included in this repository. They are preserved locally. Contact the authors for access.

## License

This project is for academic research purposes. See individual component licenses for third-party code.
