# Experiment Status and Limits

## MOSI

### What WAS Done
- ✅ Full TAV main model with 3-seed validation (87.80% mean)
- ✅ 5-variant ablation (A1-A5) with 3 seeds each
- ✅ Diagnostic probes (text-only, audio-only, vision-only, shuffled-label)
- ✅ Feature temporality audit (both modalities Route-S)
- ✅ Two failed canonical AWAF attempts documented (P6AD, P6AF)
- ✅ Counterfactual verification (audio/vision zero + permutation)

### What WAS NOT Done
- ❌ Baseline-lite matrix (TFN, LMF, MulT, SelfMM, MISA, MLCL, DLF)
- ❌ Curriculum learning (TA→TAV)
- ❌ Text-query cross-modal attention
- ❌ Confidence-calibrated fusion

### Known Limits
1. **Text backbone dominates**: Text-only (88.11%) slightly exceeds TAV (87.80%). Audio/vision corrections are within seed variation.
2. **P6K initialization**: Model uses P6K text+audio checkpoint for initialization. From-scratch text-anchored (P6AG) achieved 82.01%.
3. **Small validation set**: 229 val samples limits hyperparameter selection reliability.
4. **Different features from MOSEI**: CLIP-L14 1024d vs OpenFace2 713d; data2vec 768d vs COVAREP 74d.

## MOSEI

### What WAS Done
- ✅ TAV main model (12-epoch, 87.98%)
- ✅ 7-variant 4-epoch ablation with statistical analysis
- ✅ AWAF weight analysis
- ✅ Data pipeline verification

### What WAS NOT Done
- ❌ Strict complete-case cohort (18,571) re-training
- ❌ Multi-seed main model
- ❌ Long-epoch (12+) ablation

### Known Limits
1. **Not strict complete-case**: P6AA/P6AB used 20,680 samples (2,109 have missing/defective vision or audio). Reclassified as `aligned_or_mixed_tav_reference`.
2. **Single seed**: Main model is seed 42 only.
3. **12-epoch vs 4-epoch gap**: Main model uses 12 epochs; ablation uses 4 epochs. Not directly numerically comparable.
4. **AWAF limitation at low epochs**: Global static weights (+0.97pp) significantly outperform dynamic AWAF at 4 epochs.

## Cross-Dataset

### What the Evidence Supports
- Canonical AWAF works on larger datasets (MOSEI 13K)
- Text-anchored fusion is necessary for small datasets (MOSI 1.3K)
- Dataset size is a critical factor in multimodal fusion architecture choice

### What the Evidence Does NOT Support
- Direct numerical comparison of MOSI and MOSEI ACC2
- Claims that one architecture is universally superior
- Claims about vision/audio contribution magnitude generalizing across datasets
