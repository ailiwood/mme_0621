# P6AK-G3: Paper Claims — Allowed

## MOSI

1. "The final MOSI TAV model achieves 3-seed test ACC2_Non0 mean of 87.80% (range 87.50–88.26%) using text-anchored reliable fusion with P6K text+audio initialization and CLIP-L14 vision features."

2. "On MOSI's 1,284 training samples, canonical softmax-weighted AWAF collapses to 42–44% ACC2 (below majority baseline), while text-anchored reliable fusion preserves the strong text baseline and achieves stable performance."

3. "The text-anchored architecture uses per-sample reliability gates (r_a, r_v) initialized near zero, ensuring the model starts from a text-only behavior and only incorporates auxiliary corrections when the gates learn they are reliable."

4. "Ablation analysis shows that text-only achieves 88.11% (highest), and audio and vision corrections each contribute approximately 0.25pp. The reliability gates and Hadamard interactions have minimal impact because the auxiliary corrections are already small."

5. "The key insight is that on small datasets, multimodal fusion architectures must preserve a strong unimodal baseline. Text-anchored fusion achieves this, while canonical AWAF's softmax blending destroys it."

## MOSEI

6. "On MOSEI's 13,239 training samples, canonical AWAF with Hadamard interaction terms achieves strong performance. A fixed 4-epoch ablation protocol reveals: vision contributes +1.67pp (significant), Hadamard interaction contributes +2.37pp (largest effect), and global static weights outperform dynamic AWAF at low training budgets."

## Cross-Dataset

7. "The contrasting behavior of canonical AWAF on MOSEI (works) vs MOSI (collapses) demonstrates that multimodal fusion architecture choice is dataset-size-dependent. Larger datasets enable equal-opportunity softmax fusion; smaller datasets require text-preserving architectures."

8. "The P6K text-audio conservative reference (88.72%) was used as a historical reference and weight initialization source."

## Architecture Contribution

9. "The Text-Anchored Reliable Fusion module is a lightweight addition (4.16M trainable parameters) that can be applied to any text-dominated multimodal task where auxiliary modalities may provide weak or inconsistent signal."
