# P6AK-G3: Paper Claims — Prohibited

## Do NOT Write

1. ❌ "Canonical AWAF achieves state-of-the-art on both MOSI and MOSEI"
   - Canonical AWAF collapses on MOSI (42–44% ACC2).

2. ❌ "Audio and vision significantly improve over text-only on MOSI"
   - Text-only (88.11%) slightly exceeds full TAV (87.80%). The difference is within seed variation.

3. ❌ "The reliability gate mechanism provides critical performance gains on MOSI"
   - Gate ablation (A4) shows only 0.05pp difference. Gates are functional but corrections are already small.

4. ❌ "Dual-dataset three-modal evidence with identical architectures"
   - MOSEI uses canonical AWAF; MOSI uses text-anchored fusion. Different architectures for different dataset sizes.

5. ❌ "P6K 88.72% is the TAV result"
   - P6K is text+audio only, uses different feature version (v3_T40 with 768d vision), and is a historical reference.

6. ❌ "Vision significantly contributes to sentiment prediction on MOSI"
   - Vision contribution is 0.25pp, within seed variation (std 0.33pp).

7. ❌ "The Hadamard interaction terms are critical for MOSI performance"
   - Interaction ablation (A5) shows only 0.10pp difference. On MOSEI this was +2.37pp — the mechanism matters on larger datasets.

8. ❌ Direct numerical comparison of MOSI and MOSEI ACC2 without noting:
   - Different feature versions (CLIP-L14 vs OpenFace2, data2vec vs COVAREP)
   - Different architectures (text-anchored vs AWAF)
   - Different dataset sizes (1,284 vs 13,239 train)

9. ❌ "The final model achieves 87.80% without any historical weight initialization"
   - P6K text+audio checkpoint was used for initialization. This must be disclosed.

10. ❌ "MOSI TAV baselines have been completed"
    - Baseline-lite matrix was NOT run in P6AK. Only ablation was completed.
