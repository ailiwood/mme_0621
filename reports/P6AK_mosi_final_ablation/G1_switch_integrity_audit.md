# P6AK-G1: Ablation Switch Integrity Audit

**Overall**: ALL PASS

| Check | Status | Detail |
|-------|--------|--------|
| A1_no_audio_vision_branch | PASS | Audio=False, Vision=False |
| A2_flag_no_audio_correction | PASS | Flag=True |
| A3_flag_no_vision_correction | PASS | Flag=True |
| A4_gate_always_one | PASS | r_a=1.0000, r_v=1.0000 |
| A5_no_interaction_flag | PASS | no_interaction=True |
| F0_vs_A1_outputs_differ | PASS | |F0-A1|=0.504838 |
| ablation_outputs_not_identical | PASS | Unique outputs: 5/5 |

## Variant Summary

| Var | Mode | Params(M) | Audio | Vision | reg_mean | r_a | r_v |
|-----|------|-----------|-------|--------|----------|-----|-----|
| F0 | al_mosi_text_anchored_tav | 4.16 | Y | Y | -0.4609 | 0.1221 | 0.0582 |
| A1 | text_only | 2.23 | N | N | 0.0439 | 0.0000 | 0.0000 |
| A2 | nchored_tav_no_audio_corr | 4.16 | Y | Y | 0.2001 | 0.1566 | 0.1426 |
| A3 | chored_tav_no_vision_corr | 4.16 | Y | Y | -0.0291 | 0.1640 | 0.0770 |
| A4 | text_anchored_tav_no_gate | 4.16 | Y | Y | 0.0915 | 1.0000 | 1.0000 |
| A5 | chored_tav_no_interaction | 4.16 | Y | Y | -0.2959 | 0.1955 | 0.1182 |
