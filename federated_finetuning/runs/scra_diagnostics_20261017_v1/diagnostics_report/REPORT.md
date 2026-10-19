# scra_diagnostics_20261017_v1 diagnostics

## D0 (logged training run)

| quantity | value | criterion | pass |
|---|---:|---|---|
| relative systematic owner–SCRA gap | 0.0406 | ≥ 0.05 | False |
| adjacent 10-round window cosine of o | 0.039 | ≥ 0.3 | False |
| crossover rounds K* | 11.2 | ≤ 50 | True |
| CV² of weak dispersion within adapter | 0.0001 | ≥ 0.1 (enable E1) | False |
| update norm late/early | 0.810 | ≤ 3.0 | True |

Descriptive: adjacent-round cosine -0.014 (noise-dominated), systematic energy ratio 0.0126, CV² all blocks 0.003 (relative 0.025), shadow strong relative mode 0.435, shadow κ (last 50) 0.048, shadow a (last 50) 0.830, final macro accuracy 0.8051.

## D1 (fixed states)

| round | ‖e_SCRA‖ | ‖e_owner‖ | blocks SCRA better | rel. mode (weak / strong) | κ weak (oracle) | κ from strong | a* oracle / from strong | cos(m_w, m_s) | ‖D_core‖ / ‖D_demoted‖ |
|---:|---:|---:|---:|---|---:|---:|---|---:|---|
| 20 | 0.1686 | 0.1439 | 0.14 | 0.839 / 0.823 | 0.321 | 0.370 | 0.00 / 0.00 | 0.799 | 0.2802 / 0.3992 |
| 60 | 0.1663 | 0.1399 | 0.22 | 0.674 / 0.667 | 0.301 | 0.311 | 0.00 / 0.00 | 0.819 | 0.2903 / 0.4213 |
| 120 | 0.1058 | 0.1017 | 0.56 | 0.519 / 0.548 | 0.140 | 0.172 | 0.39 / 0.25 | 0.775 | 0.2925 / 0.4211 |

Checks per round: r20: F1_scra_beats_owner_D1=False, F11_A4_mode_cosine_ok=True, F11_A4_implied_a_agrees=True; r60: F1_scra_beats_owner_D1=False, F11_A4_mode_cosine_ok=True, F11_A4_implied_a_agrees=True; r120: F1_scra_beats_owner_D1=False, F11_A4_mode_cosine_ok=True, F11_A4_implied_a_agrees=True

Thresholds and their meaning: docs/designs/algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md §7.1, §6.
