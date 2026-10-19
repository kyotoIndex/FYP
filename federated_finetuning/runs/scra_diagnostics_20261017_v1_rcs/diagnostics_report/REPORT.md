# scra_diagnostics_20261017_v1_rcs diagnostics

## D0 (logged training run)

| quantity | value | criterion | pass |
|---|---:|---|---|
| relative systematic owner–SCRA gap | 0.0458 | ≥ 0.05 | False |
| adjacent 10-round window cosine of o | 0.043 | ≥ 0.3 | False |
| crossover rounds K* | 8.9 | ≤ 50 | True |
| CV² of weak dispersion within adapter | 0.0001 | ≥ 0.1 (enable E1) | False |
| update norm late/early | 0.891 | ≤ 3.0 | True |

Descriptive: adjacent-round cosine -0.012 (noise-dominated), systematic energy ratio 0.0130, CV² all blocks 0.011 (relative 0.026), shadow strong relative mode 0.489, shadow κ (last 50) 0.059, shadow a (last 50) 0.789, final macro accuracy 0.7998.

**A-SCRA kappa.** The per-round EMA above is biased by self-correlation (critical review C1); use the window-averaged vector estimate: κ (last 3 windows) = 0.010, implied a = 0.96; per window: 0.063, 0.008, 0.056, 0.092, 0.034, 0.118, 0.078, 0.022, 0.015, 0.067, 0.049, 0.051, 0.017, 0.009, 0.005.

**Accuracy, rounds ≥ 110 (core weak / all clients, %).** Same seed and participant schedule as the sweep.

| run | core weak | all |
|---|---:|---:|
| r8 | 80.13 | 81.22 |
| r2 | 78.39 | 79.72 |
| L0 prefix | 78.04 | 79.62 |
| L2 (main D0) | 77.72 | 79.16 |
| vref (weak train full, upload 2 of 8, Hajek) | 80.15 | 81.22 |
| this run: RCS shrink, hajek | 77.64 | 79.00 |

This run recovers -3% of the vref − L2 core-weak gap (2.43 pp); 0% = L2, 100% = vref (1 seed; per-seed noise ≈ 0.2–0.4 pp).

Share of the r8−r2 core-weak gap recovered: L0 prefix -0.20, this run -0.43; remaining gap to r8 2.48 pp (1 seed, descriptive).

Thresholds and their meaning: docs/designs/algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md §7.1, §6.
