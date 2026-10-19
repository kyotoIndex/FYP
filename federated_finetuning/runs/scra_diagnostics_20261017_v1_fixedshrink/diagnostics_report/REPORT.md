# scra_diagnostics_20261017_v1_fixedshrink diagnostics

## D0 (logged training run)

| quantity | value | criterion | pass |
|---|---:|---|---|
| relative systematic owner–SCRA gap | 0.0670 | ≥ 0.05 | True |
| adjacent 10-round window cosine of o | 0.060 | ≥ 0.3 | False |
| crossover rounds K* | 5.1 | ≤ 50 | True |
| CV² of weak dispersion within adapter | 0.0002 | ≥ 0.1 (enable E1) | False |
| update norm late/early | 0.889 | ≤ 3.0 | True |

Descriptive: adjacent-round cosine 0.009 (noise-dominated), systematic energy ratio 0.0152, CV² all blocks 0.004 (relative 0.007), shadow strong relative mode 0.414, shadow κ (last 50) 0.043, shadow a (last 50) 0.846, final macro accuracy 0.8025.

**A-SCRA kappa.** The per-round EMA above is biased by self-correlation (critical review C1); use the window-averaged vector estimate: κ (last 3 windows) = 0.025, implied a = 0.89; per window: -0.007, -0.042, 0.058, 0.067, 0.048, 0.047, 0.068, 0.021, 0.029, 0.068, 0.022, 0.044, 0.024, 0.023, 0.026.

**Accuracy, rounds ≥ 110 (core weak / all clients, %).** Same seed and participant schedule as the sweep.

| run | core weak | all |
|---|---:|---:|
| r8 | 80.13 | 81.22 |
| r2 | 78.39 | 79.72 |
| L0 prefix | 78.04 | 79.62 |
| L2 (main D0) | 77.72 | 79.16 |
| vref (weak train full, upload 2 of 8, Hajek) | 80.15 | 81.22 |
| this run: fixed shrink 0.63, hajek | 78.32 | 79.63 |

This run recovers 25% of the vref − L2 core-weak gap (2.43 pp); 0% = L2, 100% = vref (1 seed; per-seed noise ≈ 0.2–0.4 pp).

Share of the r8−r2 core-weak gap recovered: L0 prefix -0.20, this run -0.04; remaining gap to r8 1.81 pp (1 seed, descriptive).

Thresholds and their meaning: docs/designs/algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md §7.1, §6.
