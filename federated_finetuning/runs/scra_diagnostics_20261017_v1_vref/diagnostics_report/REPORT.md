# scra_diagnostics_20261017_v1_vref diagnostics

## D0 (logged training run)

| quantity | value | criterion | pass |
|---|---:|---|---|
| relative systematic owner–SCRA gap | 0.0497 | ≥ 0.05 | False |
| adjacent 10-round window cosine of o | 0.025 | ≥ 0.3 | False |
| crossover rounds K* | 11.5 | ≤ 50 | True |
| CV² of weak dispersion within adapter | 0.0002 | ≥ 0.1 (enable E1) | False |
| update norm late/early | 0.932 | ≤ 3.0 | True |

Descriptive: adjacent-round cosine 0.004 (noise-dominated), systematic energy ratio 0.0124, CV² all blocks 0.005 (relative 0.002), shadow strong relative mode 0.000, shadow κ (last 50) —, shadow a (last 50) —, final macro accuracy 0.8157.

**A-SCRA kappa.** The per-round EMA above is biased by self-correlation (critical review C1); use the window-averaged vector estimate: κ (last 3 windows) = —, implied a = —; per window: .

**Accuracy, rounds ≥ 110 (core weak / all clients, %).** Same seed and participant schedule as the sweep.

| run | core weak | all |
|---|---:|---:|
| r8 | 80.13 | 81.22 |
| r2 | 78.39 | 79.72 |
| L0 prefix | 78.04 | 79.62 |
| L2 (main D0) | 77.72 | 79.16 |
| vref (this run: weak train full, Hajek) | 80.15 | 81.22 |

Decomposition of the core-weak gap r8 − L2 = 2.41 pp: fewer contributors (r8 − vref) = -0.02 pp; restricted training mode (+ small owner bias) (vref − L2) = 2.43 pp (1 seed; per-seed noise ≈ 0.2–0.4 pp).

Share of the r8−r2 core-weak gap recovered: L0 prefix -0.20, L2 (coverage-balanced, owner) 1.01; remaining gap -0.02 pp (1 seed, descriptive).

Thresholds and their meaning: docs/designs/algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md §7.1, §6.
