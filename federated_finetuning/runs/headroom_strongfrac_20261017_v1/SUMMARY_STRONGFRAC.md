# headroom_strongfrac_20261017_v1 summary

Completed cells: 15/15.
Primary metric: client-macro accuracy (%) of the fixed core-weak set (30 clients never strong in any cell), averaged over evaluations at rounds ≥ 110.

## Capacity reference (all clients)

| seed | r8 late | r2 late | r8 − r2 | rounds to 80% (r8 / r2) |
|---|---:|---:|---:|---:|
| 2040 | 81.22 | 79.72 | 1.50 | 80 / 130 |
| 2041 | 81.34 | 79.59 | 1.75 | 90 / 150 |
| 2042 | 81.23 | 79.70 | 1.53 | 80 / 150 |

## Core-weak gap by strong fraction (primary)

| strong fraction | r8 − hetero, core weak, late (per seed) | mean | min | final round (mean) | share of r8−r2 gap recovered (mean) | decision |
|---|---|---:|---:|---:|---:|---|
| 10% | 2.09, 1.83, 1.64 | 1.85 | 1.64 | 1.63 | -0.03 | **yes** |
| 20% | 1.84, 1.48, 1.04 | 1.45 | 1.04 | 1.25 | 0.20 | **yes** |
| 40% | 1.13, 0.95, 0.46 | 0.85 | 0.46 | 1.08 | 0.53 | **no** |

## Secondary groups (r8 − hetero, late, mean over seeds)

| strong fraction | strong / all-weak / demoted clients | all clients | all weak at f | demoted (gov/travel now weak) | rounds with 0 / 1 strong participant |
|---|---|---:|---:|---:|---|
| 10% | 5 / 45 / 15 | 1.41 | 1.48 | 0.73 | 29% / 45% |
| 20% | 10 / 40 / 10 | 1.02 | 1.13 | 0.19 | 8% / 28% |
| 40% | 20 / 30 / 0 | 0.44 | 0.85 | — | 0% / 4% |

Decision: **yes** when the late core-weak gap is ≥ 1.0 pp in every seed, i.e. naive heterogeneity leaves at least that much for an adaptive method to recover on the same fixed set of clients.
