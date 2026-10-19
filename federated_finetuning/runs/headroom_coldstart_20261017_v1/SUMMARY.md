# headroom_coldstart_20261017_v1 summary

Completed cells: 8/8.

## Final round, client-macro accuracy (%)

| method | seed 2040 | seed 2041 | weak (mean) | strong (mean) | +FT (mean) | minutes (mean) |
|---|---:|---:|---:|---:|---:|---:|
| fedavg_r8 | 81.64 | 81.73 | 80.62 | 83.28 | 80.11 | 12.1 |
| fedavg_r2 | 80.25 | 80.04 | 78.86 | 82.09 | 79.07 | 12.6 |
| hetero_r8r2 | 80.05 | 81.39 | 79.21 | 82.99 | 80.54 | 12.3 |
| local_only | 71.37 | 69.94 | 66.83 | 76.40 | — | 12.8 |

## Paired gaps (percentage points; per seed, mean)

| contrast | seed 2040 | seed 2041 | mean |
|---|---:|---:|---:|
| capacity_r8_minus_r2 | 1.38 | 1.69 | 1.54 |
| hetero_cost_r8_minus_hetero | 1.59 | 0.34 | 0.97 |
| hetero_weak_cost_r8_minus_hetero | 2.03 | 0.79 | 1.41 |
| hetero_gain_over_all_r2 | -0.21 | 1.35 | 0.57 |
| federation_hetero_minus_local | 8.68 | 11.45 | 10.07 |
| federation_weak_hetero_minus_local | 10.97 | 13.80 | 12.38 |
| federation_hetero_ft_minus_local | 9.20 | 10.58 | 9.89 |

## Pre-registered decision (gap ≥ 1.0 pp in every seed)

- rank_capacity_headroom: **yes**
- weak_clients_lose_under_heterogeneity: **no**
- federation_beats_local_only: **yes**
- federation_beats_local_after_personalization: **yes**
