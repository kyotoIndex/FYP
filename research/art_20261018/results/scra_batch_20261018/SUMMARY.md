# Validation batch 2026-10-18

Core weak accuracy (%) averaged over rounds ≥ 110; `core10` = the fixed 30-client core set of the 10%-strong setting; `osc` = mean |Δ all-client accuracy| between evaluations from round 50. 1 seed noise ≈ 0.2–0.4 pp.

## Main comparison (10% strong, R=8, r=2)

| method | core s2040 | core s2041 | core s2042 | core mean ± sd | all mean | osc mean |
|---|---:|---:|---:|---:|---:|---:|
| r8 (upper) | 80.13 | 80.26 | 80.19 | 80.19 ± 0.06 | 81.26 | 0.34 |
| r2 | 78.39 | 78.32 | 78.45 | 78.39 ± 0.05 | 79.67 | 0.46 |
| prefix hetero (L0) | 78.04 | 78.43 | 78.54 | 78.34 ± 0.22 | 79.85 | 0.92 |
| L2 restricted, owner | 77.72 | 77.28 | 78.56 | 77.85 ± 0.53 | 79.28 | 2.69 |
| vref (full train, upload r) | 80.15 | 80.32 | 80.02 | 80.17 ± 0.12 | 81.27 | 0.37 |
| **amp (amplified restricted, Hájek)** | 80.05 | 80.32 | 79.99 | 80.12 ± 0.14 | 81.24 | 0.35 |

## Ablations (seed 2040)

| run | core | Δ vs amp | closure L2→vref | osc |
|---|---:|---:|---:|---:|
| amp_s2040 | 80.05 | 0.00 | 0.96 | 0.37 |
| amp_owner_s2040 | 79.61 | -0.45 | 0.78 | 0.40 |
| amp_prefix_s2040 | 78.51 | -1.55 | 0.32 | 0.48 |
| amp_a2_s2040 | 79.47 | -0.58 | 0.72 | 0.97 |
| amp_a8_s2040 | 79.91 | -0.14 | 0.90 | 0.36 |

## Robustness (seed 2040)

| setting | L2 | vref | amp | closure | core10 L2 / vref / amp | osc amp |
|---|---:|---:|---:|---:|---|---:|
| r = 1 (upload 1/8) | 78.34 | 80.00 | 80.01 | 1.01 | 78.34 / 80.00 / 80.01 | 0.35 |
| ranks {1,2,4} | 79.02 | 80.12 | 80.00 | 0.89 | 79.02 / 80.12 / 80.00 | 0.40 |
| 20% strong | 78.92 | 80.09 | 80.00 | 0.92 | 78.92 / 80.09 / 80.00 | 0.36 |

## Cost per run

| run | peak memory MiB | seconds |
|---|---:|---:|
| L2_het_s2040 | 1557 | 811 |
| L2_r1_s2040 | 1557 | 763 |
| L2_s2040 | 1557 | 827 |
| L2_s2041 | 1557 | 758 |
| L2_s2042 | 1556 | 756 |
| L2_sf20_s2040 | 1557 | 787 |
| amp_a2_s2040 | 1558 | 858 |
| amp_a8_s2040 | 1558 | 821 |
| amp_het_s2040 | 1558 | 856 |
| amp_owner_s2040 | 1558 | 876 |
| amp_prefix_s2040 | 1558 | 817 |
| amp_r1_s2040 | 1558 | 875 |
| amp_s2040 | 1558 | 828 |
| amp_s2041 | 1558 | 817 |
| amp_s2042 | 1558 | 909 |
| amp_sf20_s2040 | 1558 | 805 |
| vref_het_s2040 | 1557 | 736 |
| vref_r1_s2040 | 1557 | 812 |
| vref_s2040 | 1557 | 741 |
| vref_s2041 | 1557 | 786 |
| vref_s2042 | 1556 | 746 |
| vref_sf20_s2040 | 1557 | 756 |

## Pre-registered reading

| question | criterion | result |
|---|---|---|
| M1 amp matches vref and beats L2 | 3-seed mean ≥ vref − 0.3, ≥ L2 + 1.0, amp > L2 in every seed | True |
| M2 amp is stable | mean osc ≤ 1.5 × vref | True |
| A1 Hájek needed | amp_owner ≤ amp − 0.3 | True |
| A2 balanced allocation needed | amp_prefix ≤ amp − 0.3 | True |
| A3 α = R/r is right | amp ≥ max(α=2, α=8) − 0.3 | True |
| R1 rank 1 | closure ≥ 0.7 (gap ≥ 0.5 pp) | True |
| R2 heterogeneous ranks | closure ≥ 0.7 | True |
| R3 20% strong | closure ≥ 0.7 | True |

Reading: M1 and M2 true → amplified restricted training is the main method; next are baselines (FLASC, FSLoRA, Fed-PLoRA), literature check and scale. A1/A2 false → that component is not needed and is dropped from the method (single seed, so only a gap ≥ 0.3 pp counts). M1 false while D3 passed → the one-step view missed a multi-round effect; probe amp states with D3 before changing the method. A closure of None means a run is missing or the setting has < 0.5 pp headroom.
