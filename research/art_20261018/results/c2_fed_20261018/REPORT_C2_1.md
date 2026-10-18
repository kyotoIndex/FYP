# C2-1 decoder-LLM federated runs

Core-weak accuracy (%) on the evaluation subset. Scope as in C2-0: no resource measurement.

## Stage A: learning-rate check (homogeneous R=64, development seed)

| lr | core r0 | core r5 | core r10 | weak local loss: full | restricted (replay) | full ≤ restricted |
|---|---:|---:|---:|---:|---:|---:|
| 2.5e-05 | 57.75 | 77.08 | 78.00 | 0.8391 | 0.8414 | True |
| 5e-05 | 57.75 | 77.33 | 78.67 | 0.8833 | 0.7414 | False |
| 0.0001 | 57.75 | 78.83 | 79.25 | 1.0334 | 0.7066 | False |
| 0.0002 | 57.75 | 76.25 | 79.33 | 1.2420 | 0.7769 | False |

Selection rule (pre-registered): highest core accuracy at round 10; within 0.5 pp of the best, the larger lr. **Chosen lr = 0.0002.** The loss columns are a diagnostic, not part of the rule: in C2-0 (lr 2e-4) full training had a higher local loss than restricted training on the same batches, a sign that the full-rank reference steps too far; at a well-chosen lr full training should not lose to restricted training locally.

## Stage B: main comparison

`late` = mean over evaluations in rounds 40–60; `early` = rounds 5–20; `osc` = mean |Δ all-client accuracy| between evaluations from round 20.

| run | method | α | late | early | final | osc | lr |
|---|---|---:|---:|---:|---:|---:|---:|
| r8 | r8 | — | 82.52 | 79.40 | 82.58 | 0.37 | 0.0002 |
| vref | vref | — | 79.92 | 74.10 | 80.67 | 0.69 | 0.0002 |
| L2 | L2 | — | 33.70 | 39.04 | 34.00 | 0.97 | 0.0002 |
| amp | amp | 8 | 78.45 | 77.60 | 78.75 | 0.63 | 0.0002 |
| fslora_st | fslora_st | — | 74.88 | 76.27 | 74.50 | 0.60 | 0.0002 |
| fslora | fslora | — | 78.45 | 77.48 | 77.33 | 0.51 | 0.0002 |
| amp_a4 | amp | 4 | 77.35 | 75.31 | 77.17 | 3.51 | 0.0002 |
| amp_a16 | amp | 16 | 45.18 | 58.06 | 34.67 | 10.38 | 0.0002 |
| amp_a32 | amp | 32 | 31.30 | 36.12 | 31.33 | 0.41 | 0.0002 |

Curves (core %):

| round | r8 | vref | L2 | amp | fslora_st | fslora | amp_a4 | amp_a16 | amp_a32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.75 | 57.75 | 57.75 | 57.75 | 57.75 | 57.75 | 57.75 | 57.75 | 57.75 |
| 5 | 76.67 | 66.83 | 34.00 | 76.92 | 76.25 | 75.50 | 77.08 | 50.42 | 44.50 |
| 10 | 80.08 | 72.83 | 53.50 | 77.00 | 76.75 | 78.25 | 75.17 | 53.92 | 31.33 |
| 15 | 80.58 | 78.00 | 34.00 | 78.17 | 76.83 | 77.58 | 78.83 | 71.17 | 34.00 |
| 20 | 80.25 | 78.75 | 34.67 | 78.33 | 75.25 | 78.58 | 70.17 | 56.75 | 34.67 |
| 25 | 81.42 | 78.00 | 34.00 | 78.17 | 76.00 | 77.33 | 77.75 | 65.17 | 31.33 |
| 30 | 82.50 | 78.58 | 34.67 | 77.42 | 75.17 | 78.08 | 76.50 | 60.92 | 31.33 |
| 35 | 82.58 | 79.17 | 34.67 | 79.17 | 75.83 | 78.58 | 80.25 | 44.92 | 31.33 |
| 40 | 82.42 | 78.75 | 34.50 | 78.75 | 75.17 | 79.08 | 74.33 | 62.58 | 31.17 |
| 45 | 82.33 | 80.08 | 34.67 | 78.17 | 74.50 | 78.92 | 79.08 | 51.58 | 31.33 |
| 50 | 82.33 | 79.42 | 34.00 | 78.67 | 74.83 | 78.50 | 76.83 | 45.75 | 31.33 |
| 55 | 82.92 | 80.67 | 31.33 | 77.92 | 75.42 | 78.42 | 79.33 | 31.33 | 31.33 |
| 60 | 82.58 | 80.67 | 34.00 | 78.75 | 74.50 | 77.33 | 77.17 | 34.67 | 31.33 |

## Mechanism diagnostics (replayed full training on the same batches; not aggregated)

Per weak client (medians): its update on its assigned components against its own full run (product step / cos), and the function-space step against its own full run (≈ r/R when matched component by component, more when overshooting). Aggregate: the method's global update against this round's r8 aggregate (all participants full).

| run | round | client step | client cos | client function step | local loss (method / full) | aggregate step | aggregate cos | aggregate function step | aggregate function cos |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| L2 | 1 | 0.42 | 0.41 | 0.861 | 1.258 / 1.852 | 0.95 | 0.83 | 1.06 | 0.97 |
| L2 | 10 | 0.74 | 0.44 | 0.914 | 1.703 / 1.382 | 1.26 | 0.64 | 1.19 | 0.39 |
| L2 | 20 | 1.03 | 0.72 | 0.485 | 6.259 / 3.169 | 1.10 | 0.88 | 1.06 | 0.74 |
| L2 | 30 | 1.01 | 0.61 | 0.181 | 6.283 / 3.689 | 1.19 | 0.93 | 1.03 | 1.00 |
| L2 | 40 | 1.33 | 0.79 | 0.244 | 30.708 / 12.452 | 1.44 | 0.81 | 0.94 | 0.99 |
| L2 | 50 | 2.28 | 0.72 | 0.571 | 2.269 / 1.483 | 1.94 | 0.87 | 1.42 | 0.58 |
| L2 | 60 | 1.22 | 0.71 | 0.189 | 4.420 / 2.642 | 1.24 | 0.73 | 0.70 | 0.62 |
| amp | 1 | 0.80 | 0.80 | 0.577 | 1.922 / 1.852 | 0.96 | 0.72 | 0.99 | 1.00 |
| amp | 10 | 0.88 | 0.89 | 0.095 | 0.938 / 0.943 | 0.83 | 0.38 | 0.86 | 0.81 |
| amp | 20 | 0.89 | 0.89 | 0.083 | 0.856 / 0.897 | 0.81 | 0.39 | 1.04 | 0.95 |
| amp | 30 | 0.88 | 0.88 | 0.070 | 0.822 / 0.808 | 0.76 | 0.42 | 0.62 | 0.84 |
| amp | 40 | 0.86 | 0.85 | 0.061 | 0.758 / 0.753 | 0.68 | 0.34 | 0.95 | 0.96 |
| amp | 50 | 0.87 | 0.86 | 0.068 | 0.764 / 0.750 | 0.77 | 0.40 | 0.76 | 0.82 |
| amp | 60 | 0.84 | 0.84 | 0.055 | 0.804 / 0.816 | 0.67 | 0.35 | 0.70 | 0.75 |
| fslora_st | 1 | 0.80 | 0.80 | 0.577 | 1.922 / 1.852 | 0.96 | 0.72 | 0.99 | 1.00 |
| fslora_st | 10 | 0.40 | 0.41 | 0.053 | 0.992 / 0.957 | 0.54 | 0.25 | 0.72 | 0.76 |
| fslora_st | 20 | 0.28 | 0.27 | 0.062 | 0.954 / 0.862 | 0.41 | 0.19 | 0.53 | 0.70 |
| fslora_st | 30 | 0.22 | 0.22 | 0.034 | 0.934 / 0.812 | 0.43 | 0.25 | 0.51 | 0.94 |
| fslora_st | 40 | 0.18 | 0.18 | 0.029 | 0.949 / 0.875 | 0.16 | 0.08 | 0.33 | 0.67 |
| fslora_st | 50 | 0.13 | 0.13 | 0.042 | 0.982 / 0.829 | 0.02 | 0.01 | 0.41 | 0.33 |
| fslora_st | 60 | 0.13 | 0.13 | 0.018 | 0.983 / 0.817 | 0.01 | 0.01 | 0.32 | 0.48 |
| fslora | 1 | 0.84 | 0.82 | 0.563 | 1.966 / 1.852 | 0.96 | 0.71 | 1.00 | 1.00 |
| fslora | 10 | 0.37 | 0.38 | 0.033 | 0.966 / 0.940 | 0.52 | 0.23 | 0.74 | 0.75 |
| fslora | 20 | 0.30 | 0.29 | 0.038 | 0.867 / 0.859 | 0.45 | 0.19 | 0.43 | 0.56 |
| fslora | 30 | 0.22 | 0.22 | 0.022 | 0.864 / 0.723 | 0.43 | 0.20 | 0.48 | 0.73 |
| fslora | 40 | 0.18 | 0.18 | 0.020 | 0.860 / 0.766 | 0.28 | 0.12 | 0.09 | 0.34 |
| fslora | 50 | 0.17 | 0.17 | 0.047 | 0.874 / 0.697 | 0.32 | 0.14 | 0.64 | 0.64 |
| fslora | 60 | 0.16 | 0.16 | 0.026 | 0.852 / 0.664 | 0.21 | 0.08 | 0.32 | 0.49 |
| amp_a4 | 1 | 0.55 | 0.58 | 0.700 | 1.683 / 1.852 | 0.91 | 0.70 | 1.03 | 0.99 |
| amp_a4 | 10 | 0.72 | 0.71 | 0.130 | 0.689 / 0.959 | 0.78 | 0.35 | 1.25 | 0.71 |
| amp_a4 | 20 | 0.74 | 0.71 | 0.294 | 0.810 / 0.977 | 0.92 | 0.45 | 1.26 | 0.99 |
| amp_a4 | 30 | 0.74 | 0.73 | 0.215 | 0.700 / 0.893 | 0.90 | 0.46 | 1.17 | 0.99 |
| amp_a4 | 40 | 0.76 | 0.74 | 0.158 | 0.715 / 0.804 | 0.81 | 0.39 | 1.25 | 0.98 |
| amp_a4 | 50 | 0.72 | 0.72 | 0.083 | 0.626 / 0.887 | 0.71 | 0.33 | 0.91 | 0.65 |
| amp_a4 | 60 | 0.73 | 0.72 | 0.145 | 0.622 / 0.799 | 0.76 | 0.35 | 1.25 | 0.97 |
| amp_a16 | 1 | 0.52 | 0.51 | 0.543 | 2.262 / 1.852 | 0.85 | 0.63 | 0.95 | 0.99 |
| amp_a16 | 10 | 0.40 | 0.42 | 0.016 | 1.324 / 1.091 | 0.27 | 0.17 | -0.06 | -0.26 |
| amp_a16 | 20 | 0.29 | 0.29 | 0.004 | 1.303 / 1.122 | -0.17 | -0.12 | -0.14 | -0.26 |
| amp_a16 | 30 | 0.21 | 0.22 | 0.009 | 1.296 / 1.053 | 0.09 | 0.12 | 0.23 | 0.60 |
| amp_a16 | 40 | 0.03 | 0.03 | 0.019 | 1.279 / 1.158 | -0.21 | -0.21 | 0.16 | 0.62 |
| amp_a16 | 50 | 0.18 | 0.13 | 0.100 | 1.207 / 1.188 | -0.49 | -0.43 | 0.01 | 0.02 |
| amp_a16 | 60 | -0.11 | -0.04 | 0.000 | 1.196 / 1.183 | -0.05 | -0.03 | -0.11 | -0.55 |
| amp_a32 | 1 | 0.39 | 0.38 | 0.463 | 2.686 / 1.852 | 0.76 | 0.57 | 0.94 | 0.99 |
| amp_a32 | 10 | 0.17 | 0.16 | -0.015 | 1.950 / 1.257 | 0.11 | 0.07 | -0.09 | -0.42 |
| amp_a32 | 20 | 0.07 | 0.05 | -0.043 | 2.311 / 1.339 | -0.15 | -0.09 | -0.13 | -0.10 |
| amp_a32 | 30 | 0.09 | 0.06 | -0.028 | 1.928 / 1.225 | 0.31 | 0.21 | 0.33 | 1.00 |
| amp_a32 | 40 | 0.02 | 0.01 | -0.000 | 1.613 / 1.243 | 0.02 | 0.01 | 0.15 | 0.31 |
| amp_a32 | 50 | 0.18 | 0.10 | -0.033 | 1.502 / 1.190 | 0.24 | 0.16 | -0.20 | -0.76 |
| amp_a32 | 60 | 0.38 | 0.22 | -0.000 | 1.443 / 1.158 | 0.37 | 0.23 | -0.70 | -0.92 |

## Pre-registered reading (single seed)

| question | criterion | result |
|---|---|---|
| H0 restricted training costs accuracy over many rounds | L2 late ≤ r8 late − 1.0 pp | True |
| H1 amp matches vref | amp late ≥ vref late − 0.5 pp | False |
| H2 amp beats L2 | amp late ≥ L2 late + 1.0 pp | True |
| H3 cold start acceptable | amp early ≥ vref early − 2.0 pp | True |
| H4 amp stable | osc(amp) ≤ 1.5 × osc(vref) | True |
| H5a amplifying only the increment beats sketching the state | amp late ≥ fslora_st late + 0.3 pp | True |
| H5b gradient path (fslora_st − fslora), reported as an implementation effect | \|diff\| ≥ 0.3 pp | True (-3.57 pp) |
| H6 α = R/r is right (not re-selected) | amp(α=R/r) late ≥ best of α ∈ {4, 16, 32} − 0.3 pp | True (best run: amp) |

Reading: H0–H2 true → the method holds on the decoder LLM; add seeds 2041 / 2042. H0 false → restricted training costs little here over many rounds (scope statement). H0 true, H1 false → use H3, H6 and the mechanism table to separate cold start, α, direction and the method itself. H5a is the state-sketch claim; H5b is an implementation effect, not part of the method claim. H6 false → report that R/r is not the best α here; the method's α is not re-chosen on these results. One seed: differences below ~0.5 pp are not interpreted.

## Stage C: three seeds (2040 from stage B; 2041, 2042), late core-weak accuracy

Same plan and lr (stage A choice) for every seed; seeds share schedule and initialisation across methods, so differences are paired per seed. 95% t interval with 2 degrees of freedom (t = 4.30).

| method | s2040 | s2041 | s2042 | mean ± sd |
|---|---:|---:|---:|---:|
| r8 | 82.52 | 82.27 | 81.80 | 82.19 ± 0.36 |
| vref | 79.92 | 80.30 | 79.38 | 79.87 ± 0.46 |
| amp | 78.45 | 78.35 | 78.27 | 78.36 ± 0.09 |
| fslora | 78.45 | 78.07 | 78.10 | 78.21 ± 0.21 |

| comparison | per seed | mean ± sd | 95% CI | criterion | result |
|---|---|---:|---|---|---|
| MS1: amp − vref | -1.47, -1.95, -1.12 | -1.51 ± 0.42 | [-2.55, -0.47] | mean ≥ −0.5 pp → amp matches vref | False |
| MS2: amp − fslora | +0.00, +0.28, +0.17 | +0.15 ± 0.14 | [-0.20, +0.50] | |mean| < 0.5 pp → equivalent; ≥ 0.5 pp with all seeds same sign → that side better | equivalent |
| MS3: r8 − amp | +4.07, +3.92, +3.53 | +3.84 ± 0.28 | [+3.16, +4.52] | descriptive (gap to the upper bound) | descriptive |
| MS4: r8 − vref | +2.60, +1.97, +2.42 | +2.33 ± 0.33 | [+1.52, +3.14] | descriptive (cost of uploading only assigned components) | descriptive |

Reading: MS1 true → amp matches vref across seeds (the single-seed H1 gap was noise); false → report the gap with its interval. MS2 decides amp vs FSLoRA's forward with true gradient (both with Hájek): 'equivalent', 'amp better' or 'fslora better'. MS3 / MS4 are descriptive (distance to the homogeneous upper bound; cost of uploading only assigned components).
