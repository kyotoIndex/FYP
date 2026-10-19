# D2 single-round dispersion (scra_diagnostics_20261017_v1)

One simulated round from each snapshot, repeated over the real schedule of the following rounds; every rule is applied once to the same snapshot. `diff` = accuracy minus r8 on the same draw (removes participant-sampling noise shared by all rules). Product-space numbers are relative to |mean r8 update|^2.

## Snapshot round 20 (24 draws, shrink 0.63; before update: acc 66.72, core 63.89)

| rule | acc mean | acc std | diff mean | diff std | core diff mean | core diff std | min acc | LoRA var | LoRA bias² | LoRA paired | step along r8 (mean ± std) | head paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r8 | 74.42 | 0.25 | +0.00 | 0.00 | +0.00 | 0.00 | 73.50 | 0.682 | 0.000 | 0.000 | 1.00 ± 0.06 | 0.000 |
| vref | 74.41 | 0.26 | -0.00 | 0.05 | -0.00 | 0.07 | 73.49 | 1.594 | 0.040 | 0.953 | 1.00 ± 0.05 | 0.000 |
| L2 | 64.65 | 5.28 | -9.77 | 5.18 | -9.57 | 5.03 | 55.29 | 1.730 | 0.724 | 1.825 | 1.70 ± 0.26 | 1.174 |
| hajek | 62.33 | 3.71 | -12.09 | 3.61 | -11.77 | 3.56 | 55.29 | 1.750 | 1.041 | 2.175 | 1.86 ± 0.14 | 1.174 |
| shrink | 70.71 | 1.27 | -3.70 | 1.16 | -3.83 | 1.16 | 68.13 | 0.724 | 0.160 | 0.707 | 1.20 ± 0.07 | 1.174 |
| shrink_fullhead | 73.14 | 0.65 | -1.28 | 0.56 | -1.24 | 0.59 | 71.49 | 0.724 | 0.160 | 0.707 | 1.20 ± 0.07 | 0.000 |
| vref_restrhead | 73.25 | 0.56 | -1.16 | 0.43 | -1.30 | 0.49 | 71.71 | 1.594 | 0.040 | 0.953 | 1.00 ± 0.05 | 1.174 |

Checks: acc_spread_ratio_shrink_vs_vref = 25.63, lora_paired_error_ratio_shrink_vs_vref = 0.74, lora_variance_ratio_shrink_vs_vref = 0.45, lora_bias_over_variance_shrink = 0.22, head_swap_shrink_fullhead_over_shrink = 0.48, head_swap_vref_restrhead_over_vref = 9.46
Weak single-pair restricted vs full (assigned blocks, median over draws): cos 0.89, norm ratio 1.12 (D1 pair-averaged: cos ≈ 0.9, ratio ≈ 1.1).

## Snapshot round 60 (24 draws, shrink 0.63; before update: acc 70.07, core 67.47)

| rule | acc mean | acc std | diff mean | diff std | core diff mean | core diff std | min acc | LoRA var | LoRA bias² | LoRA paired | step along r8 (mean ± std) | head paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r8 | 78.80 | 0.19 | +0.00 | 0.00 | +0.00 | 0.00 | 78.31 | 0.371 | 0.000 | 0.000 | 1.00 ± 0.07 | 0.000 |
| vref | 78.78 | 0.20 | -0.02 | 0.05 | -0.03 | 0.07 | 78.29 | 0.751 | 0.016 | 0.396 | 1.00 ± 0.07 | 0.000 |
| L2 | 72.09 | 2.19 | -6.71 | 2.12 | -6.81 | 2.27 | 67.76 | 0.813 | 0.366 | 0.844 | 1.55 ± 0.17 | 0.560 |
| hajek | 70.51 | 1.50 | -8.29 | 1.40 | -8.44 | 1.49 | 67.76 | 0.803 | 0.556 | 1.026 | 1.69 ± 0.11 | 0.560 |
| shrink | 76.45 | 0.59 | -2.35 | 0.50 | -2.28 | 0.58 | 75.29 | 0.336 | 0.041 | 0.281 | 1.09 ± 0.06 | 0.560 |
| shrink_fullhead | 78.22 | 0.28 | -0.58 | 0.20 | -0.45 | 0.22 | 77.69 | 0.336 | 0.041 | 0.281 | 1.09 ± 0.06 | 0.000 |
| vref_restrhead | 77.56 | 0.50 | -1.24 | 0.36 | -1.20 | 0.46 | 76.34 | 0.751 | 0.016 | 0.396 | 1.00 ± 0.07 | 0.560 |

Checks: acc_spread_ratio_shrink_vs_vref = 10.21, lora_paired_error_ratio_shrink_vs_vref = 0.71, lora_variance_ratio_shrink_vs_vref = 0.45, lora_bias_over_variance_shrink = 0.12, head_swap_shrink_fullhead_over_shrink = 0.41, head_swap_vref_restrhead_over_vref = 7.49
Weak single-pair restricted vs full (assigned blocks, median over draws): cos 0.92, norm ratio 1.14 (D1 pair-averaged: cos ≈ 0.9, ratio ≈ 1.1).

## Snapshot round 120 (24 draws, shrink 0.63; before update: acc 78.40, core 76.82)

| rule | acc mean | acc std | diff mean | diff std | core diff mean | core diff std | min acc | LoRA var | LoRA bias² | LoRA paired | step along r8 (mean ± std) | head paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r8 | 80.86 | 0.21 | +0.00 | 0.00 | +0.00 | 0.00 | 80.18 | 0.936 | 0.000 | 0.000 | 1.00 ± 0.11 | 0.000 |
| vref | 80.86 | 0.24 | +0.01 | 0.06 | +0.02 | 0.07 | 80.11 | 1.796 | 0.038 | 0.903 | 1.00 ± 0.11 | 0.000 |
| L2 | 77.46 | 1.68 | -3.39 | 1.64 | -3.51 | 1.74 | 73.87 | 1.964 | 0.490 | 1.562 | 1.60 ± 0.25 | 0.653 |
| hajek | 76.41 | 1.42 | -4.45 | 1.30 | -4.61 | 1.42 | 73.61 | 1.928 | 0.752 | 1.790 | 1.77 ± 0.17 | 0.653 |
| shrink | 79.68 | 0.66 | -1.18 | 0.57 | -1.20 | 0.61 | 78.15 | 0.819 | 0.086 | 0.623 | 1.15 ± 0.10 | 0.653 |
| shrink_fullhead | 80.43 | 0.43 | -0.43 | 0.32 | -0.43 | 0.33 | 79.34 | 0.819 | 0.086 | 0.623 | 1.15 ± 0.10 | 0.000 |
| vref_restrhead | 80.52 | 0.42 | -0.34 | 0.27 | -0.35 | 0.28 | 79.31 | 1.796 | 0.038 | 0.903 | 1.00 ± 0.11 | 0.653 |

Checks: acc_spread_ratio_shrink_vs_vref = 10.20, lora_paired_error_ratio_shrink_vs_vref = 0.69, lora_variance_ratio_shrink_vs_vref = 0.46, lora_bias_over_variance_shrink = 0.11, head_swap_shrink_fullhead_over_shrink = 0.56, head_swap_vref_restrhead_over_vref = 4.80
Weak single-pair restricted vs full (assigned blocks, median over draws): cos 0.93, norm ratio 1.08 (D1 pair-averaged: cos ≈ 0.9, ratio ≈ 1.1).

## Pre-registered reading (≥ 2 of 3 snapshots)

| question | criterion | result |
|---|---|---|
| H1: the gap is single-round dispersion from stitching | acc spread ratio shrink/vref ≥ 2 and LoRA paired-error ratio ≥ 2 | False |
| H1 rejected: instability builds up over rounds instead | acc spread ratio < 1.5 | False |
| expected bias is not the main term after shrinking | LoRA bias² / variance < 1 | True |
| where it lives | head: shrink_fullhead/shrink ≤ 0.6 or vref_restrhead/vref ≥ 2; LoRA: ≥ 0.8 and < 1.5 | head |

Next step by outcome: H1 true + LoRA → fix the local training dynamics of the restricted components; H1 true + head → fix head aggregation / co-adaptation (cheaper); H1 rejected → multi-round rollout diagnostic before any method.
