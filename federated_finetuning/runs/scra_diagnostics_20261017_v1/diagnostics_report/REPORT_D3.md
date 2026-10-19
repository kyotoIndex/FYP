# D3 realizable-fix screening (scra_diagnostics_20261017_v1)

Same protocol as D2: one update from each snapshot over the real schedule of the following rounds. `core diff` = core-weak accuracy minus r8 on the same draw. `closure` = share of the shrink → vref core gap closed (0 = shrink, 1 = vref). LoRA / head numbers are relative to |mean r8 update|².

## Snapshot round 20 (24 draws, shrink 0.63, amp alpha 4.0; before update: acc 66.72, core 63.89)

| rule | core diff mean | core diff std | diff mean | min acc | closure | LoRA paired | LoRA step along r8 | head paired | head step along r8 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| r8 | +0.00 | 0.00 | +0.00 | 74.04 | 1.00 | 0.000 | 1.00 ± 0.06 | 0.000 | 1.00 ± 0.07 |
| vref | +0.01 | 0.07 | -0.00 | 74.05 | 1.00 | 0.924 | 1.00 ± 0.06 | 0.000 | 1.00 ± 0.07 |
| shrink | -3.76 | 1.18 | -3.65 | 67.41 | 0.00 | 0.678 | 1.17 ± 0.08 | 1.114 | 1.91 ± 0.13 |
| shrink_fullhead | -1.27 | 0.61 | -1.30 | 71.55 | 0.66 | — | — | — | — |
| shrink_h40 | -1.21 | 0.63 | -1.22 | 71.56 | 0.68 | — | — | — | — |
| shrink_h55 | -1.71 | 0.75 | -1.68 | 70.55 | 0.54 | 0.678 | 1.17 ± 0.08 | 0.195 | 1.09 ± 0.06 |
| shrink_h70 | -2.33 | 0.88 | -2.26 | 69.72 | 0.38 | — | — | — | — |
| shrink_hstrong | -0.93 | 0.68 | -0.98 | 72.39 | 0.75 | — | — | — | — |
| shrink_hnone | -0.34 | 0.36 | -0.49 | 73.10 | 0.91 | — | — | — | — |
| hajek_h55 | -8.16 | 2.47 | -8.48 | 58.84 | -1.17 | — | — | — | — |
| fh_hajek | -15.95 | 3.74 | -16.53 | 49.89 | -3.23 | 4.571 | 2.49 ± 0.20 | 2.342 | 0.57 ± 0.50 |
| fh_shrink | -4.44 | 1.36 | -4.34 | 68.10 | -0.18 | — | — | — | — |
| amp_hajek | -0.00 | 0.08 | -0.03 | 74.08 | 1.00 | 0.941 | 1.00 ± 0.06 | 0.004 | 1.01 ± 0.07 |
| amp_fullhead | +0.01 | 0.07 | -0.02 | 74.09 | 1.00 | — | — | — | — |
| rcs2 | -0.77 | 0.53 | -0.82 | 72.28 | 0.79 | 0.674 | 1.07 ± 0.07 | 0.193 | 1.00 ± 0.06 |
| oracle2 | -0.36 | 0.31 | -0.39 | 73.31 | 0.90 | 0.537 | 0.96 ± 0.06 | 0.200 | 0.95 ± 0.06 |

Pooled coefficients: strong replays (rcs2) LoRA mean 0.630, head 0.501; weak oracle LoRA mean 0.521, head 0.473.
Per weak client vs its full run (median): restricted block cos 0.89, norm ratio 1.11, head step 1.13; amplified block cos 1.00, norm ratio 1.00, head step 1.00.
Mean local training loss (weak): full 0.7030, restricted 0.6888, frozen head 0.7292, amplified 0.7038.

## Snapshot round 60 (24 draws, shrink 0.63, amp alpha 4.0; before update: acc 70.07, core 67.47)

| rule | core diff mean | core diff std | diff mean | min acc | closure | LoRA paired | LoRA step along r8 | head paired | head step along r8 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| r8 | +0.00 | 0.00 | +0.00 | 78.15 | 0.99 | 0.000 | 1.00 ± 0.05 | 0.000 | 1.00 ± 0.07 |
| vref | +0.01 | 0.05 | +0.00 | 78.19 | 1.00 | 0.412 | 1.00 ± 0.05 | 0.000 | 1.00 ± 0.07 |
| shrink | -2.34 | 0.47 | -2.40 | 75.01 | 0.00 | 0.289 | 1.10 ± 0.04 | 0.568 | 1.66 ± 0.10 |
| shrink_fullhead | -0.44 | 0.22 | -0.58 | 77.62 | 0.81 | — | — | — | — |
| shrink_h40 | -0.15 | 0.21 | -0.23 | 78.08 | 0.93 | — | — | — | — |
| shrink_h55 | -0.49 | 0.24 | -0.61 | 77.63 | 0.79 | 0.289 | 1.10 ± 0.04 | 0.090 | 0.96 ± 0.05 |
| shrink_h70 | -0.97 | 0.31 | -1.08 | 76.92 | 0.58 | — | — | — | — |
| shrink_hstrong | -0.27 | 0.44 | -0.30 | 77.24 | 0.88 | — | — | — | — |
| shrink_hnone | +0.08 | 0.21 | +0.19 | 78.85 | 1.03 | — | — | — | — |
| hajek_h55 | -5.25 | 0.86 | -5.32 | 71.57 | -1.24 | — | — | — | — |
| fh_hajek | -10.91 | 1.18 | -11.10 | 64.88 | -3.65 | 2.728 | 2.34 ± 0.15 | 1.620 | 0.64 ± 0.48 |
| fh_shrink | -2.57 | 1.04 | -2.82 | 72.95 | -0.10 | — | — | — | — |
| amp_hajek | -0.00 | 0.06 | -0.01 | 78.11 | 0.99 | 0.420 | 1.00 ± 0.05 | 0.003 | 1.00 ± 0.07 |
| amp_fullhead | +0.01 | 0.06 | -0.00 | 78.09 | 1.00 | — | — | — | — |
| rcs2 | -0.41 | 0.23 | -0.53 | 77.78 | 0.82 | 0.303 | 1.09 ± 0.04 | 0.087 | 0.99 ± 0.05 |
| oracle2 | -0.02 | 0.21 | -0.07 | 78.23 | 0.98 | 0.253 | 0.99 ± 0.04 | 0.088 | 0.97 ± 0.05 |

Pooled coefficients: strong replays (rcs2) LoRA mean 0.652, head 0.573; weak oracle LoRA mean 0.564, head 0.558.
Per weak client vs its full run (median): restricted block cos 0.92, norm ratio 1.15, head step 1.17; amplified block cos 1.00, norm ratio 1.00, head step 1.00.
Mean local training loss (weak): full 0.6073, restricted 0.6055, frozen head 0.6596, amplified 0.6072.

## Snapshot round 120 (24 draws, shrink 0.63, amp alpha 4.0; before update: acc 78.40, core 76.82)

| rule | core diff mean | core diff std | diff mean | min acc | closure | LoRA paired | LoRA step along r8 | head paired | head step along r8 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| r8 | +0.00 | 0.00 | +0.00 | 80.70 | 1.01 | 0.000 | 1.00 ± 0.12 | 0.000 | 1.00 ± 0.13 |
| vref | -0.01 | 0.06 | -0.00 | 80.71 | 1.00 | 0.948 | 1.00 ± 0.12 | 0.000 | 1.00 ± 0.13 |
| shrink | -1.08 | 0.67 | -1.08 | 78.40 | 0.00 | 0.665 | 1.14 ± 0.11 | 0.593 | 1.61 ± 0.17 |
| shrink_fullhead | -0.38 | 0.32 | -0.39 | 79.62 | 0.66 | — | — | — | — |
| shrink_h40 | -0.25 | 0.24 | -0.28 | 80.00 | 0.77 | — | — | — | — |
| shrink_h55 | -0.44 | 0.34 | -0.44 | 79.58 | 0.60 | 0.665 | 1.14 ± 0.11 | 0.164 | 0.93 ± 0.10 |
| shrink_h70 | -0.63 | 0.45 | -0.63 | 79.21 | 0.42 | — | — | — | — |
| shrink_hstrong | -0.23 | 0.22 | -0.24 | 79.98 | 0.79 | — | — | — | — |
| shrink_hnone | -0.01 | 0.14 | -0.00 | 80.56 | 1.00 | — | — | — | — |
| hajek_h55 | -3.14 | 1.37 | -3.09 | 75.21 | -1.93 | — | — | — | — |
| fh_hajek | -5.93 | 1.60 | -5.80 | 72.80 | -4.53 | 3.304 | 2.27 ± 0.25 | 3.174 | 0.67 ± 0.58 |
| fh_shrink | -1.25 | 0.62 | -1.28 | 78.19 | -0.16 | — | — | — | — |
| amp_hajek | +0.00 | 0.06 | -0.00 | 80.69 | 1.01 | 0.974 | 0.99 ± 0.12 | 0.007 | 1.00 ± 0.13 |
| amp_fullhead | +0.00 | 0.06 | -0.00 | 80.69 | 1.01 | — | — | — | — |
| rcs2 | -0.58 | 0.41 | -0.60 | 79.35 | 0.46 | 0.779 | 1.15 ± 0.11 | 0.153 | 1.00 ± 0.10 |
| oracle2 | -0.08 | 0.18 | -0.10 | 80.44 | 0.93 | 0.587 | 0.98 ± 0.09 | 0.160 | 0.95 ± 0.10 |

Pooled coefficients: strong replays (rcs2) LoRA mean 0.706, head 0.597; weak oracle LoRA mean 0.552, head 0.561.
Per weak client vs its full run (median): restricted block cos 0.93, norm ratio 1.08, head step 1.09; amplified block cos 0.99, norm ratio 1.00, head step 1.00.
Mean local training loss (weak): full 0.5502, restricted 0.5313, frozen head 0.5412, amplified 0.5495.

## Pre-registered reading (≥ 2 of 3 snapshots)

| question | criterion | result |
|---|---|---|
| a scalar on weak heads fixes most of it | best of shrink_h40/55/70 closure ≥ 0.6 | True |
| calibration from strong replays is realizable | rcs2 closure ≥ 0.6 and within 0.3 pp of oracle2 | False |
| amplified restricted training works without constants | amp_hajek closure ≥ 0.6 | True |
| amplification is not worse than calibration | amp_hajek ≥ rcs2 − 0.2 pp | True |
| weak heads carry signal (dropping them hurts) | closure(hstrong) ≤ best head scalar − 0.2 | False |

Next step by outcome: amplification works and is not worse → main candidate is amplified restricted training, confirm with a full D0 run; else rcs2 realizable → RCS v2 (head + LoRA calibration) full D0 run; head scalar works but rcs2 not → estimator transfer problem, inspect coefficients; nothing closes ≥ 0.6 → the one-step view misses the mechanism, rethink before any method.
