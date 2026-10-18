# Batch 2: positioning controls (2026-10-18)

Core weak accuracy (%); `late` = rounds 110–150 (the batch-1 metric), `early` = 10–50, `mid` = 60–100. Seeds share participant schedule and initialisation, so differences are paired per seed. With 3 seeds the 95% t interval is wide (t = 4.30); verdicts use the mean threshold plus sign agreement. FSLoRA's forward equals amp's while the adapter is zero only in the Adam eps → 0 limit (FIX_BATCH2_CONTROLS_2026-10-18 §2); A1a removes that confound.

| method | late s2040 | late s2041 | late s2042 | late mean ± sd | early | mid | osc |
|---|---:|---:|---:|---:|---:|---:|---:|
| r8 (upper) | 80.13 | 80.26 | 80.19 | 80.19 ± 0.07 | 72.88 | 78.69 | 0.34 |
| vref (full train, upload r/R, Hájek) | 80.15 | 80.32 | 80.02 | 80.17 ± 0.15 | 72.79 | 78.66 | 0.37 |
| **amp (ours)** | 80.05 | 80.32 | 79.99 | 80.12 ± 0.17 | 72.76 | 78.58 | 0.35 |
| A1a FSLoRA forward, amp gradient path, balanced + Hájek | 79.71 | 79.75 | 79.63 | 79.70 ± 0.06 | 72.43 | 78.19 | 0.33 |
| A1b FSLoRA forward, true gradient, balanced + Hájek | 79.71 | 79.91 | 79.78 | 79.80 ± 0.10 | 72.53 | 78.34 | 0.32 |
| A1c FSLoRA forward, true gradient, balanced + zero-fill | 77.18 | 77.35 | 77.02 | 77.18 ± 0.17 | 65.65 | 75.12 | 0.49 |
| A2 LoRA lr × R/r + server ÷ R/r | 79.36 | 79.31 | 79.40 | 79.36 ± 0.05 | 71.59 | 77.79 | 0.41 |
| ext. FSLoRA-style (our AdamW protocol; random, zero-fill) | 77.09 | 77.30 | 77.11 | 77.17 ± 0.11 | 65.69 | 75.11 | 0.50 |
| ext. Fed-PLoRA style (random, owner) | 77.89 | 76.17 | 77.42 | 77.16 ± 0.89 | 65.63 | 74.98 | 2.72 |
| ext. FLASC-style (full train, global top-r/R upload by weak, zero-fill) | 79.28 | 79.51 | 79.21 | 79.33 ± 0.16 | 71.12 | 77.83 | 0.36 |
| L2 (balanced, owner) | 77.72 | 77.28 | 78.56 | 77.85 ± 0.65 | 64.16 | 75.57 | 2.69 |

## Mechanism controls (one factor differs per pair)

| comparison | a − b | per seed (late) | mean ± sd | 95% CI | early | mid | criterion | result |
|---|---|---|---:|---|---:|---:|---|---|
| A1a_state_sketch | amp − fslora_st | +0.35, +0.56, +0.36 | +0.42 ± 0.12 | [+0.13, +0.72] | +0.34 | +0.39 | mean ≥ 0.3 pp and every seed > 0 | True |
| A1b_gradient_path | fslora_st − fslora_hajek | -0.01, -0.16, -0.16 | -0.11 ± 0.08 | [-0.32, +0.10] | -0.11 | -0.15 | |mean| ≥ 0.3 pp, all seeds same sign | False |
| A1c_aggregation | fslora_hajek − fslora_zerofill | +2.53, +2.56, +2.77 | +2.62 ± 0.13 | [+2.30, +2.94] | +6.89 | +3.22 | mean ≥ 0.3 pp and every seed > 0 | True |
| A1_total_vs_fslora_forward | amp − fslora_hajek | +0.34, +0.41, +0.21 | +0.32 ± 0.10 | [+0.07, +0.57] | +0.23 | +0.24 | mean ≥ 0.3 pp and every seed > 0 | True |
| A2_lr_compensation | amp − lrcomp | +0.70, +1.00, +0.59 | +0.76 ± 0.22 | [+0.23, +1.30] | +1.18 | +0.79 | |mean| < 0.3 pp | False |

## External positioning (several factors differ; not mechanism evidence)

| comparison | a − b | per seed (late) | mean ± sd | 95% CI | early | mid | criterion | result |
|---|---|---|---:|---|---:|---:|---|---|
| E1_vs_fslora_style | amp − fslora_paper | +2.96, +3.02, +2.88 | +2.95 ± 0.07 | [+2.78, +3.12] | +7.08 | +3.48 | mean ≥ 0.3 pp and every seed > 0 | True |
| E2_vs_fedplora_style | amp − fedplora | +2.17, +4.14, +2.58 | +2.96 ± 1.04 | [+0.37, +5.55] | +7.14 | +3.61 | mean ≥ 1.0 pp and every seed > 0 | True |
| E3_vs_flasc_style | amp − flasc | +0.78, +0.81, +0.78 | +0.79 ± 0.02 | [+0.75, +0.83] | +1.65 | +0.75 | mean ≥ -0.3 pp | True |

## Reading (pre-registered)

- A1a true → amplifying only the increment (not sketching the state) has value beyond FSLoRA's forward; A1a false → narrow the claim to a new explanation of FSLoRA-style amplification (Adam progress matching, head / component overshoot) plus the diagnostics.
- A1b true → FSLoRA's true-gradient implementation itself changes the outcome under AdamW (gradient scale / eps); report it as an implementation effect, not as the state-sketch effect.
- A1c true → Hájek helps FSLoRA-style training too.
- A2 true → present the method as matching local progress to full-rank training, forward amplification being one implementation; A2 false → the forward form matters, report which side wins.
- E1–E3 position the method against published-method styles under our unified protocol only.
- A result of None means a run is missing.
