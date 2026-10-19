# C2-0 decoder-LLM mechanism probe (c2_probe_20261018)

One update from each snapshot of a homogeneous rank-R run, over the real schedule of the following rounds (same protocol as D2 / D3). Steps are projections on the full-training reference: per weak client against its own full run on its assigned components, per aggregate against r8 on the same draw. `product` = exact LoRA ΔW (all adapters), `function` = label-word log-probabilities on a fixed probe batch. Accuracy diffs are core-weak, one step, paired per draw: mean ± sd (negative/positive draws).

**Scope.** This probe does not measure resource savings (unassigned components are not folded; peak memory here is not evidence for C1). The model has no newly initialised classification head, but the frozen tied output embedding still computes the label-word logits. M1 checks update scale, not task gain; task gain is C2-1 / C2-2.

**Learnability gate.** M2 is read only on snapshots whose core-weak accuracy before the update is ≥ 45% (chance 33.3%).

Snapshot run (all / core accuracy): r0 58.25 / 57.75, r10 81.45 / 80.08, r20 81.80 / 80.25, r30 83.65 / 82.50.

## Snapshot round 0 (12 draws, R=64, r=8, α=8.0; before update: acc 58.25, core 57.75 — gate passed)

| rule | product step vs r8 | function step vs r8 | paired rel. error | core diff vs r8 (pp) | all diff (pp) |
|---|---:|---:|---:|---:|---:|
| r8 | 1.00 | 1.00 | 0.000 | +0.00 ± 0.00 (0−/0+) | +0.00 |
| vref | 1.00 | 1.00 | 0.790 | -0.47 ± 2.01 (7−/5+) | -0.24 |
| L2 | 0.92 | 1.04 | 0.525 | -17.97 ± 10.09 (12−/0+) | -19.17 |
| hajek | 0.83 | 1.02 | 1.163 | -21.01 ± 9.87 (12−/0+) | -22.52 |
| amp | 0.96 | 1.01 | 0.870 | -6.30 ± 7.01 (12−/0+) | -6.45 |

amp − vref (core, pp): -5.83 ± 6.80 (11−/1+).
Per weak client (medians, assigned components): restricted product step 0.41, cos 0.41; amplified product step 0.79, cos 0.79. Function step vs own full run (≈ r/R = 0.125 when matched): restricted 0.873, amplified 0.668. Local loss: full 1.9529, restricted 1.2162, amplified 1.9443.

## Snapshot round 5 (12 draws, R=64, r=8, α=8.0; before update: acc 78.65, core 76.67 — gate passed)

| rule | product step vs r8 | function step vs r8 | paired rel. error | core diff vs r8 (pp) | all diff (pp) |
|---|---:|---:|---:|---:|---:|
| r8 | 1.00 | 1.00 | 0.000 | +0.00 ± 0.00 (0−/0+) | +0.00 |
| vref | 1.00 | 1.05 | 4.721 | -0.17 ± 0.49 (8−/4+) | -0.24 |
| L2 | 0.83 | 1.54 | 3.318 | -6.90 ± 7.60 (11−/1+) | -6.83 |
| hajek | 0.72 | 2.53 | 6.124 | -12.26 ± 8.05 (10−/2+) | -12.52 |
| amp | 0.93 | 0.90 | 4.910 | -0.14 ± 0.45 (8−/4+) | -0.37 |

amp − vref (core, pp): +0.03 ± 0.58 (6−/6+).
Per weak client (medians, assigned components): restricted product step 0.66, cos 0.62; amplified product step 0.90, cos 0.90. Function step vs own full run (≈ r/R = 0.125 when matched): restricted 0.097, amplified 0.086. Local loss: full 0.9016, restricted 0.5627, amplified 0.9065.

## Snapshot round 30 (12 draws, R=64, r=8, α=8.0; before update: acc 83.65, core 82.50 — gate passed)

| rule | product step vs r8 | function step vs r8 | paired rel. error | core diff vs r8 (pp) | all diff (pp) |
|---|---:|---:|---:|---:|---:|
| r8 | 1.00 | 1.00 | 0.000 | +0.00 ± 0.00 (0−/0+) | +0.00 |
| vref | 1.00 | 1.11 | 4.778 | -0.33 ± 0.41 (8−/4+) | -0.14 |
| L2 | 0.83 | 1.15 | 4.677 | -3.42 ± 4.62 (12−/0+) | -3.15 |
| hajek | 0.79 | 1.12 | 5.948 | -4.45 ± 4.43 (12−/0+) | -4.19 |
| amp | 0.96 | 1.08 | 4.959 | -0.33 ± 0.51 (9−/3+) | -0.20 |

amp − vref (core, pp): +0.01 ± 0.28 (4−/6+).
Per weak client (medians, assigned components): restricted product step 0.79, cos 0.74; amplified product step 0.94, cos 0.94. Function step vs own full run (≈ r/R = 0.125 when matched): restricted 0.102, amplified 0.101. Local loss: full 0.6538, restricted 0.4657, amplified 0.6510.

## Pre-registered reading (≥ 2 of 3 snapshots)

| question | criterion | result |
|---|---|---|
| M0 restricted training overshoots without a new head | hajek aggregate product or function step ≥ 1.2 | False |
| M1 amplification corrects the update scale | amp aggregate step in [0.9, 1.1], per-client amp step in [0.95, 1.05], cos ≥ 0.95 | False |
| M2 one-step accuracy signal (gated snapshots only: 3) | L2 or hajek core diff ≤ −0.5 pp and \|amp − vref\| ≤ 0.2 pp | True |

Next step: M0 and M1 → C2-1 (headroom run). M0 but not M1 → inspect the per-client probe for the failing assumption before changing the method. Not M0 → state the scope (settings with co-trained new parameters); C2-1 only checks that restricted training costs nothing here. M0 and M1 without M2 → the mechanism is present but one step does not show it in accuracy; go to C2-1. M2 not interpretable → the task was not learned; fix the prompt / labels or lengthen the snapshot run before reading M2. A marginal M2 (few draws in the same direction) is not a strong result.
