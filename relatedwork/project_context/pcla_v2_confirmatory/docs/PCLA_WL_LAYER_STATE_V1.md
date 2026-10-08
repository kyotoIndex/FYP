# PCLA-WL layer-state repair (v1)

Date: 2026-09-21.

`layer_only` (reported as PCLA-WL) now stores and ranks **layer** EMA utilities.
It no longer applies per-rank z-scoring, UCB, or age bonuses and then sums
those terms into a layer score.

## Why

Whole-layer allocation is the decision. Rank-level bandit state was leftover
from component PCLA. A dedicated `LayerUtilityState` already existed for
`pcla_lite` and is now shared with `layer_only`.

## Policies

| Policy | State | Layer score |
|---|---|---|
| `layer_only` | `LayerUtilityState` | EMA of summed component probes |
| `layer_only_rank_state` | `UtilityState` | sum of z + UCB + age over ranks |

`layer_only_rank_state` exists only to reproduce the pre-repair selector.

## Frozen evidence

`results/pcla_v2_confirmatory_v1/` and later `layer_only` screens used the rank
state. Do not mix those numbers with new `layer_only` runs. New experiments
need a new results directory.

Resume of an old `layer_only` checkpoint with this code is refused: the client
state schema changed.

## Required comparison before a new quality claim

Development screen `results/pcla_wl_layer_state_screening_v1/` (seed 101)
is complete. Layer EMA vs rank-state changed last-five accuracy by
+0.14 pp with mean layer Jaccard 0.91. Unused-seed confirmation of this
contrast is not justified. See that directory's `DEVELOPMENT_DECISION.md`.
