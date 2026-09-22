# PCLA aggregation handoff v3

Batch identity: `aggregation-causal-20260923-v3`

This directory is a corrected replacement for the unexecuted v2 handoff. It is
kept on an isolated branch and does not change the project trainer, old results,
paper, Issue #1, or the merged evidence in `main`.

## What changed from v2

1. The primary magnitude control now matches the **effective LoRA matrix update**
   rather than only the A/B parameter-vector norm. The old parameter-norm control
   remains an auxiliary fixed-state diagnostic.
2. The wrapper no longer changes deterministic-algorithm or TF32 flags. It
   records the live numerical environment and requires exact fixed-run parity
   before old fixed-denominator controls can be reused.
3. Execution is staged: two parity runs and seed-7 fixed-state smoke first, all
   fixed-state diagnostics second, a two-seed trajectory pilot third, and the
   remaining seeds only after an explicit gate.
4. The queue freezes `site.json`, records attempts, supports inspected recovery,
   marks dependency-blocked jobs, and never silently retries.
5. Analysis independently recomputes new-run metrics from logits, audits traces,
   checkpoints, aggregation records, coverage and contributor composition, and
   distinguishes old fixed-run GPU NLL from newly recomputed float64 NLL.
6. The two independent discovery ideas remain bounded numerical utilities. They
   are not represented as completed Qwen experiments.

## Read order

1. `REVIEW_CHANGES_V2_TO_V3.md`
2. `SCIENTIFIC_PROTOCOL.md`
3. `CODEX_NEXT.md`
4. `RESOURCE_PLAN.md`
5. `site.example.json`

## Code

- `common.py`: source identity, atomic records, safe archives, frozen site.
- `aggregation.py`: F/C controls and parameter/function-norm matching.
- `pcla_runner.py`: original-runtime integration and exact-support trajectories.
- `fixed_state.py`: same-local-update interventions and coverage diagnostics.
- `dispatch.py`: phased queue, claims, attempts, recovery and materialization.
- `gate.py`: predeclared phase reports; no result-driven hyperparameter changes.
- `analyze.py`: complete paired analysis and independent metric audit.
- `tests.py`: numerical, synthetic replay, queue and analysis rejection tests.
- `discovery_math.py`: compact adapter-compression and product-delta primitives.
- `build_package.py`: syntax-check and package this directory after checkout.

## Execution status

No new CPU, BERT/CUDA, aggregation, or discovery experiment has been executed by
this branch creation. Codex must create fresh test and execution evidence in the
actual PCLA environment. The branch is an implementation handoff, not a result.
