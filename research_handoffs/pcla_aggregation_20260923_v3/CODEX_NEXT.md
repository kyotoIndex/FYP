# Codex execution instructions

## 0. Boundary

This branch is a handoff only. Do not edit `pcla_repair/`, old results, the paper,
Issue #1, PR #2, or `main`. Do not push additional changes or post messages.

User-reported current capacity is 12 RTX 4090 and 2 A100. Live allocation,
UUID, CPU quota, memory and process occupancy must be refreshed before launch.

## 1. Prepare an isolated checkout

Checkout this branch in a new directory. The inspected project source remains at
commit `e53078b638ee30a633bad98195c1e2d4476523f5`; handoff files are external
wrappers and analysis tools.

Copy `site.example.json` to a private path outside Git. Resolve:

- unchanged project checkout at the inspected commit;
- complete original `core_shared_hh_20260918_v3` checkpoint tree;
- a new shared queue root;
- node-local scratch prefix;
- the original PCLA Python executable;
- environment variables matching the original run.

Do not infer the old checkpoint tree from the lightweight GitHub results. It
must contain complete `checkpoint.pt` and `probe_round_*.pt` files.

## 2. Tests before queue creation

Run in the original PCLA environment:

```bash
PY=/absolute/original/venv/bin/python
H=/absolute/repo/research_handoffs/pcla_aggregation_20260923_v3

$PY $H/tests.py \
  --repo /absolute/pcla-experiments \
  --out /absolute/evidence/cpu_tests.json
```

The report must be fresh, passed, and bound to the current handoff source hashes.
Syntax compilation alone is not sufficient.

Required tests include:

- F/C/G_P/G_W arithmetic and classifier identity;
- G_W reciprocal-gauge invariance;
- one-factor validity and both-factor rejection;
- full-coverage F=C=G;
- exact four-round support replay;
- fixed-state toy intervention;
- archive traversal/symlink rejection;
- frozen-site identity;
- staged DAG counts and dependency blocking;
- recovery bookkeeping;
- analyzer rejection of altered logits, trace and checkpoint identity.

## 3. Freeze site and initialize queue

```bash
$PY $H/dispatch.py init \
  --site /private/site.json \
  --test-report /absolute/evidence/cpu_tests.json
```

Initialization copies a canonical `frozen_site.json` into the queue and records
its SHA. Workers read only that frozen copy. Modifying the original site file
after initialization has no effect.

Inspect:

```bash
$PY $H/dispatch.py doctor --queue /shared/new_queue
```

The initial ready set must contain exactly four Phase-1 jobs.

## 4. Launch workers inside existing allocations

One process per GPU; no DDP, MPS, threads running multiple experiments, or
unrecorded dual-process packing.

Example shape, adapted to the actual site:

```bash
srun --jobid EXISTING_ID --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=4 --exact --cpu-bind=cores \
  $PY $H/dispatch.py worker \
  --queue /shared/new_queue \
  --gpu-uuid GPU-... \
  --max-jobs 20
```

Slurm owns `CUDA_VISIBLE_DEVICES`. The worker verifies its physical UUID and
records host, allocation, environment and attempt identity.

Raw files remain under node-local scratch. A completed or failed job is packed
into one verified archive and a small shared record.

## 5. Phase 1

Run only:

- `parity_pcla_7`;
- `parity_replay_random_rank_7`;
- `diagnose_7_5`;
- `diagnose_7_15`.

Then:

```bash
$PY $H/gate.py phase1 --queue /shared/new_queue --out phase1_gate.json
```

Phase 1 opens Phase 2 only when exact replay and technical fixed-state checks
pass. It never uses whether PCLA accuracy is favorable.

## 6. Phase 2

Start workers again. Exactly eight remaining fixed-state jobs become ready.
After completion:

```bash
$PY $H/dispatch.py materialize --queue /shared/new_queue --out materialized_phase2
$PY $H/analyze.py fixed-state \
  --queue /shared/new_queue \
  --root materialized_phase2 \
  --out fixed_state_analysis
$PY $H/gate.py phase2 \
  --queue /shared/new_queue \
  --analysis fixed_state_analysis \
  --out phase2_review.json
```

Phase 2 intentionally stops for scientific review. Do not create an approval
record because GPUs are idle or because one method looks good.

If approved, open Phase 3 with an explicit reviewed record:

```bash
$PY $H/gate.py approve \
  --queue /shared/new_queue \
  --open-phase 3 \
  --review phase2_review.json \
  --decision continue \
  --reason "<scientific reason>"
```

## 7. Phase 3 and Phase 4

Phase 3 contains 16 seeds7/13 pilot trajectories. After completion, materialize,
run complete analysis, and produce `phase3_review.json`.

Phase 4 remains closed until a second explicit reviewed approval. If the
intervention is negligible or fully explained by G_W, record `stop` and do not
run the remaining seeds.

## 8. Recovery

The worker never automatically retries.

For an external interruption:

```bash
$PY $H/dispatch.py doctor --queue /shared/new_queue
$PY $H/dispatch.py recover \
  --queue /shared/new_queue \
  --job JOB_ID \
  --reason external_interruption \
  --confirm-process-stopped
```

Same-node recovery may resume the verified checkpoint. If node-local scratch is
unavailable, the second attempt starts from scratch and the first attempt stays
in evidence. Scientific exceptions, NaN, identity mismatch and corrupted files
are not classified as external interruptions.

## 9. Final analysis

After a complete approved matrix:

```bash
$PY $H/dispatch.py materialize --queue /shared/new_queue --out materialized_all
$PY $H/analyze.py complete \
  --queue /shared/new_queue \
  --root materialized_all \
  --out final_analysis
```

Return one evidence package containing:

- planned/completed/failed/blocked inventory;
- fresh CPU tests and all source/environment identities;
- replay parity;
- fixed-state logits, recomputed metrics, coverage and composition analysis;
- all predeclared five-seed contrasts and both seed groups;
- parameter and effective-matrix magnitude controls;
- online versus fixed-support comparison;
- timings, archives, attempts and failures;
- `REPORT_TO_PRO.md` answering the original aggregation questions.

Stop. Do not alter the paper or start another scientific batch.
