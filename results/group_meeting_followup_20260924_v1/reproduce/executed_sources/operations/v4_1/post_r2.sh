#!/bin/bash
# Materialize queue_v41_r2, run the V4.1 analyzer (independent re-audit of every job), then the
# pre-registered extra readouts. Runs inside an existing allocation step; writes POST_R2_DONE.json.
set -u
R=$HPC_HOME/pcla_aggregation_20260923_v41
PY=$HPC_HOME/pcla_repair_20260918/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $R/handoff_r2
t0=$(date +%s)
$PY dispatch.py materialize --queue $R/queue_v41_r2 --out $R/materialized_r2 > $R/evidence_r2/materialize_r2.log 2>&1; m=$?
t1=$(date +%s)
a=99; e=99
if [ $m -eq 0 ]; then
  $PY analyze.py --root $R/materialized_r2 --repo $HPC_HOME/pcla_repair_20260918/repo \
     --source-root $HPC_HOME/pcla_repair_20260918/repo/results/core_shared_hh_20260918_v3 \
     --out $R/analysis_r2 > $R/evidence_r2/analyze_r2.log 2>&1; a=$?
fi
t2=$(date +%s)
I=$R/analysis_extra_r2_inputs
$PY $I/extra_readouts.py --materialized $R/materialized_r2 --out $R/analysis_extra_r2 \
   --v3-rounds $I/per_round.csv --cohort $I/cohort_label_share.csv > $R/evidence_r2/extra_r2.log 2>&1; e=$?
t3=$(date +%s)
printf '{"materialize_exit":%d,"analyze_exit":%d,"extra_exit":%d,"materialize_s":%d,"analyze_s":%d,"extra_s":%d,"host":"%s","step":"%s.%s"}\n' \
  $m $a $e $((t1-t0)) $((t2-t1)) $((t3-t2)) "$(hostname)" "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" > $R/evidence_r2/POST_R2_DONE.json
