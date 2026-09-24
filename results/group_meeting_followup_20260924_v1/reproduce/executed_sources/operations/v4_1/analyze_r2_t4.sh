#!/bin/bash
# Re-run the V4.1 analyzer with 4 CPU threads, the thread count of original_environment() used by
# every in-job audit; the G0 readout check compares floats exactly and fails under 8 threads.
set -u
R=$HPC_HOME/pcla_aggregation_20260923_v41
PY=$HPC_HOME/pcla_repair_20260918/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $R/handoff_r2
t0=$(date +%s)
$PY analyze.py --root $R/materialized_r2 --repo $HPC_HOME/pcla_repair_20260918/repo \
   --source-root $HPC_HOME/pcla_repair_20260918/repo/results/core_shared_hh_20260918_v3 \
   --out $R/analysis_r2 > $R/evidence_r2/analyze_r2_t4.log 2>&1; a=$?
t1=$(date +%s)
I=$R/analysis_extra_r2_inputs
rm -rf $R/analysis_extra_r2_v2
$PY $I/extra_readouts_v2.py --materialized $R/materialized_r2 --out $R/analysis_extra_r2_v2 \
   --v3-rounds $I/per_round.csv --cohort $I/cohort_label_share.csv > $R/evidence_r2/extra_r2_v2.log 2>&1; e=$?
t2=$(date +%s)
printf '{"analyze_exit":%d,"extra_v2_exit":%d,"analyze_s":%d,"extra_s":%d,"threads":4,"host":"%s","step":"%s.%s"}\n' \
  $a $e $((t1-t0)) $((t2-t1)) "$(hostname)" "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" > $R/evidence_r2/ANALYZE_R2_T4_DONE.json
