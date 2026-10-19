#!/bin/bash
#SBATCH --job-name=c2-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03,hpcgpu08
#SBATCH --output=logs/c2-%A_%a.out
#SBATCH --error=logs/c2-%A_%a.err
# C2-0 decoder-LLM probe. Usage (from the experiment directory), one step at a time:
#   sbatch --time=00:30:00 run_c2.sbatch check             # asset check -> runs/c2_probe_20261018/c2check/check.json
#   sbatch --time=01:30:00 run_c2.sbatch snap              # snapshots (resumable); check core accuracy at round 30
#   sbatch --time=02:00:00 --array=0 run_c2.sbatch probe   # one probe first: note time and memory
#   sbatch --time=02:00:00 --array=1-2 run_c2.sbatch probe # then the other two (resumable per draw)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="${SCRA_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
MODE="${1:?mode: check, snap or probe}"
case "$MODE" in
  check) args=(c2check --plan "${SCRA_PLAN:-plan_c2.json}" --device cuda) ;;
  snap) args=(c2snap --plan "${SCRA_PLAN:-plan_c2.json}" --device cuda) ;;
  probe) args=(c2probe --plan "${SCRA_PLAN:-plan_c2.json}" --device cuda --task-index "${SLURM_ARRAY_TASK_ID:?probe needs --array}") ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
if [[ -n "${SCRA_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$SCRA_PROJECT_ROOT"); fi
echo "host=$(hostname) job=${SLURM_JOB_ID} mode=${MODE} task=${SLURM_ARRAY_TASK_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "$PY" run.py "${args[@]}"
