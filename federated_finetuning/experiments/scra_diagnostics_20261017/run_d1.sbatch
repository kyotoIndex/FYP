#!/bin/bash
#SBATCH --job-name=scra-d1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03
#SBATCH --array=0-2%3
#SBATCH --output=logs/d1-%A_%a.out
#SBATCH --error=logs/d1-%A_%a.err
# D1: fixed-state replay on D0 snapshots (rounds 20/60/120 -> task 0/1/2). Needs D0 snapshots to exist.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${SCRA_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
args=(d1 --plan "${SCRA_PLAN:-plan.json}" --device cuda --task-index "$SLURM_ARRAY_TASK_ID")
if [[ -n "${SCRA_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$SCRA_PROJECT_ROOT"); fi
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
exec "$PY" run.py "${args[@]}"
