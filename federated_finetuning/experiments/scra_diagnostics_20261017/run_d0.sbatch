#!/bin/bash
#SBATCH --job-name=scra-d0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03
#SBATCH --output=logs/d0-%j.out
#SBATCH --error=logs/d0-%j.err
# D0: one logged federated run (10% strong, seed 2040, 150 rounds). Resubmitting resumes from checkpoint.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${SCRA_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
args=(d0 --plan "${SCRA_PLAN:-plan.json}" --device cuda --resume)
if [[ -n "${SCRA_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$SCRA_PROJECT_ROOT"); fi
echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
exec "$PY" run.py "${args[@]}"
