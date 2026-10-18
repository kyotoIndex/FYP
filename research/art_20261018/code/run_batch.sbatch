#!/bin/bash
#SBATCH --job-name=scra-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03,hpcgpu08
#SBATCH --array=0-18%2
#SBATCH --output=logs/batch-%A_%a.out
#SBATCH --error=logs/batch-%A_%a.err
# Validation batch 2026-10-18: task i runs plans_batch/<i>_*.json (one full D0, ~15 min). Resubmitting resumes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${SCRA_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
mapfile -t PLANS < <(ls plans_batch/*.json | sort)
PLAN="${PLANS[$SLURM_ARRAY_TASK_ID]:?no plan for task $SLURM_ARRAY_TASK_ID}"
args=(d0 --plan "$PLAN" --device cuda --resume)
if [[ -n "${SCRA_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$SCRA_PROJECT_ROOT"); fi
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} plan=${PLAN}"
exec "$PY" run.py "${args[@]}"
