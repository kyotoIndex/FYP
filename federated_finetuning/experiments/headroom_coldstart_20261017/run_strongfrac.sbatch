#!/bin/bash
#SBATCH --job-name=headroom-sf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03
#SBATCH --array=0-14%3
#SBATCH --output=logs/task-%A_%a.out
#SBATCH --error=logs/task-%A_%a.err
# 15 tasks = 3 seeds x (fedavg_r8, fedavg_r2, hetero at 10%/20%/40% strong); at most 3 GPUs at a time.
# Submit from this directory:  mkdir -p logs && sbatch run_strongfrac.sbatch
# Resubmitting a task continues from its checkpoint (identity-checked); set HEADROOM_RESUME=0 to forbid.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
PY="${HEADROOM_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
args=(--plan "${HEADROOM_PLAN:-plan_strongfrac.json}" --task-index "$SLURM_ARRAY_TASK_ID" --device cuda)
if [[ -n "${HEADROOM_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$HEADROOM_PROJECT_ROOT"); fi
if [[ "${HEADROOM_RESUME:-1}" == "1" ]]; then args+=(--resume); fi
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
exec "$PY" train.py "${args[@]}"
