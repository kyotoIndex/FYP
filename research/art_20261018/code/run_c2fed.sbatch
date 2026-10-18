#!/bin/bash
#SBATCH --job-name=c2-fed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03,hpcgpu08
#SBATCH --output=logs/c2fed-%A_%a.out
#SBATCH --error=logs/c2fed-%A_%a.err
# C2-1 runs: task i runs the i-th plan (sorted) matching $PLANS. Resubmitting resumes from the checkpoint.
#   python make_c2fed.py lrcheck && PLANS='plans_c2fed/lr_*.json' sbatch --time=00:40:00 --array=0-3%2 run_c2fed.sbatch
#   python make_c2fed.py main --lr <chosen> && PLANS='plans_c2fed/main_*.json' sbatch --array=0-8%2 run_c2fed.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="${SCRA_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
mapfile -t LIST < <(ls ${PLANS:?set PLANS, e.g. plans_c2fed/main_*.json} | sort)
PLAN="${LIST[$SLURM_ARRAY_TASK_ID]:?no plan for task $SLURM_ARRAY_TASK_ID}"
args=(c2fed --plan "$PLAN" --device cuda)
if [[ -n "${SCRA_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$SCRA_PROJECT_ROOT"); fi
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} plan=${PLAN}"
exec "$PY" run.py "${args[@]}"
