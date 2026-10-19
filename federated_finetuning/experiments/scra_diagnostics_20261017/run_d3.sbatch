#!/bin/bash
#SBATCH --job-name=scra-d3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --exclude=hpcgpu01,hpcgpu02,hpcgpu03,hpcgpu08
#SBATCH --array=0-2%2
#SBATCH --output=logs/d3-%A_%a.out
#SBATCH --error=logs/d3-%A_%a.err
# D3: realizable-fix screening on the L2 D0 snapshots (rounds 20/60/120 -> task 0/1/2), ~50-70 min each.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the experiment directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PY="${SCRA_PYTHON:-/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python}"
args=(d3 --plan "${SCRA_PLAN:-plan_d3.json}" --device cuda --task-index "$SLURM_ARRAY_TASK_ID")
if [[ -n "${SCRA_PROJECT_ROOT:-}" ]]; then args+=(--project-root "$SCRA_PROJECT_ROOT"); fi
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
exec "$PY" run.py "${args[@]}"
