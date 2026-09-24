#!/bin/bash
set -euo pipefail
root=$HPC_HOME/pcla_v5_20260924
python_pcla=$HPC_HOME/pcla_repair_20260918/venv/bin/python
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 srun --jobid=172165 --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 --exact --cpu-bind=cores "$python_pcla" "$root/analyze_v5e.py" --materialized "$root/materialized_v5e" --out "$root/analysis_v5e"
