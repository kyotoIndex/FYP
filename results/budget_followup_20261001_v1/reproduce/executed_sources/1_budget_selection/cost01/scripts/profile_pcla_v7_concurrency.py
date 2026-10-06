#!/usr/bin/env python3
"""Measure one or two independent technical runs on one allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--processes', type=int, choices=[1, 2], required=True)
    p.add_argument('--rounds', type=int, default=4)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Requires one allocated GPU')
    a.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    procs, logs, samples = [], [], []
    started = time.time()
    for i in range(a.processes):
        log = (a.out / f'process_{i}.log').open('wb'); logs.append(log)
        argv = [sys.executable, str(a.source / 'scripts/run_pcla_v7.py'), '--config', str(a.config),
                '--out', str(a.out / f'process_{i}'), '--device', 'cuda', '--stop-after', str(a.rounds)]
        procs.append(subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT))
    pids = {p.pid for p in procs}
    while any(p.poll() is None for p in procs):
        raw = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                              '--format=csv,noheader,nounits'], text=True, capture_output=True)
        records = []
        for line in raw.stdout.splitlines():
            fields = [v.strip() for v in line.split(',')]
            if len(fields) == 3 and fields[0].isdigit() and int(fields[0]) in pids:
                records.append({'pid': int(fields[0]), 'gpu': fields[1], 'memory_mib': int(fields[2])})
        samples.append({'time': time.time(), 'processes': records})
        time.sleep(1)
    elapsed = time.time() - started
    for f in logs: f.close()
    statuses = [json.loads((a.out / f'process_{i}/status.json').read_text())
                for i in range(a.processes)]
    result = {'processes': a.processes, 'rounds_each': a.rounds, 'wall_seconds': elapsed,
              'rounds_per_second': a.processes * a.rounds / elapsed,
              'exit_codes': [p.returncode for p in procs], 'statuses': statuses,
              'peak_total_process_memory_mib': max((sum(x['memory_mib'] for x in s['processes'])
                                                   for s in samples), default=0),
              'slurm_job': os.environ['SLURM_JOB_ID'], 'slurm_step': os.environ.get('SLURM_STEP_ID'),
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'samples': samples}
    (a.out / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))
    if any(p.returncode != 0 for p in procs) or any(s['completed_round'] != a.rounds for s in statuses):
        raise RuntimeError('Technical profile did not finish all requested rounds')


if __name__ == '__main__':
    main()
