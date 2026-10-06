#!/usr/bin/env python3
"""Real-model exact continuous/resumed check inside one allocated GPU step."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for directory, stop, resume in [('continuous', 2, False), ('resumed', 1, False), ('resumed', 2, True)]:
        argv = [sys.executable, str(a.source / 'scripts/run_pcla_v7.py'), '--config', str(a.config),
                '--out', str(a.out / directory), '--device', 'cuda', '--stop-after', str(stop)]
        if resume: argv.append('--resume')
        with (a.out / (directory + '.log')).open('ab') as f:
            subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    import torch
    x, y = [torch.load(a.out / d / 'latest.pt', weights_only=True, map_location='cpu')
            for d in ('continuous', 'resumed')]
    checks = {'round': x['round'] == y['round'] == 2,
              'adapter': all(torch.equal(x['adapter'][n], y['adapter'][n]) for n in x['adapter']),
              'ema': all(torch.equal(x['ema_effective'][n], y['ema_effective'][n]) for n in x['ema_effective']),
              'wire_cache': x['cache'] == y['cache'],
              'deployed_bytes': x['deployed_bytes'] == y['deployed_bytes'],
              'torch_rng': torch.equal(x['torch_rng'], y['torch_rng']),
              'metrics': all(all(l[key] == r[key] for key in
                  ['round', 'selected_clients', 'deployed_bytes', 'diagnostic_after_budget_bytes',
                   'evaluation', 'ema_evaluation']) for l, r in zip(x['metrics'], y['metrics']))}
    result = {'passed': all(checks.values()), 'checks': checks,
              'source': str(a.source), 'config': str(a.config),
              'slurm_job': os.environ.get('SLURM_JOB_ID'), 'slurm_step': os.environ.get('SLURM_STEP_ID')}
    (a.out / 'resume_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']: raise RuntimeError('Continuous/resumed comparison differs')


if __name__ == '__main__':
    main()
