#!/usr/bin/env python3
"""Launch hash-checked queue workers inside the six authorized GPU allocations."""
import argparse
import getpass
import json
from pathlib import Path
import shlex
import subprocess
import time

ALLOCATIONS = [('157325', 'A100', 2), ('160347', '4090D', 2),
               ('172164', '4090D', 4), ('172165', '4090D', 4),
               ('172168', '4090D', 2), ('172169', '4090D', 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dispatcher', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--only', nargs='*', default=[])
    p.add_argument('--generation', required=True)
    p.add_argument('--concurrency-gate', type=Path, required=True)
    args = p.parse_args()
    queue, logs = args.root / 'queue', args.root / 'workers'
    identity = json.loads((queue / 'matrix_identity.json').read_text())
    gate = json.loads(args.concurrency_gate.read_text())
    if gate.get('passed') is not True or gate.get('throughput_ratio', 0) < 1.25:
        raise ValueError('Single-GPU concurrency has not passed technical checks')
    if identity['count'] != 351 or not args.dispatcher.is_file():
        raise ValueError('Queue or dispatcher is not ready')
    if not args.generation.isalnum():
        raise ValueError('Generation must be alphanumeric')
    allocations = [a for a in ALLOCATIONS if not args.only or a[0] in args.only]
    if set(args.only) - {a[0] for a in ALLOCATIONS}:
        raise ValueError('Allocation outside authorized inventory')
    # Validate the whole requested set before creating any worker.
    snapshots = {}
    for job, pool, count in allocations:
        live = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
        fields = dict(s.split('=', 1) for s in live.split() if '=' in s)
        if fields.get('JobState') != 'RUNNING' or fields.get('UserId', '').split('(')[0] != getpass.getuser():
            raise RuntimeError(f'{job}: expected a running allocation owned by current user')
        allocated = fields.get('AllocTRES', '').split(',')
        ngpu = next((int(x.split('=')[1]) for x in allocated if x.startswith('gres/gpu=')), None)
        if ngpu != count:
            raise RuntimeError(f'{job}: live GPU count differs from approved inventory')
        snapshots[job] = live
        for i in range(count):
            session = f'pclav7_{args.generation}_{job}_{i}'
            if subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode == 0:
                raise RuntimeError(f'Existing worker session {session}')
    logs.mkdir(parents=True, exist_ok=True)
    receipt = logs / f'launch_{args.generation}.json'
    if receipt.exists():
        raise ValueError('Launch generation already recorded')
    result = {'allocations': snapshots, 'workers': [], 'launched_at': time.time()}
    for job, pool, count in allocations:
        for i in range(count):
            session = f'pclav7_{args.generation}_{job}_{i}'
            log = logs / (session + '.log')
            shared = job in {'172165', '172168', '172169'}
            argv = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1',
                    '--gpus-per-task=1', f'--cpus-per-task={8 if shared else 4}', '--exact', '--cpu-bind=cores',
                    args.python, str(args.dispatcher), '--queue', str(queue)]
            argv += ['pair', '--python', args.python] if shared else [
                'worker', '--pool', pool, '--python', args.python, '--mode', 'exclusive']
            command = shlex.join(argv) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
            result['workers'].append({'session': session, 'allocation': job, 'pool': pool,
                                      'log': str(log), 'argv': argv, 'processes': 2 if shared else 1,
                                      'gpu_execution_mode': 'shared' if shared else 'exclusive', 'time': time.time()})
            receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'workers': len(result['workers']), 'receipt': str(receipt)}))


if __name__ == '__main__':
    main()
