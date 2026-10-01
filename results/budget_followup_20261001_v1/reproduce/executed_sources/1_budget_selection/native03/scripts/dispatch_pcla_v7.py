#!/usr/bin/env python3
"""Small, locked job queue for explicitly released v7 configs in existing allocations.

Does not allocate GPUs, select scientific winners, or silently retry failures.
Each released config points at an immutable, hash-checked source snapshot.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def verify_source(source):
    manifest = read(source / 'SOURCE_MANIFEST.json')
    for rel, expected in manifest['files'].items():
        p = source / rel
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError(f'source snapshot mismatch: {rel}')
    return sha(source / 'SOURCE_MANIFEST.json')


def initialize(root, manifest, pool_map):
    jobs = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
    if len(jobs) != 351 or len({j['job_id'] for j in jobs}) != 351:
        raise ValueError('expected frozen 351-job D0 matrix')
    with lock(root):
        for j in jobs:
            path = root / 'jobs' / (j['job_id'] + '.json')
            pool = pool_map[j['hardware_block']]
            item = {'job': j, 'pool': pool, 'state': 'blocked', 'reason': 'integration_not_released',
                    'attempts': [], 'created_at': time.time()}
            if path.exists():
                old = read(path)
                if old['job'] != j or old['pool'] != pool:
                    raise ValueError('existing queue identity differs')
            else:
                write(path, item)
        write(root / 'matrix_identity.json', {'manifest_sha256': sha(manifest),
              'pool_map': pool_map, 'count': len(jobs)})


def release(root, ids, config_dir, source, output_root):
    source_hash = verify_source(source)
    changes = []
    with lock(root):
        for jid in ids:
            p = root / 'jobs' / (jid + '.json')
            v = read(p)
            if v['state'] != 'blocked':
                raise ValueError(f'{jid}: release requires blocked state, found {v["state"]}')
            cfg = config_dir / (jid + '.json')
            if not cfg.is_file():
                raise FileNotFoundError(cfg)
            # Compilation and integration are separate, recorded release gates.
            changes.append((p, dict(v, state='pending', reason=None,
                config=str(cfg), config_sha256=sha(cfg), source=str(source),
                source_manifest_sha256=source_hash, out=str(output_root / jid),
                released_at=time.time())))
        for p, v in changes:
            write(p, v)


def claim(root, pool, worker_id, mode=None):
    with lock(root):
        candidates = [(p, read(p)) for p in sorted((root / 'jobs').glob('*.json'))]
        candidates = [(p, v) for p, v in candidates if v['pool'] == pool and v['state'] == 'pending']
        if mode is not None:
            candidates = [(p, v) for p, v in candidates
                          if ('exclusive' if pool == 'A100' or v['job']['seed'] == 401 else 'shared') == mode]
        candidates.sort(key=lambda pv: (pv[1]['job']['priority'], pv[1]['job']['hardware_block'], pv[1]['job']['job_id']))
        if not candidates:
            return None
        p, v = candidates[0]
        if sha(v['config']) != v['config_sha256']:
            raise RuntimeError('released config mutated')
        if verify_source(Path(v['source'])) != v['source_manifest_sha256']:
            raise RuntimeError('released source manifest mutated')
        v.update(state='running', worker=worker_id, started_at=time.time())
        v['attempts'].append({'worker': worker_id, 'started_at': v['started_at'],
                              'slurm_job': os.getenv('SLURM_JOB_ID'), 'slurm_step': os.getenv('SLURM_STEP_ID')})
        write(p, v)
        return p, v


def finish(root, p, worker, state, **fields):
    with lock(root):
        v = read(p)
        if v['state'] != 'running' or v['worker'] != worker:
            raise RuntimeError('lost job ownership')
        v.update(state=state, finished_at=time.time(), **fields)
        write(p, v)


def worker(root, pool, python, idle_seconds=45, mode='exclusive'):
    if not os.getenv('SLURM_JOB_ID') or not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('worker requires an existing Slurm GPU step')
    wid = f'{socket.gethostname()}:{os.getpid()}:{os.getenv("SLURM_JOB_ID")}.{os.getenv("SLURM_STEP_ID")}'
    start_idle = time.monotonic()
    while True:
        item = claim(root, pool, wid, mode)
        if item is None:
            if time.monotonic() - start_idle >= idle_seconds:
                return
            time.sleep(3)
            continue
        start_idle = time.monotonic()
        path, v = item
        out = Path(v['out']); out.mkdir(parents=True, exist_ok=True)
        argv = [python, str(Path(v['source']) / 'scripts/run_pcla_v7.py'),
                '--config', v['config'], '--out', str(out), '--device', 'cuda']
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   PCLA_GPU_SHARING=mode)
        write(out / 'dispatch_identity.json', {'worker': wid, 'argv': argv,
              'config_sha256': v['config_sha256'], 'source_manifest_sha256': v['source_manifest_sha256'],
              'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'), 'pool': pool,
              'gpu_execution_mode': mode})
        with (out / 'worker.log').open('ab') as log:
            proc = subprocess.run(argv, cwd=v['source'], env=env, stdout=log, stderr=subprocess.STDOUT)
        # A zero exit is execution success only; completion is audited separately.
        state = 'executed_pending_audit' if proc.returncode == 0 else 'failed_needs_review'
        finish(root, path, wid, state, exit_code=proc.returncode)
        print(json.dumps({'job': v['job']['job_id'], 'state': state, 'exit': proc.returncode}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', type=Path, required=True)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init'); p.add_argument('--manifest', required=True); p.add_argument('--pool-map', required=True)
    p = sub.add_parser('release'); p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--config-dir', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p = sub.add_parser('worker'); p.add_argument('--pool', choices=['4090D', 'A100'], required=True)
    p.add_argument('--python', default=sys.executable); p.add_argument('--idle-seconds', type=int, default=45)
    p.add_argument('--mode', choices=['exclusive', 'shared'], default='exclusive')
    p = sub.add_parser('pair'); p.add_argument('--python', default=sys.executable)
    sub.add_parser('status')
    a = ap.parse_args()
    if a.command == 'init': initialize(a.queue, a.manifest, read(a.pool_map))
    elif a.command == 'release': release(a.queue, a.ids, a.config_dir, a.source, a.output_root)
    elif a.command == 'worker': worker(a.queue, a.pool, a.python, a.idle_seconds, a.mode)
    elif a.command == 'pair':
        argv = [a.python, str(Path(__file__).resolve()), '--queue', str(a.queue),
                'worker', '--pool', '4090D', '--python', a.python, '--mode', 'shared']
        pair = [subprocess.Popen(argv) for _ in range(2)]
        codes = [p.wait() for p in pair]
        if any(codes): raise SystemExit(max(codes))
    else:
        jobs = [read(p) for p in (a.queue/'jobs').glob('*.json')]
        print(json.dumps({'counts': dict(Counter(j['state'] for j in jobs if j['job'].get('phase', 'D0') == 'D0')),
              'technical_counts': dict(Counter(j['state'] for j in jobs if j['job'].get('phase') == 'TECH')),
              'running': [{'job_id': j['job']['job_id'], 'worker': j['worker']} for j in jobs if j['state']=='running']}, indent=2))


if __name__ == '__main__':
    main()
