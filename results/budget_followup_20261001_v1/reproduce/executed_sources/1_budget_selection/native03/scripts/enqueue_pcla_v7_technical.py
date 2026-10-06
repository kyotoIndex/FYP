#!/usr/bin/env python3
"""Enqueue explicitly bounded TECH checks ahead of the next scientific job."""
import argparse
import copy
import json
from pathlib import Path
import time

from dispatch_pcla_v7 import lock, read, write, sha, verify_source

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--configs', type=Path, nargs='+', required=True)
p.add_argument('--generation', required=True)
p.add_argument('--rounds', type=int, choices=[2, 4], default=2)
a = p.parse_args()
source_sha = verify_source(a.source)
receipt = []
with lock(a.root / 'queue'):
    for config in a.configs:
        expanded = copy.deepcopy(read(config))
        job = expanded['job']
        if job.get('phase') != 'TECH' or job['seed'] not in {9001, 9002}:
            raise ValueError('Only declared technical seeds may enter this queue')
        if not expanded.get('technical_profile_only'):
            raise ValueError('Missing technical provenance')
        jid = job['job_id'] + '_' + a.generation
        job.update(job_id=jid, priority=0, hardware_block='TECH_4090D')
        expanded['technical_stop_after'] = a.rounds
        expanded.pop('expanded_sha256', None)
        cfg = a.root / 'configs/queued_technical' / (jid + '.json')
        record = a.root / 'queue/jobs' / (jid + '.json')
        if cfg.exists() or record.exists():
            raise ValueError('Technical identity already exists; inspect it before another attempt')
        write(cfg, expanded)
        out = a.root / 'checks/queued_technical' / jid
        write(record, {'job': job, 'pool': '4090D', 'state': 'pending', 'reason': None,
                       'attempts': [], 'created_at': time.time(), 'released_at': time.time(),
                       'config': str(cfg), 'config_sha256': sha(cfg), 'source': str(a.source),
                       'source_manifest_sha256': source_sha, 'out': str(out)})
        receipt.append({'job': jid, 'round_limit': a.rounds, 'source_manifest_sha256': source_sha})
write(a.root / 'queue' / ('technical_release_' + a.generation + '.json'), receipt)
print(json.dumps({'technical_jobs': len(receipt), 'priority': 0, 'formal_matrix_count': 351}))
