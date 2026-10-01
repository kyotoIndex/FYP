#!/usr/bin/env python3
"""Run a frozen cohort support or bank/readout stage on a verified V anchor.

Stage success is not a complete 144-object diagnostic result. The independent
prediction comparison and direction runner have separate integration gates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from pcla_repair.runtime import atomic_checkpoint, atomic_json
from pcla_research_v7.diagnostic_configuration import SEEDS, sha
from pcla_research_v7.diagnostic_model import prepare_context
from pcla_research_v7.diagnostic_supports import construct_diagnostic_supports
from pcla_research_v7.diagnostic_pipeline import run_bank_readout


def verify_source(path: Path) -> str:
    if path.resolve().parent != ROOT:
        raise ValueError('diagnostic manifest must bind the executing source directory')
    files = json.loads(path.read_text())['files']
    for relative, expected in files.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or sha(ROOT / rel) != expected:
            raise ValueError(f'diagnostic source differs: {relative}')
    if str(Path(__file__).relative_to(ROOT)) not in files:
        raise ValueError('diagnostic driver not in source manifest')
    return sha(path)


def validate_config(config):
    if (config.get('schema') != 'pcla-v7-diagnostic-config-1'
            or config.get('line') != 'cohort' or config.get('seed') not in SEEDS
            or config.get('anchor_round') not in (10, 60) or config.get('factor') not in ('a', 'b')
            or config.get('calls') != {'clients': 10, 'local_calls': 40, 'optimizer_steps': 640}):
        raise ValueError('expected a frozen cohort diagnostic object')
    training = config['training']
    required = dict(batch_size=16, local_examples=256, local_optimizer_steps=16,
                    learning_rate=.001, head_learning_rate=.001, optimizer='adamw',
                    betas=[.9, .999], eps=1e-8, weight_decay=.01, prefix_steps=[1, 4, 16],
                    paired_rng_key_scope=['diagnostic_object_id', 'client_id'])
    if any(training.get(k) != v for k, v in required.items()):
        raise ValueError('diagnostic training differs from fixed cohort protocol')


def run_stage(config, *, out: Path, config_sha: str, source_sha: str,
              device: str, stage: str, prepare=prepare_context,
              construct=construct_diagnostic_supports, pipeline=run_bank_readout):
    validate_config(config)
    identity = dict(schema='pcla-v7-diagnostic-attempt-1', object_id=config['job_id'],
                    config_sha256=config_sha, source_sha256=source_sha)
    path = out / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('diagnostic output belongs to a different frozen identity')
    atomic_json(path, identity)
    context = prepare(config, device=device)
    if device.startswith('cuda'):
        name = context['provenance']['hardware']['name']
        expected = config['hardware_family']
        if not ((expected == '4090D' and name == 'NVIDIA GeForce RTX 4090 D') or
                (expected == 'A100' and 'A100' in name)):
            raise ValueError('diagnostic hardware differs from frozen seed block')
    anchor_receipt = out / 'anchor.json'
    if anchor_receipt.exists() and json.loads(anchor_receipt.read_text()) != context['anchor']:
        raise ValueError('diagnostic anchor changed during recovery')
    atomic_json(anchor_receipt, context['anchor'])
    bundle_path, receipt_path = out / 'supports.pt', out / 'supports.json'
    if bundle_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['sha256'] != sha(bundle_path) or receipt['identity'] != identity:
            raise ValueError('saved support bundle hash/identity differs')
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    else:
        started = time.perf_counter()
        bundle = construct(model=context['model'], parent_state=context['parent_state'],
                           object_id=config['job_id'], anchor_identity=context['anchor']['checkpoint_identity'],
                           anchor_sha256=context['anchor']['anchor_sha256'], seed=config['seed'],
                           round_id=config['anchor_round']+1, factor=config['factor'],
                           participants=config['participants'], clients=context['clients'],
                           public=context['public'], visits=context['visits'], capacities=context['capacities'],
                           locked_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_checkpoint(bundle_path, bundle)
        atomic_json(receipt_path, dict(identity=identity, sha256=sha(bundle_path),
                                       seconds=time.perf_counter()-started,
                                       public_rule=bundle['public_linear_rule_id'],
                                       jq_lock=asdict(bundle['jq_lock'])))
    if bundle['anchor_sha256'] != context['anchor']['anchor_sha256']:
        raise ValueError('support bundle belongs to a different anchor')
    result = dict(stage=stage, state='supports_complete', diagnostic_object_complete=False)
    if stage == 'bank-readout':
        result['bank_stage'] = pipeline(config=config, context=context, supports=bundle['supports'],
                                      jq_lock=bundle['jq_lock'], output=out/'bank_stage',
                                      config_sha256=config_sha, source_sha256=source_sha)
        result['state'] = 'bank_readout_complete'
    atomic_json(out/'stage_status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=['supports', 'bank-readout'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    source_sha = verify_source(args.source_manifest)
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'.driver.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time(); stamp = time.time_ns()
        attempt = dict(pid=os.getpid(), started_at=started, stage=args.stage,
                       slurm_job=os.environ.get('SLURM_JOB_ID'), slurm_step=os.environ.get('SLURM_STEP_ID'),
                       source_sha256=source_sha, config_sha256=sha(args.config))
        atomic_json(args.out/f'attempt_{stamp}.json', attempt)
        try:
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
            torch.use_deterministic_algorithms(True)
            result = run_stage(config, out=args.out, config_sha=sha(args.config), source_sha=source_sha,
                               device=args.device, stage=args.stage)
            attempt.update(state=result['state'], ended_at=time.time())
        except BaseException:
            attempt.update(state='failed', ended_at=time.time(), traceback=traceback.format_exc())
            atomic_json(args.out/f'attempt_{stamp}.json', attempt)
            raise
        atomic_json(args.out/f'attempt_{stamp}.json', attempt)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
