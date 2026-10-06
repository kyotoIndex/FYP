#!/usr/bin/env python3
"""Independent 36-job conditional V extension queue; no automatic release."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dispatch_pcla_v7 import lock, read, sha, verify_source, write

PROTOCOL = 'pcla-global-budget-20260928-v7'
QUEUE_NAME = 'conditional_validation_queue'
SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
SLOTS = {3: ('R_local', 'D035', 'local'),
         4: ('R_cyclic', 'D042', 'cyclic'),
         16: ('R_replay', 'D035', 'quota_replay_v1')}
JOB_RE = re.compile(r'V_S(03|04|16)_(D035|D042)_s(\d+)')
HEX = re.compile(r'[0-9a-f]{64}')
MODES = ('exclusive', 'shared')
REPLAY_FILES = {'donor_config', 'source_manifest', 'data_manifest', 'metrics',
                'initial_identity', 'run_identity', 'status', 'donor_quota_seal'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def family(seed: int) -> str:
    return 'A100' if seed in (281, 307) else '4090D'


def queue_path(run_root: Path) -> Path:
    root = run_root.resolve()
    path = root / QUEUE_NAME
    if path.is_symlink() or (root / 'V').is_symlink():
        raise ValueError('conditional V queue/output cannot be symlinks')
    return path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _identity(job: dict, cfg: dict | None = None) -> tuple[int, int]:
    jid = job.get('job_id', '')
    match = JOB_RE.fullmatch(jid)
    if not match:
        raise ValueError(f'not an exact conditional V ID: {jid!r}')
    slot, recipe, seed = int(match[1]), match[2], int(match[3])
    label, chosen, policy = SLOTS[slot]
    if (type(seed) is not int or seed not in SEEDS or recipe != chosen or
            job.get('recipe_id') != recipe or job.get('seed') != seed or
            job.get('phase') != 'V' or job.get('protocol') != PROTOCOL or
            job.get('budget_id') != 'B8' or job.get('min_rounds') != 100 or
            job.get('max_rounds') != 200 or job.get('hardware_block') != f'V_s{seed}' or
            job.get('output_dir') != f'runs/{PROTOCOL}/V/{jid}' or
            (job.get('depends_on') or '') !=
                (f'V_S03_D035_s{seed}' if slot == 16 else '') or
            job.get('method') != ('cyclic' if slot == 4 else 'local') or
            job.get('representation') != 'T'):
        raise ValueError(f'conditional V job identity differs: {jid}')
    if cfg is not None:
        meta = cfg.get('conditional_validation', {})
        if (cfg.get('job') != job or meta.get('slot') != slot or
                meta.get('slot_name') != label or
                meta.get('execution_policy') != policy or
                cfg.get('recipe', {}).get('recipe_id') != recipe or
                cfg.get('expanded_sha256') != canonical_hash(
                    {k: v for k, v in cfg.items() if k != 'expanded_sha256'})):
            raise ValueError(f'conditional V config identity differs: {jid}')
        if slot == 16 and meta.get('donor', {}).get('job_id') != f'V_S03_D035_s{seed}':
            raise ValueError(f'replay donor identity differs: {jid}')
    return slot, seed


def compile_matrix(compilation: Path, manifest: Path, plan: Path,
                   draft_dir: Path) -> tuple[dict, dict[str, dict]]:
    record = read(compilation)
    if (record.get('schema') != 'pcla-v7-conditional-validation-draft-1' or
            record.get('job_count') != 36 or record.get('seeds') != list(SEEDS) or
            record.get('resolution', {}).get('R_local_recipe_id') != 'D035' or
            record.get('resolution', {}).get('R_cyclic_recipe_id') != 'D042' or
            sha(plan) not in record.get('input_sha256', {}).values()):
        raise ValueError('accepted conditional compilation/Pro cost resolution differs')
    jobs, plan_rows = _jsonl(manifest), _jsonl(plan)
    if len(jobs) != 36 or len(plan_rows) != 192 or len({j.get('job_id') for j in plan_rows}) != 192:
        raise ValueError('conditional matrix or full Pro plan cardinality differs')
    plans = {(x.get('slot'), x.get('seed')): x for x in plan_rows}
    if len(plans) != 192:
        raise ValueError('Pro plan duplicates a slot/seed')
    by_id, pairs = {}, set()
    for job in jobs:
        slot, seed = _identity(job)
        jid = job['job_id']
        if jid in by_id or (slot, seed) in pairs or job.get('status') != 'draft_blocked_dependencies':
            raise ValueError(f'conditional draft duplicate or unblocked: {jid}')
        path = draft_dir / f'{jid}.json'
        if path.is_symlink() or sha(path) != record.get('draft_config_sha256', {}).get(path.name):
            raise ValueError(f'conditional draft SHA differs: {jid}')
        cfg = read(path)
        _identity(job, cfg)
        label, recipe, policy = SLOTS[slot]
        planned = plans.get((label, seed), {})
        if (planned.get('phase') != 'V' or planned.get('protocol') != PROTOCOL or
                planned.get('d0_recipe_id') != recipe or
                planned.get('execution_policy') != policy or
                planned.get('hardware_block') != f'V_s{seed}' or
                planned.get('hardware_family') != ('A100-PCIE-40GB' if seed in (281, 307) else 'RTX4090D') or
                planned.get('plan_only') is not True or
                (planned.get('depends_on') or []) !=
                    ([f'V_R_local_s{seed}'] if slot == 16 else [])):
            raise ValueError(f'Pro plan and conditional draft differ: {jid}')
        by_id[jid] = {'job': job, 'draft_config_sha256': sha(path),
                      'draft_config': str(path.resolve()), 'pool': family(seed),
                      'slot': slot}
        pairs.add((slot, seed))
    if pairs != {(slot, seed) for slot in SLOTS for seed in SEEDS}:
        raise ValueError('conditional V 3x12 matrix has a gap')
    identity = {'schema': 'pcla-v7-conditional-queue-1', 'protocol': PROTOCOL,
                'count': 36, 'slots': sorted(SLOTS), 'seeds': list(SEEDS),
                'compilation_sha256': sha(compilation), 'manifest_sha256': sha(manifest),
                'plan_sha256': sha(plan),
                'draft_config_sha256': {jid: by_id[jid]['draft_config_sha256'] for jid in sorted(by_id)},
                'pool_by_seed': {str(seed): family(seed) for seed in SEEDS}}
    return identity, by_id


def initialize(run_root: Path, compilation: Path, manifest: Path,
               plan: Path, draft_dir: Path) -> dict:
    queue = queue_path(run_root)
    identity, by_id = compile_matrix(compilation, manifest, plan, draft_dir)
    identity_path = queue / 'matrix_identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('existing conditional matrix identity differs')
    with lock(queue):
        jobs_dir = queue / 'jobs'
        if identity_path.exists():
            if read(identity_path) != identity or {p.name for p in jobs_dir.iterdir()} != {
                    f'{jid}.json' for jid in by_id}:
                raise ValueError('existing conditional queue job set differs')
            for jid, expected in by_id.items():
                old = read(jobs_dir / f'{jid}.json')
                if any(old.get(k) != expected[k] for k in expected):
                    raise ValueError(f'existing conditional job identity differs: {jid}')
            return identity
        if jobs_dir.exists() and any(jobs_dir.iterdir()):
            raise ValueError('foreign partial conditional queue')
        if any(p.name not in {'queue.lock', 'jobs'} for p in queue.iterdir()):
            raise ValueError('foreign conditional queue files')
        jobs_dir.mkdir(exist_ok=True)
        for jid, base in by_id.items():
            write(jobs_dir / f'{jid}.json', {**base, 'state': 'blocked',
                 'reason': 'conditional_release_gates_unmet', 'attempts': [],
                 'created_at': time.time()})
        write(identity_path, identity)
    return identity


def checked_queue(queue: Path) -> tuple[Path, dict]:
    if queue.is_symlink():
        raise ValueError('conditional queue cannot be a symlink')
    queue = queue.resolve()
    if queue.name != QUEUE_NAME:
        raise ValueError('not the independent conditional V queue')
    identity = read(queue / 'matrix_identity.json')
    if (identity.get('schema') != 'pcla-v7-conditional-queue-1' or
            identity.get('protocol') != PROTOCOL or identity.get('count') != 36 or
            identity.get('slots') != sorted(SLOTS) or identity.get('seeds') != list(SEEDS)):
        raise ValueError('conditional queue identity differs')
    if {p.name for p in (queue / 'jobs').glob('*.json')} != {
            f'{jid}.json' for jid in identity.get('draft_config_sha256', {})} or \
            len(identity.get('draft_config_sha256', {})) != 36:
        raise ValueError('conditional queue job set differs')
    return queue, identity


def checked_job(queue: Path, matrix: dict, jid: str) -> tuple[Path, dict]:
    match = JOB_RE.fullmatch(jid)
    if not match:
        raise ValueError('invalid conditional V job ID')
    path = queue / 'jobs' / f'{jid}.json'
    item = read(path)
    slot, seed = _identity(item['job'])
    if (slot != item.get('slot') or item.get('pool') != matrix['pool_by_seed'][str(seed)] or
            item.get('draft_config_sha256') != matrix['draft_config_sha256'].get(jid) or
            sha(item['draft_config']) != item['draft_config_sha256']):
        raise ValueError(f'conditional queue job/draft identity differs: {jid}')
    if path.is_symlink() or (queue.parent / 'V' / jid).is_symlink():
        raise ValueError('conditional queue job/output symlink')
    return path, item


def _release_config(draft: dict, actual: dict, jid: str, source_manifest: Path) -> None:
    slot, _ = _identity(actual['job'], actual)
    if actual['job'].get('status') == 'draft_blocked_dependencies':
        raise ValueError(f'{jid}: released config remains blocked')
    if actual.get('source_manifest') != str(source_manifest.resolve()):
        raise ValueError(f'{jid}: source manifest path differs')
    expected = json.loads(json.dumps(draft))
    expected['job']['status'] = actual['job']['status']
    expected['source_manifest'] = actual['source_manifest']
    expected['conditional_validation']['release_state'] = actual['conditional_validation']['release_state']
    expected['conditional_validation']['unmet_dependencies'] = actual['conditional_validation']['unmet_dependencies']
    if slot == 16:
        expected['quota_replay'] = actual.get('quota_replay')
    expected['expanded_sha256'] = actual['expanded_sha256']
    if expected != actual or actual['job']['status'] not in {'released', 'ready'} or \
            actual['conditional_validation']['release_state'] != 'released' or \
            actual['conditional_validation']['unmet_dependencies']:
        raise ValueError(f'{jid}: released config changed scientific fields or has dependencies')


def _donor_ready(queue: Path, item: dict, cfg: dict) -> tuple[str, str]:
    jid = item['job']['job_id']
    _, seed = _identity(item['job'])
    donor_id = f'V_S03_D035_s{seed}'
    _, donor = checked_job(queue, read(queue / 'matrix_identity.json'), donor_id)
    if donor.get('state') != 'completed':
        raise ValueError(f'{jid}: same-seed R-local donor is not audited completed')
    audit = donor.get('audit', {})
    report_path = Path(audit.get('report', ''))
    if (not report_path.is_file() or sha(report_path) != audit.get('report_sha256') or
            read(report_path).get('ok') is not True or
            read(report_path).get('errors') != [] or
            audit.get('config_sha256') != donor.get('config_sha256') or
            audit.get('source_manifest_sha256') != donor.get('source_manifest_sha256') or
            read(report_path).get('checks', {}).get('planned_job_id') != donor_id):
        raise ValueError(f'{jid}: donor audit receipt differs')
    binding_plan = cfg.get('quota_replay', {})
    binding_path = Path(binding_plan.get('binding_path', ''))
    if (binding_plan.get('schema') != 'pcla-v7-quota-replay-config-1' or
            not binding_path.is_absolute() or not binding_path.is_file() or
            binding_plan.get('binding_sha256') != sha(binding_path)):
        raise ValueError(f'{jid}: replay binding path/SHA differs')
    binding = read(binding_path)
    if binding.get('schema') != 'pcla-v7-quota-replay-binding-1' or set(binding.get('files', {})) != REPLAY_FILES:
        raise ValueError(f'{jid}: donor binding originals incomplete')
    files = {}
    for key, entry in binding['files'].items():
        path = Path(entry.get('path', ''))
        if (not path.is_absolute() or not path.is_file() or
                not isinstance(entry.get('sha256'), str) or
                not HEX.fullmatch(entry['sha256']) or sha(path) != entry['sha256']):
            raise ValueError(f'{jid}: donor {key} original SHA differs')
        files[key] = path
    out = queue.parent / 'V' / donor_id
    if (files['metrics'].parent != out or any(files[k].parent != out for k in
            ('initial_identity', 'run_identity', 'status', 'donor_quota_seal')) or
            files['donor_config'] != Path(donor.get('config', '')) or
            files['source_manifest'] != Path(donor.get('source', '')) / 'SOURCE_MANIFEST.json' or
            donor.get('config_sha256') != sha(files['donor_config'])):
        raise ValueError(f'{jid}: donor originals do not belong to audited queue run')
    status, metrics, seal = (read(files[k]) for k in ('status', 'metrics', 'donor_quota_seal'))
    if (status.get('state') != 'completed' or not isinstance(metrics, list) or
            not item['job']['min_rounds'] <= len(metrics) <= item['job']['max_rounds'] or
            status.get('completed_round') != metrics[-1].get('round') or
            [row.get('round') for row in metrics] !=
                list(range(1, len(metrics) + 1)) or
            seal.get('completed_round') != metrics[-1].get('round') or
            seal.get('job_id') != donor_id or
            seal.get('identities', {}).get('raw_metrics_file_sha256') != sha(files['metrics']) or
            seal.get('identities', {}).get('donor_config_file_sha256') != sha(files['donor_config'])):
        raise ValueError(f'{jid}: donor seal does not cover completed trace')
    return str(binding_path), sha(binding_path)


def release(queue: Path, ids: list[str], config_dir: Path, source: Path,
            receipt_path: Path, receipt_sha256: str, technical_path: Path,
            technical_sha256: str, hardware_family: str,
            mode: str = 'exclusive') -> list[str]:
    queue, matrix = checked_queue(queue)
    if not ids or len(ids) != len(set(ids)) or hardware_family not in {'4090D', 'A100'} or mode not in MODES:
        raise ValueError('release needs unique IDs, hardware family and mode')
    if mode != 'exclusive':
        raise ValueError('conditional shared mode needs a separate reviewed profile gate')
    source = source.resolve(strict=True)
    source_sha = verify_source(source)
    closure = read(source / 'SOURCE_MANIFEST.json').get('files', {})
    if not {'pcla_research_v7/runtime.py', 'pcla_research_v7/conditional_replay.py',
            'scripts/run_pcla_v7.py', 'scripts/conditional_v_queue.py',
            'scripts/audit_pcla_v7_conditional_run.py',
            'scripts/audit_pcla_v7_conditional_queue.py',
            'scripts/audit_pcla_v7_run.py', 'scripts/dispatch_pcla_v7.py'} <= set(closure):
        raise ValueError('conditional V frozen source closure incomplete')
    if any((source / name).is_symlink() for name in closure):
        raise ValueError('conditional V source closure contains symlink')
    if sha(receipt_path) != receipt_sha256 or sha(technical_path) != technical_sha256:
        raise ValueError('conditional release/technical receipt SHA differs')
    receipt, technical = read(receipt_path), read(technical_path)
    if (receipt.get('schema') != 'pcla-v7-conditional-release-gates-1' or
            receipt.get('protocol') != PROTOCOL or receipt.get('phase') != 'V' or
            receipt.get('job_count') != 36 or
            receipt.get('matrix_sha256') != sha(queue / 'matrix_identity.json') or
            receipt.get('review_gate') != 'approved' or
            receipt.get('configuration_gate') != 'passed' or
            receipt.get('source_gate') != 'passed' or
            receipt.get('technical_receipt_sha256') != technical_sha256 or
            receipt.get('source_manifest_sha256') != source_sha or
            technical.get('schema') != 'pcla-v7-conditional-technical-gate-1' or
            technical.get('protocol') != PROTOCOL or technical.get('phase') != 'V' or
            technical.get('status') != 'passed' or
            technical.get('source_manifest_sha256') != source_sha):
        raise ValueError('conditional review/config/source/technical gate unmet')
    changes = []
    with lock(queue):
        for jid in ids:
            path, item = checked_job(queue, matrix, jid)
            if item['state'] != 'blocked' or item['pool'] != hardware_family:
                raise ValueError(f'{jid}: blocked state or hardware family differs')
            cfg_path = config_dir / f'{jid}.json'
            cfg = read(cfg_path)
            draft = read(item['draft_config'])
            _release_config(draft, cfg, jid, source / 'SOURCE_MANIFEST.json')
            cfg_sha = sha(cfg_path)
            if (receipt.get('config_sha256_by_job', {}).get(jid) != cfg_sha or
                    technical.get('config_sha256_by_job', {}).get(jid) != cfg_sha):
                raise ValueError(f'{jid}: released config receipt SHA differs')
            if (queue.parent / 'V' / jid).exists() and any((queue.parent / 'V' / jid).iterdir()):
                raise ValueError(f'{jid}: output is not empty before release')
            binding, binding_sha = (None, None)
            if item['slot'] == 16:
                binding, binding_sha = _donor_ready(queue, item, cfg)
            changes.append((path, {**item, 'state': 'pending', 'reason': None,
                'config': str(cfg_path.resolve()), 'config_sha256': cfg_sha,
                'source': str(source), 'source_manifest_sha256': source_sha,
                'release_receipt': str(receipt_path.resolve()), 'release_receipt_sha256': receipt_sha256,
                'technical_receipt': str(technical_path.resolve()), 'technical_receipt_sha256': technical_sha256,
                'binding': binding, 'binding_sha256': binding_sha,
                'hardware_family': hardware_family, 'mode': mode,
                'out': str(queue.parent / 'V' / jid), 'released_at': time.time()}))
        for path, item in changes:
            write(path, item)
    return ids


def claim(queue: Path, hardware_family: str, worker: str,
          mode: str = 'exclusive') -> tuple[Path, dict] | None:
    queue, matrix = checked_queue(queue)
    if hardware_family not in {'4090D', 'A100'} or mode not in MODES:
        raise ValueError('unknown hardware family/mode')
    with lock(queue):
        candidates = []
        for path in sorted((queue / 'jobs').glob('*.json')):
            _, item = checked_job(queue, matrix, path.stem)
            if item['state'] == 'pending' and item.get('hardware_family') == hardware_family and item.get('mode') == mode:
                candidates.append((path, item))
        candidates.sort(key=lambda p: (p[1]['job'].get('priority', 100), p[1]['job']['job_id']))
        for path, item in candidates:
            if sha(item['config']) != item['config_sha256'] or \
                    verify_source(Path(item['source'])) != item['source_manifest_sha256'] or \
                    sha(item['release_receipt']) != item['release_receipt_sha256'] or \
                    sha(item['technical_receipt']) != item['technical_receipt_sha256']:
                raise RuntimeError('released conditional config/source/receipt mutated')
            if item['slot'] == 16:
                binding, binding_sha = _donor_ready(queue, item, read(item['config']))
                if binding != item['binding'] or binding_sha != item['binding_sha256']:
                    raise RuntimeError('released donor binding mutated')
            now = time.time()
            item.update(state='running', worker=worker, started_at=now)
            item['attempts'].append({'worker': worker, 'started_at': now,
                'slurm_job': os.getenv('SLURM_JOB_ID'), 'slurm_step': os.getenv('SLURM_STEP_ID')})
            write(path, item)
            return path, item
    return None


def finish(queue: Path, path: Path, worker: str, exit_code: int) -> str:
    queue, matrix = checked_queue(queue)
    with lock(queue):
        path, item = checked_job(queue, matrix, path.stem)
        if item['state'] != 'running' or item.get('worker') != worker:
            raise RuntimeError('lost conditional V job ownership')
        status_path = Path(item['out']) / 'status.json'
        status = read(status_path) if status_path.is_file() else {}
        state = ('executed_pending_audit' if exit_code == 0 else
                 'blocked_donor_suffix' if item['slot'] == 16 and status.get('state') == 'needs_donor_suffix' else
                 'failed_needs_review')
        now = time.time()
        item.update(state=state, finished_at=now, exit_code=exit_code)
        item['attempts'][-1].update(finished_at=now, exit_code=exit_code,
                                    status_state=status.get('state'))
        write(path, item)
    return state


def resume(queue: Path, jid: str, review_path: Path, review_sha256: str) -> str:
    queue, matrix = checked_queue(queue)
    if sha(review_path) != review_sha256:
        raise ValueError('resume review SHA differs')
    review = read(review_path)
    if review != {'schema': 'pcla-v7-conditional-resume-review-1',
                  'job_id': jid, 'decision': 'resume_same_identity'}:
        raise ValueError('resume requires explicit same-identity review')
    with lock(queue):
        path, item = checked_job(queue, matrix, jid)
        if item['state'] != 'failed_needs_review' or not (Path(item['out']) / 'latest.pt').is_file():
            raise ValueError('conditional resume requires failed checkpointed run')
        if sha(item['config']) != item['config_sha256'] or \
                verify_source(Path(item['source'])) != item['source_manifest_sha256']:
            raise ValueError('resume source/config changed')
        if item['slot'] == 16:
            binding, binding_sha = _donor_ready(queue, item, read(item['config']))
            if binding != item['binding'] or binding_sha != item['binding_sha256']:
                raise ValueError('resume donor binding changed')
        item.update(state='pending', reason=None, resume_requested=True,
                    resume_review=str(review_path.resolve()), resume_review_sha256=review_sha256)
        write(path, item)
    return jid


def worker(queue: Path, hardware_family: str, python: str = sys.executable,
           idle_seconds: int = 900, mode: str = 'exclusive') -> None:
    if not os.getenv('SLURM_JOB_ID') or not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('conditional V worker needs existing Slurm GPU step')
    if mode != 'exclusive':
        raise ValueError('conditional shared mode has no reviewed profile gate')
    wid = f'{socket.gethostname()}:{os.getpid()}:{os.getenv("SLURM_JOB_ID")}.{os.getenv("SLURM_STEP_ID")}'
    idle_since = time.monotonic()
    while True:
        claimed = claim(queue, hardware_family, wid, mode)
        if claimed is None:
            if time.monotonic() - idle_since >= idle_seconds:
                return
            time.sleep(3)
            continue
        idle_since = time.monotonic()
        path, item = claimed
        out = Path(item['out'])
        argv = [python, str(Path(item['source']) / 'scripts/run_pcla_v7.py'),
                '--config', item['config'], '--out', str(out), '--device', 'cuda']
        if item.get('binding'):
            argv.extend(['--quota-replay-binding', item['binding']])
        if item.get('resume_requested'):
            argv.append('--resume')
        try:
            if verify_source(Path(item['source'])) != item['source_manifest_sha256']:
                raise RuntimeError('conditional V source changed before executable launch')
            if out.exists() and any(out.iterdir()) and not item.get('resume_requested'):
                raise RuntimeError('conditional V first output is not empty')
            if item.get('resume_requested') and not (out / 'latest.pt').is_file():
                raise RuntimeError('conditional V resume lacks checkpoint')
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PCLA_GPU_SHARING=mode)
            dispatch_path = (out / f'dispatch_identity_attempt_{len(item["attempts"]):02d}.json'
                             if item.get('resume_requested') else out / 'dispatch_identity.json')
            write(dispatch_path, {'worker': wid, 'argv': argv,
                  'config_sha256': item['config_sha256'],
                  'source_manifest_sha256': item['source_manifest_sha256'],
                  'release_receipt_sha256': item['release_receipt_sha256'],
                  'technical_receipt_sha256': item['technical_receipt_sha256'],
                  'binding_sha256': item.get('binding_sha256'),
                  'hardware_family': hardware_family, 'gpu_execution_mode': mode,
                  'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES')})
            with (out / 'worker.log').open('ab') as log:
                code = subprocess.run(argv, cwd=item['source'], env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
        except Exception as exc:
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'worker.log').open('ab') as log:
                log.write(f'\nconditional V worker launch error: {exc!r}\n'.encode())
            code = 127
        print(json.dumps({'job': item['job']['job_id'], 'state': finish(queue, path, wid, code),
                          'exit': code}), flush=True)


def status(queue: Path) -> dict:
    queue, matrix = checked_queue(queue)
    with lock(queue):
        jobs = [checked_job(queue, matrix, p.stem)[1]
                for p in sorted((queue / 'jobs').glob('*.json'))]
    return {'count': len(jobs), 'counts': dict(Counter(j['state'] for j in jobs)),
            'by_hardware_family': {name: dict(Counter(j['state'] for j in jobs if j['pool'] == name))
                                   for name in ('4090D', 'A100')},
            'running': [{'job_id': j['job']['job_id'], 'worker': j.get('worker')}
                        for j in jobs if j['state'] == 'running']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    matrix = sub.add_parser('plan')
    init = sub.add_parser('init')
    for p in (matrix, init):
        p.add_argument('--compilation', type=Path, required=True)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--draft-dir', type=Path, required=True)
    rel = sub.add_parser('release')
    rel.add_argument('--ids', nargs='+', required=True)
    rel.add_argument('--config-dir', type=Path, required=True)
    rel.add_argument('--source', type=Path, required=True)
    rel.add_argument('--release-receipt', type=Path, required=True)
    rel.add_argument('--release-receipt-sha256', required=True)
    rel.add_argument('--technical-receipt', type=Path, required=True)
    rel.add_argument('--technical-receipt-sha256', required=True)
    rel.add_argument('--hardware-family', choices=['4090D', 'A100'], required=True)
    rel.add_argument('--mode', choices=MODES, default='exclusive')
    wrk = sub.add_parser('worker')
    wrk.add_argument('--hardware-family', choices=['4090D', 'A100'], required=True)
    wrk.add_argument('--mode', choices=MODES, default='exclusive')
    wrk.add_argument('--python', default=sys.executable)
    wrk.add_argument('--idle-seconds', type=int, default=900)
    res = sub.add_parser('resume')
    res.add_argument('--job-id', required=True)
    res.add_argument('--review', type=Path, required=True)
    res.add_argument('--review-sha256', required=True)
    sub.add_parser('status')
    args = parser.parse_args()
    queue = queue_path(args.run_root)
    if args.command == 'plan':
        identity, _ = compile_matrix(args.compilation, args.manifest, args.plan, args.draft_dir)
        print(json.dumps(identity, sort_keys=True))
    elif args.command == 'init':
        print(json.dumps(initialize(args.run_root, args.compilation, args.manifest,
                                    args.plan, args.draft_dir), sort_keys=True))
    elif args.command == 'release':
        print(json.dumps({'released': release(queue, args.ids, args.config_dir, args.source,
             args.release_receipt, args.release_receipt_sha256, args.technical_receipt,
             args.technical_receipt_sha256, args.hardware_family, args.mode)}))
    elif args.command == 'worker':
        worker(queue, args.hardware_family, args.python, args.idle_seconds, args.mode)
    elif args.command == 'resume':
        print(json.dumps({'resume': resume(queue, args.job_id, args.review, args.review_sha256)}))
    else:
        print(json.dumps(status(queue), sort_keys=True))


if __name__ == '__main__':
    main()
