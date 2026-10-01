#!/usr/bin/env python3
"""Audit executed conditional V jobs without touching the fixed 156 queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dispatch_pcla_v7 import lock, read, sha, write, verify_source
from scripts.conditional_v_queue import checked_job, checked_queue


def audit_pending(queue: Path, ids: list[str] | None = None, limit: int = 24,
                  python: str = sys.executable) -> list[dict]:
    queue, matrix = checked_queue(queue)
    if limit < 1 or (ids and len(ids) != len(set(ids))):
        raise ValueError('audit limit/IDs invalid')
    with lock(queue):
        waiting = [(p, checked_job(queue, matrix, p.stem)[1])
                   for p in sorted((queue / 'jobs').glob('*.json'))]
    waiting = [(p, v) for p, v in waiting if v['state'] == 'executed_pending_audit'
               and (not ids or v['job']['job_id'] in ids)][:limit]
    results = []
    for path, old in waiting:
        jid = old['job']['job_id']
        source = Path(old['source'])
        if sha(old['config']) != old['config_sha256'] or \
                verify_source(source) != old['source_manifest_sha256'] or \
                sha(old['release_receipt']) != old['release_receipt_sha256'] or \
                sha(old['technical_receipt']) != old['technical_receipt_sha256']:
            raise RuntimeError(f'{jid}: source/config changed before audit')
        auditor = source / 'scripts/audit_pcla_v7_conditional_run.py'
        auditor_sha = sha(auditor)
        report = queue.parent / 'checks' / 'conditional_validation_audits' / \
                 f'{jid}.{auditor_sha[:12]}.{time.time_ns()}.json'
        argv = [python, str(auditor), '--config', old['config'], '--run', old['out'],
                '--source', str(source), '--out', str(report)]
        process = subprocess.run(argv, capture_output=True, text=True)
        if not report.is_file():
            raise RuntimeError(f'{jid}: conditional auditor produced no report: {process.stderr[-1000:]}')
        result = read(report)
        if sha(auditor) != auditor_sha:
            raise RuntimeError(f'{jid}: auditor changed while reading evidence')
        good = (process.returncode == 0 and result.get('ok') is True and
                result.get('errors') == [] and
                result.get('checks', {}).get('planned_job_id') == jid and
                result.get('checks', {}).get('conditional_slot') == old['slot'] and
                Path(result.get('run_dir', '')).resolve() == Path(old['out']).resolve())
        receipt = {'report': str(report), 'report_sha256': sha(report),
                   'auditor_sha256': auditor_sha, 'argv': argv,
                   'exit_code': process.returncode, 'time': time.time(),
                   'config_sha256': sha(old['config']),
                   'source_manifest_sha256': sha(source / 'SOURCE_MANIFEST.json')}
        with lock(queue):
            _, live = checked_job(queue, matrix, jid)
            if live != old:
                raise RuntimeError(f'{jid}: queue record changed during audit')
            live.update(state='completed' if good else 'audit_failed_needs_review',
                        audit=receipt)
            write(path, live)
        results.append({'job_id': jid, 'state': live['state'],
                        'errors': result.get('errors', [])[:4]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    for result in audit_pending(args.queue, args.ids, args.limit, args.python):
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
