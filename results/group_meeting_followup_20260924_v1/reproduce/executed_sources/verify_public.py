#!/usr/bin/env python3
"""Offline integrity check for the path-redacted PCLA source export."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
read = lambda path: json.loads(path.read_text())


def main():
    manifest = read(ROOT / 'PUBLIC_SOURCE_MANIFEST.json')
    records = {r['path']: r for r in manifest['records']}
    external = {r['path']: r for r in manifest['external_sources']}
    assert len(records) == len(manifest['records'])
    for rel, r in records.items():
        data = (ROOT / rel).read_bytes()
        assert sha(data) == r['published_sha256'], rel
        if not r['redacted']:
            assert r['published_sha256'] == r['executed_sha256'], rel
        old = (ROOT / 'baseline' / r['baseline_repo_path']).read_bytes() if r['baseline_repo_path'] else None
        assert (sha(old) if old is not None else None) == r['old_sha256'], rel
        status = r['diff_status']
        if status == 'unified_diff':
            logical = r['comparison_path'] if r['category'] == 'snapshot' else rel
            patch = ''.join(difflib.unified_diff(
                [] if old is None else old.decode().splitlines(keepends=True),
                data.decode().splitlines(keepends=True),
                fromfile='a/' + logical if old is not None else '/dev/null',
                tofile='b/' + logical, n=3)).encode()
            assert patch == (ROOT / r['diff_path']).read_bytes(), rel
            assert sha(patch) == r['diff_sha256'], rel
        elif status == 'identical_to_baseline':
            assert old == data and r['diff_path'] is None, rel
        elif status == 'empty_file':
            assert data == b'' and r['diff_path'] is None, rel
        elif status == 'not_applicable_frozen_queue_generated_at_execution':
            assert r['category'] == 'queue' and r['diff_path'] is None
        else:
            raise AssertionError((rel, status))

    jobs = bindings = 0
    for group, info in manifest['queues'].items():
        q = ROOT / 'queues' / group
        frozen = read(q / 'FROZEN.json')
        execution_hash = read(q / 'FROZEN_SHA256.json')['sha256']
        assert execution_hash == info['frozen_sha256']
        assert execution_hash == records[f'queues/{group}/FROZEN.json']['executed_sha256']
        assert len(frozen['jobs']) == info['jobs']
        assert len(frozen['source_files']) == info['bound_source_files']
        for rel, expected in frozen['source_files'].items():
            path = f'snapshots/{group}/{rel}'
            record = records.get(path) or external.get(path)
            assert record and record['executed_sha256'] == expected, path
        lock = read(ROOT / 'snapshots' / group / 'SOURCE_LOCK.json')
        assert lock['commit'] == manifest['baseline_commit']
        assert len(lock['files']) == manifest['source_lock_file_counts'][group]
        for rel, identity in lock['files'].items():
            assert sha((ROOT / 'snapshots' / group / 'reference_snapshot' / rel).read_bytes()) == identity['sha256']
            assert sha((ROOT / 'baseline' / rel).read_bytes()) == identity['sha256']
        jobs += len(frozen['jobs'])
        bindings += len(frozen['source_files'])

    for pair in read(ROOT / 'LINEAGE_MANIFEST.json')['pairs']:
        parts = []
        for c in pair['files_changed']:
            rel = c['path']
            a = (ROOT / 'snapshots' / pair['from'] / rel).read_bytes() if c['old_sha256'] else b''
            b = (ROOT / 'snapshots' / pair['to'] / rel).read_bytes() if c['new_sha256'] else b''
            assert (sha(a) if c['old_sha256'] else None) == c['old_sha256']
            assert (sha(b) if c['new_sha256'] else None) == c['new_sha256']
            parts.append(''.join(difflib.unified_diff(
                a.decode().splitlines(keepends=True), b.decode().splitlines(keepends=True),
                fromfile='a/' + rel if c['old_sha256'] else '/dev/null',
                tofile='b/' + rel if c['new_sha256'] else '/dev/null', n=3)))
        patch = (ROOT / pair['patch']).read_bytes()
        assert patch == ''.join(parts).encode()
        assert sha(patch) == pair['patch_sha256']

    archives = read(ROOT / 'RUN_ARCHIVE_HASHES.json')['records']
    assert len(archives) == jobs
    assert all(r['state'] == 'completed' and r['archive_sha256'] for r in archives)
    print(f'PASS: {len(records)} published files, {len(external)} external references, '
          f'{bindings} frozen source bindings, {jobs} receipt-recorded archives')


if __name__ == '__main__':
    main()
