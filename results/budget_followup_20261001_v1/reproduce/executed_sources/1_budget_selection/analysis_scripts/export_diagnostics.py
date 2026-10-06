#!/usr/bin/env python3
"""Fail-closed, read-only PCLA diagnostic exports (no scientific aggregation)."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

SOURCE_SHA = 'beba39274e00668d6900c50267bdbee956e932f1d85cf8ac8360576351d415d7'
AUDITOR_SHA = '9a61d52c68fbc397870c540c3fbd4144f08ab8385b6018eaafd7208ae7f719b7'
FROZEN_CSV_SHA = 'f3e56cd8b3a0c21a8a53ab928275017d351eec952d1dbb206296af3f5e0d24e0'
WARNING = 'direction raw bank has no explicit readout source/row hash; role is bound by frozen driver/model source and manifest'
SEEDS = {101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307}
DIRECTION_LABELS = {'N', 'D', 'S', 'T', 'S_random_nonzero', 'S_zero_removed'}
COHORT_CHOICES = {'local', 'cyclic', 'JL_h32', 'public512_linear', 'JQ_xi1', 'finite_bank_search'}


class ExportError(ValueError):
    pass


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path, cache: dict[Path, str] | None = None) -> str:
    path = path.resolve(strict=True)
    if cache is not None and path in cache:
        return cache[path]
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    result = sha.hexdigest()
    if cache is not None:
        cache[path] = result
    return result


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ExportError(message)


def file_under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    require(path.is_absolute(), f'nonabsolute evidence path: {path}')
    require(path.is_file() and not path.is_symlink(), f'missing or linked evidence: {path}')
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root), f'evidence escapes root: {path}')
    return resolved


def expect_hash(path: Path, expected: str, cache: dict[Path, str]) -> str:
    require(isinstance(expected, str) and len(expected) == 64, f'no expected SHA256: {path}')
    actual = digest(path, cache)
    require(actual == expected, f'SHA256 mismatch: {path}')
    return actual


def _source(root: Path, expected_sha: str, cache: dict[Path, str]) -> dict:
    base = root / 'sources/diagnostic03'
    manifest = base / 'SOURCE_MANIFEST.json'
    expect_hash(manifest, expected_sha, cache)
    record = read(manifest)
    files = record.get('files')
    require(isinstance(files, dict) and files, 'frozen source file map absent')
    for relative, expected in files.items():
        original = base / relative
        require(not original.is_symlink(), f'frozen source linked: {relative}')
        path = original.resolve(strict=True)
        require(path.is_relative_to(base) and path.is_file(),
                f'frozen source missing/escaped: {relative}')
        expect_hash(path, expected, cache)
    return record


def _manifest(root: Path, expected_total: int, expected_layout: tuple[int, int],
              expected_sha: str, expected_csv_sha: str) -> dict[str, str]:
    ready_base = root / 'checks/diagnostic_ready01'
    matrix_path = root / 'diagnostic_queue/matrix_identity.json'
    matrix = read(matrix_path)
    compile_record = read(ready_base / 'compilation.json')
    ready = read(ready_base / 'READY_RECEIPT.json')
    ids = matrix.get('config_sha256_by_job')
    require(isinstance(ids, dict) and len(ids) == expected_total and matrix.get('count') == expected_total,
            'matrix count or unique ID map differs')
    inputs = ready.get('input_sha256', {})
    csv_path = root / 'checks/pro_received_20260928/jobs/cohort_diagnostics_144.csv'
    with csv_path.open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_ids = [row['job_id'] for row in csv_rows]
    csv_sha = digest(csv_path)
    require(len(csv_ids) == len(set(csv_ids)) == expected_total and set(csv_ids) == set(ids) and
            csv_sha == compile_record.get('csv_sha256') == inputs.get('diagnostic_csv') ==
            expected_csv_sha,
            'frozen diagnostic CSV identities/hash differ')
    hardware_path = root / 'checks/v_execution_intake_20260928/seed_hardware_map.json'
    require(digest(hardware_path) == compile_record.get('hardware_map_sha256') ==
            inputs.get('seed_hardware_map'), 'frozen seed hardware map hash differs')
    require(matrix.get('source_manifest_sha256') == expected_sha and
            inputs.get('diagnostic03_source_manifest') == expected_sha and
            inputs.get('data_manifest') == compile_record.get('manifest_sha256') and
            inputs.get('compilation_receipt') == digest(ready_base / 'compilation.json') and
            inputs.get('queue_matrix_identity') == digest(matrix_path) and
            ready.get('count') == compile_record.get('count') == expected_total,
            'matrix/source/data/ready identity differs')
    require(ready.get('config_sha256_by_job') == ids and
            compile_record.get('config_sha256') == {f'{jid}.json': sha for jid, sha in ids.items()},
            'compiled configuration hash map differs')
    counts = Counter('direction' if jid.startswith('Direction_') else
                     'cohort' if jid.startswith('Cohort_') else 'unknown' for jid in ids)
    require((counts['cohort'], counts['direction']) == expected_layout and not counts['unknown'],
            f'diagnostic line coverage differs: {dict(counts)}')
    return ids


def _readouts_direction(stage: dict) -> None:
    values = stage.get('readouts')
    require(isinstance(values, dict) and set(values) == DIRECTION_LABELS,
            'direction six readout representations differ')
    for label, policies in values.items():
        require(set(policies) == {'local', 'cyclic'}, f'direction policy differs: {label}')
        for policy, prefixes in policies.items():
            require(set(prefixes) == {'1', '4', '16'}, f'direction prefixes differ: {label}/{policy}')
            for prefix, row in prefixes.items():
                status = row['normmatch_status']
                require(status in {'matched', 'NA_zero_reference', 'NA_zero_candidate'},
                        f'unknown normmatch status: {label}/{policy}/{prefix}')
                if status == 'matched':
                    require(row['normmatched_loss'] is not None and
                            row['normmatched_benefit'] is not None and
                            row['normmatch_multiplier'] is not None,
                            f'matched readout incomplete: {label}/{policy}/{prefix}')
                else:
                    require(all(row[key] is None for key in
                                ('normmatched_loss', 'normmatched_benefit', 'normmatch_multiplier')),
                            f'NA readout was filled: {label}/{policy}/{prefix}')


def _readouts_cohort(result: dict) -> None:
    values = result.get('values')
    require(isinstance(values, dict) and set(values) == COHORT_CHOICES,
            'cohort readout menu differs')
    for choice, prefixes in values.items():
        require(set(prefixes) == ({'16'} if choice == 'finite_bank_search' else {'1', '4', '16'}),
                f'cohort prefixes differ: {choice}')
        for prefix, row in prefixes.items():
            require(len(row['local64_benefit_by_client']) == 10 and
                    len(row['weighted_readout2048_benefit_by_client']) == 10,
                    f'cohort client readout count differs: {choice}/{prefix}')
            status = row['cancellation_status']
            require(status in {'measured', 'NA_zero_denominator'}, 'unknown cancellation status')
            require((status == 'measured' and row['cancellation'] is not None) or
                    (status == 'NA_zero_denominator' and row['cancellation'] is None and
                     row['denominator_mean_effective_norm_sq'] == 0),
                    'cohort cancellation NA was filled or reason differs')


def _load_pt(path: Path, source_root: Path) -> Any:
    """Load one already-audited auxiliary PT on CPU; never load a bank here."""
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def _read_current(path: Path, evidence: str) -> tuple[Any, dict[str, str]]:
    """Hash the exact bytes parsed for metadata lacking an audit-time file SHA."""
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path),
                             'sha256_at_export': hashlib.sha256(raw).hexdigest(),
                             'evidence_scope': evidence}


def validate_collection(root: Path, *, expected_total: int = 144,
                        expected_layout: tuple[int, int] = (96, 48),
                        expected_source_sha: str = SOURCE_SHA,
                        expected_csv_sha: str = FROZEN_CSV_SHA,
                        auditor_sha: str = AUDITOR_SHA,
                        load_auxiliary: Callable[[Path, Path], Any] = _load_pt) -> list[dict]:
    """Verify all objects; after the 144 gate, check audited auxiliary PT/JSON pairs."""
    root = root.resolve(strict=True)
    ids = _manifest(root, expected_total, expected_layout, expected_source_sha,
                    expected_csv_sha)
    job_dir = root / 'diagnostic_queue/jobs'
    observed = {p.stem for p in job_dir.glob('*.json')}
    require(observed == set(ids), f'missing/extra queue jobs: missing={sorted(set(ids)-observed)}, extra={sorted(observed-set(ids))}')
    # Fail fast while a scientific object is not formally accepted; avoid hashing large banks.
    job_records = {jid: _read_current(job_dir / f'{jid}.json',
                   'current_completed_queue_record; attempts_not_separately_audit_time_hashed')
                   for jid in ids}
    jobs = {jid: record for jid, (record, _) in job_records.items()}
    incomplete = {jid: j.get('state') for jid, j in jobs.items()
                  if j.get('state') != 'completed' or not j.get('acceptance')}
    require(not incomplete, f'144-object formal gate incomplete: {incomplete}')
    cache: dict[Path, str] = {}
    _source(root, expected_source_sha, cache)
    objects = []
    for jid in sorted(ids):
        j = jobs[jid]
        line = 'direction' if jid.startswith('Direction_') else 'cohort'
        config_path = root / 'checks/diagnostic_ready01/diagnostic_configs' / f'{jid}.json'
        expect_hash(config_path, ids[jid], cache)
        config = read(config_path)
        audit_path = root / 'diagnostic_queue/audits' / f'{jid}.json'
        require(file_under(root, j['audit']['receipt']) == audit_path.resolve(),
                f'queue audit receipt path differs: {jid}')
        expect_hash(audit_path, j['audit']['receipt_sha256'], cache)
        audit = read(audit_path)
        acceptance_entry = j['acceptance']
        acceptance_path = file_under(root, acceptance_entry.get('receipt', acceptance_entry.get('path')))
        expect_hash(acceptance_path,
                    acceptance_entry.get('receipt_sha256', acceptance_entry.get('sha256')), cache)
        acceptance = read(acceptance_path)
        require(config.get('job_id') == j.get('job_id') == audit.get('object_id') ==
                acceptance.get('job_id') == jid and config.get('line') == audit.get('line') == line,
                f'object/line identity differs: {jid}')
        require(config.get('seed') == j.get('seed') and config.get('seed') in SEEDS and
                config.get('factor') in {'a', 'b'} and config.get('anchor_round') in {10, 60} and
                len(config.get('participants', [])) == 10 and
                len(set(config['participants'])) == 10, f'config seed/factor/cohort differs: {jid}')
        require(j.get('config_sha256') == ids[jid] == audit['hashes']['config'] ==
                acceptance.get('config_sha256') and
                j.get('source_manifest_sha256') == audit['hashes']['source_manifest'] ==
                acceptance.get('source_manifest_sha256') == expected_source_sha and
                j.get('anchor_sha256') == audit.get('anchor_sha256') ==
                acceptance.get('anchor_sha256') == audit['hashes']['anchor'] ==
                config['parent']['anchor_check']['anchor_sha256'] and
                config.get('frozen_data_manifest_sha256') == audit['hashes']['frozen_data_manifest'],
                f'config/source/anchor/data hash chain differs: {jid}')
        require(j['audit'].get('auditor_sha256') == audit.get('auditor_sha256') == auditor_sha and
                j['audit'].get('status') == audit.get('status') == 'audit_pass_candidate' and
                j['audit'].get('returncode') == 0 and audit.get('errors') == [] and
                audit.get('warnings') == ([WARNING] if line == 'direction' else []),
                f'original audit pass/warning differs: {jid}')
        require(acceptance.get('decision') == 'accepted' and
                acceptance.get('audit_sha256') == digest(audit_path, cache) and
                file_under(root, acceptance['audit_path']) == audit_path.resolve() and
                (line != 'direction' or acceptance.get('warnings_preserved') == [WARNING]),
                f'root acceptance/hash/warning differs: {jid}')
        require(config.get('hardware_family') == j.get('hardware_family') and
                audit['coverage']['local_calls'] == config['calls']['local_calls'] ==
                (120 if line == 'direction' else 40) and
                audit['coverage']['optimizer_steps'] == config['calls']['optimizer_steps'] ==
                (1920 if line == 'direction' else 640) and
                audit['coverage']['snapshots_per_trajectory'] == 3 and
                audit['role_counts'] == {'client_train': 2560, 'client_local_readout': 640,
                                          'public_training': 512, 'selection': 1024,
                                          'readout': 2048}, f'coverage/role/hardware differs: {jid}')
        out = root / 'diagnostics' / jid
        anchor = read(out / 'anchor.json')
        require(anchor['anchor_sha256'] == j['anchor_sha256'] and
                anchor['checkpoint_identity'] == config['parent']['anchor_check']['checkpoint_identity'],
                f'parent anchor receipt differs: {jid}')
        attempts = j.get('attempts', [])
        require(isinstance(attempts, list) and attempts and
                any(a.get('returncode') == 0 and a.get('cpu_threads') == 4 for a in attempts),
                f'no original successful four-CPU attempt: {jid}')
        driver_paths = sorted(out.glob('attempt_*.json'))
        driver_attempts = []
        provenance = {'queue_job': job_records[jid][1]}
        provenance['driver_attempts'] = []
        for path in driver_paths:
            record, source = _read_current(path, 'current_file_at_export_no_audit_time_sha')
            driver_attempts.append(record)
            provenance['driver_attempts'].append(source)
        require(driver_attempts and all(a.get('source_sha256') == expected_source_sha and
                                        a.get('config_sha256') == ids[jid] for a in driver_attempts),
                f'driver attempt identity differs: {jid}')
        extra = {}
        if line == 'cohort':
            stage_dir = out / 'bank_stage'
            stage = read(stage_dir / 'bank_readout_stage.json')
            result_path = file_under(root, stage['result_path'])
            require(result_path.parent == stage_dir.resolve(), f'cohort readout path differs: {jid}')
            expect_hash(result_path, stage['result_sha256'], cache)
            require(stage['result_sha256'] == audit['hashes']['readout_json'] and
                    stage['bank_sha256'] == audit['hashes']['bank_pt'],
                    f'cohort result/bank SHA differs: {jid}')
            for relative, key in [('supports.pt', 'supports_pt'),
                                  ('bank_stage/prediction_lock.pt', 'prediction_lock_pt'),
                                  ('bank_stage/bank.pt', 'bank_pt'),
                                  ('bank_stage/selection_lock.pt', 'selection_lock_pt')]:
                expect_hash(out / relative, audit['hashes'][key], cache)
            supports_record, provenance['supports_json'] = _read_current(
                out / 'supports.json', 'current_file_at_export_no_audit_time_sha; seconds_not_COST1_B8')
            bank_record, provenance['bank_receipt_json'] = _read_current(
                stage_dir / 'bank_receipt.json', 'current_file_at_export_no_audit_time_sha; seconds_not_COST1_B8')
            prediction_record, provenance['prediction_lock_json'] = _read_current(
                stage_dir / 'prediction_lock.json', 'cross_checked_with_audited_prediction_lock_pt')
            selection, provenance['selection_lock_json'] = _read_current(
                stage_dir / 'selection_lock.json', 'cross_checked_with_audited_selection_lock_pt')
            jq, provenance['jq_pretraining_lock_json'] = _read_current(
                stage_dir / 'jq_pretraining_lock.json', 'cross_checked_with_audited_supports_pt_and_stage_identity')
            require(bank_record['sha256'] == audit['hashes']['bank_pt'] and
                    supports_record['sha256'] == audit['hashes']['supports_pt'] and
                    prediction_record['sha256'] == audit['hashes']['prediction_lock_pt'] and
                    selection['file_sha256'] == audit['hashes']['selection_lock_pt'],
                    f'cohort artifact receipt differs: {jid}')
            source_root = root / 'sources/diagnostic03'
            supports_pt = load_auxiliary(out / 'supports.pt', source_root)
            prediction_pt = load_auxiliary(stage_dir / 'prediction_lock.pt', source_root)
            selection_pt = load_auxiliary(stage_dir / 'selection_lock.pt', source_root)
            require(_plain(supports_pt['jq_lock']) == jq and
                    supports_pt['object_id'] == jid and
                    supports_pt['anchor_sha256'] == j['anchor_sha256'] and
                    _plain(prediction_pt) == prediction_record['prediction'] and
                    _plain(selection_pt) == {k: v for k, v in selection.items()
                                             if k != 'file_sha256'},
                    f'cohort auxiliary JSON differs from audited PT: {jid}')
            del supports_pt, prediction_pt, selection_pt
            stage_identity = read(stage_dir / 'bank_stage_identity.json')
            result = read(result_path)
            _readouts_cohort(result)
            require(result['object_id'] == jid and result['anchor_sha256'] == j['anchor_sha256'] and
                    result['prediction_lock_sha256'] == audit['hashes']['prediction_lock_pt'] and
                    result['selection_lock_sha256'] == selection['sha256'] and
                    stage_identity['jq_lock_sha256'] == jq['sha256'] and
                    len(result['local64_source_ids']) == 10 and
                    set(result['local64_source_ids']) == {str(cid) for cid in config['participants']} and
                    set(result['local64_source_ids'].values()) == {result['readout2048_source_id']},
                    f'cohort readout identity differs: {jid}')
            stamp = result_path.stem.removeprefix('bank_readout_')
            eval_path = stage_dir / f'evaluations_{stamp}.json'
            require(eval_path.is_file(), f'cohort evaluation log absent: {jid}')
            evaluations, provenance['evaluations_json'] = _read_current(
                eval_path, 'current_file_at_export_no_audit_time_sha; evaluation_seconds_not_COST1_B8')
            extra = {'result': result, 'result_path': result_path,
                     'result_sha256': stage['result_sha256'],
                     'jq': jq, 'selection': selection,
                     'prediction': prediction_record['prediction'],
                     'evaluations': evaluations,
                     'supports_seconds': supports_record.get('seconds'),
                     'bank_seconds': bank_record.get('seconds')}
        else:
            stage_path = out / 'direction_stage.json'
            stage, provenance['direction_stage_json'] = _read_current(
                stage_path, 'current_file_at_export_no_audit_time_sha; seconds_not_COST1_B8; readouts_rechecked_against_audited_bank_in_projection')
            require(stage == read(out / 'stage_status.json') and
                    stage['identity']['object_id'] == jid and
                    stage['identity']['config_sha256'] == ids[jid] and
                    stage['identity']['source_sha256'] == expected_source_sha and
                    stage['bank_sha256'] == audit['hashes']['direction_bank_pt'] and
                    stage['local_calls'] == 120 and stage['optimizer_steps'] == 1920,
                    f'direction stage identity differs: {jid}')
            expect_hash(out / 'direction_bank.pt', stage['bank_sha256'], cache)
            _readouts_direction(stage)
            extra = {'stage': stage, 'stage_sha256': digest(stage_path, cache)}
        objects.append({'job_id': jid, 'line': line, 'config': config, 'config_sha256': ids[jid],
                        'job': j, 'audit': audit, 'audit_sha256': digest(audit_path, cache),
                        'acceptance': acceptance, 'acceptance_sha256': digest(acceptance_path, cache),
                        'out': out, 'anchor': anchor, 'attempts': attempts,
                        'driver_attempts': driver_attempts, 'auxiliary_provenance': provenance,
                        **extra})
    return objects


def _json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'


def _write_jsonl(path: Path, rows: list[dict]) -> tuple[int, str]:
    with path.open('x') as stream:
        for row in rows:
            stream.write(_json_line(row))
    return len(rows), digest(path)


def _basis(item: dict) -> dict:
    c, j = item['config'], item['job']
    return {'job_id': item['job_id'], 'line': item['line'], 'seed': c['seed'],
            'parent_job_id': c['parent']['job_id'], 'parent_recipe_id': c['parent']['recipe_id'],
            'anchor_round': c['anchor_round'], 'factor': c['factor'],
            'hardware_family': c['hardware_family'], 'config_sha256': item['config_sha256'],
            'source_manifest_sha256': j['source_manifest_sha256'],
            'anchor_sha256': j['anchor_sha256'], 'audit_receipt_sha256': item['audit_sha256'],
            'acceptance_receipt_sha256': item['acceptance_sha256']}


def object_row(item: dict) -> dict:
    base = _basis(item)
    out = item['out']
    costs = {'boundary': 'diagnostic_attempt_wall_and_nonexclusive_stage_fragments_not_COST1_B8',
             'driver_attempts': item['driver_attempts']}
    if item['line'] == 'cohort':
        costs.update(supports_seconds=item['supports_seconds'], bank_seconds=item['bank_seconds'],
                     evaluation_seconds_by_role=[{'role': r['role'], 'rows': r['rows'],
                                                  'seconds': r['seconds']} for r in item['evaluations']])
    else:
        costs.update(direction_collect_and_save_seconds=item['stage']['seconds'])
    return {**base, 'record_type': 'object', 'state': item['job']['state'],
            'participants': item['config']['participants'], 'training': item['config']['training'],
            'coverage': item['audit']['coverage'], 'role_counts': item['audit']['role_counts'],
            'warnings': item['audit']['warnings'], 'errors': item['audit']['errors'],
            'attempts_all_including_failures': item['attempts'], 'cost_scope': costs,
            'auxiliary_json_provenance': item['auxiliary_provenance'],
            'readout_source_id': (item['result']['readout2048_source_id'] if item['line'] == 'cohort'
                                  else None), 'direction_readout_source_row_hash': None,
            'readout_source_limitation': (WARNING if item['line'] == 'direction' else None),
            'result_sha256': (item['result_sha256'] if item['line'] == 'cohort'
                              else item['stage_sha256'])}


def json_rows(objects: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {name: [] for name in
        ('objects', 'cohort_readouts', 'cohort_clients', 'cohort_locks',
         'cohort_evaluations', 'direction_readouts')}
    for item in objects:
        b = _basis(item)
        result['objects'].append(object_row(item))
        if item['line'] == 'cohort':
            raw = item['result']
            for choice, prefixes in raw['values'].items():
                for prefix, value in prefixes.items():
                    common = {**b, 'choice': choice, 'prefix': int(prefix),
                              'readout_json_sha256': item['result_sha256'],
                              'parent_readout2048_loss': raw['parent_readout2048_loss']}
                    row = {key: value[key] for key in value if key not in
                           ('local64_benefit_by_client', 'weighted_readout2048_benefit_by_client')}
                    result['cohort_readouts'].append({**common, **row})
                    for cid, local in value['local64_benefit_by_client'].items():
                        result['cohort_clients'].append({**common, 'client_id': int(cid),
                            'local64_benefit': local,
                            'weighted_readout2048_benefit':
                                value['weighted_readout2048_benefit_by_client'][cid]})
            for kind, payload, source in (
                    ('JQ_pretraining', item['jq'], item['auxiliary_provenance']['jq_pretraining_lock_json']),
                    ('selection1024', item['selection'], item['auxiliary_provenance']['selection_lock_json']),
                    ('prediction_h32_full', item['prediction'], item['auxiliary_provenance']['prediction_lock_json']),
                    ('full_gradient_choice_readout', raw['full_gradient_choice_readout'],
                     {'path': str(item['result_path']),
                      'sha256_at_export': item['result_sha256'],
                      'evidence_scope': 'result_json_sha256_in_original_audit'})):
                result['cohort_locks'].append({**b, 'lock_kind': kind, 'payload': payload,
                                               'payload_provenance': source,
                                               'readout_json_sha256': item['result_sha256']})
            for ordinal, row in enumerate(item['evaluations']):
                result['cohort_evaluations'].append({**b, 'ordinal': ordinal,
                    'log_provenance': item['auxiliary_provenance']['evaluations_json'], **row})
        else:
            for label, policies in item['stage']['readouts'].items():
                for policy, prefixes in policies.items():
                    for prefix, row in prefixes.items():
                        result['direction_readouts'].append({**b, 'representation': label,
                            'policy': policy, 'prefix': int(prefix),
                            'stage_sha256': item['stage_sha256'], **row})
    return result


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if hasattr(value, 'detach') and hasattr(value, 'tolist'):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite PT projection scalar')
    return value


def project_bank(bank: dict, item: dict) -> list[dict]:
    """Project existing geometry and T comparisons; never derive a new scientific score."""
    jid = item['job_id']
    stage = item['stage']
    c = item['config']
    require(bank.get('schema') == 'pcla-v7-direction-diagnostic-bank-1' and
            bank.get('object_id') == jid and bank.get('anchor_sha256') == item['job']['anchor_sha256'] and
            bank.get('factor') == c['factor'] and bank.get('participants') == c['participants'] and
            bank.get('local_calls') == 120 and bank.get('optimizer_steps') == 1920 and
            _plain(bank.get('readouts')) == stage['readouts'],
            f'direction bank/stage differs: {jid}')
    representations = bank['representations']
    require(set(representations) == DIRECTION_LABELS, f'direction geometry menu differs: {jid}')
    base = {**_basis(item), 'bank_sha256': stage['bank_sha256'],
            'stage_sha256': item['stage_sha256']}
    rows: list[dict] = []
    rows.append({**base, 'record_type': 'direction_parent',
                 'parent_readout2048_loss': bank['parent_readout2048_loss'],
                 'actual_visits': bank['actual_visits'], 'capacities': bank['capacities'],
                 'training_seed_by_client': bank['training_seed_by_client'],
                 'selection_seed_by_client': bank['selection_seed_by_client'],
                 'probe_seed_by_client': bank['probe_seed_by_client'],
                 't_probe': {'cbar_sha256': bank['t_probe']['cbar_sha256'],
                             'full_gradients_sha256': bank['t_probe']['full_gradients_sha256'],
                             'examples_by_client': bank['t_probe']['examples_by_client'],
                             'seed_by_client': bank['t_probe']['seed_by_client']}})
    for label, representation in representations.items():
        ranks = representation['numerical_ranks']
        require(len(ranks) == 24 and set(ranks) == set(representation['rank_thresholds']) ==
                set(representation['relative_errors']), f'direction 24 modules differ: {jid}/{label}')
        for module in sorted(ranks):
            rotation = representation['random_rotations'].get(module)
            rows.append({**base, 'record_type': 'representation_module',
                         'representation': label, 'module': module,
                         'active': representation['active'],
                         'parent_state_sha256': representation['parent_state_sha256'],
                         'numerical_rank': ranks[module],
                         'rank_threshold': representation['rank_thresholds'][module],
                         'relative_error': representation['relative_errors'][module],
                         'random_rotation': _plain(rotation) if rotation is not None else None})
    comparator = bank['t_full_h32_comparator']
    require(comparator['status'] == 'diagnostic_only_no_extra_training' and
            set(comparator['one_step_linear_gain']) == {'local', 'cyclic'},
            f'T full/h32 comparator differs: {jid}')
    rows.append({**base, 'record_type': 'T_comparator_identity',
                 'status': comparator['status'], 'prediction_kind': comparator['prediction_kind'],
                 'full_cbar_sha256': comparator['full_cbar_sha256'],
                 'h32_cbar_sha256': comparator['h32_cbar_sha256']})
    for module, record in sorted(comparator['rotations_and_grams'].items()):
        rows.append({**base, 'record_type': 'T_module_comparison', 'module': module,
                     'gram_frobenius_relative_error': record['gram_frobenius_relative_error'],
                     'gram_frobenius_absolute_error': record['gram_frobenius_absolute_error'],
                     'column_absolute_cosine_descriptive_only':
                         _plain(record['column_absolute_cosine_descriptive_only']),
                     'o_full_untrained': _plain(record['o_full_untrained']),
                     'o_h32_deployed': _plain(record['o_h32_deployed']),
                     'full_gram': _plain(record['full_gram']),
                     'h32_gram': _plain(record['h32_gram'])})
    for policy, prediction in comparator['one_step_linear_gain'].items():
        rows.append({**base, 'record_type': 'T_prediction_aggregate', 'policy': policy,
                     'full': prediction['full'], 'h32': prediction['h32']})
        for cid, value in prediction['by_client'].items():
            rows.append({**base, 'record_type': 'T_prediction_client', 'policy': policy,
                         'client_id': int(cid), 'full': value['full'], 'h32': value['h32']})
    return rows


def _temporary_directory(out: Path) -> Path:
    require(not out.exists(), f'formal output already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=out.name + '.building-', dir=out.parent))


def project_direction(root: Path, out: Path, *, expected_total: int = 144,
                      expected_layout: tuple[int, int] = (96, 48),
                      expected_source_sha: str = SOURCE_SHA,
                      expected_csv_sha: str = FROZEN_CSV_SHA,
                      auditor_sha: str = AUDITOR_SHA,
                      load_auxiliary: Callable[[Path, Path], Any] = _load_pt,
                      load_bank: Callable[[Path], dict] | None = None) -> dict:
    objects = validate_collection(root, expected_total=expected_total,
        expected_layout=expected_layout, expected_source_sha=expected_source_sha,
        expected_csv_sha=expected_csv_sha, auditor_sha=auditor_sha,
        load_auxiliary=load_auxiliary)
    directions = [o for o in objects if o['line'] == 'direction']
    if load_bank is None:
        # Only this command imports torch. CUDA libraries are never requested.
        source = (root.resolve() / 'sources/diagnostic03')
        sys.path.insert(0, str(source))
        import torch
        load_bank = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    temporary = _temporary_directory(out)
    try:
        projection_path = temporary / 'direction_geometry.jsonl'
        receipts = []
        count = 0
        with projection_path.open('x') as stream:
            for item in directions:
                path = item['out'] / 'direction_bank.pt'
                require(digest(path) == item['stage']['bank_sha256'],
                        f'direction bank changed before projection: {item["job_id"]}')
                before = path.stat()
                bank = load_bank(path)
                rows = project_bank(bank, item)
                for row in rows:
                    stream.write(_json_line(row))
                count += len(rows)
                del bank, rows
                gc.collect()
                after = path.stat()
                require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and
                        digest(path) == item['stage']['bank_sha256'],
                        f'direction bank changed during projection: {item["job_id"]}')
                receipts.append({'job_id': item['job_id'], 'bank_sha256': item['stage']['bank_sha256'],
                                 'stage_sha256': item['stage_sha256'],
                                 'audit_receipt_sha256': item['audit_sha256']})
        manifest = {'schema': 'pcla-v7-diagnostic-direction-projection-1',
                    'source_manifest_sha256': expected_source_sha,
                    'object_count': len(directions), 'row_count': count,
                    'geometry_sha256': digest(projection_path), 'objects': receipts,
                    'cost_scope': 'CPU serialization of already audited banks; no new scientific training'}
        (temporary / 'direction_projection_manifest.json').write_text(_json_line(manifest))
        temporary.rename(out)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def export_json(root: Path, out: Path, projection_dir: Path, *, expected_total: int = 144,
                expected_layout: tuple[int, int] = (96, 48),
                expected_source_sha: str = SOURCE_SHA,
                expected_csv_sha: str = FROZEN_CSV_SHA,
                auditor_sha: str = AUDITOR_SHA,
                load_auxiliary: Callable[[Path, Path], Any] = _load_pt) -> dict:
    objects = validate_collection(root, expected_total=expected_total,
        expected_layout=expected_layout, expected_source_sha=expected_source_sha,
        expected_csv_sha=expected_csv_sha, auditor_sha=auditor_sha,
        load_auxiliary=load_auxiliary)
    projection = read(projection_dir / 'direction_projection_manifest.json')
    require(projection.get('schema') == 'pcla-v7-diagnostic-direction-projection-1' and
            projection.get('source_manifest_sha256') == expected_source_sha and
            projection.get('object_count') == expected_layout[1] and
            digest(projection_dir / 'direction_geometry.jsonl') == projection.get('geometry_sha256'),
            'direction projection identity/hash differs')
    expected = {o['job_id']: (o['stage']['bank_sha256'], o['stage_sha256'], o['audit_sha256'])
                for o in objects if o['line'] == 'direction'}
    found = {r['job_id']: (r['bank_sha256'], r['stage_sha256'], r['audit_receipt_sha256'])
             for r in projection['objects']}
    require(len(projection['objects']) == len(found) == expected_layout[1] and found == expected,
            'direction projection object/bank/stage/audit hashes differ')
    temporary = _temporary_directory(out)
    try:
        tables = json_rows(objects)
        output = {}
        for name, rows in tables.items():
            count, sha = _write_jsonl(temporary / f'{name}.jsonl', rows)
            output[name] = {'rows': count, 'sha256': sha}
        geometry_out = temporary / 'direction_geometry.jsonl'
        shutil.copyfile(projection_dir / 'direction_geometry.jsonl', geometry_out)
        require(digest(geometry_out) == projection['geometry_sha256'],
                'direction geometry changed during final packaging')
        output['direction_geometry'] = {'rows': projection['row_count'],
                                        'sha256': projection['geometry_sha256']}
        projection_out = temporary / 'direction_projection_manifest.json'
        shutil.copyfile(projection_dir / 'direction_projection_manifest.json', projection_out)
        require(digest(projection_out) == digest(projection_dir / 'direction_projection_manifest.json'),
                'projection manifest changed during final packaging')
        manifest = {'schema': 'pcla-v7-diagnostic-json-export-1',
                    'object_count': len(objects), 'cohort_count': expected_layout[0],
                    'direction_count': expected_layout[1],
                    'source_manifest_sha256': expected_source_sha,
                    'direction_projection_manifest_sha256':
                        digest(projection_out),
                    'direction_geometry_sha256': projection['geometry_sha256'],
                    'tables': output,
                    'scope': 'raw audited observations only; no seed statistic, threshold, winner or COST1 B8 claim'}
        (temporary / 'EXPORT_MANIFEST.json').write_text(_json_line(manifest))
        temporary.rename(out)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'project-direction', 'export-json'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--projection-dir', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            objects = validate_collection(args.root)
            result = {'status': 'ready', 'objects': len(objects),
                      'lines': dict(Counter(o['line'] for o in objects))}
        elif args.command == 'project-direction':
            require(args.out is not None, '--out required for project-direction')
            result = project_direction(args.root, args.out)
        else:
            require(args.out is not None and args.projection_dir is not None,
                    '--out and --projection-dir required for export-json')
            result = export_json(args.root, args.out, args.projection_dir)
        print(_json_line(result), end='')
    except (ExportError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(_json_line({'status': 'rejected', 'error': str(error)}), end='')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
