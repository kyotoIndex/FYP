"""Pure CPU compiler for the frozen nine-slot/twelve-seed v8 validation plan.

Consumes root-pinned complete accepted DEV selection; never selects a winner,
creates a queue, generates data, loads a model or starts an experiment.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

from .configuration import digest, verified_receipt
from pcla_research_v7.data import digest as data_digest

VAL_SEEDS = (601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661)
DEV_SEEDS = {503, 509, 521, 541}
PRO_SHA = {
    'configs/common.json': '9fe4878c0db0700fd809e52f4a7868f8c602be54534ed90f0665f67d40aef6be',
    'configs/recipes.json': '93a4a2a518a1f9fafd2978ca357192deb6e1026eaf4aacfc3a08fc45b9debc97',
    'jobs/validation_slots.csv': '1be0e9481896ac74605e0b7e74e539104d20da32835489c4519b38d70b4c9842',
    'configs/phase_plan.json': '16cd823e978f5ac0ea5dea28590dd87811f5936a32511bf7652aaa0546d10680',
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pinned_json(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, f'external input SHA required: {path}')
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f'input SHA differs: {path}')
    return json.loads(raw)


def read_protocol(package_dir):
    package_dir = Path(package_dir)
    values = {rel: pinned_json(package_dir / rel, expected)
              for rel, expected in PRO_SHA.items() if rel.endswith('.json')}
    path = package_dir / 'jobs/validation_slots.csv'; raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PRO_SHA['jobs/validation_slots.csv'], 'original VAL slots SHA differs')
    slots = list(csv.DictReader(io.StringIO(raw.decode())))
    common = values['configs/common.json']; plan = values['configs/phase_plan.json']['VAL']
    require(plan['slots'] == 9 and tuple(plan['seeds']) == VAL_SEEDS and plan['jobs_max'] == 108 and
            plan['same_configuration_alias'] == 'deduplicate only exact configs, retain alias map', 'Pro VAL contract differs')
    recipes = {v['recipe_id']: v for v in values['configs/recipes.json']}
    require(len(recipes) == 32 and all(v['config_sha256'] == digest({k: x for k, x in v.items() if k != 'config_sha256'})
                                     for v in recipes.values()), 'original recipe identity differs')
    require(len(slots) == 108 and len({v['job_id'] for v in slots}) == 108, 'VAL logical matrix incomplete')
    return common, recipes, slots


def validate_selection(selection, *, recipes, slots, scope_sha256, accepted_snapshot_sha256):
    require(selection.get('status') == 'ready_for_independent_review' and selection.get('blocked') == [] and
            selection.get('dev_count') == 128 and selection.get('val_jobs_max') == 108 and
            selection.get('hardware') == 'RTX4090D' and selection.get('launch_performed') is False and
            selection.get('selection_performed') is True and tuple(selection.get('val_seeds', ())) == VAL_SEEDS,
            'selection is not complete unblocked ready collector output')
    require(selection.get('recipe_source_sha256') == PRO_SHA['configs/recipes.json'] and
            selection.get('slot_source_sha256') == PRO_SHA['jobs/validation_slots.csv'] and
            selection.get('scope_sha256') == scope_sha256 and
            selection.get('accepted_snapshot_sha256') == accepted_snapshot_sha256 and
            isinstance(accepted_snapshot_sha256, str) and len(accepted_snapshot_sha256) == 64,
            'selection root-pinned source/scope/snapshot binding differs')
    evidence = selection.get('evidence_rows', [])
    expected = {(rid, seed) for rid in recipes for seed in DEV_SEEDS}
    require(len(evidence) == 128 and {(r['recipe_id'], r['seed']) for r in evidence} == expected,
            'accepted DEV selection evidence coverage differs')
    for row in evidence:
        require(row.get('state') == 'completed' and row.get('independent_accepted') is True and
                row.get('endpoint') == 'B8_original_parameters' and row.get('samples') == 4096 and
                type(row.get('correct')) is int and 0 <= row['correct'] <= 4096 and
                isinstance(row.get('artifact_sha256'), dict) and row['artifact_sha256'],
                'selection includes unaccepted or non-original B8 DEV evidence')
    winners = selection.get('winners', [])
    require(len(winners) == 9 and {v['slot'] for v in winners} == set(range(1, 10)), 'nine resolved winners required')
    by_slot = {w['slot']: w for w in winners}
    for slot in slots:
        w = by_slot[int(slot['slot'])]
        recipe = recipes.get(w['recipe_id'])
        require(recipe is not None and w['group'] == slot['selection_group'] and
                slot['recipe_rule'].startswith('best within ') and
                recipe['family'] in slot['recipe_rule'][len('best within '):].split(','),
                'selected recipe does not belong to the prescribed slot')
    return by_slot


def checked_source(path, expected):
    path = Path(path).resolve(); source = pinned_json(path, expected)
    require(isinstance(source.get('files'), dict) and source['files'], 'source manifest closure missing')
    for rel, pinned in source['files'].items():
        file = (path.parent / rel).resolve()
        require(file.is_relative_to(path.parent) and file.is_file() and sha(file) == pinned,
                f'source closure differs: {rel}')
    return source


def check_data(paths, expected, common, model_files):
    manifest = pinned_json(paths['manifest'], expected['manifest'])
    require(sha(paths['train_file']) == expected['train_file'] == manifest['source_sha256'], 'training source binding differs')
    roles = manifest.get('roles', {})
    sizes = {'client_pool': 16000, 'public_training': 512, 'cohort_select': 1024,
             'cohort_readout': 2048, 'recipe_dev': 4096}
    require(set(roles) == set(sizes) and all(len(roles[k]) == n for k, n in sizes.items()), 'original data role coverage differs')
    flattened = [row for ids in roles.values() for row in ids]
    require(len(set(flattened)) == len(flattened) == 23680, 'data role overlap/duplicate')
    pool = set(roles['client_pool'])
    for seed in VAL_SEEDS:
        part = manifest.get('partitions', {}).get(str(seed), {})
        train, readout = part.get('client_train', {}), part.get('client_local_readout', {})
        require(len(train) == len(readout) == 50 and set(train) == set(readout) and
                all(len(v) == 256 for v in train.values()) and all(len(v) == 64 for v in readout.values()),
                f'existing VAL partition missing/changed: {seed}')
        partition_rows = [x for values in (train, readout) for ids in values.values() for x in ids]
        require(len(partition_rows) == len(set(partition_rows)) == 16000 and set(partition_rows) == pool,
                f'VAL partition role isolation differs: {seed}')
    metadata = pinned_json(paths['token_cache_metadata'], expected['token_cache_metadata'])
    identity = metadata['identity']
    require(sha(paths['token_cache']) == expected['token_cache'] == metadata['cache_sha256'] and
            metadata['identity_sha256'] == data_digest(identity) and identity['source_sha256'] == expected['train_file'] and
            identity['split_sha256'] == manifest['split_sha256'] and identity['max_length'] == common['model']['max_length'] and
            metadata['rows'] == 23680 and metadata['shape'] == [23680, 64], 'existing token/data/model identity differs')
    for name, pinned in identity['tokenizer_files'].items():
        require(model_files.get(name) == pinned, 'model tokenizer differs from frozen metadata')
    wire = pinned_json(paths['b8_wire'], expected['b8_wire'])
    require(wire.get('total_bytes') == common['budget']['B8'] == 2382653750 and
            wire.get('wire_manifest_sha256') == data_digest({k: v for k, v in wire.items() if k != 'wire_manifest_sha256'}),
            'original B8 wire contract differs')
    return manifest, metadata


def check_model(path, expected_files):
    path = Path(path).resolve()
    require(path.is_dir() and expected_files and 'config.json' in expected_files,
            'local original model files must be externally pinned')
    actual_files = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    require(actual_files == set(expected_files) and
            any(name.endswith(('.safetensors', '.bin')) for name in actual_files), 'model file closure differs')
    for name, expected in expected_files.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and
                sha(path / relative) == expected, f'original model SHA differs: {name}')


def deduplicate_exact_configs(configs):
    """Deduplicate execution equality only; retain every logical slot identity."""
    canonical, by_key, aliases = [], {}, {}
    for item in configs:
        item = copy.deepcopy(item); logical_id = item['job']['job_id']
        execution = copy.deepcopy(item)
        for k in ('expanded_sha256', 'validation_slot', 'selection', 'validation_aliases'):
            execution.pop(k, None)
        for k in ('job_id', 'slot', 'selection_group', 'output_dir'):
            execution['job'].pop(k, None)
        key = digest(execution)
        if key not in by_key:
            by_key[key] = item; item['validation_aliases'] = []; canonical.append(item)
        target = by_key[key]; target['validation_aliases'].append(logical_id)
        aliases[logical_id] = target['job']['job_id']
    for item in canonical:
        item.pop('expanded_sha256', None); item['expanded_sha256'] = digest(item)
    return canonical, aliases


def compile_validation(*, package_dir: Path, selection_path: Path, selection_sha256: str,
                       accepted_snapshot_sha256: str, authorization_path: Path,
                       loader_template_path: Path, source_manifest: Path, output_root: Path,
                       expected_input_sha256: dict, model_files_sha256: dict,
                       source_manifest_by_family: dict | None = None) -> dict:
    """Return 108 logical slots, immutable execution configs and exact alias map.

    Root supplies selection_sha256 only after accepting selection evidence.
    expected_input_sha256 pins existing loader/data/scope/source files; source
    overrides use keys source_manifest:<family>. Model hashes describe disk
    files, never the seed-dependent initialized classifier tensor hash.
    """
    common, recipes, slots = read_protocol(package_dir)
    authorization = pinned_json(authorization_path, expected_input_sha256['scope_authorization'])
    # Reuse scope semantics, then additionally prove the VAL scientific range.
    require(verified_receipt(Path(authorization_path)) == authorization, 'scope changed during verification')
    require('up to108 VAL' in authorization.get('scientific_scope', ''), 'VAL is outside scope')
    selection = pinned_json(selection_path, selection_sha256)
    winners = validate_selection(selection, recipes=recipes, slots=slots,
                                 scope_sha256=expected_input_sha256['scope_authorization'],
                                 accepted_snapshot_sha256=accepted_snapshot_sha256)
    loader = pinned_json(loader_template_path, expected_input_sha256['loader_template'])
    paths = {k: Path(loader[k]).resolve() for k in ('manifest', 'train_file', 'token_cache', 'token_cache_metadata', 'b8_wire')}
    model_path = Path(loader['model']['repository']).resolve()
    require(loader['model'].get('local_files_only') is True and
            loader['model'].get('max_length') == common['model']['max_length'] and
            loader['model'].get('dropout') == common['model']['lora_dropout'], 'loader model contract differs')
    check_model(model_path, model_files_sha256)
    _, metadata = check_data(paths, expected_input_sha256, common, model_files_sha256)
    sources = {None: (Path(source_manifest).resolve(), expected_input_sha256['source_manifest'])}
    for family, path in (source_manifest_by_family or {}).items():
        require(family in {r['family'] for r in recipes.values()}, 'unknown family source override')
        sources[family] = (Path(path).resolve(), expected_input_sha256[f'source_manifest:{family}'])
    require('fedsb_r32' in sources, 'selected native family requires explicit VAL-capable source override')
    source_records = {key: checked_source(path, pinned) for key, (path, pinned) in sources.items()}
    require(Path(output_root).is_absolute(), 'VAL output root must be absolute')
    configs, logical_slots = [], []
    for row in slots:
        seed, slot = int(row['seed']), int(row['slot']); winner = winners[slot]
        recipe = recipes[winner['recipe_id']]; source_key = recipe['family'] if recipe['family'] in sources else None
        source_path, source_sha = sources[source_key]
        require(row['phase'] == 'VAL' and row['job_id'] == f'VAL_S{slot:02d}_s{seed}' and seed in VAL_SEEDS and
                row['recipe_id'] == 'RESOLVE_AFTER_COMPLETE_DEV' and row['launch_status'] == 'NOT_LAUNCHED' and
                int(row['byte_cap']) == common['budget']['B8'] and int(row['max_rounds']) == common['training']['maximum_rounds'],
                'original VAL slot identity/parameters differ')
        job = {'job_id': row['job_id'], 'phase': 'VAL', 'slot': slot, 'selection_group': row['selection_group'],
               'recipe_id': recipe['recipe_id'], 'seed': seed, 'hardware_family': 'RTX4090D',
               'method': 'fedsb' if recipe['server_rule'] == 'fedsb_r32' else 'full_both',
               'protocol': common['protocol'], 'eval_role': 'recipe_dev', 'representation': 'N',
               'period': recipe['period'], 'xi': 0.0, 'rank': recipe['rank'], 'lr': recipe['client_lr'],
               'head_lr': recipe['head_lr'], 'min_rounds': common['training']['minimum_research_rounds'],
               'max_rounds': int(row['max_rounds']), 'priority': 20,
               'output_dir': str(Path(output_root) / row['job_id'])}
        runtime = {'byte_cap': common['budget']['B8'], 'clients_per_round': common['federation']['clients_per_round'],
                   'batch_size': common['training']['batch_size'], 'local_epochs': common['training']['local_epochs'],
                   'betas': common['training']['betas'], 'eps': common['training']['eps'],
                   'weight_decay': common['training']['weight_decay'], 'eval_batch_size': common['controller']['eval_batch_size'],
                   'min_rounds': job['min_rounds'], 'max_rounds': job['max_rounds'], 'server_rule': recipe['server_rule'],
                   'period': recipe['period'], 'public_training': recipe['public_training'], 'controller': common['controller']}
        for key in ('capacity_by_tier', 'probe_batch_size', 'probe_batches'):
            require(key in loader['runtime'], f'original loader runtime missing: {key}')
            runtime[key] = copy.deepcopy(loader['runtime'][key])
        item = {'schema': 'pcla-v8-expanded-1', 'job': job, 'recipe': copy.deepcopy(recipe), 'common': copy.deepcopy(common),
                **{k: str(path) for k, path in paths.items()}, 'source_manifest': str(source_path),
                'model': {'repository': str(model_path), 'revision': None, 'local_files_only': True,
                          'max_length': common['model']['max_length'], 'dropout': common['model']['lora_dropout']},
                'runtime': runtime, 'validation_slot': copy.deepcopy(row), 'selection': copy.deepcopy(winner),
                'binding': {'status': 'bound_unverified', 'data_manifest_sha256': expected_input_sha256['manifest'],
                            'b8_wire_sha256': expected_input_sha256['b8_wire'],
                            'token_cache_sha256': expected_input_sha256['token_cache'],
                            'token_cache_metadata_sha256': expected_input_sha256['token_cache_metadata'],
                            'train_source_sha256': expected_input_sha256['train_file'],
                            'source_manifest_sha256': source_sha, 'source_file_hashes': source_records[source_key]['files'],
                            'common_sha256': PRO_SHA['configs/common.json'], 'recipes_sha256': PRO_SHA['configs/recipes.json'],
                            'validation_slots_sha256': PRO_SHA['jobs/validation_slots.csv'],
                            'phase_plan_sha256': PRO_SHA['configs/phase_plan.json'],
                            'scope_authorization_sha256': expected_input_sha256['scope_authorization'],
                            'loader_template_sha256': expected_input_sha256['loader_template'],
                            'selection_sha256': selection_sha256, 'accepted_dev_snapshot_sha256': accepted_snapshot_sha256,
                            'model_revision': common['model']['revision'], 'model_files_sha256': copy.deepcopy(model_files_sha256),
                            'token_cache_identity_sha256': metadata['identity_sha256'],
                            'original_hardware_family': row['hardware_family'], 'approved_hardware_override': 'all seeds all methods use4090'},
                'launch_status': 'blocked_pending_VAL_integration_and_audit'}
        configs.append(item)
        logical_slots.append({'job_id': row['job_id'], 'slot': slot, 'seed': seed,
                              'selection_group': row['selection_group'], 'recipe_id': recipe['recipe_id']})
    canonical, aliases = deduplicate_exact_configs(configs)
    for slot in logical_slots:
        slot['canonical_job_id'] = aliases[slot['job_id']]
    require(len(logical_slots) == 108 and len(aliases) == 108 and len(canonical) <= 108 and
            len({c['job']['output_dir'] for c in canonical}) == len(canonical), 'compiled VAL coverage/output collision')
    return {'schema': 'pcla-v8-validation-compilation-1', 'status': 'compiled_pending_integration_and_audit',
            'logical_slots': logical_slots, 'configs': canonical, 'alias_map': aliases,
            'logical_jobs': 108, 'execution_jobs': len(canonical), 'alias_jobs': 108 - len(canonical),
            'selection_sha256': selection_sha256, 'accepted_dev_snapshot_sha256': accepted_snapshot_sha256,
            'selection_performed': False, 'queue_initialized': False, 'launch_performed': False}
