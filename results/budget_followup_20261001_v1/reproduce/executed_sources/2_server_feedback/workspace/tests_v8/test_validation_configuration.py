import copy
import json
import os
from pathlib import Path
import shutil

import pytest

from pcla_research_v8 import validation_configuration as m
from pcla_research_v8.configuration import digest
from pcla_research_v7.data import digest as data_digest

WORKSPACE = Path(__file__).resolve().parents[1]
RUN = WORKSPACE.parents[1]
PACKAGE = RUN.parents[1] / 'handoffs/PCLA_PostV_Received_20260929/attachment/PCLA_PostV_Research_20260929_v8'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


@pytest.fixture
def fixture(tmp_path):
    pack = tmp_path / 'pack'; shutil.copytree(PACKAGE, pack)
    common, recipes, slots = m.read_protocol(pack)
    selection = {'status': 'ready_for_independent_review', 'blocked': [], 'dev_count': 128,
                 'val_jobs_max': 108, 'hardware': 'RTX4090D', 'launch_performed': False,
                 'selection_performed': True, 'val_seeds': list(m.VAL_SEEDS),
                 'recipe_source_sha256': m.PRO_SHA['configs/recipes.json'],
                 'slot_source_sha256': m.PRO_SHA['jobs/validation_slots.csv'],
                 'accepted_snapshot_sha256': 'a' * 64, 'winners': [],
                 'evidence_rows': [{'recipe_id': rid, 'seed': seed, 'state': 'completed',
                     'independent_accepted': True, 'endpoint': 'B8_original_parameters',
                     'samples': 4096, 'correct': 3500, 'artifact_sha256': {'accepted-raw-fixture': 'b' * 64}}
                    for rid in recipes for seed in m.DEV_SEEDS]}
    # Fixed synthetic inputs; never call the winner selector or real collector.
    for slot in range(1, 10):
        row = next(r for r in slots if int(r['slot']) == slot)
        families = row['recipe_rule'][len('best within '):].split(',')
        rid = next(rid for rid, recipe in recipes.items() if recipe['family'] in families)
        selection['winners'].append({'slot': slot, 'group': row['selection_group'], 'recipe_id': rid})
    scope = tmp_path / 'scope_authorization.json'; shutil.copyfile(RUN / 'scope_authorization.json', scope)
    selection['scope_sha256'] = m.sha(scope)
    selection_path = tmp_path / 'selection.json'; put(selection_path, selection)
    train = tmp_path / 'SST-2/train.tsv'; train.parent.mkdir(); train.write_text('sentence\tlabel\nfixture\t0\n')
    model = tmp_path / 'model'; model.mkdir()
    for name, data in {'config.json': b'{}', 'model.safetensors': b'fixture weights', 'vocab.txt': b'fixture vocabulary'}.items():
        (model / name).write_bytes(data)
    model_hash = {p.name: m.sha(p) for p in model.iterdir()}
    all_ids = [f'train:{i}' for i in range(23680)]
    roles = {}; start = 0
    for name, count in [('client_pool', 16000), ('public_training', 512), ('cohort_select', 1024),
                        ('cohort_readout', 2048), ('recipe_dev', 4096)]:
        roles[name] = all_ids[start:start + count]; start += count
    part = {'client_train': {}, 'client_local_readout': {}}
    for client in range(50):
        ids = roles['client_pool'][client * 320:(client + 1) * 320]
        part['client_train'][str(client)] = ids[:256]; part['client_local_readout'][str(client)] = ids[256:]
    manifest = {'source_sha256': m.sha(train), 'split_sha256': 'c' * 64, 'roles': roles,
                'partitions': {str(s): copy.deepcopy(part) for s in m.VAL_SEEDS}}
    manifest_path = tmp_path / 'data/train_roles.json'; put(manifest_path, manifest)
    token = tmp_path / 'tokens.npz'; token.write_bytes(b'existing opaque cached token fixture; never rebuilt')
    identity = {'source_sha256': m.sha(train), 'split_sha256': manifest['split_sha256'], 'max_length': 64,
                'tokenizer_files': {'vocab.txt': model_hash['vocab.txt']}}
    metadata = {'identity': identity, 'identity_sha256': data_digest(identity), 'cache_sha256': m.sha(token),
                'rows': 23680, 'shape': [23680, 64]}
    metadata_path = tmp_path / 'data/train_tokens.json'; put(metadata_path, metadata)
    wire = {'total_bytes': 2382653750, 'codec': 'synthetic contract'}
    wire['wire_manifest_sha256'] = data_digest(wire)
    wire_path = tmp_path / 'b8_wire.json'; put(wire_path, wire)
    paths = {'manifest': manifest_path, 'train_file': train, 'token_cache': token,
             'token_cache_metadata': metadata_path, 'b8_wire': wire_path}
    loader = {**{k: str(v) for k, v in paths.items()},
              'model': {'repository': str(model), 'local_files_only': True, 'max_length': 64, 'dropout': .1},
              'runtime': {'capacity_by_tier': [96, 128, 192], 'probe_batch_size': 16, 'probe_batches': 2}}
    loader_path = tmp_path / 'loader.json'; put(loader_path, loader)
    sources = {}
    for kind in ('core', 'native'):
        directory = tmp_path / kind; directory.mkdir(); code = directory / 'runtime.py'
        code.write_text(f'# synthetic {kind} source; no execution\n')
        path = directory / 'SOURCE_MANIFEST.json'; put(path, {'files': {'runtime.py': m.sha(code)}})
        sources[kind] = path
    pins = {k: m.sha(v) for k, v in paths.items()}
    pins.update(scope_authorization=m.sha(scope), loader_template=m.sha(loader_path),
                source_manifest=m.sha(sources['core']))
    pins['source_manifest:fedsb_r32'] = m.sha(sources['native'])
    args = dict(package_dir=pack, selection_path=selection_path, selection_sha256=m.sha(selection_path),
                accepted_snapshot_sha256='a' * 64, authorization_path=scope,
                loader_template_path=loader_path, source_manifest=sources['core'], output_root=tmp_path / 'outputs',
                expected_input_sha256=pins, model_files_sha256=model_hash,
                source_manifest_by_family={'fedsb_r32': sources['native']})
    return {'args': args, 'selection': selection, 'paths': paths, 'manifest': manifest, 'sources': sources,
            'model': model, 'metadata': metadata, 'loader': loader, 'slots': slots, 'recipes': recipes}


def amend_selection(fixture, change):
    selection = fixture['selection']; change(selection)
    put(fixture['args']['selection_path'], selection)
    fixture['args']['selection_sha256'] = m.sha(fixture['args']['selection_path'])


def test_full_original_matrix_exact_recipe_runtime_and_native_source_binding(fixture):
    before = {p: m.sha(p) for p in fixture['paths'].values()}
    args = fixture['args']; output = m.compile_validation(**args)
    assert output['logical_jobs'] == output['execution_jobs'] == 108 and output['alias_jobs'] == 0
    assert not output['selection_performed'] and not output['queue_initialized'] and not output['launch_performed']
    assert set(output['alias_map']) == {r['job_id'] for r in fixture['slots']}
    for config in output['configs']:
        job = config['job']; winner = next(w for w in fixture['selection']['winners'] if w['slot'] == job['slot'])
        recipe = fixture['recipes'][winner['recipe_id']]
        assert config['recipe'] == recipe
        assert job['seed'] in m.VAL_SEEDS and job['phase'] == 'VAL' and job['hardware_family'] == 'RTX4090D'
        assert job['eval_role'] == 'recipe_dev' and (job['min_rounds'], job['max_rounds']) == (100, 200)
        assert (job['rank'], job['lr'], job['head_lr'], job['period']) == (
            recipe['rank'], recipe['client_lr'], recipe['head_lr'], recipe['period'])
        assert config['runtime']['byte_cap'] == 2382653750 and config['runtime']['batch_size'] == 16
        assert config['runtime']['server_rule'] == recipe['server_rule']
        assert config['runtime']['controller'] == config['common']['controller']
        assert config['binding']['model_files_sha256'] == args['model_files_sha256']
        assert config['binding']['status'] == 'bound_unverified' and 'integration' in config['launch_status']
        expected = fixture['sources']['native' if recipe['family'] == 'fedsb_r32' else 'core']
        assert config['source_manifest'] == str(expected) and config['binding']['source_manifest_sha256'] == m.sha(expected)
        assert config['expanded_sha256'] == digest({k: v for k, v in config.items() if k != 'expanded_sha256'})
    assert {p: m.sha(p) for p in fixture['paths'].values()} == before
    assert not args['output_root'].exists()
    a100_originals = [c for c in output['configs'] if c['job']['seed'] in (659, 661)]
    assert all(c['binding']['original_hardware_family'] == 'A100-PCIE-40GB' for c in a100_originals)
    assert all(c['job']['hardware_family'] == 'RTX4090D' for c in a100_originals)


@pytest.mark.parametrize('change,reason', [
    (lambda s: s.update(status='blocked_timing_evidence'), 'unblocked ready'),
    (lambda s: s['blocked'].append({'slot': 2}), 'unblocked ready'),
    (lambda s: s['winners'].pop(), 'resolved winners'),
    (lambda s: s['evidence_rows'].pop(), 'evidence coverage'),
    (lambda s: s['evidence_rows'].__setitem__(1, s['evidence_rows'][0].copy()), 'evidence coverage'),
    (lambda s: s['evidence_rows'][0].update(state='executed_pending_audit'), 'unaccepted'),
    (lambda s: s['evidence_rows'][0].update(independent_accepted=False), 'unaccepted'),
    (lambda s: s['evidence_rows'][0].update(endpoint='EMA'), 'unaccepted'),
    (lambda s: s.update(accepted_snapshot_sha256='0' * 64), 'root-pinned'),
    (lambda s: s['winners'][0].update(recipe_id='U005'), 'prescribed slot'),
    (lambda s: s.update(val_seeds=[503] * 12), 'unblocked ready'),
])
def test_incomplete_unaccepted_blocked_or_wrong_slot_refused(fixture, change, reason):
    amend_selection(fixture, change)
    with pytest.raises(ValueError, match=reason):
        m.compile_validation(**fixture['args'])


def test_external_selection_hash_and_original_pro_slot_hash_are_mandatory(fixture):
    fixture['args']['selection_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='input SHA differs'):
        m.compile_validation(**fixture['args'])
    fixture['args']['selection_sha256'] = m.sha(fixture['args']['selection_path'])
    slotpath = fixture['args']['package_dir'] / 'jobs/validation_slots.csv'
    slotpath.write_text(slotpath.read_text().replace('2382653750', '1'))
    with pytest.raises(ValueError, match='original VAL slots SHA'):
        m.compile_validation(**fixture['args'])


def test_missing_existing_val_seed_is_error_without_generation(fixture):
    fixture['manifest']['partitions'].pop('661')
    path = fixture['paths']['manifest']; put(path, fixture['manifest'])
    fixture['args']['expected_input_sha256']['manifest'] = m.sha(path)
    with pytest.raises(ValueError, match='existing VAL partition missing/changed: 661'):
        m.compile_validation(**fixture['args'])
    assert '661' not in json.loads(path.read_text())['partitions']


def test_token_cache_mutation_and_model_weight_mutation_refused(fixture):
    fixture['paths']['token_cache'].write_bytes(b'altered token bytes')
    with pytest.raises(ValueError, match='existing token/data/model'):
        m.compile_validation(**fixture['args'])
    fixture['paths']['token_cache'].write_bytes(b'existing opaque cached token fixture; never rebuilt')
    (fixture['model'] / 'model.safetensors').write_bytes(b'altered original weights')
    with pytest.raises(ValueError, match='original model SHA'):
        m.compile_validation(**fixture['args'])


def test_bad_token_split_and_b8_wire_contract_refused(fixture):
    fixture['metadata']['identity']['split_sha256'] = '0' * 64
    fixture['metadata']['identity_sha256'] = data_digest(fixture['metadata']['identity'])
    path = fixture['paths']['token_cache_metadata']; put(path, fixture['metadata'])
    fixture['args']['expected_input_sha256']['token_cache_metadata'] = m.sha(path)
    with pytest.raises(ValueError, match='token/data/model'):
        m.compile_validation(**fixture['args'])
    fixture['metadata']['identity']['split_sha256'] = fixture['manifest']['split_sha256']
    fixture['metadata']['identity_sha256'] = data_digest(fixture['metadata']['identity']); put(path, fixture['metadata'])
    fixture['args']['expected_input_sha256']['token_cache_metadata'] = m.sha(path)
    wire = {'total_bytes': 1}; wire['wire_manifest_sha256'] = data_digest(wire)
    path = fixture['paths']['b8_wire']; put(path, wire); fixture['args']['expected_input_sha256']['b8_wire'] = m.sha(path)
    with pytest.raises(ValueError, match='original B8 wire'):
        m.compile_validation(**fixture['args'])


def test_actual_v8_metadata_identity_uses_authoritative_compact_digest():
    identity = {'max_length': 64,
        'source_sha256': '48eac26399f8456935f81dd9e3a745c1430adb6c62e233ccfe22db4a0d15d1d1',
        'split_sha256': '4636c2b3c94ceaf0b1a855b71ecf2448f894d7f53bf99de34ea28733ca8afb26',
        'tokenizer_files': {
            'tokenizer.json': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
            'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'}}
    assert data_digest(identity) == 'c2ad0474f161915b2408a5a499cf8576c121e5c1e59075792a298f088e70bd41'
    assert digest(identity) == '21d4e825881227d27ed7a471e1935c47a5aa86c46ec84e415685db133c67dbfb'


@pytest.mark.parametrize('field', ['token_cache_metadata', 'b8_wire'])
def test_configuration_digest_cannot_substitute_for_original_data_identity(fixture, field):
    path = fixture['paths'][field]; value = json.loads(path.read_text())
    if field == 'token_cache_metadata':
        value['identity_sha256'] = digest(value['identity'])
        reason = 'existing token/data/model'
    else:
        value['wire_manifest_sha256'] = digest({k: v for k, v in value.items() if k != 'wire_manifest_sha256'})
        reason = 'original B8 wire'
    put(path, value); fixture['args']['expected_input_sha256'][field] = m.sha(path)
    with pytest.raises(ValueError, match=reason):
        m.compile_validation(**fixture['args'])
    assert not fixture['args']['output_root'].exists()


def test_native_val_source_override_and_source_closure_required(fixture):
    args = fixture['args']; args['source_manifest_by_family'] = {}
    with pytest.raises(ValueError, match='native family requires explicit'):
        m.compile_validation(**args)
    args['source_manifest_by_family'] = {'fedsb_r32': fixture['sources']['native']}
    fixture['sources']['native'].with_name('runtime.py').write_text('# changed source\n')
    with pytest.raises(ValueError, match='source closure differs'):
        m.compile_validation(**args)


def test_exact_alias_preserves_two_slots_without_extra_training_or_seed_deduplication():
    base = {'schema': 'pcla-v8-expanded-1', 'job': {'job_id': 'VAL_S01_s601', 'slot': 1,
            'selection_group': 'slot1', 'output_dir': '/output/VAL_S01_s601', 'recipe_id': 'U001', 'seed': 601,
            'rank': 8, 'lr': .001}, 'recipe': {'recipe_id': 'U001', 'lr': .001}, 'runtime': {'byte_cap': 2382653750},
            'binding': {'source_manifest_sha256': 'a' * 64}, 'selection': {'slot': 1}, 'validation_slot': {'slot': '1'}}
    alias = copy.deepcopy(base); alias['job'].update(job_id='VAL_S02_s601', slot=2, selection_group='slot2',
                                                  output_dir='/output/VAL_S02_s601')
    alias['selection']['slot'] = 2; alias['validation_slot']['slot'] = '2'
    other_seed = copy.deepcopy(base); other_seed['job'].update(job_id='VAL_S01_s607', seed=607)
    changed_lr = copy.deepcopy(alias); changed_lr['job'].update(job_id='VAL_S03_s601', lr=.0003)
    configs, aliases = m.deduplicate_exact_configs([base, alias, other_seed, changed_lr])
    assert len(configs) == 3 and aliases['VAL_S02_s601'] == 'VAL_S01_s601'
    assert configs[0]['validation_aliases'] == ['VAL_S01_s601', 'VAL_S02_s601']
    assert aliases['VAL_S01_s607'] != aliases['VAL_S01_s601']
    assert aliases['VAL_S03_s601'] != aliases['VAL_S01_s601']
    assert 'validation_aliases' not in base  # pure compiler does not mutate caller input
