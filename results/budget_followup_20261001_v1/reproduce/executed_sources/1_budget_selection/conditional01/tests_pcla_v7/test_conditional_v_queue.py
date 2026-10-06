"""Pure-CPU conditional queue tests; all queue writes are isolated in tmp_path."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import tempfile
import unittest

from scripts import conditional_v_queue as q
from scripts.audit_pcla_v7_conditional_queue import audit_pending
from scripts.audit_pcla_v7_conditional_run import audit as audit_run
from scripts.dispatch_pcla_v7 import read, sha, write

RUN = Path('$FYP_ROOT/10_projects/pcla/runs/pcla-global-budget-20260928-v7')
INPUT = RUN / 'checks/v_conditional_configuration_20260928/accepted_01'
COMPILATION = INPUT / 'compilation.json'
MANIFEST = INPUT / 'conditional_jobs.jsonl'
PLAN = RUN / 'checks/cost_independent_audit_20260928/accepted_01/decision_after_cost/V_JOBS.jsonl'
DRAFT = INPUT / 'draft_configs'


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)


def _fixture(tmp_path):
    root = tmp_path / q.PROTOCOL
    root.mkdir()
    q.initialize(root, COMPILATION, MANIFEST, PLAN, DRAFT)
    source = tmp_path / 'source'
    names = ['pcla_research_v7/runtime.py', 'pcla_research_v7/conditional_replay.py',
             'scripts/run_pcla_v7.py', 'scripts/conditional_v_queue.py',
             'scripts/audit_pcla_v7_conditional_run.py',
             'scripts/audit_pcla_v7_conditional_queue.py',
             'scripts/audit_pcla_v7_run.py', 'scripts/dispatch_pcla_v7.py']
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    _put(source / 'SOURCE_MANIFEST.json', {'files': {name: sha(source / name) for name in names}})
    config_dir = tmp_path / 'released_configs'
    config_dir.mkdir()
    technical = tmp_path / 'technical.json'
    release = tmp_path / 'release.json'
    return root, source, config_dir, technical, release


def _released(draft_path, source, *, binding=None):
    cfg = read(draft_path)
    cfg['job']['status'] = 'released'
    cfg['conditional_validation']['release_state'] = 'released'
    cfg['conditional_validation']['unmet_dependencies'] = []
    cfg['source_manifest'] = str((source / 'SOURCE_MANIFEST.json').resolve())
    if binding is not None:
        cfg['quota_replay'] = {'schema': 'pcla-v7-quota-replay-config-1',
                               'binding_path': str(binding.resolve()),
                               'binding_sha256': sha(binding)}
    cfg['expanded_sha256'] = q.canonical_hash({k: v for k, v in cfg.items()
                                                if k != 'expanded_sha256'})
    return cfg


def _gates(root, source, config_dir, technical, release):
    hashes = {p.stem: sha(p) for p in config_dir.glob('*.json')}
    _put(technical, {'schema': 'pcla-v7-conditional-technical-gate-1',
                     'protocol': q.PROTOCOL, 'phase': 'V',
                     'status': 'passed', 'source_manifest_sha256': sha(source / 'SOURCE_MANIFEST.json'),
                     'config_sha256_by_job': hashes})
    _put(release, {'schema': 'pcla-v7-conditional-release-gates-1',
                   'protocol': q.PROTOCOL, 'phase': 'V', 'job_count': 36,
                   'matrix_sha256': sha(root / q.QUEUE_NAME / 'matrix_identity.json'),
                   'review_gate': 'approved', 'configuration_gate': 'passed',
                   'source_gate': 'passed',
                   'technical_receipt_sha256': sha(technical),
                   'source_manifest_sha256': sha(source / 'SOURCE_MANIFEST.json'),
                   'config_sha256_by_job': hashes})


def _release(root, source, config_dir, technical, receipt, ids, family='4090D'):
    return q.release(root / q.QUEUE_NAME, ids, config_dir, source,
                     receipt, sha(receipt), technical, sha(technical), family)


def _claim_once(args):
    queue, family, worker = args
    claimed = q.claim(Path(queue), family, worker)
    return claimed[1]['job']['job_id'] if claimed else None


def test_real_plan_is_exact_36_and_hardware_map_without_queue_write(tmp_path):
    identity, jobs = q.compile_matrix(COMPILATION, MANIFEST, PLAN, DRAFT)
    assert len(jobs) == identity['count'] == 36
    assert {v['slot'] for v in jobs.values()} == {3, 4, 16}
    assert sum(v['pool'] == 'A100' for v in jobs.values()) == 6
    assert not (tmp_path / q.QUEUE_NAME).exists()


def test_initializer_is_idempotent_and_keeps_fixed_queue_unmodified(tmp_path):
    root = tmp_path / q.PROTOCOL
    root.mkdir()
    fixed = root / 'validation_queue'
    _put(fixed / 'matrix_identity.json', {'count': 156})
    before = (fixed / 'matrix_identity.json').read_bytes()
    first = q.initialize(root, COMPILATION, MANIFEST, PLAN, DRAFT)
    assert q.initialize(root, COMPILATION, MANIFEST, PLAN, DRAFT) == first
    assert len(list((root / q.QUEUE_NAME / 'jobs').glob('*.json'))) == 36
    assert (fixed / 'matrix_identity.json').read_bytes() == before
    bad = root / q.QUEUE_NAME / 'jobs/V_S03_D035_s101.json'
    item = read(bad)
    item['pool'] = 'A100'
    write(bad, item)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'identity differs'):
        q.initialize(root, COMPILATION, MANIFEST, PLAN, DRAFT)


def test_local_and_cyclic_release_first_with_unique_locked_claim(tmp_path):
    root, source, configs, tech, receipt = _fixture(tmp_path)
    ids = [f'V_S{slot:02d}_{recipe}_s{seed}' for seed in q.SEEDS
           for slot, recipe in ((3, 'D035'), (4, 'D042'))]
    for jid in ids:
        write(configs / f'{jid}.json', _released(DRAFT / f'{jid}.json', source))
    _gates(root, source, configs, tech, receipt)
    family_ids = [jid for jid in ids if jid.endswith('_s281') or jid.endswith('_s307')]
    _release(root, source, configs, tech, receipt, family_ids, 'A100')
    _release(root, source, configs, tech, receipt,
             [jid for jid in ids if jid not in family_ids])
    status = q.status(root / q.QUEUE_NAME)
    assert status['counts'] == {'pending': 24, 'blocked': 12}
    with multiprocessing.get_context('fork').Pool(2) as pool:
        claimed = pool.map(_claim_once, [(str(root / q.QUEUE_NAME), '4090D', f'worker-{i}')
                                         for i in (1, 2)])
    assert len(set(claimed)) == 2
    assert all(jid.startswith('V_S03_D035_') for jid in claimed)
    assert q.claim(root / q.QUEUE_NAME, 'A100', 'worker-c')[1]['job']['seed'] in (281, 307)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'unique'):
        _release(root, source, configs, tech, receipt, [ids[0], ids[0]])


def _donor_originals(root, source, donor_cfg, donor_id, tmp_path):
    out = root / 'V' / donor_id
    out.mkdir(parents=True)
    data = tmp_path / 'data_manifest.json'
    _put(data, {'role': 'synthetic'})
    files = {'donor_config': donor_cfg, 'source_manifest': source / 'SOURCE_MANIFEST.json',
             'data_manifest': data, 'metrics': out / 'metrics.json',
             'initial_identity': out / 'initial_state_identity.json',
             'run_identity': out / 'run_identity.json', 'status': out / 'status.json',
             'donor_quota_seal': out / 'donor_quota_seal.json'}
    _put(files['metrics'], [{'round': round_id, 'selected_clients': list(range(10)),
                             'support_trace': {}} for round_id in range(1, 101)])
    _put(files['initial_identity'], {'initial_state_sha256': 'd' * 64})
    _put(files['run_identity'], {'job': read(donor_cfg)['job']})
    _put(files['status'], {'state': 'completed', 'completed_round': 100})
    _put(files['donor_quota_seal'], {'job_id': donor_id, 'completed_round': 100,
        'identities': {'raw_metrics_file_sha256': sha(files['metrics']),
                       'donor_config_file_sha256': sha(donor_cfg)}})
    binding = tmp_path / 'binding.json'
    _put(binding, {'schema': 'pcla-v7-quota-replay-binding-1',
                   'files': {key: {'path': str(path.resolve()), 'sha256': sha(path)}
                             for key, path in files.items()}})
    return binding, files


def test_replay_requires_same_seed_audited_donor_and_immutable_binding(tmp_path):
    root, source, configs, tech, receipt = _fixture(tmp_path)
    donor_id, replay_id = 'V_S03_D035_s101', 'V_S16_D035_s101'
    donor_cfg = configs / f'{donor_id}.json'
    write(donor_cfg, _released(DRAFT / f'{donor_id}.json', source))
    _gates(root, source, configs, tech, receipt)
    _release(root, source, configs, tech, receipt, [donor_id])
    binding, files = _donor_originals(root, source, donor_cfg, donor_id, tmp_path)
    write(configs / f'{replay_id}.json',
          _released(DRAFT / f'{replay_id}.json', source, binding=binding))
    _gates(root, source, configs, tech, receipt)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'not audited completed'):
        _release(root, source, configs, tech, receipt, [replay_id])
    donor_path = root / q.QUEUE_NAME / 'jobs' / f'{donor_id}.json'
    donor = read(donor_path)
    report = tmp_path / 'donor_audit.json'
    _put(report, {'ok': True, 'errors': [], 'checks': {'planned_job_id': donor_id}})
    donor.update(state='completed', audit={'report': str(report), 'report_sha256': sha(report),
        'config_sha256': donor['config_sha256'],
        'source_manifest_sha256': donor['source_manifest_sha256']})
    write(donor_path, donor)
    originals = {key: path.read_bytes() for key, path in files.items()}
    original_binding = binding.read_bytes()
    _put(files['metrics'], read(files['metrics'])[:1])
    _put(files['status'], {'state': 'completed', 'completed_round': 1})
    short_seal = read(files['donor_quota_seal'])
    short_seal['completed_round'] = 1
    short_seal['identities']['raw_metrics_file_sha256'] = sha(files['metrics'])
    _put(files['donor_quota_seal'], short_seal)
    short_binding = read(binding)
    for key in ('metrics', 'status', 'donor_quota_seal'):
        short_binding['files'][key]['sha256'] = sha(files[key])
    _put(binding, short_binding)
    short_cfg = _released(DRAFT / f'{replay_id}.json', source, binding=binding)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'does not cover completed trace'):
        q._donor_ready((root / q.QUEUE_NAME).resolve(),
                       read(root / q.QUEUE_NAME / 'jobs' / f'{replay_id}.json'), short_cfg)
    for key, content in originals.items():
        files[key].write_bytes(content)
    binding.write_bytes(original_binding)
    _release(root, source, configs, tech, receipt, [replay_id])
    claimed = q.claim(root / q.QUEUE_NAME, '4090D', 'worker-a')
    assert claimed[1]['job']['job_id'] == replay_id
    assert read(files['status'])['completed_round'] == 100
    replay_out = Path(claimed[1]['out'])
    replay_out.mkdir(parents=True)
    (replay_out / 'latest.pt').write_bytes(b'completed-prefix-checkpoint')
    (replay_out / 'worker.log').write_text('donor suffix required at replay r101')
    _put(replay_out / 'status.json', {'state': 'needs_donor_suffix',
                                    'requested_round': 101, 'available_through': 100})
    assert q.finish(root / q.QUEUE_NAME, claimed[0], 'worker-a', 1) == 'blocked_donor_suffix'
    blocked = read(claimed[0])
    assert blocked['state'] == 'blocked_donor_suffix' and len(blocked['attempts']) == 1
    assert read(replay_out / 'status.json')['requested_round'] == 101
    assert (replay_out / 'latest.pt').read_bytes() == b'completed-prefix-checkpoint'
    assert 'r101' in (replay_out / 'worker.log').read_text()
    files['metrics'].write_text(files['metrics'].read_text() + '\n')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'original SHA differs'):
        q._donor_ready(root / q.QUEUE_NAME, claimed[1], read(configs / f'{replay_id}.json'))


def test_completed_100_round_donor_releases_prefix_and_r101_stays_blocked(tmp_path):
    root, source, configs, tech, receipt = _fixture(tmp_path)
    donor_id, replay_id = 'V_S03_D035_s101', 'V_S16_D035_s101'
    donor_cfg = configs / f'{donor_id}.json'
    write(donor_cfg, _released(DRAFT / f'{donor_id}.json', source))
    _gates(root, source, configs, tech, receipt)
    _release(root, source, configs, tech, receipt, [donor_id])
    binding, files = _donor_originals(root, source, donor_cfg, donor_id, tmp_path)
    write(configs / f'{replay_id}.json',
          _released(DRAFT / f'{replay_id}.json', source, binding=binding))
    _gates(root, source, configs, tech, receipt)
    donor_path = root / q.QUEUE_NAME / 'jobs' / f'{donor_id}.json'
    donor = read(donor_path)
    report = tmp_path / 'donor_audit.json'
    _put(report, {'ok': True, 'errors': [], 'checks': {'planned_job_id': donor_id}})
    donor.update(state='completed', audit={'report': str(report), 'report_sha256': sha(report),
        'config_sha256': donor['config_sha256'],
        'source_manifest_sha256': donor['source_manifest_sha256']})
    write(donor_path, donor)
    assert read(files['status'])['completed_round'] == 100
    assert _release(root, source, configs, tech, receipt, [replay_id]) == [replay_id]
    path, claimed = q.claim(root / q.QUEUE_NAME, '4090D', 'prefix-worker')
    out = Path(claimed['out'])
    out.mkdir(parents=True)
    (out / 'latest.pt').write_bytes(b'prefix')
    _put(out / 'status.json', {'state': 'needs_donor_suffix',
                              'requested_round': 101, 'available_through': 100})
    assert q.finish(root / q.QUEUE_NAME, path, 'prefix-worker', 1) == 'blocked_donor_suffix'
    assert read(path)['state'] == 'blocked_donor_suffix'
    assert read(out / 'status.json')['requested_round'] == 101
    assert (out / 'latest.pt').read_bytes() == b'prefix'
    assert q.claim(root / q.QUEUE_NAME, '4090D', 'another-worker') is None


def test_failed_attempt_preserves_original_and_manual_resume_needs_receipt(tmp_path):
    root, source, configs, tech, receipt = _fixture(tmp_path)
    jid = 'V_S03_D035_s101'
    write(configs / f'{jid}.json', _released(DRAFT / f'{jid}.json', source))
    _gates(root, source, configs, tech, receipt)
    _release(root, source, configs, tech, receipt, [jid])
    path, item = q.claim(root / q.QUEUE_NAME, '4090D', 'worker-a')
    out = Path(item['out'])
    out.mkdir(parents=True)
    (out / 'worker.log').write_text('original failure')
    (out / 'latest.pt').write_bytes(b'checkpoint')
    assert q.finish(root / q.QUEUE_NAME, path, 'worker-a', 7) == 'failed_needs_review'
    review = tmp_path / 'resume_review.json'
    _put(review, {'schema': 'pcla-v7-conditional-resume-review-1',
                  'job_id': jid, 'decision': 'resume_same_identity'})
    with unittest.TestCase().assertRaisesRegex(ValueError, 'SHA differs'):
        q.resume(root / q.QUEUE_NAME, jid, review, '0' * 64)
    q.resume(root / q.QUEUE_NAME, jid, review, sha(review))
    resumed = q.claim(root / q.QUEUE_NAME, '4090D', 'worker-b')
    assert resumed[1]['resume_requested'] is True
    assert len(resumed[1]['attempts']) == 2
    assert (out / 'worker.log').read_text() == 'original failure'


def test_queue_auditor_retains_failure_report_and_never_marks_bad_run_complete(tmp_path):
    root, source, configs, tech, receipt = _fixture(tmp_path)
    jid = 'V_S03_D035_s101'
    write(configs / f'{jid}.json', _released(DRAFT / f'{jid}.json', source))
    _gates(root, source, configs, tech, receipt)
    _release(root, source, configs, tech, receipt, [jid])
    path = root / q.QUEUE_NAME / 'jobs' / f'{jid}.json'
    item = read(path)
    item['state'] = 'executed_pending_audit'
    write(path, item)
    def fake_run(argv, **kwargs):
        _put(Path(argv[argv.index('--out') + 1]), {'ok': False,
            'errors': ['B8 return differs'], 'run_dir': item['out'],
            'checks': {'planned_job_id': jid, 'conditional_slot': 3}})
        return type('Result', (), {'returncode': 1, 'stderr': ''})()
    with patch('scripts.audit_pcla_v7_conditional_queue.subprocess.run', side_effect=fake_run):
        result = audit_pending(root / q.QUEUE_NAME, [jid])
    assert result[0]['state'] == 'audit_failed_needs_review'
    saved = read(path)
    assert Path(saved['audit']['report']).is_file()
    assert sha(saved['audit']['report']) == saved['audit']['report_sha256']


def test_auditor_and_release_require_actual_frozen_source_root(tmp_path):
    result = audit_run(tmp_path / 'config.json', tmp_path / 'run', tmp_path / 'other_source')
    assert result['ok'] is False and 'execution root differs' in result['errors'][0]
    root, source, configs, tech, receipt = _fixture(tmp_path)
    jid = 'V_S03_D035_s101'
    write(configs / f'{jid}.json', _released(DRAFT / f'{jid}.json', source))
    _gates(root, source, configs, tech, receipt)
    script = source / 'scripts/run_pcla_v7.py'
    target = tmp_path / 'mutable_runner.py'
    target.write_bytes(script.read_bytes())
    script.unlink()
    script.symlink_to(target)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'source closure contains symlink'):
        _release(root, source, configs, tech, receipt, [jid])


class ConditionalQueueTests(unittest.TestCase):
    def _run_case(self, function):
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_plan(self):
        self._run_case(test_real_plan_is_exact_36_and_hardware_map_without_queue_write)

    def test_init(self):
        self._run_case(test_initializer_is_idempotent_and_keeps_fixed_queue_unmodified)

    def test_release_claim(self):
        self._run_case(test_local_and_cyclic_release_first_with_unique_locked_claim)

    def test_replay_block(self):
        self._run_case(test_replay_requires_same_seed_audited_donor_and_immutable_binding)

    def test_completed_prefix(self):
        self._run_case(test_completed_100_round_donor_releases_prefix_and_r101_stays_blocked)

    def test_resume(self):
        self._run_case(test_failed_attempt_preserves_original_and_manual_resume_needs_receipt)

    def test_audit(self):
        self._run_case(test_queue_auditor_retains_failure_report_and_never_marks_bad_run_complete)

    def test_source_root(self):
        self._run_case(test_auditor_and_release_require_actual_frozen_source_root)


if __name__ == '__main__':
    unittest.main()
