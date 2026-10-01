"""Independent fixture tests for diagnostic export gates and raw-value preservation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostic_export_under_test', HERE/'export_diagnostics.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


def raw(path, value=b'fixture bytes'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        source = self.root/'sources/diagnostic03'
        src = raw(source/'module.py', b'fixed source')
        self.source_sha = mod.digest(put(source/'SOURCE_MANIFEST.json',
            {'files': {'module.py': mod.digest(src)}}))
        self.ids = ['Cohort_s101_full_alt_r10_a', 'Direction_s101_r10_a']
        self.config_sha = {}
        self.stage = None
        for line, jid in zip(('cohort', 'direction'), self.ids):
            self._object(line, jid)
        base = self.root/'checks/diagnostic_ready01'
        frozen_csv = self.root/'checks/pro_received_20260928/jobs/cohort_diagnostics_144.csv'
        frozen_csv.parent.mkdir(parents=True, exist_ok=True)
        frozen_csv.write_text('job_id\n' + '\n'.join(self.ids) + '\n')
        self.csv_sha = mod.digest(frozen_csv)
        hardware_map = put(self.root/'checks/v_execution_intake_20260928/seed_hardware_map.json',
                           {'101': '4090D'})
        matrix = put(self.root/'diagnostic_queue/matrix_identity.json',
                     {'count': 2, 'source_manifest_sha256': self.source_sha,
                      'config_sha256_by_job': self.config_sha})
        compile_record = put(base/'compilation.json',
                             {'count': 2, 'manifest_sha256': 'd'*64,
                              'csv_sha256': mod.digest(frozen_csv),
                              'hardware_map_sha256': mod.digest(hardware_map),
                              'config_sha256': {f'{k}.json': v for k,v in self.config_sha.items()}})
        put(base/'READY_RECEIPT.json',
            {'count': 2, 'config_sha256_by_job': self.config_sha,
             'input_sha256': {'diagnostic03_source_manifest': self.source_sha,
                              'data_manifest': 'd'*64,
                              'diagnostic_csv': mod.digest(frozen_csv),
                              'seed_hardware_map': mod.digest(hardware_map),
                              'compilation_receipt': mod.digest(compile_record),
                              'queue_matrix_identity': mod.digest(matrix)}})
        cohort_stage = self.root/'diagnostics'/self.ids[0]/'bank_stage'
        self.sealed_auxiliary = {
            'jq_lock': mod.read(cohort_stage/'jq_pretraining_lock.json'),
            'prediction': mod.read(cohort_stage/'prediction_lock.json')['prediction'],
            'selection': {k: v for k,v in mod.read(cohort_stage/'selection_lock.json').items()
                          if k != 'file_sha256'},
        }

    def _object(self, line, jid):
        r = self.root
        anchor = ('a' if line == 'cohort' else 'b')*64
        config = {'job_id': jid, 'line': line, 'seed': 101, 'factor': 'a',
                  'anchor_round': 10, 'participants': list(range(10)),
                  'hardware_family': '4090D',
                  'frozen_data_manifest_sha256': 'd'*64,
                  'parent': {'job_id': 'V_S10_D072_s101', 'recipe_id': 'D072',
                             'anchor_check': {'anchor_sha256': anchor,
                                              'checkpoint_identity': 'checkpoint'}},
                  'calls': {'local_calls': 40 if line == 'cohort' else 120,
                            'optimizer_steps': 640 if line == 'cohort' else 1920},
                  'training': {'learning_rate': .001 if line == 'cohort' else .0003}}
        cfg = put(r/'checks/diagnostic_ready01/diagnostic_configs'/f'{jid}.json', config)
        self.config_sha[jid] = mod.digest(cfg)
        out = r/'diagnostics'/jid
        put(out/'anchor.json', {'anchor_sha256': anchor,
                                'checkpoint_identity': 'checkpoint'})
        put(out/'attempt_1.json', {'source_sha256': self.source_sha,
                                   'config_sha256': self.config_sha[jid],
                                   'state': 'failed', 'started_at': 1, 'ended_at': 2})
        put(out/'attempt_2.json', {'source_sha256': self.source_sha,
                                   'config_sha256': self.config_sha[jid],
                                   'state': 'finished', 'started_at': 3, 'ended_at': 4})
        hashes = {'config': self.config_sha[jid], 'source_manifest': self.source_sha,
                  'anchor': anchor, 'frozen_data_manifest': 'd'*64}
        if line == 'cohort':
            hashes.update(self._cohort_artifacts(out, jid, anchor))
        else:
            hashes.update(self._direction_artifacts(out, jid))
        audit = {'status': 'audit_pass_candidate', 'object_id': jid, 'line': line,
                 'errors': [], 'warnings': [] if line == 'cohort' else [mod.WARNING],
                 'auditor_sha256': mod.AUDITOR_SHA, 'anchor_sha256': anchor,
                 'hashes': hashes,
                 'coverage': {'local_calls': config['calls']['local_calls'],
                              'optimizer_steps': config['calls']['optimizer_steps'],
                              'snapshots_per_trajectory': 3},
                 'role_counts': {'client_train': 2560, 'client_local_readout': 640,
                                 'public_training': 512, 'selection': 1024, 'readout': 2048}}
        audit_path = put(r/'diagnostic_queue/audits'/f'{jid}.json', audit)
        acceptance = {'decision': 'accepted', 'job_id': jid, 'audit_path': str(audit_path),
                      'audit_sha256': mod.digest(audit_path),
                      'config_sha256': self.config_sha[jid],
                      'source_manifest_sha256': self.source_sha,
                      'anchor_sha256': anchor}
        if line == 'direction':
            acceptance['warnings_preserved'] = [mod.WARNING]
        acceptance_path = put(r/'checks/accepted'/f'{jid}.json', acceptance)
        job = {'job_id': jid, 'state': 'completed', 'seed': 101,
               'hardware_family': '4090D', 'config_sha256': self.config_sha[jid],
               'source_manifest_sha256': self.source_sha, 'anchor_sha256': anchor,
               'attempts': [{'returncode': 1, 'cpu_threads': 4, 'cost_bytes': 17},
                            {'returncode': 0, 'cpu_threads': 4, 'cost_bytes': 19}],
               'audit': {'receipt': str(audit_path), 'receipt_sha256': mod.digest(audit_path),
                         'auditor_sha256': mod.AUDITOR_SHA,
                         'status': 'audit_pass_candidate', 'returncode': 0},
               'acceptance': {'receipt': str(acceptance_path),
                              'receipt_sha256': mod.digest(acceptance_path)}}
        put(r/'diagnostic_queue/jobs'/f'{jid}.json', job)

    def _cohort_artifacts(self, out, jid, anchor):
        stage = out/'bank_stage'
        artifacts = {}
        for path, key in [('supports.pt','supports_pt'),
                          ('bank_stage/prediction_lock.pt','prediction_lock_pt'),
                          ('bank_stage/bank.pt','bank_pt'),
                          ('bank_stage/selection_lock.pt','selection_lock_pt')]:
            artifacts[key] = mod.digest(raw(out/path, path.encode()))
        prefix_values = {}
        for choice in mod.COHORT_CHOICES:
            prefixes = (16,) if choice == 'finite_bank_search' else (1,4,16)
            prefix_values[choice] = {}
            for prefix in prefixes:
                na = choice == 'local' and prefix == 1
                prefix_values[choice][str(prefix)] = {
                    'local64_benefit_by_client': {str(i): .1 for i in range(10)},
                    'weighted_readout2048_benefit_by_client': {str(i): .01 for i in range(10)},
                    'aggregate_readout2048_benefit': .2, 'nonadditivity': .1,
                    'head_only_readout2048_benefit': .02,
                    'lora_only_readout2048_benefit': .15,
                    'cancellation': None if na else .5,
                    'cancellation_status': 'NA_zero_denominator' if na else 'measured',
                    'numerator_effective_norm_sq': 0 if na else .1,
                    'denominator_mean_effective_norm_sq': 0 if na else .2,
                    'max_effective_linearity_relative_error': 0.0}
        result = {'object_id': jid, 'anchor_sha256': anchor,
                  'readout2048_source_id': 'source_id',
                  'local64_source_ids': {str(i): 'source_id' for i in range(10)},
                  'parent_readout2048_loss': .5,
                  'prediction_lock_sha256': artifacts['prediction_lock_pt'],
                  'selection_lock_sha256': 's'*64,
                  'values': prefix_values,
                  'full_gradient_choice_readout': {'actual_bank_benefit': .2}}
        result_path = put(stage/'bank_readout_123.json', result)
        artifacts['readout_json'] = mod.digest(result_path)
        put(stage/'bank_readout_stage.json',
            {'result_path': str(result_path), 'result_sha256': artifacts['readout_json'],
             'bank_sha256': artifacts['bank_pt']})
        put(stage/'bank_receipt.json', {'sha256': artifacts['bank_pt'], 'seconds': 3,
                                        'local_calls': 40, 'optimizer_steps': 640})
        put(out/'supports.json', {'sha256': artifacts['supports_pt'], 'seconds': 2})
        put(stage/'prediction_lock.json', {'sha256': artifacts['prediction_lock_pt'],
                                            'prediction': {'spearman_mean_defined': None}})
        put(stage/'selection_lock.json', {'file_sha256': artifacts['selection_lock_pt'],
                                          'sha256': 's'*64, 'choices': [], 'evaluations': 1})
        put(stage/'jq_pretraining_lock.json', {'sha256': 'j'*64,
                                               'locked_at_utc': '2026-09-29T00:00:00Z'})
        put(stage/'bank_stage_identity.json', {'jq_lock_sha256': 'j'*64})
        put(stage/'evaluations_123.json', [{'role': 'readout2048', 'rows': 2048,
                                             'loss': .3, 'seconds': .5}])
        return artifacts

    def _direction_artifacts(self, out, jid):
        bank_path = raw(out/'direction_bank.pt', b'direction fixture bank')
        self.stage = out/'direction_stage.json'
        readouts = {}
        for label in mod.DIRECTION_LABELS:
            readouts[label] = {}
            for policy in ('local','cyclic'):
                readouts[label][policy] = {}
                for prefix in (1,4,16):
                    na = label == 'N' and policy == 'local' and prefix == 1
                    readouts[label][policy][str(prefix)] = {
                        'original_loss': .3, 'original_benefit': .2,
                        'normmatched_loss': None if na else .31,
                        'normmatched_benefit': None if na else .19,
                        'normmatch_status': 'NA_zero_candidate' if na else 'matched',
                        'normmatch_multiplier': None if na else 1.0,
                        'reference_n_effective_norm': 1.0,
                        'candidate_effective_norm': 0.0 if na else 1.0}
        stage = {'identity': {'object_id': jid, 'config_sha256': self.config_sha[jid],
                              'source_sha256': self.source_sha},
                 'bank_sha256': mod.digest(bank_path), 'local_calls': 120,
                 'optimizer_steps': 1920, 'seconds': 4.0, 'readouts': readouts}
        put(self.stage, stage)
        put(out/'stage_status.json', stage)
        return {'direction_bank_pt': stage['bank_sha256']}

    def fake_bank(self):
        stage = mod.read(self.stage)
        jid = self.ids[1]
        modules = [f'module{i}' for i in range(24)]
        representations = {}
        for label in mod.DIRECTION_LABELS:
            representations[label] = {
                'active': label != 'S_zero_removed', 'parent_state_sha256': 'p'*64,
                'numerical_ranks': {m: 1 for m in modules},
                'rank_thresholds': {m: .1 for m in modules},
                'relative_errors': {m: 0.0 for m in modules},
                'random_rotations': {}}
        return {'schema': 'pcla-v7-direction-diagnostic-bank-1', 'object_id': jid,
                'anchor_sha256': 'b'*64, 'factor': 'a', 'participants': list(range(10)),
                'local_calls': 120, 'optimizer_steps': 1920,
                'readouts': stage['readouts'], 'parent_readout2048_loss': .5,
                'actual_visits': {}, 'capacities': {},
                'training_seed_by_client': {}, 'selection_seed_by_client': {},
                'probe_seed_by_client': {},
                't_probe': {'cbar_sha256': 'c'*64, 'full_gradients_sha256': 'f'*64,
                            'examples_by_client': {}, 'seed_by_client': {}},
                'representations': representations,
                't_full_h32_comparator': {
                    'status': 'diagnostic_only_no_extra_training',
                    'prediction_kind': 'fixed_masks', 'full_cbar_sha256': 'f'*64,
                    'h32_cbar_sha256': 'c'*64,
                    'rotations_and_grams': {m: {
                        'gram_frobenius_relative_error': None,
                        'gram_frobenius_absolute_error': 0.0,
                        'column_absolute_cosine_descriptive_only': [1.0],
                        'o_full_untrained': [[1.0]], 'o_h32_deployed': [[1.0]],
                        'full_gram': [[1.0]], 'h32_gram': [[1.0]]} for m in modules},
                    'one_step_linear_gain': {p: {'full': 1.0, 'h32': .9,
                        'by_client': {i: {'full': .1, 'h32': .09} for i in range(10)}}
                        for p in ('local','cyclic')}}}

    def fake_auxiliary(self, path, _source_root):
        if path.name == 'supports.pt':
            return {'jq_lock': copy.deepcopy(self.sealed_auxiliary['jq_lock']),
                    'object_id': self.ids[0], 'anchor_sha256': 'a'*64}
        if path.name == 'prediction_lock.pt':
            return copy.deepcopy(self.sealed_auxiliary['prediction'])
        if path.name == 'selection_lock.pt':
            return copy.deepcopy(self.sealed_auxiliary['selection'])
        raise AssertionError(path)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name)/'run')
        self.root = self.fixture.root

    def validate(self):
        return mod.validate_collection(self.root, expected_total=2,
            expected_layout=(1,1), expected_source_sha=self.fixture.source_sha,
            expected_csv_sha=self.fixture.csv_sha,
            load_auxiliary=self.fixture.fake_auxiliary)

    def test_full_gate_json_and_projection_preserve_na_and_failed_attempt(self):
        objects = self.validate()
        rows = mod.json_rows(objects)
        self.assertEqual(len(rows['objects']), 2)
        self.assertEqual(rows['objects'][0]['attempts_all_including_failures'][0]['cost_bytes'], 17)
        self.assertEqual(len(rows['cohort_readouts']), 16)
        self.assertEqual(len(rows['cohort_clients']), 160)
        self.assertEqual(len(rows['direction_readouts']), 36)
        self.assertIsNone(next(r for r in rows['cohort_readouts'] if r['choice']=='local' and r['prefix']==1)['cancellation'])
        self.assertIsNone(next(r for r in rows['direction_readouts'] if r['representation']=='N' and r['policy']=='local' and r['prefix']==1)['normmatched_loss'])
        self.assertIsNone(rows['objects'][1]['direction_readout_source_row_hash'])
        calls = []
        projection = self.root.parent/'projection'
        mod.project_direction(self.root, projection, expected_total=2,
            expected_layout=(1,1), expected_source_sha=self.fixture.source_sha,
            expected_csv_sha=self.fixture.csv_sha,
            load_auxiliary=self.fixture.fake_auxiliary,
            load_bank=lambda path: calls.append(path) or self.fixture.fake_bank())
        self.assertEqual(len(calls), 1)
        self.assertEqual(mod.read(projection/'direction_projection_manifest.json')['object_count'], 1)
        output = self.root.parent/'formal'
        manifest = mod.export_json(self.root, output, projection,
            expected_total=2, expected_layout=(1,1), expected_source_sha=self.fixture.source_sha,
            expected_csv_sha=self.fixture.csv_sha,
            load_auxiliary=self.fixture.fake_auxiliary)
        self.assertEqual(manifest['object_count'], 2)
        self.assertEqual(manifest['tables']['direction_geometry']['sha256'],
                         mod.digest(output/'direction_geometry.jsonl'))
        self.assertTrue((output/'direction_projection_manifest.json').is_file())
        with (output/'objects.jsonl').open() as stream:
            self.assertEqual(sum(1 for _ in stream), 2)

    def test_missing_object_fails_closed(self):
        (self.root/'diagnostic_queue/jobs'/f'{self.fixture.ids[1]}.json').unlink()
        with self.assertRaisesRegex(mod.ExportError, 'missing/extra queue jobs'):
            self.validate()
        self.assertFalse((self.root.parent/'formal').exists())

    def test_unaccepted_object_fails_before_projection(self):
        p = self.root/'diagnostic_queue/jobs'/f'{self.fixture.ids[1]}.json'
        job = mod.read(p)
        job['state'] = 'executed_pending_audit'
        put(p, job)
        with self.assertRaisesRegex(mod.ExportError, 'formal gate incomplete'):
            self.validate()

    def test_legacy_acceptance_location_is_verified(self):
        p = self.root/'diagnostic_queue/jobs'/f'{self.fixture.ids[0]}.json'
        job = mod.read(p)
        job['acceptance'] = {'path': job['acceptance']['receipt'],
                             'sha256': job['acceptance']['receipt_sha256']}
        put(p, job)
        self.assertEqual(len(self.validate()), 2)

    def test_duplicate_manifest_identity_collapses_and_is_rejected(self):
        p = self.root/'diagnostic_queue/matrix_identity.json'
        h = self.fixture.config_sha[self.fixture.ids[0]]
        p.write_text('{"count":2,"source_manifest_sha256":"'+self.fixture.source_sha+'",'
                     '"config_sha256_by_job":{"'+self.fixture.ids[0]+'":"'+h+'",'
                     '"'+self.fixture.ids[0]+'":"'+h+'"}}')
        with self.assertRaisesRegex(mod.ExportError, 'unique ID map differs'):
            self.validate()

    def test_readout_hash_change_rejected(self):
        p = self.root/'diagnostics'/self.fixture.ids[0]/'bank_stage/bank_readout_123.json'
        p.write_text(p.read_text().replace('0.5', '0.6'))
        with self.assertRaisesRegex(mod.ExportError, 'SHA256 mismatch'):
            self.validate()

    def test_scientific_auxiliary_json_change_rejected_against_audited_pt(self):
        stage = self.root/'diagnostics'/self.fixture.ids[0]/'bank_stage'
        for name, mutate in (
            ('prediction_lock.json', lambda x: x['prediction'].update(spearman_mean_defined=.7)),
            ('selection_lock.json', lambda x: x['choices'].append([1, 'local'])),
            ('jq_pretraining_lock.json', lambda x: x.update(locked_at_utc='changed')),
        ):
            with self.subTest(name=name):
                path = stage/name
                original = path.read_bytes()
                value = mod.read(path)
                mutate(value)
                put(path, value)
                with self.assertRaisesRegex(mod.ExportError, 'auxiliary JSON differs from audited PT'):
                    self.validate()
                path.write_bytes(original)

    def test_unsealed_cost_metadata_is_labeled_and_hashed_at_export(self):
        out = self.root/'diagnostics'/self.fixture.ids[0]
        evaluation = out/'bank_stage/evaluations_123.json'
        value = mod.read(evaluation)
        value[0]['seconds'] = 999
        put(evaluation, value)
        attempt = out/'attempt_1.json'
        value = mod.read(attempt)
        value['ended_at'] = 999
        put(attempt, value)
        item = self.validate()[0]
        provenance = item['auxiliary_provenance']
        self.assertEqual(provenance['evaluations_json']['sha256_at_export'], mod.digest(evaluation))
        self.assertIn('current_file_at_export_no_audit_time_sha',
                      provenance['evaluations_json']['evidence_scope'])
        self.assertEqual(provenance['driver_attempts'][0]['sha256_at_export'], mod.digest(attempt))
        self.assertEqual(item['evaluations'][0]['seconds'], 999)
        self.assertEqual(item['driver_attempts'][0]['ended_at'], 999)

    def test_csv_drift_rejected_even_if_generated_receipts_are_updated(self):
        csv_path = self.root/'checks/pro_received_20260928/jobs/cohort_diagnostics_144.csv'
        csv_path.write_text(csv_path.read_text()+'\n')
        compilation = self.root/'checks/diagnostic_ready01/compilation.json'
        ready = self.root/'checks/diagnostic_ready01/READY_RECEIPT.json'
        c = mod.read(compilation)
        c['csv_sha256'] = mod.digest(csv_path)
        put(compilation, c)
        r = mod.read(ready)
        r['input_sha256']['diagnostic_csv'] = mod.digest(csv_path)
        r['input_sha256']['compilation_receipt'] = mod.digest(compilation)
        put(ready, r)
        with self.assertRaisesRegex(mod.ExportError, 'frozen diagnostic CSV identities/hash differ'):
            self.validate()

    def test_projection_hash_change_blocks_formal_export(self):
        projection = self.root.parent/'projection'
        mod.project_direction(self.root, projection, expected_total=2,
            expected_layout=(1,1), expected_source_sha=self.fixture.source_sha,
            expected_csv_sha=self.fixture.csv_sha,
            load_auxiliary=self.fixture.fake_auxiliary,
            load_bank=lambda _: self.fixture.fake_bank())
        p = projection/'direction_geometry.jsonl'
        p.write_text(p.read_text() + '{}\n')
        with self.assertRaisesRegex(mod.ExportError, 'projection identity/hash differs'):
            mod.export_json(self.root, self.root.parent/'formal', projection,
                expected_total=2, expected_layout=(1,1), expected_source_sha=self.fixture.source_sha,
                expected_csv_sha=self.fixture.csv_sha,
                load_auxiliary=self.fixture.fake_auxiliary)
        self.assertFalse((self.root.parent/'formal').exists())


if __name__ == '__main__':
    unittest.main()
