#!/usr/bin/env python3
"""Focused read-only fixture mutations for diagnostic export evidence binding."""
import json
from pathlib import Path
import tempfile

from test_export_diagnostics import Fixture, mod, put


def run_case(name, mutate, extract):
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory) / 'run')
        before = mod.validate_collection(fixture.root, expected_total=2,
            expected_layout=(1, 1), expected_source_sha=fixture.source_sha,
            expected_csv_sha=mod.digest(fixture.root / 'checks/pro_received_20260928/jobs/cohort_diagnostics_144.csv'),
            load_auxiliary=fixture.fake_auxiliary)
        original = extract(mod.json_rows(before))
        mutate(fixture)
        try:
            after = mod.validate_collection(fixture.root, expected_total=2,
                expected_layout=(1, 1), expected_source_sha=fixture.source_sha,
                expected_csv_sha=mod.digest(fixture.root / 'checks/pro_received_20260928/jobs/cohort_diagnostics_144.csv'),
                load_auxiliary=fixture.fake_auxiliary)
            changed = extract(mod.json_rows(after))
            return {'case': name, 'accepted': True, 'exported_value_changed': original != changed,
                    'before': original, 'after': changed}
        except Exception as error:
            return {'case': name, 'accepted': False, 'error': str(error)}


def edit_eval(fixture):
    path = fixture.root / 'diagnostics' / fixture.ids[0] / 'bank_stage/evaluations_123.json'
    value = mod.read(path)
    value[0]['seconds'] = 987.0
    put(path, value)


def edit_prediction(fixture):
    path = fixture.root / 'diagnostics' / fixture.ids[0] / 'bank_stage/prediction_lock.json'
    value = mod.read(path)
    value['prediction']['spearman_mean_defined'] = 0.999
    put(path, value)


def edit_attempt(fixture):
    path = fixture.root / 'diagnostics' / fixture.ids[0] / 'attempt_1.json'
    value = mod.read(path)
    value['ended_at'] = 987
    put(path, value)


cases = [
    run_case('evaluation_seconds', edit_eval,
             lambda rows: rows['cohort_evaluations'][0]['seconds']),
    run_case('prediction_lock', edit_prediction,
             lambda rows: next(x['payload'] for x in rows['cohort_locks']
                               if x['lock_kind'] == 'prediction_h32_full')),
    run_case('failed_attempt_wall', edit_attempt,
             lambda rows: rows['objects'][0]['cost_scope']['driver_attempts'][0]['ended_at']),
]
here = Path(__file__).resolve().parent
(here / 'REVIEWS_02_DIAGNOSTIC_MUTATIONS.json').write_text(json.dumps(cases, indent=2, sort_keys=True) + '\n')
print([(case['case'], case['accepted'], case.get('exported_value_changed')) for case in cases])
