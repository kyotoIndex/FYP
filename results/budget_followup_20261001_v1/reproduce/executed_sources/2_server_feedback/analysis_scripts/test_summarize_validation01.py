import math
import pytest
import summarize_validation01 as s


def rows():
    return [{'recipe_id': r, 'seed': seed, 'slot': i,
             'job_id': f'VAL_S{i:02d}_s{seed}', 'hardware_family': 'RTX4090D',
             'eval_role': 'recipe_dev', 'gpu_uuid': f'GPU-{seed}',
             **{m: .8 + i*.001 for m in s.ACCURACY}, 'raw_B8_NLL': .3,
             'required_prefix_seconds': 100*i, 'returned_round': 100}
            for i, r in enumerate(s.base.RECIPES, 1) for seed in s.base.SEEDS]


def test_student_interval_independent_reference():
    r = s.describe([1., 2., 3.])
    assert r['mean'] == 2
    assert r['sample_sd'] == 1
    assert r['t95_high'] == pytest.approx(2+4.302652729911275/math.sqrt(3))


def test_seed_pair_units_and_equal_weight():
    methods, pairs, raw, times = s.summaries(rows())
    assert len(raw) == 36*12*len(s.METRICS)
    p = next(x for x in pairs if x['left'] == 'U002' and x['right'] == 'U010'
             and x['metric'] == 'raw_B8_accuracy')
    assert p['n_seeds'] == 12 and p['mean'] == pytest.approx(-.1)
    assert p['t95_low'] == pytest.approx(p['t95_high'])
    assert times[0]['n_seeds'] == 3 and times[0]['mean'] == .5


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'hardware', 'role', 'nonfinite'])
def test_bad_matrix_rejected(change):
    data = rows()
    if change == 'missing': data.pop()
    elif change == 'duplicate': data[-1] = dict(data[0])
    elif change == 'hardware': data[0]['hardware_family'] = 'A100'
    elif change == 'role': data[0]['eval_role'] = 'public512'
    else: data[0]['raw_B8_accuracy'] = float('nan')
    with pytest.raises(ValueError): s.summaries(data)


def test_budget_lengths_do_not_change_seed_weight():
    data = rows()
    for row in data:
        if row['recipe_id'] == 'U006': row['returned_round'] = 200
    methods, _, _, _ = s.summaries(data)
    r = next(x for x in methods if x['recipe_id'] == 'U006'
             and x['metric'] == 'raw_B8_accuracy')
    assert r['n_seeds'] == 12 and r['mean'] == pytest.approx(80.9)


@pytest.mark.parametrize('failure', ['incomplete', 'hash_changed'])
def test_bad_export_creates_no_summary(tmp_path, monkeypatch, failure):
    import json
    import sys
    src = tmp_path/'base'; src.mkdir()
    data = rows()
    if failure == 'incomplete': data.pop()
    s.base.write_csv(src/'endpoints.csv', data)
    manifest = {'jobs': 108, 'exporter_sha256': s.BASE_SHA,
                'source_index_sha256': s.base.INDEX_SHA,
                'files': {'endpoints.csv': s.base.digest(src/'endpoints.csv')}}
    (src/'MANIFEST.json').write_text(json.dumps(manifest))
    if failure == 'hash_changed':
        with (src/'endpoints.csv').open('a') as f: f.write('\n')
    out = tmp_path/'summary'
    monkeypatch.setattr(sys, 'argv', ['summary', '--base-export', str(src), '--out', str(out)])
    with pytest.raises(ValueError): s.main()
    assert not out.exists()
