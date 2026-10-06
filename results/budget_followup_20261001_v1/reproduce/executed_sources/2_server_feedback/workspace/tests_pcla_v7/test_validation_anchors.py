"""Persisted diagnostic anchors must preserve the actual parent trajectory."""
import shutil
import torch
import pytest
from pcla_repair.testing import fixture
from pcla_research_v7.runtime import run_experiment, _load_checkpoint
from tests_pcla_v7.test_runtime import FixedWire, _config, _job, _same_state


@pytest.mark.parametrize('method', ['full_alt', 'full_both'])
def test_anchor_is_complete_and_resumes_same_parent(tmp_path, method):
    job = dict(_job(method), phase='V', min_rounds=60, max_rounds=60)
    def run(out):
        model, _, clients, capacities, evaluation = fixture(9)
        return run_experiment(model, clients, evaluation, capacities, job,
                              _config(), out, {'evaluation_role':'synthetic'}, FixedWire())
    original = tmp_path/'original'
    run(original)
    end = _load_checkpoint(original/'latest.pt')
    for r in (10,60):
        anchor = _load_checkpoint(original/f'anchor_round_{r:03d}.pt')
        assert anchor['round'] == r
        assert len(anchor['metrics']) == r
        assert all(k in anchor for k in ('adapter','cache','torch_rng','python_rng','cuda_rng','identity'))
    resumed = tmp_path/'resumed'; resumed.mkdir()
    shutil.copyfile(original/'anchor_round_010.pt', resumed/'latest.pt')
    shutil.copyfile(original/'evaluation_labels.json', resumed/'evaluation_labels.json')
    run(resumed)
    result = _load_checkpoint(resumed/'latest.pt')
    assert _same_state(end['adapter'], result['adapter'])
    assert torch.equal(end['torch_rng'], result['torch_rng'])
    assert end['python_rng'] == result['python_rng']
    assert end['cache'] == result['cache']
    assert [x['evaluation'] for x in end['metrics']] == [x['evaluation'] for x in result['metrics']]
