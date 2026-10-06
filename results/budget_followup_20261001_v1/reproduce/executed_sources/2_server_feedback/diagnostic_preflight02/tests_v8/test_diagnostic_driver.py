import copy
import json

import pytest
import torch
from torch.nn import functional as F

from pcla_repair.testing import TinyDataset, TinyModel
from pcla_research_v7.runtime import state
from scripts import run_pcla_v8_diagnostic as driver


def test_readout_double_nll_raw_predictions_and_caller_restored():
    torch.manual_seed(88)
    model = TinyModel()
    model.train()
    original = state(model)
    candidate = {key: value + .01 for key, value in original.items()}
    result = driver.raw_readout(model, candidate, TinyDataset(54, 64), expected_samples=64)
    logits = torch.tensor(result['logits'], dtype=torch.float32)
    labels = torch.tensor(result['labels'])
    assert result['loss'] == F.cross_entropy(logits.double(), labels, reduction='mean').item()
    assert sum(result['prediction_counts']) == 64
    assert result['accuracy'] == int((logits.argmax(1) == labels).sum()) / 64
    assert model.training
    assert all(torch.equal(value, state(model)[key]) for key, value in original.items())
    with pytest.raises(ValueError, match='row count'):
        driver.raw_readout(model, candidate, TinyDataset(54, 63), expected_samples=64)


def test_cohort_recovery_reuses_saved_update_and_rejects_changed_identity(tmp_path, monkeypatch):
    parent = state(TinyModel())
    value = {'local_calls': 10, 'optimizer_steps': 160, 'elapsed_seconds': 2.3,
             'aggregate_updates': [{'children': [parent], 'proposal': parent}]}
    calls = []
    def collect(context, config):
        calls.append(1)
        return copy.deepcopy(value)
    monkeypatch.setattr(driver, 'collect_rank8', collect)
    context = {'parent_state': parent}
    first, first_receipt = driver.restore_or_collect(context, {}, tmp_path, 'a' * 64)
    second, second_receipt = driver.restore_or_collect(context, {}, tmp_path, 'a' * 64)
    assert len(calls) == 1
    assert first_receipt['reused'] is False and second_receipt['reused'] is True
    assert first['optimizer_steps'] == second['optimizer_steps'] == 160
    with pytest.raises(ValueError, match='differs'):
        driver.restore_or_collect(context, {}, tmp_path, 'b' * 64)
    assert len(calls) == 1


def test_sealed_cohort_corruption_is_preserved_and_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'new_cohort.pt'
    path.write_bytes(b'original partial evidence')
    called = []
    monkeypatch.setattr(driver, 'collect_rank8', lambda *a: called.append(1))
    with pytest.raises(ValueError, match='partial cohort'):
        driver.restore_or_collect({'parent_state': state(TinyModel())}, {}, tmp_path, 'x')
    assert path.read_bytes() == b'original partial evidence' and not called


def test_transfer_chain_weights_actual_improvements_once(monkeypatch):
    parent = {'classifier.bias': torch.tensor([0., 0.])}
    children = [{'classifier.bias': torch.tensor([float(i + 1), 0.])} for i in range(10)]
    proposal = {'classifier.bias': torch.tensor([5.5, 0.])}
    def reader(candidate):
        return {'loss': 20. - float(candidate['classifier.bias'][0]), 'samples': 2048}
    monkeypatch.setattr(driver, 'raw_readout', lambda m, c, d, **kw: reader(c))
    context = {'model': object(), 'participants': list(range(10)), 'parent_state': parent,
               'local_readout_by_client': {i: object() for i in range(10)}}
    update = {'children': children, 'proposal': proposal, 'weights': [.1] * 10}
    chain = driver.transfer_chain(context, update, reader)
    assert chain['weighted_individual_global_NLL_gain'] == pytest.approx(5.5)
    assert chain['aggregate_global_NLL_gain'] == 5.5
    assert [r['individual_global_NLL_gain'] for r in chain['clients']] == list(range(1, 11))
    update['weights'][0] = .01
    with pytest.raises(ValueError, match='equally weighted'):
        driver.transfer_chain(context, update, reader)
