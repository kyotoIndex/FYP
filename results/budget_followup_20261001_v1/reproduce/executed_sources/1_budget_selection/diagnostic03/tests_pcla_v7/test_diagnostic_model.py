from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from torch import nn

from pcla_repair.core import BudgetLoRALinear
from pcla_research_v7.diagnostic_model import ModelLoss, diagnostic_views, source_rows
from pcla_research_v7.diagnostic_readout import DiagnosticView
from pcla_research_v7.runtime import state


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = BudgetLoRALinear(nn.Linear(3, 3), 2, 2, .1)
        self.classifier = nn.Linear(3, 2)

    def forward(self, input_ids, labels):
        logits = self.classifier(self.query(input_ids.float()))
        return SimpleNamespace(logits=logits, loss=nn.functional.cross_entropy(logits, labels))


def test_real_loss_restores_model_rng_modes_and_caches_exact_values():
    model = Model(); model.train(); model.query.base.eval()
    data = [{'input_ids': torch.tensor([1., 2., 3.]), 'labels': torch.tensor(0)},
            {'input_ids': torch.tensor([2., 1., 0.]), 'labels': torch.tensor(1)}]
    v = DiagnosticView('local64', data, 'source', (10, 12))
    loss = ModelLoss(model, [v], batch_size=2)
    base = state(model)
    changed = {k: t.clone() for k, t in base.items()}; changed['classifier.bias'][0] += 2
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone(); py = random.getstate(); numpy = np.random.get_state()
    value = loss(changed, data)
    model.eval()
    with torch.no_grad():
        logits = model.classifier(model.query(torch.stack([x['input_ids'] for x in data])))
        logits[:, 0] += 2
        expected = nn.functional.cross_entropy(logits, torch.tensor([0, 1])).item()
    assert value == pytest.approx(expected)
    # Restore test-side mode changes before checking evaluator preservation.
    for m, mode in zip(model.modules(), modes): m.training = mode
    assert all(torch.equal(base[k], t) for k, t in state(model).items())
    assert torch.equal(rng, torch.get_rng_state()) and py == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1])
    assert loss(changed, data) == value and len(loss.evaluations) == 1
    changed['classifier.bias'][1] -= 1
    loss(changed, data)
    assert len(loss.evaluations) == 2
    with pytest.raises(PermissionError): loss(changed, list(data))


def test_loss_failure_also_restores_parent_and_randomness(monkeypatch):
    import pcla_research_v7.diagnostic_model as module
    model = Model(); base = state(model); rng = torch.get_rng_state().clone()
    data = [0]
    loss = ModelLoss(model, [DiagnosticView('local64', data, 'src', (1,))])
    child = {k: v + 1 for k, v in base.items()}
    def fail(*a, **kw):
        torch.rand(5)
        raise RuntimeError('readout failure')
    monkeypatch.setattr(module, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='readout failure'): loss(child, data)
    assert all(torch.equal(base[k], t) for k, t in state(model).items())
    assert torch.equal(rng, torch.get_rng_state()) and loss.cache == {}


def test_role_views_preserve_global_source_identity_and_access_actor():
    manifest = {'source_sha256': 'one-source', 'roles':
                {'cohort_select': ['train:7'], 'cohort_readout': ['train:9']},
                'partitions': {'101': {'client_local_readout': {'3': ['train:5']}}}}
    class Tokens:
        def __init__(self): self.calls = []
        def dataset(self, m, role, actor, seed, cid):
            self.calls.append((role, actor)); return [0]
    tokens = Tokens()
    local, select, readout = diagnostic_views(tokens, manifest, 101, [3])
    assert local[3].row_ids == (5,) and select.row_ids == (7,) and readout.row_ids == (9,)
    assert {v.source_id for v in [local[3], select, readout]} == {'one-source'}
    assert tokens.calls == [('client_local_readout', 'local_diagnostic'),
                            ('cohort_select', 'finite_bank_search'), ('cohort_readout', 'cohort_report')]
    manifest['roles']['cohort_readout'] = ['train:7']
    with pytest.raises(ValueError, match='overlap'): diagnostic_views(tokens, manifest, 101, [3])
    for ids in (['train:01'], ['dev:1'], ['train:1', 'train:1']):
        with pytest.raises(ValueError): source_rows(ids)
