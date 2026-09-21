import importlib.util
from pathlib import Path
import pytest
import torch
from pcla_repair.testing import fixture
from pcla_repair.core import adapter_state, load_adapter, lora_modules

spec = importlib.util.spec_from_file_location('rank_signal', Path(__file__).parents[1]/'scripts/diagnose_rank_signal.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_split_is_disjoint_deterministic_and_source_order_independent():
    ids = [f'train:{i}' for i in range(256)]
    split = diagnostic.split_ids(ids, 7, 3)
    assert split == diagnostic.split_ids(list(reversed(ids)), 7, 3)
    assert [len(v) for v in split.values()] == [16,16,16,16,64]
    assert len(set(x for group in split.values() for x in group)) == 128
    assert split != diagnostic.split_ids(ids, 7, 4)


@pytest.mark.parametrize('factor', ['a','b'])
def test_reciprocal_scaling_preserves_logits_and_invariant_scores(factor):
    model, layout, clients, *_ = fixture()
    model.double()
    for data in clients.values(): data.x = data.x.double()
    for module in lora_modules(model).values():
        with torch.no_grad(): module.lora_b.normal_(0,.1)
    original = adapter_state(model)
    scores, logits = diagnostic.score_batch(model,layout,clients[0],factor,torch.device('cpu'))
    scales = diagnostic.scale_adapter(model,layout)
    scaled, logits2 = diagnostic.score_batch(model,layout,clients[0],factor,torch.device('cpu'))
    torch.testing.assert_close(logits,logits2,rtol=1e-12,atol=1e-12)
    for kind in ['normalized_projection','gate_shared_qv']:
        torch.testing.assert_close(scores[kind],scaled[kind],rtol=1e-10,atol=1e-12)
    for layer in range(4):
        c = torch.tensor(scales[f'blocks.{layer}.query'],dtype=torch.float64)
        multiplier = c if factor == 'b' else 1/c
        torch.testing.assert_close(scaled['gradient_norm'][layer],scores['gradient_norm'][layer]*multiplier)
        torch.testing.assert_close(scaled['legacy'][layer],scores['legacy'][layer]*multiplier**2)
    load_adapter(model,original)


def test_shared_gate_matches_joint_qv_finite_difference():
    model,layout,clients,*_ = fixture()
    model.double()
    for data in clients.values(): data.x = data.x.double()
    for module in lora_modules(model).values():
        with torch.no_grad(): module.lora_b.normal_(0,.1)
    original = adapter_state(model)
    scores,_ = diagnostic.score_batch(model,layout,clients[0],'b',torch.device('cpu'))
    vals=[]
    for delta in (-1e-5,1e-5):
        load_adapter(model,original)
        with torch.no_grad():
            for name,module in lora_modules(model).items():
                if layout[name] == 1: module.lora_b[:,2].mul_(1+delta)
        vals.append(sum(diagnostic.losses(model,layout,clients[0],torch.device('cpu')))/len(clients[0]))
    derivative=abs((vals[1]-vals[0])/2e-5)
    assert derivative == pytest.approx(float(scores['gate_shared_qv'][1,2]),rel=1e-5,abs=1e-9)
