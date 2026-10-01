import copy
import random
import numpy as np
import pytest
import torch

from pcla_repair.core import lora_modules
from pcla_repair.testing import fixture, TinyDataset
from pcla_research_v7.runtime import (
    active_state, attach_slices, release_slices, state, full_support, local_train)


def same(a, b):
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_active_state_matches_committed_state_without_mutation():
    model, *_ = fixture(73)
    support = full_support(model)
    for masks in support.values():
        masks['a'][1:] = False
        masks['b'][::2] = False
    before = state(model)
    attach_slices(model, support)
    with torch.no_grad():
        for module in lora_modules(model).values():
            module.active_a.add_(.125)
            module.active_b.add_(.25)
    rng = torch.get_rng_state().clone()
    snapshot = active_state(model)
    same(state(model), before)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(m._v7_active for m in lora_modules(model).values())
    release_slices(model)
    same(snapshot, state(model))
    for value in snapshot.values():
        assert value.device.type == 'cpu' and not value.requires_grad


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_prefix_capture_preserves_final_parameters_rng_and_optimizer_steps(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model, *_ = fixture(27)
    model.to(device)
    baseline = copy.deepcopy(model)
    support = full_support(model, 'b')
    data = TinyDataset(81, 256)
    ref = local_train(baseline, data, support, seed=123, lr=.001)
    expected_rng = torch.get_rng_state().clone()
    expected_cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
    expected_py_rng = random.getstate()
    expected_np_rng = np.random.get_state()
    # Independent observation after optimizer.step: clone and commit only the
    # copy, so snapshots are checked against the actual current active tensors.
    reference_states = {}
    real_step = torch.optim.AdamW.step
    counter = 0
    def observe_step(optimizer, *args, **kwargs):
        nonlocal counter
        result = real_step(optimizer, *args, **kwargs)
        counter += 1
        if counter in (1, 4, 16):
            clone = copy.deepcopy(model)
            release_slices(clone)
            reference_states[counter] = state(clone)
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', observe_step)
    from pcla_research_v7.execution_timing import ExecutionTimingLedger
    timer = ExecutionTimingLedger({'local_training': 'necessary_algorithm',
                                   'local_setup': 'necessary_algorithm',
                                   'prefix_capture': 'research_observation'},
                                  synchronize=torch.cuda.synchronize if device == 'cuda' else None)
    result = local_train(model, data, support, seed=123, lr=.001,
                         capture_steps=(1, 4, 16), timing=timer)
    times = timer.snapshot()
    assert times['category_seconds']['necessary_algorithm'] > 0
    assert times['category_seconds']['research_observation'] > 0
    assert times['total_seconds'] == pytest.approx(sum(times['label_seconds'].values()))
    assert times['bucket_category_seconds']['diagnostic_after_budget']['necessary_algorithm'] == 0
    assert counter == ref['optimizer_steps'] == result['optimizer_steps'] == 16
    assert set(result['prefix_states']) == {1, 4, 16}
    for step, snapshot in result['prefix_states'].items():
        same(snapshot, reference_states[step])
    same(state(model), state(baseline))
    same(result['prefix_states'][16], state(model))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    if device == "cuda":
        assert all(torch.equal(a, b) for a, b in zip(expected_cuda_rng, torch.cuda.get_rng_state_all()))
    assert random.getstate() == expected_py_rng
    actual_np_rng = np.random.get_state()
    assert actual_np_rng[0] == expected_np_rng[0]
    assert np.array_equal(actual_np_rng[1], expected_np_rng[1])
    assert actual_np_rng[2:] == expected_np_rng[2:]
    assert result['prefix_capture_seconds'] >= 0
    assert 'prefix_states' not in ref


@pytest.mark.parametrize('steps', [(0,), (17,), (1, 1), (True,), (1.5,)])
def test_invalid_prefix_rejected_before_training(steps):
    model, *_ = fixture(93)
    before = state(model)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        local_train(model, TinyDataset(92,256), full_support(model), seed=1,
                    lr=.001, capture_steps=steps)
    same(state(model), before)
    assert torch.equal(rng, torch.get_rng_state())
