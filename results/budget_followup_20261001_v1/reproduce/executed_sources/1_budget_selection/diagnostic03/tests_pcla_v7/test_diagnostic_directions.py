"""Fixed-parent direction and effective-matrix invariants; no model training."""
import math
import json
import random

import numpy as np

import pytest
import torch

from pcla_repair.core import keyed_seed
from pcla_research_v7.diagnostic_directions import (
    VARIANTS, build_direction_variants, matched_training_seed, norm_match_to_n,
    read_norm_matched, sample_s_nonzero_rotation,
)
from pcla_research_v7.diagnostic_bank import _seed as bank_seed
from pcla_research_v7.data import PROTOCOL


def parent_state(*, with_null: bool = True):
    state = {}
    for i in range(24):
        name = f"layer{i:02d}.q"
        b = torch.tensor([[10., 0.], [0., 1.], [0., 0.]], dtype=torch.float32)
        a = torch.tensor([[.1, 0., 0.], [0., 1., 0.]], dtype=torch.float32)
        if with_null and i == 0:
            b[:, 1] = 0
        state[name + ".lora_b"] = b
        state[name + ".lora_a"] = a
    state["classifier.weight"] = torch.tensor([[1., 2., 3.]], dtype=torch.float32)
    state["classifier.bias"] = torch.tensor([.25], dtype=torch.float32)
    return state


def effective(state, name):
    return state[name + ".lora_b"].double() @ state[name + ".lora_a"].double()


def build(parent, *, object_id="diagnostic-object-1"):
    observed = []

    def sketch_on_d(d):
        observed.append(d)
        # D is a reciprocal rebalance of the imbalanced original factors.
        assert not torch.equal(d["layer01.q.lora_b"], parent["layer01.q.lora_b"])
        return {name: torch.tensor([[2., 0.], [0., 1.]] + [[0., 0.]] * 30,
                                   dtype=torch.float64)
                for name in sorted(key[:-7] for key in d if key.endswith(".lora_a"))}

    variants = build_direction_variants(parent, object_id=object_id,
                                        sketch_for_d_parent=sketch_on_d)
    return variants, observed


def test_six_variants_share_effective_parent_and_head_without_mutation():
    parent = parent_state()
    before = {key: value.clone() for key, value in parent.items()}
    variants, observed = build(parent)
    assert tuple(variants) == VARIANTS
    assert len(observed) == 1
    assert torch.equal(observed[0]["layer01.q.lora_b"], variants["D"].state["layer01.q.lora_b"])
    for label, variant in variants.items():
        assert len(variant.numerical_ranks) == 24
        assert torch.equal(variant.state["classifier.weight"], parent["classifier.weight"])
        assert torch.equal(variant.state["classifier.bias"], parent["classifier.bias"])
        for i in range(24):
            name = f"layer{i:02d}.q"
            assert torch.allclose(effective(variant.state, name), effective(parent, name), atol=1e-5)
    assert all(torch.equal(parent[key], value) for key, value in before.items())
    assert variants["S_zero_removed"].active
    assert variants["S"].numerical_ranks["layer00.q"] == 1
    assert variants["S_zero_removed"].state["layer00.q.lora_a"][1].abs().sum() == 0
    assert variants["S"].state["layer00.q.lora_a"][1].abs().sum() > 0
    assert torch.equal(variants["S_random_nonzero"].state["layer00.q.lora_a"][1],
                       variants["S"].state["layer00.q.lora_a"][1])


def test_no_numerical_zero_direction_marks_ablation_inactive():
    parent = parent_state(with_null=False)
    variants, _ = build(parent)
    assert not variants["S_zero_removed"].active
    assert all(variants["S"].numerical_ranks.values())
    assert all(torch.equal(variants["S"].state[key], variants["S_zero_removed"].state[key])
               for key in parent)


def test_t_requires_current_complete_h32():
    parent = parent_state()
    with pytest.raises(ValueError, match="h32"):
        build_direction_variants(parent, object_id="diagnostic-object-1",
            sketch_for_d_parent=lambda d: {"layer00.q": torch.zeros((32, 2))})
    with pytest.raises(ValueError, match="h32"):
        build_direction_variants(parent, object_id="diagnostic-object-1",
            sketch_for_d_parent=lambda d: {name: torch.zeros((16, 2))
                for name in sorted(key[:-7] for key in d if key.endswith(".lora_a"))})


def test_training_rng_key_is_shared_by_representation_and_local_cyclic():
    # There is no representation or selector argument: all twelve trials for
    # one client use the exact v7 local_train seed, but different clients do not.
    seed = matched_training_seed("diagnostic-object-1", 7)
    assert seed == keyed_seed("diagnostic-object-1", 7) == bank_seed("diagnostic-object-1", 7)
    assert seed != matched_training_seed("diagnostic-object-1", 8)
    with pytest.raises(ValueError):
        matched_training_seed("diagnostic-object-1", -1)


def test_s_rotation_k0_k1_qr_sign_and_serialized_provenance():
    object_id = "diagnostic-object-1"
    empty = sample_s_nonzero_rotation(object_id, "layer00.q", 0)
    assert empty.matrix.shape == (0, 0) and empty.matrix.dtype == torch.float64
    assert empty.seed == keyed_seed(PROTOCOL, "diagnostic_S_nonzero_rotation_v1",
                                    object_id, "layer00.q")
    assert empty.as_dict()["matrix"] == []
    one = sample_s_nonzero_rotation(object_id, "layer00.q", 1)
    generator = torch.Generator(device="cpu").manual_seed(one.seed)
    draw = torch.randn((1, 1), dtype=torch.float64, generator=generator)
    expected = torch.sign(draw)
    expected[expected == 0] = 1.
    assert torch.equal(one.matrix, expected)
    assert abs(one.matrix.item()) == 1.
    assert json.loads(json.dumps(one.as_dict()))["matrix_sha256"] == one.matrix_sha256


def test_s_rotation_is_orthogonal_deterministic_independent_and_rng_pure():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    one = sample_s_nonzero_rotation("object-A", "layer01.q", 5)
    same = sample_s_nonzero_rotation("object-A", "layer01.q", 5)
    other_module = sample_s_nonzero_rotation("object-A", "layer02.q", 5)
    other_object = sample_s_nonzero_rotation("object-B", "layer01.q", 5)
    assert torch.equal(one.matrix, same.matrix) and one.matrix_sha256 == same.matrix_sha256
    assert one.seed != other_module.seed and one.seed != other_object.seed
    assert not torch.equal(one.matrix, other_module.matrix)
    assert torch.allclose(one.matrix.T @ one.matrix, torch.eye(5, dtype=torch.float64),
                          atol=1e-12, rtol=0)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


def test_variant_records_each_rotation_and_handles_rank_zero():
    parent = parent_state()
    parent["layer00.q.lora_b"][:] = 0
    variants, _ = build(parent)
    records = variants["S_random_nonzero"].random_rotations
    assert len(records) == 24
    assert records["layer00.q"].matrix.shape == (0, 0)
    assert records["layer00.q"].matrix_sha256
    again, _ = build(parent)
    assert all(records[name].matrix_sha256 == again["S_random_nonzero"].random_rotations[name].matrix_sha256
               for name in records)


def test_normmatch_uses_one_effective_multiplier_and_retains_actual_head():
    parent = parent_state(with_null=False)
    n = {key: value.clone() for key, value in parent.items()}
    candidate = {key: value.clone() for key, value in parent.items()}
    n["layer00.q.lora_b"][0, 0] += 2.
    n["layer01.q.lora_a"][1, 1] += 1.
    candidate["layer00.q.lora_b"][0, 0] += .5
    candidate["layer00.q.lora_a"][0, 0] += .2  # cross term must be included
    candidate["layer01.q.lora_a"][1, 1] += .1
    candidate["classifier.weight"] += 3.
    candidate["classifier.bias"] -= .2
    before = {key: value.clone() for key, value in candidate.items()}
    result = norm_match_to_n(parent, n, candidate)
    assert result.status == "matched"
    assert result.multiplier is not None and result.multiplier > 0
    assert math.isclose(result.candidate_norm * result.multiplier, result.reference_norm,
                        rel_tol=1e-12)
    matched_sq = sum(float((result.matched_effective[name] - effective(parent, name)).square().sum())
                     for name in result.matched_effective)
    assert math.isclose(math.sqrt(matched_sq), result.reference_norm, rel_tol=1e-12)
    for name in result.matched_effective:
        delta = effective(candidate, name) - effective(parent, name)
        assert torch.allclose(result.matched_effective[name],
                              effective(parent, name) + result.multiplier * delta,
                              atol=1e-12, rtol=1e-12)
    assert torch.equal(result.head["classifier.weight"], candidate["classifier.weight"])
    assert torch.equal(result.head["classifier.bias"], candidate["classifier.bias"])
    assert all(torch.equal(candidate[key], value) for key, value in before.items())


def test_zero_effective_update_is_na_without_epsilon_or_factor_scaling():
    parent = parent_state(with_null=False)
    n = {key: value.clone() for key, value in parent.items()}
    n["layer00.q.lora_b"][0, 0] += 1.
    zero = norm_match_to_n(parent, n, parent)
    assert zero.status == "NA_zero_candidate"
    assert zero.multiplier is None and zero.matched_effective is None
    no_reference = norm_match_to_n(parent, parent, n)
    assert no_reference.status == "NA_zero_reference"
    assert no_reference.multiplier is None and no_reference.matched_effective is None
    assert read_norm_matched(zero, lambda *_: pytest.fail("NA must not invoke evaluator")) is None


def test_readout_bridge_receives_effective_matrices_and_unscaled_head():
    parent = parent_state(with_null=False)
    n = {key: value.clone() for key, value in parent.items()}
    n["layer00.q.lora_b"][0, 0] += 1.
    n["classifier.bias"] += 2.
    match = norm_match_to_n(parent, n, n)
    calls = []

    def evaluate_effective(matrices, head):
        calls.append((matrices, head))
        assert len(matrices) == 24
        assert torch.equal(head["classifier.bias"], n["classifier.bias"])
        return 1.25

    assert read_norm_matched(match, evaluate_effective) == 1.25
    assert len(calls) == 1
    with pytest.raises(ValueError, match="nonfinite"):
        read_norm_matched(match, lambda *_: float("nan"))
