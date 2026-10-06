"""Executable direction bridge invariants on CPU; no GPU or queue work."""
import copy
import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from pcla_repair.core import BudgetLoRALinear, keyed_seed, lora_modules
from pcla_repair.core import tensor_digest
from pcla_research_v7.coordination import (aggregate_gradient_sketches, client_payload,
                                            projection, sketch, virtual_adamw_step)
from pcla_research_v7.coordinates import rotate_gradient, rotation_from_sketch
from pcla_research_v7.data import PROTOCOL
from pcla_research_v7.diagnostic_directions import VARIANTS, build_direction_variants
from pcla_research_v7.diagnostic_direction_runtime import (
    TemporaryEffectiveReadout, collect_direction_diagnostic, compare_t_full_h32_directions,
    norm_match_distinct_parents,
)
from pcla_research_v7.diagnostic_readout import DiagnosticView
from pcla_research_v7.runtime import aggregate


def _state():
    values = {}
    for i in range(24):
        name = f"layers.m{i:02d}"
        b = torch.tensor([[4., 0.], [0., 1.], [0., 0.]], dtype=torch.float32)
        a = torch.tensor([[.25, 0., 0.], [0., 1., 0.]], dtype=torch.float32)
        if i == 0:
            b[:, 1] = 0
        values[name + ".lora_b"] = b
        values[name + ".lora_a"] = a
    values["classifier.weight"] = torch.tensor([[.5, .3, .2]])
    values["classifier.bias"] = torch.tensor([.1])
    return values


def _clone(values):
    return {k: v.clone() for k, v in values.items()}


def _equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


class StateModel:
    def __init__(self, values):
        self.values = _clone(values)


def _readout(values):
    total = sum(float((values[name + ".lora_b"].double() @
                       values[name + ".lora_a"].double()).sum())
                for name in sorted(k[:-7] for k in values if k.endswith(".lora_a")))
    return total + sum(float(values[key].double().sum())
                       for key in ("classifier.weight", "classifier.bias"))


def _fixture():
    values = _state()
    model = StateModel(values)
    clients = tuple(range(10))
    view = DiagnosticView("readout2048", list(range(2048)), "frozen-source", tuple(range(2048)))
    config = {
        "schema": "pcla-v7-diagnostic-config-1", "protocol": PROTOCOL,
        "line": "direction", "phase": "diagnostic", "job_id": "Direction_s101_r10_b",
        "seed": 101, "factor": "b", "anchor_round": 10,
        "participant_schedule_round": 11, "participants": list(clients),
        "calls": {"clients": 10, "local_calls": 120, "optimizer_steps": 1920},
        "direction": {
            "representations": ["N", "D", "S", "T", "S_nonzero_nullspace_rotation",
                                "S_zero_direction"],
            "support_policies": ["local", "cyclic"],
            "forced_coordinate_intervention": True, "norm_matched_readout": True,
            "selection_evaluations": 0,
        },
        "training": {"paired_rng_key_scope": ["diagnostic_object_id", "client_id"],
                     "prefix_steps": [1, 4, 16], "local_optimizer_steps": 16,
                     "local_examples": 256, "batch_size": 16, "optimizer": "adamw",
                     "learning_rate": .0003, "head_learning_rate": .0003,
                     "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01},
        "visits": None,
    }
    context = {"model": model, "parent_state": values,
               "anchor": {"state": "anchor_verified", "participants": list(clients),
                          "anchor_sha256": "anchor-sha"},
               "clients": {cid: [cid] * 256 for cid in clients},
               "capacities": {cid: 1 for cid in clients},
               "visits": {cid: cid + 10 for cid in clients}, "readout": view}
    return config, context


def _execute(*, bad_prefix=False):
    config, context = _fixture()
    model = context["model"]
    parent = _clone(model.values)
    trace = {"probes": [], "selectors": [], "training": [], "readout": []}

    def load(model, values):
        model.values = _clone(values)

    def state(model):
        return _clone(model.values)

    def probe(model, dataset, factor, *, seed, batch_size):
        cid = dataset[0]
        assert factor == "b" and batch_size == 16
        trace["probes"].append((cid, seed, _clone(model.values)))
        return {key[:-7]: torch.ones_like(model.values[key[:-7] + ".lora_b"]) *
                (cid + 1) / 10 for key in model.values if key.endswith(".lora_a")}, 32

    def selector(model, dataset, method, factor, budget, **kwargs):
        cid = dataset[0]
        assert method in {"local", "cyclic"} and factor == "b" and budget == 1
        trace["selectors"].append((cid, method, kwargs, _clone(model.values)))
        names = sorted(k[:-7] for k in model.values if k.endswith(".lora_a"))
        out = {name: {"a": torch.zeros(2, dtype=torch.bool),
                      "b": torch.zeros(2, dtype=torch.bool)} for name in names}
        out[names[0]]["b"][0] = True
        return out

    def train(model, dataset, support, **kwargs):
        cid = dataset[0]
        trace["training"].append((cid, kwargs, _clone(model.values)))
        names = sorted(k[:-7] for k in model.values if k.endswith(".lora_a"))
        active = names[0]
        assert support[active]["b"].tolist() == [True, False]
        snapshots = {}
        for step in (1, 4, 16):
            candidate = _clone(model.values)
            candidate[active + ".lora_b"][0, 0] += .001 * (cid + 1) * step
            candidate["classifier.bias"] += .0001 * step
            snapshots[step] = candidate
        if bad_prefix:
            snapshots.pop(4)
        # A real local_train mutates its input model, then the caller resets it.
        model.values = _clone(snapshots[16])
        return {"optimizer_steps": 16, "prefix_states": snapshots}

    def original(values, view):
        assert view is context["readout"]
        trace["readout"].append("original")
        return _readout(values)

    def effective(matrices, head, view):
        assert view is context["readout"]
        trace["readout"].append("matched")
        return sum(float(v.sum()) for v in matrices.values()) + sum(float(v.sum())
                    for v in head.values())

    kwargs = dict(probe_fn=probe, select_support_fn=selector, train_fn=train,
                  aggregate_fn=aggregate, load_state_fn=load, state_fn=state,
                  original_readout=original, effective_readout=effective,
                  scaling_by_module={key[:-7]: 1. for key in parent if key.endswith(".lora_a")})
    return config, context, trace, parent, kwargs


def test_complete_bank_uses_one_trajectory_shared_rng_actual_visits_and_fixed_t_cohort():
    config, context, trace, parent, kwargs = _execute()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = collect_direction_diagnostic(config, context, **kwargs)
    assert result["status"] == "executed_unreviewed"
    assert result["local_calls"] == 120 and result["optimizer_steps"] == 1920
    assert tuple(result["representations"]) == VARIANTS
    assert all(result["representations"][label]["parent_state_sha256"] ==
               tensor_digest(result["representations"][label]["parent_state"])
               for label in VARIANTS)
    assert len(trace["probes"]) == 10 and len(trace["selectors"]) == len(trace["training"]) == 120
    assert len(result["t_probe"]["payloads"]) == 10
    assert set(result["t_probe"]["examples_by_client"].values()) == {32}
    assert set(result["t_probe"]["cbar"]) == set(result["representations"]["N"]["numerical_ranks"])
    d_probe = trace["probes"][0][2]
    assert not torch.equal(d_probe["layers.m01.lora_b"], parent["layers.m01.lora_b"])
    assert all(_equal(row[2], d_probe) for row in trace["probes"])
    for cid, seed, _ in trace["probes"]:
        assert seed == result["probe_seed_by_client"][cid]["b"]
    for cid, method, args, _ in trace["selectors"]:
        assert args["visit"] == context["visits"][cid]
        assert args["probe_batches"] == 2
        assert args["probe_seeds"] == result["probe_seed_by_client"][cid]
        assert args["seed"] == keyed_seed(config["seed"], "selection", cid)
    for cid, args, _ in trace["training"]:
        assert args["seed"] == keyed_seed(config["job_id"], cid)
        assert args["lr"] == .0003 and args["capture_steps"] == (1, 4, 16)
    assert [x for x in trace["readout"] if x == "original"] == ["original"] * 37
    assert [x for x in trace["readout"] if x == "matched"] == ["matched"] * 36
    n = result["readouts"]["N"]["local"][16]
    assert n["normmatch_status"] == "matched"
    assert n["normmatch_multiplier"] == pytest.approx(1.)
    assert n["normmatched_loss"] == pytest.approx(n["original_loss"])
    assert result["parent_readout2048_loss"] == pytest.approx(_readout(parent))
    assert n["original_benefit"] == pytest.approx(_readout(parent) - n["original_loss"])
    assert n["normmatched_benefit"] == pytest.approx(_readout(parent) - n["normmatched_loss"])
    comparison = result["t_full_h32_comparator"]
    assert comparison["status"] == "diagnostic_only_no_extra_training"
    assert set(comparison["one_step_linear_gain"]) == {"local", "cyclic"}
    assert set(result["t_probe"]["full_gradients_by_client"]) == set(range(10))
    assert all(len(values) == 24 for values in result["t_probe"]["full_gradients_by_client"].values())
    name, cid = "layers.m00", 3
    fullbar = comparison["full_cbar_by_module"][name]
    expected_fullbar = sum((.1 * result["t_probe"]["full_gradients_by_client"][other][name].double()
                            for other in range(10)), torch.zeros_like(fullbar))
    assert torch.allclose(fullbar, expected_fullbar, atol=1e-14)
    hbar = result["t_probe"]["cbar"][name]
    oh = comparison["rotations_and_grams"][name]["o_h32_deployed"]
    of = comparison["rotations_and_grams"][name]["o_full_untrained"]
    assert torch.allclose(oh, rotation_from_sketch(hbar))
    assert torch.allclose(of, rotation_from_sketch(fullbar))
    t_parent = result["representations"]["T"]["parent_state"]
    full_grad = result["t_probe"]["full_gradients_by_client"][cid][name]
    rotated = rotate_gradient(full_grad, oh, "b")
    update = virtual_adamw_step(t_parent[name + ".lora_b"], rotated, .0003)
    mask = result["supports"]["T"]["local"][cid][name]["b"]
    masked = update.double() * mask.double()[None, :]
    expected_full = -.1 * float(((fullbar @ oh) * masked).sum())
    s = projection(101, 11, name, "b", fullbar.shape[0], h=32)
    expected_h32 = -.1 * float(((hbar @ oh) * sketch(masked, s, "b").double()).sum())
    assert comparison["one_step_linear_gain"]["local"]["by_client"][cid]["full"] == pytest.approx(expected_full)
    assert comparison["one_step_linear_gain"]["local"]["by_client"][cid]["h32"] == pytest.approx(expected_h32)
    children = [result["prefix_states"]["N"]["local"][cid][4] for cid in range(10)]
    avg = result["aggregate_states"]["N"]["local"][4]
    key = "layers.m00.lora_b"
    assert avg[key][0, 0].item() == pytest.approx(
        parent[key][0, 0].item() + sum((row[key][0, 0] - parent[key][0, 0]).item()
                                       for row in children) / 10)
    assert _equal(context["model"].values, parent)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_incomplete_prefix_rejected_and_parent_restored():
    config, context, _, parent, kwargs = _execute(bad_prefix=True)
    with pytest.raises(ValueError, match="one 16-step trajectory"):
        collect_direction_diagnostic(config, context, **kwargs)
    assert _equal(context["model"].values, parent)


def test_norm_match_uses_each_representation_parent_and_keeps_actual_head():
    n_parent = _state()
    n_child = _clone(n_parent)
    candidate_parent = _clone(n_parent)
    candidate_child = _clone(n_parent)
    key = "layers.m01.lora_b"
    n_child[key][0, 0] += 2.
    candidate_parent[key][0, 0] += 5.
    candidate_child[key][0, 0] += 6.
    candidate_child["classifier.bias"] += 3.
    scales = {k[:-7]: 1. for k in n_parent if k.endswith(".lora_a")}
    match = norm_match_distinct_parents(n_parent, n_child, candidate_parent,
                                         candidate_child, scaling_by_module=scales)
    assert match.status == "matched"
    assert match.reference_norm == pytest.approx(.5)
    assert match.candidate_norm == pytest.approx(.25)
    assert match.multiplier == pytest.approx(2.)
    assert match.matched_effective["layers.m01"][0, 0].item() == pytest.approx(2.75)
    assert torch.equal(match.head["classifier.bias"], candidate_child["classifier.bias"])
    zero = norm_match_distinct_parents(n_parent, n_child, candidate_parent,
                                        candidate_parent, scaling_by_module=scales)
    assert zero.status == "NA_zero_candidate" and zero.matched_effective is None


def test_t_full_h32_factor_a_prediction_uses_transposed_full_gradient_without_training():
    parent = _state()
    client_ids = tuple(range(10))
    gradients = {cid: {name[:-7]: torch.tensor([[.4 + cid / 100, -.2, .1],
                                               [.2, .3 + cid / 100, -.4]], dtype=torch.float64)
                       for name in parent if name.endswith(".lora_a")}
                 for cid in client_ids}
    d = build_direction_variants(parent, object_id="factor-a", sketch_for_d_parent=lambda values:
        {name[:-7]: torch.zeros((32, 2), dtype=torch.float64)
         for name in values if name.endswith(".lora_a")})["D"].state
    factors = {name[:-7]: (d[name[:-7] + ".lora_b"], d[name])
               for name in d if name.endswith(".lora_a")}
    payloads = [client_payload(cid, 10, factors, gradients[cid], "a", .0003, 101, 11, h=32)
                for cid in client_ids]
    hbar = aggregate_gradient_sketches(payloads, {cid: .1 for cid in client_ids},
                                       expected_parent_version=10)
    variants = build_direction_variants(parent, object_id="factor-a",
                                        sketch_for_d_parent=lambda _: hbar)
    names = sorted(factors)
    support = {name: {"a": torch.tensor([name == names[0], False]),
                      "b": torch.zeros(2, dtype=torch.bool)} for name in names}
    supports = {method: {cid: support for cid in client_ids} for method in ("local", "cyclic")}
    result = compare_t_full_h32_directions(
        full_gradients=gradients, h32_cbar=hbar, d_parent=variants["D"].state,
        t_parent=variants["T"].state, t_supports=supports, clients=client_ids,
        factor="a", seed=101, round_id=11)
    name = names[0]
    o = result["rotations_and_grams"][name]["o_h32_deployed"]
    full = result["full_cbar_by_module"][name]
    update = virtual_adamw_step(variants["T"].state[name + ".lora_a"],
                                rotate_gradient(gradients[0][name], o, "a"), .0003)
    masked = update.double() * support[name]["a"].double()[:, None]
    expected = -.1 * float(((full @ o) * masked.T).sum())
    assert result["one_step_linear_gain"]["local"]["by_client"][0]["full"] == pytest.approx(expected)
    assert result["one_step_linear_gain"]["local"] == result["one_step_linear_gain"]["cyclic"]


class ReadoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({f"m{i:02d}": BudgetLoRALinear(
            nn.Linear(2, 2), rank=1, alpha=1., dropout=.1) for i in range(24)})
        self.classifier = nn.Linear(2, 1)


def test_temporary_effective_readout_restores_w0_head_modes_and_rng_on_exception():
    model = ReadoutModel()
    view = DiagnosticView("readout2048", list(range(2048)), "source", tuple(range(2048)))
    modules = lora_modules(model)
    bases = {key: module.base.weight.detach().clone() for key, module in modules.items()}
    heads = {name: value.detach().clone() for name, value in model.named_parameters()
             if name.startswith("classifier.")}
    modes = {name: module.training for name, module in model.named_modules()}
    matrices = {key: torch.ones_like(module.base.weight) * .2 for key, module in modules.items()}
    future_head = {key: value + 1. for key, value in heads.items()}

    def fail(model, dataset):
        assert dataset is view.dataset
        assert all(torch.allclose(module.base.weight, bases[key] + .2)
                   and module.scaling == 0. for key, module in lora_modules(model).items())
        assert all(torch.equal(dict(model.named_parameters())[key], future_head[key])
                   for key in future_head)
        model.eval()
        random.random(); np.random.random(); torch.rand(1)
        raise RuntimeError("readout failed")

    random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="readout failed"):
        TemporaryEffectiveReadout(model, view, evaluate_fn=fail)(matrices, future_head, view)
    assert all(torch.equal(module.base.weight, bases[key]) and module.scaling == 1.
               for key, module in lora_modules(model).items())
    assert all(torch.equal(dict(model.named_parameters())[key], heads[key]) for key in heads)
    assert all(module.training == modes[name] for name, module in model.named_modules())
    assert random.getstate() == random_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
