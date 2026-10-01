"""Small actual-model checks; synthetic values are never research evidence."""

import json
import random

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pcla_repair.testing import TinyDataset, TinyModel
from pcla_research_v7.runtime import load_state, state
from pcla_research_v8.diagnostic_feedback import evaluate_feedback
from pcla_research_v8.server_control import interpolate


def fixture():
    torch.manual_seed(194)
    model = TinyModel()
    parent = state(model)
    children = [
        {key: value + (i + 1) * .015 * torch.randn_like(value)
         for key, value in parent.items()}
        for i in range(3)
    ]
    weights = [1., 2., 3.]
    proposal = {
        key: sum(weight * child[key] for weight, child in zip(weights, children)) / sum(weights)
        for key in parent
    }
    public = TinyDataset(301, 512)
    readout = TinyDataset(302, 48)
    return model, parent, proposal, children, weights, public, readout


def direct_reader(model, dataset):
    batches = list(DataLoader(dataset, batch_size=48))
    calls = []

    def readout(candidate):
        calls.append(1)
        load_state(model, candidate)
        model.eval()
        torch.manual_seed(100 + len(calls))  # caller RNG must still be restored
        with torch.no_grad():
            logits = torch.cat([model(**batch).logits.float() for batch in batches])
        labels = torch.cat([batch["labels"] for batch in batches])
        return {
            "loss": F.cross_entropy(logits.double(), labels, reduction="mean").item(),
            "accuracy": (logits.argmax(1) == labels).float().mean().item(),
            "predictions": logits.argmax(1),
        }

    return readout, calls


def test_real_grid_readout_corners_costs_and_state_preservation():
    model, parent, proposal, children, weights, public, readout_set = fixture()
    reader, calls = direct_reader(model, readout_set)
    model.train()
    before_torch = torch.random.get_rng_state().clone()
    before_py = random.getstate()
    config = {
        "job": {"job_id": "diagnostic-test"},
        "expanded_sha256": "synthetic-config",
        "controller": {
            "candidate_rules": ["ordinary_mean", "scalar4", "scalar17",
                                "block16", "separable", "old_finite_bank_reference"],
            "grid4": [0, .25, .5, 1],
            "scalar17": [i / 16 for i in range(17)],
            "public_selection_then_independent_readout": True,
            "tie_abs_nll": 1e-12,
        },
    }
    selected, record = evaluate_feedback(
        model, parent, proposal, children, weights, public, reader, 31, 4, config=config)

    assert tuple(selected) == ("ordinary_mean", "scalar4", "scalar17", "block16", "separable")
    assert all(torch.equal(selected["ordinary_mean"][k], proposal[k]) for k in parent)
    for rule, picked in selected.items():
        row = record["rules"][rule]
        expected = interpolate(parent, proposal, row["alpha"], row["beta"])
        assert all(torch.equal(picked[k], expected[k]) for k in expected)
        assert row["standalone_measured_server_wall_seconds"] > 0
        assert row["controller"]["committed_state_hash"] == row["selected_state_hash"]
        assert row["independent_gain_direction"] in (-1, 0, 1)
        assert row["public_independent_gain_direction_agrees"] == (
            row["public_gain_direction"] == row["independent_gain_direction"])
    assert record["counters"]["controller_public_dataset_passes"] == 29
    assert record["counters"]["controller_encoder_batches"] == 29 * 8
    assert record["counters"]["controller_head_evals"] == 53
    assert record["rules"]["ordinary_mean"]["controller"]["features_count"] == 0
    assert record["timers"]["measurement"].startswith("five actual standalone")
    assert record["posthoc_readout_grid"]["role"] == "diagnostic_only_after_public_selection"
    assert len(record["posthoc_readout_grid"]["grid_independent_NLL"]) == 16
    assert len(calls) == record["counters"]["independent_readout_unique_state_evals"]
    assert len(calls) <= 5 + 16
    assert len(record["independent_readout_by_state_hash"]) == len(calls)
    json.dumps(record)
    assert model.training and torch.equal(torch.random.get_rng_state(), before_torch)
    assert random.getstate() == before_py
    assert all(torch.equal(state(model)[k], parent[k]) for k in parent)

    corners = record["four_corners"]
    for name, (alpha, beta) in {"00": (0, 0), "10": (1, 0),
                                "01": (0, 1), "11": (1, 1)}.items():
        expected = direct_reader(model, readout_set)[0](interpolate(parent, proposal, alpha, beta))["loss"]
        assert corners[name]["independent_NLL"] == pytest.approx(expected, abs=1e-12)
    for key, label in (("public_NLL", "public"), ("independent_NLL", "independent")):
        expected = (corners["10"][key] + corners["01"][key]
                    - corners["11"][key] - corners["00"][key])
        assert record["interaction_I"][label] == pytest.approx(expected, abs=1e-12)
    assert record["comparisons"]["block_minus_scalar17_independent_NLL"] == pytest.approx(
        record["rules"]["block16"]["independent_NLL"]
        - record["rules"]["scalar17"]["independent_NLL"])
    assert record["comparisons"]["block_minus_ordinary_mean_independent_NLL"] == pytest.approx(
        record["rules"]["block16"]["independent_NLL"]
        - record["rules"]["ordinary_mean"]["independent_NLL"])


def test_readout_failure_restores_caller_and_does_not_replace_selection():
    model, parent, proposal, children, weights, public, _ = fixture()
    model.train()
    before_rng = torch.random.get_rng_state().clone()

    def invalid(candidate):
        load_state(model, candidate)
        model.eval()
        torch.manual_seed(99)
        return {"loss": float("nan")}

    with pytest.raises(FloatingPointError, match="nonfinite|invalid mean NLL"):
        evaluate_feedback(model, parent, proposal, children, weights, public,
                          invalid, 3, 1, posthoc_readout_grid=False)
    assert model.training and torch.equal(before_rng, torch.random.get_rng_state())
    assert all(torch.equal(state(model)[key], parent[key]) for key in parent)


def test_public_contract_rejects_wrong_count_before_selection():
    model, parent, proposal, children, weights, _, _ = fixture()
    with pytest.raises(ValueError, match="public training count"):
        evaluate_feedback(model, parent, proposal, children, weights,
                          TinyDataset(303, 10), lambda _: 1., 3, 1)


def test_readout_choice_cannot_change_public_coefficients():
    model, parent, proposal, children, weights, public, _ = fixture()
    first_reader, _ = direct_reader(model, TinyDataset(401, 48))
    second_reader, _ = direct_reader(model, TinyDataset(402, 48))
    states_a, a = evaluate_feedback(model, parent, proposal, children, weights,
                                     public, first_reader, 3, 1,
                                     posthoc_readout_grid=False)
    states_b, b = evaluate_feedback(model, parent, proposal, children, weights,
                                     public, second_reader, 3, 1,
                                     posthoc_readout_grid=False)
    for rule in states_a:
        assert all(torch.equal(states_a[rule][key], states_b[rule][key])
                   for key in parent)
        assert (a["rules"][rule]["alpha"], a["rules"][rule]["beta"]) == (
            b["rules"][rule]["alpha"], b["rules"][rule]["beta"])
    assert any(a["rules"][rule]["independent_NLL"] != b["rules"][rule]["independent_NLL"]
               for rule in states_a)


def test_mismatched_role_rejected():
    model, parent, proposal, children, weights, public, _ = fixture()
    with pytest.raises(ValueError, match="readout role"):
        evaluate_feedback(
            model, parent, proposal, children, weights, public, lambda _: .5,
            3, 1, config={"data": {"readout_role": "public_training_512"}},
        )
