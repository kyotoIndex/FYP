"""CPU contracts for three gains, 24-matrix cancellation and sealed search."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pcla_research_v7.diagnostic_bank import PREFIXES, SUPPORTS, make_jq_lock
from pcla_research_v7.diagnostic_readout import (
    DiagnosticView, cancellation_and_linearity, finite_bank_selection,
    measure_bank_readouts,
)


CLIENTS = tuple(range(10))
MODULES = tuple(f"bert.encoder.layer.{layer}.attention.self.{projection}"
                for layer in range(12) for projection in ("query", "value"))


@dataclass(frozen=True)
class Data:
    size: int
    target: float
    role: str

    def __len__(self):
        return self.size


def view(role, size, source, start=0, target=0.):
    return DiagnosticView(role, Data(size, target, role), source,
                          tuple(range(start, start + size)))


def state():
    result = {f"{name}.lora_a": torch.tensor([[1.]], dtype=torch.float64)
              for name in MODULES}
    result.update({f"{name}.lora_b": torch.tensor([[2.]], dtype=torch.float64)
                   for name in MODULES})
    result["classifier.weight"] = torch.tensor([[0.]], dtype=torch.float64)
    result["classifier.bias"] = torch.tensor([0.], dtype=torch.float64)
    return result


def aggregate(parent, children, weights):
    assert len(children) == len(weights) == 10 and weights == [1] * 10
    return {key: parent[key] + sum((child[key] - parent[key]) * weight
                                   for child, weight in zip(children, weights)) / sum(weights)
            for key in parent}


def setup():
    parent = state()
    prefixes = {support: {} for support in SUPPORTS}
    for index, support in enumerate(SUPPORTS):
        for cid in CLIENTS:
            prefixes[support][cid] = {}
            for step in PREFIXES:
                child = {key: value.clone() for key, value in parent.items()}
                for name in MODULES:
                    child[f"{name}.lora_a"] += step * (index + 1) * (cid + 1) / 10000
                child["classifier.weight"] += step * (cid + 1) / 100
                child["classifier.bias"] -= step * (index + 1) / 100
                prefixes[support][cid][step] = child
    choices = {cid: "JL_h32" for cid in CLIENTS}
    digests = {cid: "a" * 64 for cid in CLIENTS}
    lock = make_jq_lock(object_id="Cohort_s101_r10_a", participants=CLIENTS,
                        choice_by_client=choices, support_sha256_by_client=digests,
                        h32_sha256="b" * 64, locked_at_utc="2026-09-28T01:02:03Z",
                        information_access=("parent_state", "local_h32"))
    aggregate_states = {support: {} for support in (*SUPPORTS, "JQ_xi1")}
    for support in aggregate_states:
        for step in PREFIXES:
            children = [prefixes[choices[cid] if support == "JQ_xi1" else support][cid][step]
                        for cid in CLIENTS]
            aggregate_states[support][step] = aggregate(parent, children, [1] * 10)
    bank = {"schema": "pcla-v7-cohort-bank-1", "object_id": "Cohort_s101_r10_a",
            "anchor_sha256": "c" * 64, "factor": "a", "participants": list(CLIENTS),
            "prefix_states": prefixes, "aggregate_states": aggregate_states,
            "jq_lock_sha256": lock.sha256,
            "support_sha256": {cid: {support: digests[cid] for support in SUPPORTS}
                               for cid in CLIENTS}}
    return bank, parent, lock


def model_loss(parameters, dataset):
    effective = sum((parameters[f"{name}.lora_b"] @ parameters[f"{name}.lora_a"]).item()
                    for name in MODULES)
    signal = effective + parameters["classifier.weight"].sum().item() + parameters["classifier.bias"].sum().item()
    return (signal - dataset.target) ** 2


def test_selection_uses_only_1024_then_readout_three_gains_and_fixed_denominator():
    bank, parent, jq_lock = setup()
    calls = []
    def tracked_loss(parameters, dataset):
        calls.append(dataset.role)
        return model_loss(parameters, dataset)
    selection = view("selection1024", 1024, "recipe_dev", 0, 50.)
    search = finite_bank_selection(bank=bank, parent=parent, selection=selection,
                                   loss=tracked_loss, aggregate=aggregate)
    assert 1 <= search.evaluations <= 80 and calls == ["selection1024"] * search.evaluations
    local = {cid: view("local64", 64, f"client:{cid}", target=50.) for cid in CLIENTS}
    independent = view("readout2048", 2048, "recipe_dev", 1024, 50.)
    result = measure_bank_readouts(bank=bank, parent=parent, local64=local,
                                   readout=independent, selection_lock=search,
                                   jq_lock=jq_lock, loss=tracked_loss,
                                   aggregate=aggregate)
    assert result["selection_lock_sha256"] == search.sha256
    assert "readout2048" in calls and "local64" in calls
    assert set(result["values"]) == set((*SUPPORTS, "JQ_xi1", "finite_bank_search"))
    assert set(result["values"]["local"]) == set(PREFIXES)
    assert set(result["values"]["finite_bank_search"]) == {16}
    row = result["values"]["local"][1]
    assert row["cancellation_status"] == "measured"
    assert row["max_effective_linearity_relative_error"] < 1e-6
    assert row["nonadditivity"] == pytest.approx(row["aggregate_readout2048_benefit"] -
                                                  sum(row["weighted_readout2048_benefit_by_client"].values()))
    cid = 0
    expected_local = model_loss(parent, local[cid].dataset) - model_loss(
        bank["prefix_states"]["local"][cid][1], local[cid].dataset)
    assert row["local64_benefit_by_client"][cid] == pytest.approx(expected_local)
    assert row["head_only_readout2048_benefit"] != row["lora_only_readout2048_benefit"]


def test_data_roles_and_lock_gate_prevent_readout_based_selection():
    bank, parent, jq_lock = setup()
    used = []
    def loss(parameters, dataset):
        used.append(dataset.role)
        return model_loss(parameters, dataset)
    with pytest.raises(ValueError, match="selection1024"):
        finite_bank_selection(bank=bank, parent=parent,
            selection=view("readout2048", 1024, "recipe_dev"),
            loss=loss, aggregate=aggregate)
    assert used == []
    def wrong_denominator(base, children, weights):
        return {key: base[key] + sum(child[key] - base[key] for child in children) / 9
                for key in base}
    with pytest.raises(ValueError, match="fixed ten-client denominator"):
        finite_bank_selection(bank=bank, parent=parent,
            selection=view("selection1024", 1024, "recipe_dev"),
            loss=loss, aggregate=wrong_denominator)
    assert used == []
    local = {cid: view("local64", 64, f"client:{cid}") for cid in CLIENTS}
    readout = view("readout2048", 2048, "recipe_dev", 1024)
    with pytest.raises(ValueError, match="locked selection1024"):
        measure_bank_readouts(bank=bank, parent=parent, local64=local,
            readout=readout, selection_lock=None, jq_lock=jq_lock,
            loss=loss, aggregate=aggregate)
    assert used == []
    search = finite_bank_selection(bank=bank, parent=parent,
        selection=view("selection1024", 1024, "recipe_dev"),
        loss=loss, aggregate=aggregate)
    used.clear()
    with pytest.raises(ValueError, match="overlap"):
        measure_bank_readouts(bank=bank, parent=parent, local64=local,
            readout=view("readout2048", 2048, "recipe_dev", 0),
            selection_lock=search, jq_lock=jq_lock, loss=loss, aggregate=aggregate)
    assert used == []


def test_cancellation_zero_denominator_is_na_and_independent_matrix_check():
    parent = state()
    children = []
    for cid in CLIENTS:
        child = {key: value.clone() for key, value in parent.items()}
        child["classifier.weight"] += cid + 1
        children.append(child)
    combined = aggregate(parent, children, [1] * 10)
    result = cancellation_and_linearity(parent, children, combined)
    assert result["cancellation"] is None
    assert result["cancellation_status"] == "NA_zero_denominator"
    broken = {key: value.clone() for key, value in combined.items()}
    broken[f"{MODULES[0]}.lora_a"] += 1
    with pytest.raises(ValueError, match="independent deltaM"):
        cancellation_and_linearity(parent, children, broken)
