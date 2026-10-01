"""CPU checks for same-mask h32/full predictions and bank-only readout."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pcla_repair.core import tensor_digest
from pcla_research_v7.coordination import client_payload, virtual_adamw_step
from pcla_research_v7.diagnostic_bank import SUPPORTS, make_jq_lock, support_digest
from pcla_research_v7.diagnostic_prediction import (
    compare_fixed_bank_predictions, readout_locked_full_choice, spearman_four,
)
from pcla_research_v7.diagnostic_readout import DiagnosticView
from pcla_research_v7.diagnostic_supports import _payload_sha
from pcla_research_v7.runtime import aggregate


CLIENTS = tuple(range(10))
NAMES = tuple(f"layer.{layer}.{'query' if side == 0 else 'value'}"
              for layer in range(12) for side in range(2))


@dataclass(frozen=True)
class Rows:
    count: int
    target: float = 0.

    def __len__(self):
        return self.count


def fixture(factor="a"):
    parent = {}
    for name in NAMES:
        parent[f"{name}.lora_a"] = torch.tensor([[1., 0.], [0., 1.]])
        parent[f"{name}.lora_b"] = torch.tensor([[.4, .1], [.2, .3]])
    parent["classifier.weight"] = torch.zeros(2, 2)
    parent["classifier.bias"] = torch.zeros(2)
    factors = {name: (parent[f"{name}.lora_b"], parent[f"{name}.lora_a"])
               for name in NAMES}
    gradients = {}
    payloads = []
    for cid in CLIENTS:
        gradients[cid] = {}
        for index, name in enumerate(NAMES):
            parameter = factors[name][1 if factor == "a" else 0]
            base = torch.zeros_like(parameter, dtype=torch.float64)
            if index < 2:
                base[0, 0] = (cid + 1) * (index + 1) / 100
                base[1, 1] = -(cid + 2) / 200
            gradients[cid][name] = base
        payloads.append(client_payload(cid, 10, factors, gradients[cid], factor,
                       .001, 101, 11, h=32))
    locations = ((NAMES[0], 0), (NAMES[0], 1), (NAMES[1], 0), (NAMES[1], 1))
    supports = {}
    for cid in CLIENTS:
        supports[cid] = {}
        for choice, (picked_name, picked_rank) in zip(SUPPORTS, locations):
            support = {}
            for name in NAMES:
                active = torch.zeros(2, dtype=torch.bool)
                if name == picked_name:
                    active[picked_rank] = True
                support[name] = {"a": active if factor == "a" else torch.zeros_like(active),
                                 "b": active if factor == "b" else torch.zeros_like(active)}
            supports[cid][choice] = support
    jq_choices = {cid: "JL_h32" for cid in CLIENTS}
    jq = make_jq_lock(object_id="Cohort_s101_r10_a", participants=CLIENTS,
        choice_by_client=jq_choices,
        support_sha256_by_client={cid: support_digest(supports[cid]["JL_h32"], factor=factor)
                                  for cid in CLIENTS},
        h32_sha256=_payload_sha(payloads), locked_at_utc="2026-09-28T01:02:03Z",
        information_access=("parent_state", "local_h32"))
    kwargs = dict(supports=supports, jq_lock=jq, full_probe_gradients=gradients,
                  h32_payloads=payloads, h32_sha256=jq.h32_sha256,
                  object_id="Cohort_s101_r10_a", anchor_sha256="a" * 64,
                  parent=parent, factor=factor, participants=CLIENTS,
                  budgets={cid: 1 for cid in CLIENTS}, seed=101, round_id=11,
                  learning_rate=.001)
    return kwargs


def bank_from(kwargs):
    parent = kwargs["parent"]
    factor = kwargs["factor"]
    prefix_states = {choice: {} for choice in SUPPORTS}
    for choice in SUPPORTS:
        for cid in CLIENTS:
            child = {name: value.clone() for name, value in parent.items()}
            for name, masks in kwargs["supports"][cid][choice].items():
                key = name + ".lora_" + factor
                for rank in masks[factor].nonzero().flatten().tolist():
                    if factor == "a":
                        child[key][rank] += (cid + 1) / 100
                    else:
                        child[key][:, rank] += (cid + 1) / 100
            child["classifier.bias"] += (cid + 1) / 1000
            prefix_states[choice][cid] = {16: child}
    return {"schema": "pcla-v7-cohort-bank-1", "object_id": kwargs["object_id"],
            "anchor_sha256": kwargs["anchor_sha256"], "factor": factor,
            "participants": list(CLIENTS), "prefix_states": prefix_states,
            "support_sha256": {cid: {choice: support_digest(
                kwargs["supports"][cid][choice], factor=factor) for choice in SUPPORTS}
                               for cid in CLIENTS}}


@pytest.mark.parametrize("factor", ["a", "b"])
def test_same_four_masks_prediction_phi_and_locked_bank_readout(factor):
    inputs = fixture(factor)
    # Prediction has no bank outcomes or loss callback in its signature.
    result = compare_fixed_bank_predictions(**inputs)
    lock = result["full_choice_lock"]
    assert lock.prefix == 16 and lock.evaluations <= 80
    assert set(dict(lock.choices).values()) <= set(SUPPORTS)
    assert result["candidate_order"] == list(SUPPORTS)
    for direction in ("h32", "full"):
        prediction = result["prediction"][direction]
        cyclic = prediction["fixed_combinations"]["all_cyclic"]
        assert prediction["lambda_xi1"] == pytest.approx(
            max(0., cyclic["linear"]) / cyclic["quadratic"] if cyclic["quadratic"] else 0.)
        for item in prediction["fixed_combinations"].values():
            assert item["phi"] == pytest.approx(
                item["linear"] - .5 * prediction["lambda_xi1"] * item["quadratic"])
        assert prediction["objective_final"] >= prediction["objective_initial"] - 1e-12
    cid = 0
    one = result["per_client"][cid]["one_step_linear_gain"]["full"]["local"]
    name = NAMES[0]
    parameter = inputs["parent"][name + ".lora_" + factor]
    update = virtual_adamw_step(parameter, inputs["full_probe_gradients"][cid][name], .001)
    full_update = update.T if factor == "a" else update
    full_cbar = sum((inputs["full_probe_gradients"][other][name].T if factor == "a"
                     else inputs["full_probe_gradients"][other][name] for other in CLIENTS),
                    torch.zeros_like(full_update.double())) / 10
    expected = -.1 * float((full_cbar[:, 0] * full_update[:, 0]).sum())
    assert one == pytest.approx(expected)
    bank = bank_from(inputs)
    view = DiagnosticView("readout2048", Rows(2048, 5.), "train_source", tuple(range(2048)))
    seen = []
    def actual_loss(state, dataset):
        seen.append(dataset)
        return float(sum(t.square().sum() for t in state.values())) / len(dataset)
    readout = readout_locked_full_choice(lock=lock, bank=bank, parent=inputs["parent"],
                  readout=view, loss=actual_loss, aggregate=aggregate)
    assert seen == [view.dataset, view.dataset]
    assert readout["full_choice_lock_sha256"] == lock.sha256
    assert readout["actual_bank_benefit"] == pytest.approx(
        readout["parent_loss"] - readout["selected_loss"])


def test_constant_spearman_is_na_and_corrupt_h32_is_rejected():
    assert spearman_four([1., 1., 1., 1.], [1., 2., 3., 4.]) == (None, "NA_constant")
    assert spearman_four([1., 2., 3., 4.], [4., 3., 2., 1.])[0] == pytest.approx(-1.)
    inputs = fixture()
    first = inputs["h32_payloads"][0]
    first.update_sketches[NAMES[0]][0, 0] += 1
    inputs["h32_sha256"] = _payload_sha(inputs["h32_payloads"])
    with pytest.raises(ValueError, match="identity/probe"):
        compare_fixed_bank_predictions(**inputs)
    inputs["jq_lock"] = make_jq_lock(object_id=inputs["object_id"],
        participants=CLIENTS, choice_by_client={cid: "JL_h32" for cid in CLIENTS},
        support_sha256_by_client={cid: support_digest(
            inputs["supports"][cid]["JL_h32"], factor="a") for cid in CLIENTS},
        h32_sha256=inputs["h32_sha256"], locked_at_utc="2026-09-28T01:02:03Z",
        information_access=("parent_state", "local_h32"))
    with pytest.raises(ValueError, match="same full-probe"):
        compare_fixed_bank_predictions(**inputs)


def test_full_choice_cannot_be_changed_by_bank_outcome_or_mask_swap():
    inputs = fixture()
    locked = compare_fixed_bank_predictions(**inputs)["full_choice_lock"]
    bank = bank_from(inputs)
    view = DiagnosticView("readout2048", Rows(2048), "train_source", tuple(range(2048)))
    def loss(state, dataset):
        return float(sum(t.square().sum() for t in state.values()))
    baseline = readout_locked_full_choice(lock=locked, bank=bank, parent=inputs["parent"],
                  readout=view, loss=loss, aggregate=aggregate)
    cid, choice = locked.choices[0]
    bank["prefix_states"][choice][cid][16]["classifier.bias"] += 100
    changed = readout_locked_full_choice(lock=locked, bank=bank, parent=inputs["parent"],
                  readout=view, loss=loss, aggregate=aggregate)
    assert baseline["actual_bank_benefit"] != changed["actual_bank_benefit"]
    assert baseline["full_choice_lock_sha256"] == changed["full_choice_lock_sha256"]
    bank["support_sha256"][cid][choice] = "0" * 64
    with pytest.raises(ValueError, match="actual bank masks"):
        readout_locked_full_choice(lock=locked, bank=bank, parent=inputs["parent"],
                      readout=view, loss=loss, aggregate=aggregate)
    with pytest.raises(ValueError, match="readout2048"):
        readout_locked_full_choice(lock=locked, bank=bank_from(inputs),
            parent=inputs["parent"], readout=DiagnosticView("selection1024", Rows(1024),
            "train_source", tuple(range(1024))), loss=loss, aggregate=aggregate)
