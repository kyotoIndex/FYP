"""Independent CPU invariants for the diagnostic driver; no GPU or bank replay."""

import json

import pytest
import torch

from pcla_repair.testing import TinyDataset, TinyModel
from pcla_research_v7.runtime import state
from scripts import run_pcla_v8_diagnostic as driver


def test_sealed_rank8_cohort_is_reused_without_local_training(tmp_path, monkeypatch):
    torch.manual_seed(611)
    parent = state(TinyModel())
    context = {"parent_state": parent}
    config = {"job": {"phase": "R8_TRANSFER"}}
    collected = {"calls": 0}
    value = {
        "aggregate_updates": [{"support": "rank8"}],
        "local_calls": 10, "optimizer_steps": 160,
        "elapsed_seconds": 1.25,
    }

    def collect(_context, _config):
        collected["calls"] += 1
        return value

    monkeypatch.setattr(driver, "collect_rank8", collect)
    first, first_receipt = driver.restore_or_collect(context, config, tmp_path, "cfg-sha")
    assert first == value and first_receipt["reused"] is False
    assert collected["calls"] == 1
    assert driver.sha(tmp_path / "new_cohort.pt") == json.loads(
        (tmp_path / "new_cohort_seal.json").read_text())["pt_sha256"]

    def fail_if_retrained(_context, _config):
        raise AssertionError("sealed cohort attempted retraining")

    monkeypatch.setattr(driver, "collect_rank8", fail_if_retrained)
    restored, restored_receipt = driver.restore_or_collect(
        context, config, tmp_path, "cfg-sha")
    assert restored == value and restored_receipt["reused"] is True
    with pytest.raises(ValueError, match="sealed cohort/config/parent differs"):
        driver.restore_or_collect(context, config, tmp_path, "different-cfg")


def test_transfer_chain_reads_raw_children_and_applies_weight_once():
    torch.manual_seed(612)
    model = TinyModel()
    parent = state(model)
    children = [
        {key: value + (i + 1) * .002 for key, value in parent.items()}
        for i in range(10)
    ]
    proposal = {
        key: sum(child[key] for child in children) / 10
        for key in parent
    }
    independent = TinyDataset(613, 48)
    context = {
        "model": model,
        "parent_state": parent,
        "participants": list(range(10)),
        "local_readout_by_client": {i: TinyDataset(700 + i, 64)
                                    for i in range(10)},
    }
    update = {"children": children, "proposal": proposal,
              "weights": [0.1] * 10}

    def reader(candidate):
        return driver.raw_readout(model, candidate, independent,
                                  expected_samples=48)

    result = driver.transfer_chain(context, update, reader)
    assert len(result["clients"]) == 10
    assert result["weighted_individual_global_NLL_gain"] == pytest.approx(
        sum(0.1 * (result["global_parent"]["loss"]
                   - row["individual_global"]["loss"])
            for row in result["clients"]), abs=1e-12)
    assert result["aggregate_global_NLL_gain"] == pytest.approx(
        result["global_parent"]["loss"] - result["aggregate"]["loss"],
        abs=1e-12)
    assert all(row["sample_weight"] == .1 for row in result["clients"])
    assert all(row["individual_global"]["state_hash"] == row["child_state_hash"]
               for row in result["clients"])
    assert all(torch.equal(state(model)[key], parent[key]) for key in parent)
