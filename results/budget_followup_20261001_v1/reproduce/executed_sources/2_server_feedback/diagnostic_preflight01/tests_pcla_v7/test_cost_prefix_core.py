"""Cost timing must preserve the science prefix and count denied control."""

import hashlib
import json
import torch
import pytest

from pcla_repair.testing import fixture, TinyDataset
from pcla_research_v7.execution_timing import (
    COST_LABEL_CATEGORIES, ExecutionTimingLedger, required_breakdown,
)
from pcla_research_v7.runtime import _load_checkpoint, run_experiment
from scripts.run_pcla_v7 import (ROOT, persist_failed_cost_timing,
                                 verify_instrumented_source_manifest)


class FixedWire:
    def preview(self, **kwargs):
        return {"bytes": 100}

    def commit(self, preview, cache, parent, children, cohort, supports, *, diagnostic):
        return {"events": cache.get("events", []) + [(tuple(cohort), diagnostic)]}


def _job():
    return {"method": "full_both", "representation": "N", "rank": 4,
            "seed": 123, "lr": .01, "min_rounds": 3, "max_rounds": 4}


def _config():
    return {"byte_cap": 250, "clients_per_round": 2, "batch_size": 8,
            "eval_batch_size": 16, "local_epochs": 1}


def test_instrumented_manifest_binds_actual_runtime_source(tmp_path):
    required = [
        "pcla_research_v7/runtime.py", "pcla_research_v7/cohort_runtime.py",
        "pcla_research_v7/coordinates.py", "pcla_research_v7/timing_hooks.py",
        "pcla_research_v7/execution_timing.py", "pcla_research_v7/data.py",
        "pcla_research_v7/selectors.py", "pcla_research_v7/transport.py",
        "pcla_repair/core.py", "pcla_repair/runtime.py", "scripts/run_pcla_v7.py",
    ]
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
             for name in required}
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"files": files}))
    assert verify_instrumented_source_manifest(manifest) == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    files["pcla_research_v7/runtime.py"] = "0" * 64
    manifest.write_text(json.dumps({"files": files}))
    with pytest.raises(RuntimeError, match="instrumented source identity differs"):
        verify_instrumented_source_manifest(manifest)


def test_failed_cost_timing_persists_partial_events_without_acceptance(tmp_path):
    ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    ledger.begin_envelope()
    ledger.start("client_training", round_id=7)
    partial = persist_failed_cost_timing(tmp_path, ledger, "RuntimeError: client failed")
    event_file = tmp_path / "timing_events_partial.json"
    failure = json.loads((tmp_path / "cost_timing_failure.json").read_text())
    assert partial["partial"] is True
    assert partial["open_intervals"][-1]["round"] == 7
    assert failure["accepted"] is False
    assert failure["event_file_sha256"] == hashlib.sha256(event_file.read_bytes()).hexdigest()
    assert not (tmp_path / "cost_timing_unaccepted.json").exists()


def test_cost_prefix_stops_before_diagnostic_and_retains_original_return(tmp_path):
    model, _, clients, capacities, evaluation = fixture(7)
    original = run_experiment(model, clients, evaluation, capacities, _job(),
        _config(), tmp_path / "original", {"evaluation_role": "synthetic"}, FixedWire())
    replay_model, _, replay_clients, replay_capacities, replay_evaluation = fixture(7)
    ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    ledger.begin_envelope()  # prepare() begins here for a real replay.
    replay = run_experiment(replay_model, replay_clients, replay_evaluation,
        replay_capacities, _job(), _config(), tmp_path / "replay",
        {"evaluation_role": "synthetic"}, FixedWire(), timing=ledger,
        cost_prefix=True)
    assert len(original) == 3 and len(replay) == 2
    original_return = torch.load(tmp_path / "original" / "budget_return.pt",
        map_location="cpu", weights_only=True)
    replay_return = torch.load(tmp_path / "replay" / "budget_return.pt",
        map_location="cpu", weights_only=True)
    assert replay_return["round"] == original_return["round"] == 2
    assert replay_return["deployed_bytes"] == original_return["deployed_bytes"] == 200
    assert all(torch.equal(replay_return["adapter"][name], tensor)
               for name, tensor in original_return["adapter"].items())
    assert _load_checkpoint(tmp_path / "replay" / "latest.pt")["diagnostic_bytes"] == 0
    events = ledger.snapshot()["events"]
    assert any(event["round"] == 3 and event["label"] == "budget_and_return_control"
               for event in events)
    assert sum(required_breakdown(ledger.snapshot()).values()) > 0
    assert ledger.coverage("envelope")["passes"]


def test_cost_prefix_rejects_missing_envelope_before_any_output(tmp_path):
    model, _, clients, capacities, evaluation = fixture(7)
    ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    with pytest.raises(ValueError, match="open warm-W0 envelope"):
        run_experiment(model, clients, evaluation, capacities, _job(),
            _config(), tmp_path / "rejected", {"evaluation_role": "synthetic"},
            FixedWire(), timing=ledger, cost_prefix=True)
    assert not (tmp_path / "rejected").exists()


def test_v_anchor_is_complete_resumable_payload(tmp_path):
    model, _, clients, capacities, evaluation = fixture(13)
    job = {**_job(), "phase": "V", "max_rounds": 10}
    run_experiment(model, clients, evaluation, capacities, job,
        {**_config(), "byte_cap": 10**9}, tmp_path / "v",
        {"evaluation_role": "synthetic"}, FixedWire(), stop_after=10)
    for round_id in (10,):
        anchor = _load_checkpoint(tmp_path / "v" / f"anchor_round_{round_id:03d}.pt")
        assert anchor["round"] == round_id
        assert anchor["budget_stop_round"] == _load_checkpoint(tmp_path / "v" / "latest.pt")["budget_stop_round"]
        assert {"adapter", "metrics", "cache", "torch_rng", "python_rng",
                "deployed_bytes", "diagnostic_bytes", "ema_effective"} <= set(anchor)
        assert len(anchor["metrics"]) == round_id


@pytest.mark.parametrize("representation", ["D", "S"])
def test_coordinate_technical_logits_are_excluded_without_changing_return(tmp_path, representation):
    job = {**_job(), "method": "local", "representation": representation,
           "period": 1, "min_rounds": 2, "max_rounds": 2}
    config = {**_config(), "byte_cap": 10**9}
    model, _, clients, capacities, evaluation = fixture(17)
    run_experiment(model, clients, evaluation, capacities, job, config,
        tmp_path / "original", {"evaluation_role": "synthetic"}, FixedWire())
    replay_model, _, replay_clients, replay_capacities, replay_evaluation = fixture(17)
    ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    ledger.begin_envelope()
    run_experiment(replay_model, replay_clients, replay_evaluation,
        replay_capacities, job, config, tmp_path / "replay",
        {"evaluation_role": "synthetic"}, FixedWire(), timing=ledger,
        cost_prefix=True)
    original = torch.load(tmp_path / "original" / "budget_return.pt",
        map_location="cpu", weights_only=True)
    replay = torch.load(tmp_path / "replay" / "budget_return.pt",
        map_location="cpu", weights_only=True)
    assert original["round"] == replay["round"] == 2
    assert all(torch.equal(tensor, replay["adapter"][name])
               for name, tensor in original["adapter"].items())
    assert ledger.snapshot()["label_seconds"]["technical_logits_check"] > 0
    assert ledger.snapshot()["label_seconds"]["coordinate_and_server_decision"] > 0
    assert ledger.coverage("envelope")["passes"]
    assert all(event["round"] is None or event["round"] <= 2
               for event in ledger.snapshot()["events"])


@pytest.mark.parametrize("method,representation", [("local", "T"), ("JL", "N")])
def test_cohort_cost_prefix_preserves_control_packets_and_parent(tmp_path, method, representation):
    job = {**_job(), "method": method, "representation": representation,
           "period": 1, "min_rounds": 2, "max_rounds": 2, "xi": 1.}
    config = {**_config(), "byte_cap": 10**9, "probe_batch_size": 8,
              "probe_batches": 2}
    model, _, clients, capacities, evaluation = fixture(19)
    clients = {cid: TinyDataset(100 + cid, 32) for cid in clients}
    original = run_experiment(model, clients, evaluation, capacities, job, config,
        tmp_path / "original", {"evaluation_role": "synthetic"})
    replay_model, _, replay_clients, replay_capacities, replay_evaluation = fixture(19)
    replay_clients = {cid: TinyDataset(100 + cid, 32) for cid in replay_clients}
    ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    ledger.begin_envelope()
    replay = run_experiment(replay_model, replay_clients, replay_evaluation,
        replay_capacities, job, config, tmp_path / "replay",
        {"evaluation_role": "synthetic"}, timing=ledger, cost_prefix=True)
    assert [row["deployed_bytes"] for row in original] == [row["deployed_bytes"] for row in replay]
    assert [row["support_trace"] for row in original] == [row["support_trace"] for row in replay]
    a = _load_checkpoint(tmp_path / "original" / "latest.pt")
    b = _load_checkpoint(tmp_path / "replay" / "latest.pt")
    assert a["cache"] == b["cache"]
    assert all(torch.equal(a["adapter"][name], value)
               for name, value in b["adapter"].items())
    assert ledger.snapshot()["label_seconds"]["message_processing"] > 0
    assert ledger.snapshot()["label_seconds"]["scoring_and_shared_information"] > 0
    if representation == "T":
        assert ledger.snapshot()["label_seconds"]["technical_logits_check"] > 0
    assert ledger.coverage("envelope")["passes"]
