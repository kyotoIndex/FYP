"""CPU-only checks for native cost-prefix accounting and scientific parity."""
import json

import pytest
import torch

from pcla_research_v7.execution_timing import (ExecutionTimingLedger, NECESSARY_ALGORITHM,
                                                RESEARCH_OBSERVATION,
                                                required_breakdown)
from pcla_research_v7.native_runtime import NATIVE_TIMING_LABELS, NativeWire, run_native
from tests_pcla_v7.test_native_runtime import Data, setup


@pytest.mark.parametrize("method", ["fedsb", "fedsvd", "fedplora"])
def test_cost_prefix_max_round_parity_and_no_extra_preflight(method, tmp_path, monkeypatch):
    baseline_model, clients, capacities, job, recipe, config = setup(method)
    provenance = {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}}
    baseline = run_native(baseline_model, clients, Data(), capacities, job, recipe,
                          config, tmp_path / "baseline", provenance)
    model, clients, capacities, job, recipe, config = setup(method)
    calls = []
    original_preview = NativeWire.preview
    def traced(self, method, round_id, *args):
        calls.append(round_id)
        return original_preview(self, method, round_id, *args)
    monkeypatch.setattr(NativeWire, "preview", traced)
    timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS)
    timing.begin_envelope()
    measured = run_native(model, clients, Data(), capacities, job, recipe,
                          config, tmp_path / "cost", provenance,
                          timing=timing, cost_prefix=True)
    assert calls == [1, 2]
    assert [r["evaluation"] for r in measured] == [r["evaluation"] for r in baseline]
    clean = torch.load(tmp_path / "baseline/latest.pt", weights_only=True)
    cost = torch.load(tmp_path / "cost/latest.pt", weights_only=True)
    assert cost["budget_stop_round"] == clean["budget_stop_round"] == 2
    assert cost["deployed_bytes"] == clean["deployed_bytes"]
    assert all(torch.equal(clean["adapter"][name], cost["adapter"][name])
               for name in clean["adapter"])
    snap = timing.snapshot()
    assert snap["bucket_category_seconds"]["deployed_prefix"][NECESSARY_ALGORITHM] > 0
    assert snap["bucket_category_seconds"]["deployed_prefix"][RESEARCH_OBSERVATION] > 0
    assert snap["label_seconds"]["checkpoint_and_audit"] > 0
    assert sum(required_breakdown(snap).values()) == pytest.approx(
        snap["bucket_category_seconds"]["deployed_prefix"][NECESSARY_ALGORITHM])
    assert timing.coverage("envelope")["passes"]


def test_cost_prefix_preserves_refusal_and_skips_diagnostic(tmp_path):
    model, clients, capacities, job, recipe, config = setup("fedsvd")
    clients = {cid: Data(40) for cid in clients}
    job.update(min_rounds=2, max_rounds=3)
    config["byte_cap"] = 2400
    timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS)
    timing.begin_envelope()
    rows = run_native(model, clients, Data(), capacities, job, recipe,
                      config, tmp_path, {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}},
                      timing=timing, cost_prefix=True)
    status = json.loads((tmp_path / "status.json").read_text())
    assert len(rows) == 1
    assert status["budget_stop_round"] == 1
    assert status["diagnostic_after_budget_bytes"] == 0
    assert (tmp_path / "budget_return.pt").is_file()
    assert timing.snapshot()["bucket_category_seconds"]["deployed_prefix"][NECESSARY_ALGORITHM] > 0


def test_cost_prefix_rejects_missing_envelope_and_technical_stop(tmp_path):
    model, clients, capacities, job, recipe, config = setup("fedsb")
    timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS)
    provenance = {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}}
    with pytest.raises(ValueError, match="active cost envelope"):
        run_native(model, clients, Data(), capacities, job, recipe, config,
                   tmp_path / "missing", provenance, timing=timing, cost_prefix=True)
    timing.begin_envelope()
    with pytest.raises(ValueError, match="forbids technical stop"):
        run_native(model, clients, Data(), capacities, job, recipe, config,
                   tmp_path / "stop", provenance, timing=timing,
                   cost_prefix=True, stop_after=1)
    timing.end_envelope()


def test_failed_cost_prefix_keeps_intervals_and_attempted_wire(tmp_path, monkeypatch):
    from pcla_research_v7 import native_runtime
    model, clients, capacities, job, recipe, config = setup("fedplora")
    original = native_runtime.local_train_native
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected timed failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(native_runtime, "local_train_native", fail_second)
    timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS)
    timing.begin_envelope()
    with pytest.raises(RuntimeError, match="injected timed failure"):
        run_native(model, clients, Data(), capacities, job, recipe, config,
                   tmp_path, {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}},
                   timing=timing, cost_prefix=True)
    timing.end_envelope()
    assert timing.snapshot()["bucket_category_seconds"]["deployed_prefix"][NECESSARY_ALGORITHM] > 0
    journal = json.loads((tmp_path / "wire_attempt_journal.json").read_text())
    assert [event["direction"] for event in journal["events"]].count("download") == 2
    assert [event["direction"] for event in journal["events"]].count("upload") == 1
