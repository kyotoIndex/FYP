"""CPU-only native CLI timing handoff, without model loading or dispatch."""
import hashlib
import json
import sys

import torch

from scripts import run_pcla_v7_native as driver


def test_cost_driver_persists_unaccepted_interval_record(tmp_path, monkeypatch):
    config = {
        "job": {"job_id": "D107_s401", "method": "fedsvd", "rank": 16,
                "lr": .1, "seed": 401},
        "recipe": {"recipe_id": "D107", "method": "fedsvd",
                   "required_adapter": "fedsvd", "rank": 16, "lr": .1},
        "runtime": {"byte_cap": 10},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    source_manifest = tmp_path / "instrumented.json"
    source_manifest.write_text(json.dumps({"files": {name: "0" * 64 for name in (
        "pcla_research_v7/native_runtime.py", "scripts/run_pcla_v7_native.py",
        "pcla_research_v7/native_fedsb.py", "pcla_research_v7/native_fedsvd.py",
        "pcla_research_v7/native_fedplora.py")}}))
    out = tmp_path / "out"
    seen = {}

    def fake_prepare(expanded, path, device, *, timing=None):
        assert path == config_path and str(device) == "cpu"
        assert timing is not None
        timing.begin_envelope()
        with timing.span("task_initialization"):
            model = torch.nn.Linear(1, 1)
        return model, {}, {}, [], {"hardware": {"name": "cpu"}}

    def fake_run_native(model, clients, evaluation, capacities, job, recipe,
                        runtime, output, provenance, *, stop_after=None,
                        timing=None, cost_prefix=False):
        assert timing.active_labels == ("envelope",)
        assert cost_prefix and stop_after is None
        with timing.span("budget_and_return_control", round_id=1):
            seen["called"] = True
        timing.end_envelope()
        return [{"round": 1}]

    monkeypatch.setattr(driver, "prepare", fake_prepare)
    monkeypatch.setattr(driver, "run_native", fake_run_native)
    monkeypatch.setattr(driver, "verify_instrumented_source_manifest", lambda path: "source-sha")
    monkeypatch.setattr(sys, "argv", ["run_pcla_v7_native.py", "--config", str(config_path),
                                  "--out", str(out), "--device", "cpu", "--cost-prefix",
                                  "--instrumented-source-manifest", str(source_manifest)])
    driver.main()
    assert seen["called"]
    draft = json.loads((out / "cost_prefix_draft.json").read_text())
    events = (out / "exclusive_timing.json").read_bytes()
    assert draft["accepted"] is False and draft["failed"] is False
    assert draft["coverage"]["passes"]
    assert draft["required_seconds_unaccepted"] > 0
    assert draft["instrumented_source_manifest_sha256"] == "source-sha"
    assert draft["interval_file_sha256"] == hashlib.sha256(events).hexdigest()
    assert not (out / ".run.lock").exists()
