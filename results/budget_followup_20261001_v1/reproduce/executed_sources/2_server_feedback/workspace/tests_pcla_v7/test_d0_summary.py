"""Receipt, budget endpoint, and attempt-cost boundaries for D0 export."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_pcla_v7_d0.py"
SPEC = importlib.util.spec_from_file_location("d0_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_fixture(tmp_path, *, completed=True, corrupt=False, native_journal=False):
    root = tmp_path / "stage"
    jobs = [{"job_id": f"D001_s{s}", "recipe_id": "D001", "seed": s,
             "phase": "D0", "protocol": "v7"} for s in (401, 409, 419)]
    manifest = tmp_path / "first_batch.jsonl"
    manifest.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    put(root / "queue/matrix_identity.json", {"count": 3, "manifest_sha256": summary.digest(manifest)})
    auditor = root / "sources/core01/scripts/audit_pcla_v7_run.py"
    auditor.parent.mkdir(parents=True, exist_ok=True)
    auditor.write_text("frozen auditor\n")
    source_manifest = root / "sources/core01/SOURCE_MANIFEST.json"
    put(source_manifest, {"files": {}})
    job = jobs[0]
    jid = job["job_id"]
    run = root / "runs/D0" / jid
    cfg = root / "configs/core01" / (jid + ".json")
    put(cfg, {"job": job})
    put(run / "run_identity.json", {"job": job})
    status = {"state": "completed", "completed_round": 100,
              "budget_stop_round": 50, "deployed_bytes": 110,
              "diagnostic_after_budget_bytes": 10}
    if native_journal:
        status.update(actual_attempted_wire_bytes=127, failed_attempt_wire_bytes=7)
    put(run / "status.json", status)
    rows = []
    for r in range(1, 101):
        rows.append({"round": r, "evaluation": {"accuracy": r / 1000, "loss": 1 - r / 1000, "samples": 4},
                     "ema_evaluation": {"accuracy": r / 2000, "loss": 1 - r / 2000, "samples": 4},
                     "deployed_bytes": 110 if r >= 50 else 0,
                     "diagnostic_after_budget_bytes": 10 if r > 50 else 0,
                     "gpu_sharing": "exclusive",
                     "cost": {"local_training_seconds": 2.0, "cuda_peak_allocated_bytes": r}})
    put(run / "metrics.json", rows)
    put(run / "execution_attempts.json", [{"hardware": {"name": "4090D"}, "gpu_sharing": "exclusive"}])
    events = [{"event_id": "r1:c1:down", "round": 1, "direction": "download", "bucket": "deployed_prefix", "bytes": 90},
              {"event_id": "r2:c1:up" if native_journal else "failed:a2:r2:c1:up", "round": 2,
               "direction": "upload", "bucket": "deployed_prefix", "bytes": 20},
              {"event_id": "r51:c1:up", "round": 51, "direction": "upload", "bucket": "diagnostic_after_budget", "bytes": 10}]
    checkpoint = {"cache": {"events": events}}
    if native_journal:
        journal_events = [{**event, "attempt": 1} for event in events]
        journal_events.append({"event_id": "r3:c1:down", "round": 3, "direction": "download",
                               "bucket": "deployed_prefix", "bytes": 7, "attempt": 2})
        put(run / "wire_attempt_journal.json", {"format": "pcla-v7-native-wire-attempt-1",
                                               "identity": "native-fixture", "events": journal_events})
        checkpoint.update(identity="native-fixture", journal_committed_event_count=4,
                          journal_committed_sha256=summary.canonical_hash(journal_events))
    torch.save(checkpoint, run / "latest.pt")
    readouts = [{"round": r, "accuracy": r / 1000, "loss": 1 - r / 1000,
                 "samples": 4, "predicted_class_counts": [2, 2]} for r in range(101)]
    report = root / "checks/audits" / (jid + ".json")
    put(report, {"ok": True, "errors": [], "run_dir": str(run), "checks": {
        "planned_job_id": jid, "completed_round": 100, "budget_stop_round": 50,
        "half_budget_round": 25, "half_budget_bytes": 0,
        "b8_cap_bytes": 200, "deployed_bytes": 110,
        "diagnostic_after_budget_bytes": 10, "round_readouts": readouts}})
    receipt = {"exit_code": 0, "auditor_sha256": summary.digest(auditor),
               "report": str(report), "report_sha256": summary.digest(report),
               "config_sha256": summary.digest(cfg),
               "source_manifest_sha256": summary.digest(source_manifest), "argv": ["python", str(auditor)]}
    queue = {"state": "completed" if completed else "executed_pending_audit", "job": job,
             "out": str(run), "config": str(cfg), "source": str(source_manifest.parent),
             "config_sha256": summary.digest(cfg), "audit": receipt}
    if corrupt:
        queue["audit"]["report_sha256"] = "0" * 64
    put(root / "queue/jobs" / (jid + ".json"), queue)
    put(root / "queue/jobs/TECH_01.json", {"state": "failed", "job": {"phase": "TECH"}})
    tech = root / "checks/path_profiles/native03/D107/p1/process_0/wire_attempt_journal.json"
    put(tech, {"events": [{"bytes": 106792925}]})
    put(tech.with_name("status.json"), {"state": "failed", "actual_attempted_wire_bytes": 106792925,
                                        "failed_attempt_wire_bytes": 0})
    return root, manifest


def test_completed_receipt_exports_budget_endpoint_and_keeps_missing_seeds(tmp_path):
    root, manifest = make_fixture(tmp_path)
    out = tmp_path / "new-summary"
    result = summary.build(root, manifest, out, expected_count=3)
    assert result["stage_status"] == "partial_unreviewed"
    assert result["audited_completed_jobs"] == 1
    run = summary.read(out / "runs.json")[0]
    assert run["b8_raw"]["round"] == 50
    assert run["half_raw"]["round"] == 25
    assert run["b8_ema"]["accuracy"] == 50 / 2000
    assert run["actual_attempted_wire_bytes"] is None
    assert run["wire_attempt_accounting"].startswith("journal_missing")
    assert run["deployed_cost"]["sum"]["local_training_seconds"] == 100
    assert run["diagnostic_cost"]["sum"]["local_training_seconds"] == 100
    assert run["seed401_exclusive_time_block_candidate"] is True
    assert summary.read(out / "recipes.json")[0]["three_seed_complete"] is False
    assert len(summary.read(out / "rounds.json")) == 101
    wire = summary.read(out / "wire.json")
    assert sum(x["bytes"] for x in wire if x["failed_attempt"]) == 20
    assert sum(x["bytes"] for x in wire if x["bucket"] == "diagnostic_after_budget") == 10
    tech = summary.read(out / "technical_artifacts.json")
    assert tech["queued_tech_count"] == 1
    assert tech["profile_or_wire_artifacts"][0]["attempted_wire_bytes"] == 106792925
    assert tech["status_artifacts"][0]["state"] == "failed"
    with pytest.raises(ValueError, match="new directory"):
        summary.build(root, manifest, out, expected_count=3)


@pytest.mark.parametrize("completed,corrupt", [(False, False), (True, True)])
def test_unfinished_or_unverified_is_never_numeric(tmp_path, completed, corrupt):
    root, manifest = make_fixture(tmp_path, completed=completed, corrupt=corrupt)
    out = tmp_path / "new-summary"
    result = summary.build(root, manifest, out, expected_count=3)
    assert result["audited_completed_jobs"] == 0
    assert summary.read(out / "runs.json") == []
    assert summary.read(out / "rounds.json") == []


def test_native_attempt_journal_adds_failed_send_outside_checkpoint_cache(tmp_path):
    root, manifest = make_fixture(tmp_path, native_journal=True)
    out = tmp_path / "new-summary"
    result = summary.build(root, manifest, out, expected_count=3)
    assert result["audited_completed_jobs"] == 1
    run = summary.read(out / "runs.json")[0]
    assert run["actual_attempted_wire_bytes"] == 127
    assert run["failed_attempt_wire_bytes"] == 7
    assert run["wire_attempt_accounting"] == "native_journal_reconciled"
    assert run["wire_attempt_journal_sha256"] == summary.digest(root / "runs/D0/D001_s401/wire_attempt_journal.json")
    wire = summary.read(out / "wire.json")
    assert sum(x["bytes"] for x in wire if x["ledger"] == "committed") == 120
    assert sum(x["bytes"] for x in wire if x["ledger"] == "uncommitted_attempt") == 7


def test_native_attempt_status_mismatch_quarantines_numeric_run(tmp_path):
    root, manifest = make_fixture(tmp_path, native_journal=True)
    path = root / "runs/D0/D001_s401/status.json"
    status = summary.read(path)
    status["failed_attempt_wire_bytes"] = 0
    put(path, status)
    out = tmp_path / "new-summary"
    result = summary.build(root, manifest, out, expected_count=3)
    assert result["audited_completed_jobs"] == 0
    assert result["summary_states"]["invalid_evidence"] == 1


def test_manifest_identity_mismatch_rejected_before_output(tmp_path):
    root, manifest = make_fixture(tmp_path)
    put(root / "queue/matrix_identity.json", {"count": 3, "manifest_sha256": "wrong"})
    out = tmp_path / "new-summary"
    with pytest.raises(ValueError, match="matrix identity"):
        summary.build(root, manifest, out, expected_count=3)
    assert not out.exists()
