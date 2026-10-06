"""Independent fixture checks for the fail-closed COST1 export."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import export_cost as ex


def write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return ex.digest(path)


def test_historical_receipt_object_encodings_are_both_verified():
    receipt = {"a": 1, "nested": {"b": "x"}}
    for kwargs in ({}, {"separators": (",", ":")}):
        sha = hashlib.sha256(json.dumps(receipt, sort_keys=True, **kwargs).encode()).hexdigest()
        assert ex.object_digest(receipt, sha) == sha
    with pytest.raises(ex.ExportError, match="receipt object SHA"):
        ex.object_digest(receipt, "0" * 64)


def make_roster(root: Path):
    jobs = [{"replay_id": f"COST1_D{i:03d}_s401", "recipe_id": f"D{i:03d}", "seed": 401}
            for i in range(1, 118)]
    plan = root / "plan.json"
    plan_sha = write(plan, {"schema": "pcla-cost1-bound-plan-1", "quality_results_replaced": False,
                            "jobs": jobs})
    identities = root / "identities.json"
    identity_sha = write(identities, {job["recipe_id"]: {} for job in jobs})
    matrix = root / "cost_queue/matrix_identity.json"
    write(matrix, {"schema": "pcla-v7-cost-queue-1", "count": 117,
                   "plan": str(plan), "plan_sha256": plan_sha,
                   "expected_identities_sha256": identity_sha})
    for job in jobs:
        write(root / "cost_queue/jobs" / (job["replay_id"] + ".json"),
              {"job": job, "state": "completed", "cost_acceptance": {},
               "expected_identities": str(identities),
               "expected_identities_sha256": identity_sha})
    return jobs


def test_roster_rejects_missing_and_duplicate_recipe(tmp_path):
    jobs = make_roster(tmp_path)
    # An empty acceptance is deliberately not an accepted receipt.
    with pytest.raises(ex.ExportError, match="not root-accepted"):
        ex.roster(tmp_path, {})
    for job in jobs:
        path = tmp_path / "cost_queue/jobs" / (job["replay_id"] + ".json")
        item = ex.read(path)
        item["cost_acceptance"] = {"acceptance": "sealed"}
        write(path, item)
    assert len(ex.roster(tmp_path, {})) == 117
    missing = tmp_path / "cost_queue/jobs" / (jobs[-1]["replay_id"] + ".json")
    missing.unlink()
    with pytest.raises(ex.ExportError, match="incomplete"):
        ex.roster(tmp_path, {})
    write(missing, {"job": jobs[0], "state": "completed",
                    "cost_acceptance": {"acceptance": "sealed"},
                    "expected_identities": str(tmp_path / "identities.json"),
                    "expected_identities_sha256": ex.digest(tmp_path / "identities.json")})
    with pytest.raises(ex.ExportError, match="duplicate"):
        ex.roster(tmp_path, {})


def make_accepted_run(root: Path, *, pair_lane: bool = False):
    root = root.resolve()
    jid, recipe = "COST1_D035_s401", "D035"
    original = root / "d0"
    source = root / "sources/cost02"
    out = root / "COST1" / jid
    old_manifest = original / "SOURCE_MANIFEST.json"
    old_sha = write(old_manifest, {"files": {"x.py": write(original / "x.py", {})}})
    source_manifest = source / "SOURCE_MANIFEST.json"
    source_sha = write(source_manifest, {"files": {"x.py": write(source / "x.py", {})}})
    config = root / "configs/D035.json"
    config_sha = write(config, {"recipe": recipe})
    original_audit = original / "audit.json"
    original_audit_sha = write(original_audit, {"ok": True})
    metrics_sha = write(original / "metrics.json", [{"round": 0}])
    original_identity_sha = write(original / "run_identity.json", {"D0": True})
    gate = root / "gate.json"
    gate_sha = write(gate, {"ok": True})
    identities = root / "identities.json"
    identities_sha = write(identities, {recipe: {}})
    interval_sha = write(out / "exclusive_timing.json", {"intervals": []})
    write(out / "cost_prefix_draft.json", {"unaccepted": True})
    write(out / "status.json", {"budget_return_round": 114, "deployed_bytes": 222})
    write(out / "run_identity.json", {"job_id": jid})
    write(out / "execution_attempts.json", [{"slurm_job_id": "123", "gpu_sharing": "exclusive"}])
    log = root / "worker.log"
    log.write_text("completed\n")
    breakdown = {name: 1.0 for name in ex.TIMING}
    receipt = {"required_breakdown_seconds": breakdown, "required_seconds": 7.0,
               "interval_coverage_passed": True, "exclusive_gpu": True,
               "original_quality_retained": True, "parity_passed": True,
               "recipe_id": recipe, "seed": 401, "budget_return_round": 114,
               "deployed_bytes": 222, "interval_file_sha256": interval_sha,
               "original_config_sha256": config_sha,
               "original_source_manifest_sha256": old_sha,
               "instrumented_source_manifest_sha256": source_sha}
    receipt_path = root / "cost_queue/accepted_receipts" / f"{jid}.json"
    write(receipt_path, receipt)
    object_sha = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    checks = {name: {"status": "pass"} for name in ex.CHECKS}
    checks["timing"].update({"required_breakdown_seconds": breakdown,
                             "required_seconds": 7.0, "coverage": {"passes": True}})
    checks["exclusive"].update({"gpu_uuid": "GPU-test",
                                 "continuous_process_occupancy": "unknown_unsampled_interval"})
    checks["prefix_cache"].update({"digest_verified_clients": 4,
                                   "digest_unverifiable_clients": [1, 2],
                                   "rng_status": "unknown"})
    if pair_lane:
        checks["pair_lane"] = {"status": "pass"}
    report_path = root / "audit.json"
    report_sha = write(report_path, {"accepted": True, "job_id": jid,
                                     "original_results_modified": False,
                                     "receipt": receipt, "checks": checks})
    main_path = root / "MAIN_ACCEPTANCE.json"
    main_sha = write(main_path, {"decision": "accepted_under_COST_REPLAY_section4",
                                 "original_quality_retained": True,
                                 "frozen_source_unchanged": True,
                                 "evidence": {recipe: {"audit_sha256": report_sha,
                                                       "receipt_sha256": object_sha}}})
    job = {"replay_id": jid, "recipe_id": recipe, "seed": 401,
           "original_config": str(config), "original_config_sha256": config_sha,
           "original_source_manifest": str(old_manifest),
           "original_source_manifest_sha256": old_sha,
           "original_audit_report": str(original_audit),
           "original_audit_report_sha256": original_audit_sha,
           "original_run": str(original), "original_metrics_sha256": metrics_sha,
           "original_run_identity_sha256": original_identity_sha,
           "expected_budget_return_round": 114, "expected_deployed_bytes": 222,
           "budget_cap_bytes": 300}
    item = {"job": job, "config": str(config), "config_sha256": config_sha,
            "source": str(source), "source_manifest_sha256": source_sha,
            "old_source_manifest": str(old_manifest), "old_source_manifest_sha256": old_sha,
            "gate": str(gate), "gate_sha256": gate_sha,
            "expected_identities": str(identities),
            "expected_identities_sha256": identities_sha,
            "out": str(out), "state": "completed", "exit_code": 0,
            "gpu_uuid": "GPU-test",
            "attempts": [{"exit_code": 0, "gpu_uuid": "GPU-test", "slurm_job": "123",
                          "worker_log": str(log), "raw_output": str(out)}],
            "cost_acceptance": {"acceptance": str(main_path),
                                "acceptance_sha256": main_sha,
                                "report": str(report_path), "report_sha256": report_sha,
                                "receipt": str(receipt_path),
                                "receipt_object_sha256": object_sha}}
    return item


def test_seven_timing_and_unknowns_with_optional_pair_check(tmp_path):
    item = make_accepted_run(tmp_path, pair_lane=True)
    result = ex.extract_one(tmp_path.resolve(), item, {})
    endpoint = result["endpoint"]
    assert endpoint["required_seconds"] == 7.0
    assert endpoint["continuous_process_occupancy"] == "unknown_unsampled_interval"
    assert endpoint["rng_status"] == "unknown"
    assert endpoint["digest_unverifiable_clients"] == 2
    assert result["attempts"][0]["failed"] is False
    altered = copy.deepcopy(item)
    report = Path(altered["cost_acceptance"]["report"])
    data = ex.read(report)
    data["checks"]["pair_lane"]["status"] = "fail"
    report_sha = write(report, data)
    altered["cost_acceptance"]["report_sha256"] = report_sha
    acceptance = Path(altered["cost_acceptance"]["acceptance"])
    main = ex.read(acceptance)
    main["evidence"]["D035"]["audit_sha256"] = report_sha
    altered["cost_acceptance"]["acceptance_sha256"] = write(acceptance, main)
    with pytest.raises(ex.ExportError, match="acceptance/audit/receipt chain"):
        ex.extract_one(tmp_path.resolve(), altered, {})


def test_acceptance_sha_mismatch_rejected(tmp_path):
    item = make_accepted_run(tmp_path)
    item["cost_acceptance"]["acceptance_sha256"] = "0" * 64
    with pytest.raises(ex.ExportError, match="SHA mismatch"):
        ex.extract_one(tmp_path.resolve(), item, {})
