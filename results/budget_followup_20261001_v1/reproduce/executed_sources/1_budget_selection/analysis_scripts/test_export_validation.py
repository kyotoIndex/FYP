"""Small independent fixtures for the fail-closed V evidence export."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import export_validation as ex


def write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jid(slot: int, seed: int = 101) -> str:
    recipe = ex.EXPECTED_RECIPE[slot]
    return f"V_S{slot:02d}_{recipe}_s{seed}"


def job(slot: int, seed: int = 101) -> dict:
    return {"job_id": jid(slot, seed), "seed": seed,
            "recipe_id": jid(slot, seed).split("_")[2], "phase": "V",
            "protocol": ex.PROTOCOL, "budget_id": "B8", "method": "local",
            "representation": "T", "hardware_block": f"V_s{seed}",
            "status": "draft_blocked_dependencies" if slot in ex.CONDITIONAL_SLOTS else "draft_not_dispatched",
            "depends_on": jid(3, seed) if slot == 16 else ""}


def roster_items() -> tuple[list[dict], list[dict]]:
    fixed, conditional = [], []
    for seed in sorted(ex.EXPECTED_SEEDS):
        for slot in range(1, 17):
            item = {"job": job(slot, seed), "state": "completed", "audit": {}}
            if slot in ex.CONDITIONAL_SLOTS:
                item["slot"] = slot
                conditional.append(item)
            else:
                fixed.append(item)
    return fixed, conditional


def make_run(root: Path, slot: int, *, seed: int = 101,
             recovery: bool = False, donor: dict | None = None) -> dict:
    identity = job(slot, seed)
    name = identity["job_id"]
    source = root / "sources" / "test"
    source_sha = write(source / "SOURCE_MANIFEST.json", {"test": True})
    auditor_name = ("audit_pcla_v7_conditional_run.py" if slot in ex.CONDITIONAL_SLOTS
                    else "audit_pcla_v7_run.py")
    auditor = source / "scripts" / auditor_name
    auditor.parent.mkdir(parents=True, exist_ok=True)
    auditor.write_text("# fixture auditor\n")
    out = root / "V" / name
    config_path = root / "configs" / f"{name}.json"
    released_identity = copy.deepcopy(identity)
    if slot in ex.CONDITIONAL_SLOTS:
        released_identity["status"] = "ready"
    config = {"job": released_identity, "source_manifest": str(source / "SOURCE_MANIFEST.json")}
    donor_book = "b" * 64
    binding_path = root / "checks" / f"binding_{seed}.json"
    if slot == 16:
        assert donor is not None
        donor_out = Path(donor["out"])
        files = {
            "donor_config": Path(donor["config"]),
            "source_manifest": Path(donor["source"]) / "SOURCE_MANIFEST.json",
            "data_manifest": root / "data" / "train_roles.json",
            "metrics": donor_out / "metrics.json",
            "initial_identity": donor_out / "initial_state_identity.json",
            "run_identity": donor_out / "run_identity.json",
            "status": donor_out / "status.json",
            "donor_quota_seal": donor_out / "donor_quota_seal.json",
        }
        write(files["data_manifest"], {"test": True})
        binding = {"schema": "pcla-v7-quota-replay-binding-1",
                   "files": {key: {"path": str(path), "sha256": ex.digest(path)}
                             for key, path in files.items()}}
        binding_sha = write(binding_path, binding)
        config["quota_replay"] = {"binding_path": str(binding_path),
                                   "binding_sha256": binding_sha}
    config_sha = write(config_path, config)
    metrics = []
    for round_id in range(1, 101):
        row = {"round": round_id, "deployed_bytes": min(round_id, 80) * 10,
               "diagnostic_after_budget_bytes": 0,
               "cost": {"local": round_id}, "cumulative_cost": {"local": round_id},
               "ema_evaluation": {"accuracy": .4, "loss": .7, "samples": 4096}}
        if slot == 16:
            row["quota_replay"] = {"donor_job_id": donor["job"]["job_id"],
                                   "donor_book_sha256": donor_book,
                                   "replayed_round": round_id}
        metrics.append(row)
    write(out / "metrics.json", metrics)
    write(out / "status.json", {"state": "completed", "completed_round": 100,
                                "budget_stop_round": 80, "deployed_bytes": 800,
                                "diagnostic_after_budget_bytes": 0})
    write(out / "run_identity.json", {"job": released_identity, "provenance": {
        "config_sha256": config_sha, "source_manifest_sha256": source_sha,
        "hardware": {"name": "NVIDIA GeForce RTX 4090 D", "uuid": "test-uuid"}}})
    if slot == 3:
        write(out / "initial_state_identity.json", {"test": True})
        write(out / "donor_quota_seal.json", {"donor_book_sha256": donor_book})
    readouts = [{"round": r, "accuracy": r / 200, "loss": 1 - r / 200,
                 "predicted_class_counts": [r, 4096-r], "samples": 4096}
                for r in range(101)]
    checks = {"planned_job_id": name, "phase": "V", "completed_round": 100,
              "budget_stop_round": 80, "half_budget_round": 40,
              "budget_return_bytes": 800, "deployed_bytes": 800,
              "half_budget_bytes": 400, "b8_cap_bytes": 1200,
              "diagnostic_after_budget_bytes": 0, "round_readouts": readouts}
    if slot == 16:
        checks.update({"replay_masks_verified_rounds": 100,
                       "donor_book_sha256": donor_book})
    report_path = root / "checks" / f"audit_{name}.json"
    report_sha = write(report_path, {"ok": True, "errors": [],
                                     "run_dir": str(out), "checks": checks})
    item = {"job": identity, "state": "completed", "exit_code": 0,
            "audit": {"exit_code": 0, "argv": ["python", str(auditor)],
                      "auditor_sha256": ex.digest(auditor), "config_sha256": config_sha,
                      "source_manifest_sha256": source_sha,
                      "report": str(report_path), "report_sha256": report_sha},
            "config": str(config_path), "config_sha256": config_sha,
            "source": str(source), "source_manifest_sha256": source_sha,
            "out": str(out), "hardware_family": "4090D", "attempts": []}
    if slot in ex.CONDITIONAL_SLOTS:
        item["slot"] = slot
        draft_path = root / "configs" / "draft" / f"{name}.json"
        draft_sha = write(draft_path, {"draft_job": name})
        release_path = root / "checks" / f"release_{name}.json"
        technical_path = root / "checks" / f"technical_{name}.json"
        item.update({"draft_config": str(draft_path), "draft_config_sha256": draft_sha,
                     "release_receipt": str(release_path),
                     "release_receipt_sha256": write(release_path, {"job_id": name}),
                     "technical_receipt": str(technical_path),
                     "technical_receipt_sha256": write(technical_path, {"job_id": name})})
        matrix_path = root / "conditional_validation_queue" / "matrix_identity.json"
        matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {"draft_config_sha256": {}}
        matrix["draft_config_sha256"][name] = draft_sha
        write(matrix_path, matrix)
    if slot == 16:
        item.update({"binding": str(binding_path), "binding_sha256": binding_sha})
    if recovery:
        package = root / "checks" / f"recovery_{name}"
        write(package / "original_exception.txt", "failed after scientific return")
        item.update({"exit_code": 1,
                     "attempts": [{"exit_code": 1, "status_state": "failed"}]})
        write(package / "original_execution_attempts.json", [{"fixture": True}])
        write(package / "original_queue_job.json", item)
        write(package / "original_status.json", {"state": "failed"})
        names = {"original_exception.txt": "exception", "original_execution_attempts.json": "attempts",
                 "original_queue_job.json": "queue_job", "original_status.json": "failed_status"}
        plan = {"job_id": name, "original_run": str(out),
                "queue_job": str(root / "conditional_validation_queue" / "jobs" / f"{name}.json"),
                "original_exit_code": 1, "original_attempt_count": 1,
                "source_manifest_sha256": source_sha, "config_sha256": config_sha,
                "original_sha256": {key: ex.digest(package / file) for file, key in names.items()},
                "package_sha256": {file: ex.digest(package / file) for file in names}}
        plan_sha = write(package / "RECOVERY_PLAN.json", plan)
        receipt_path = out / "metadata_recovery.json"
        receipt = {"package": str(package), "plan_sha256": plan_sha,
                   "job_id": name, "original_process_exit_code": 1,
                   "original_attempts_preserved": True,
                   "training_reexecuted": False, "recovery_applied": True,
                   "installed_sha256": {"status.json": ex.digest(out / "status.json")}}
        receipt_sha = write(receipt_path, receipt)
        item.update({"metadata_recovery": {"receipt": str(receipt_path),
                                           "receipt_sha256": receipt_sha,
                                           "plan_sha256": plan_sha}})
    return item


def test_complete_roster_and_missing_duplicate_rejected():
    fixed, conditional = roster_items()
    assert len(ex.validate_roster(fixed, conditional)) == 192
    with pytest.raises(ex.ExportError, match="156 fixed"):
        ex.validate_roster(fixed[:-1], conditional)
    altered = copy.deepcopy(conditional)
    altered[-1] = copy.deepcopy(altered[0])
    with pytest.raises(ex.ExportError, match="duplicate"):
        ex.validate_roster(fixed, altered)
    fixed[0]["state"] = "executed_pending_audit"
    with pytest.raises(ex.ExportError, match="not independently audited"):
        ex.validate_roster(fixed, conditional)
    fixed[0]["state"] = "completed"
    fixed[0]["job"]["recipe_id"] = "D999"
    with pytest.raises(ex.ExportError, match="scientific identity"):
        ex.validate_roster(fixed, conditional)


def test_formal_export_refuses_incomplete_queue_without_output(tmp_path):
    root = tmp_path / "run"
    matrix = {"count": 156, "protocol": ex.PROTOCOL,
              "seeds": sorted(ex.EXPECTED_SEEDS), "fixed_slots": sorted(ex.FIXED_SLOTS)}
    write(root / "validation_queue" / "matrix_identity.json", matrix)
    fixed, _ = roster_items()
    for item in fixed[:-1]:
        write(root / "validation_queue" / "jobs" / (item["job"]["job_id"] + ".json"), item)
    out = tmp_path / "export_01"
    with pytest.raises(ex.ExportError, match="incomplete queue"):
        ex.export(root, out)
    assert not out.exists()
    out.mkdir()
    with pytest.raises(ex.ExportError, match="new directory"):
        ex.export(root, out)


def test_audited_budget_endpoint_is_not_round_100(tmp_path):
    item = make_run(tmp_path, 1)
    result = ex.extract_one(tmp_path, item, {}, {})
    endpoint = result["endpoint"]
    assert (endpoint["b8_round"], endpoint["b8_accuracy"], endpoint["b8_deployed_bytes"]) == (80, .4, 800)
    assert (endpoint["r100_round"], endpoint["r100_accuracy"]) == (100, .5)
    assert (endpoint["half_round"], endpoint["half_accuracy"]) == (40, .2)
    assert len(result["curve"]) == 101


def test_wrong_audit_sha_refused(tmp_path):
    item = make_run(tmp_path, 1)
    item["audit"]["report_sha256"] = "0" * 64
    with pytest.raises(ex.ExportError, match="SHA mismatch"):
        ex.extract_one(tmp_path, item, {}, {})


def test_srun_prefixed_auditor_command_binds_exact_script(tmp_path):
    item = make_run(tmp_path, 4)
    original = item["audit"]["argv"]
    item["audit"]["argv"] = ["srun", "--jobid=173697", "--exclusive",
                              "/venv/bin/python", original[1], "--config", item["config"]]
    assert ex.extract_one(tmp_path, item, {}, {})["endpoint"]["job_id"] == item["job"]["job_id"]
    item["audit"]["argv"][4] = str(Path(item["source"]) / "scripts" / "other.py")
    with pytest.raises(ex.ExportError, match="auditor executable path differs"):
        ex.extract_one(tmp_path, item, {}, {})
    item["audit"]["argv"][4] = original[1]
    item["audit"]["argv"].append(original[1])
    with pytest.raises(ex.ExportError, match="auditor executable path differs"):
        ex.extract_one(tmp_path, item, {}, {})


def test_queue_scientific_labels_must_match_released_config(tmp_path):
    item = make_run(tmp_path, 4)
    item["job"]["method"] = "fedsb"
    item["job"]["representation"] = "U"
    with pytest.raises(ex.ExportError, match="released config identity"):
        ex.extract_one(tmp_path, item, {}, {})


def test_conditional_release_chain_and_draft_hash_required(tmp_path):
    item = make_run(tmp_path, 4)
    altered = copy.deepcopy(item)
    for field in ("release_receipt", "release_receipt_sha256", "technical_receipt",
                  "technical_receipt_sha256", "draft_config", "draft_config_sha256"):
        altered.pop(field)
    with pytest.raises(ex.ExportError, match="conditional release evidence missing"):
        ex.extract_one(tmp_path, altered, {}, {})
    matrix_path = tmp_path / "conditional_validation_queue" / "matrix_identity.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["draft_config_sha256"][item["job"]["job_id"]] = "0" * 64
    write(matrix_path, matrix)
    with pytest.raises(ex.ExportError, match="draft compilation SHA differs"):
        ex.extract_one(tmp_path, item, {}, {})


def test_missing_actual_cost_round_refused(tmp_path):
    item = make_run(tmp_path, 1)
    metrics_path = Path(item["out"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    del metrics[9]["cost"]
    del metrics[9]["cumulative_cost"]
    write(metrics_path, metrics)
    with pytest.raises(ex.ExportError, match="per-round actual cost missing.*r10"):
        ex.extract_one(tmp_path, item, {}, {})


def test_recovered_exit_one_is_preserved(tmp_path):
    item = make_run(tmp_path, 3, recovery=True)
    result = ex.extract_one(tmp_path, item, {}, {})
    provenance = result["provenance"]
    assert provenance["queue_exit_code"] == 1
    assert provenance["failed_attempts"] == [{"exit_code": 1, "status_state": "failed"}]
    assert provenance["metadata_recovery"]["receipt"]["training_reexecuted"] is False
    assert provenance["metadata_recovery"]["failure_originals"]["original_exception.txt"]
    package = Path(provenance["metadata_recovery"]["receipt"]["package"])
    (package / "original_exception.txt").write_text("changed after seal\n")
    with pytest.raises(ex.ExportError, match="original failure SHA differs"):
        ex.extract_one(tmp_path, item, {}, {})


def test_replay_binds_same_seed_donor_and_rounds(tmp_path):
    donor = make_run(tmp_path, 3, seed=163)
    replay = make_run(tmp_path, 16, seed=163, donor=donor)
    result = ex.extract_one(tmp_path, replay, {(163, 3): donor}, {})
    provenance = result["provenance"]["replay"]
    assert provenance["donor_job_id"] == donor["job"]["job_id"]
    assert provenance["replay_masks_verified_rounds"] == 100
    assert provenance["binding_files"]["metrics"]["sha256"] == ex.digest(Path(donor["out"]) / "metrics.json")
    wrong_donor = copy.deepcopy(donor)
    wrong_donor["job"]["job_id"] = jid(3, 101)
    with pytest.raises(ex.ExportError, match="dependency differs"):
        ex.extract_one(tmp_path, replay, {(163, 3): wrong_donor}, {})
