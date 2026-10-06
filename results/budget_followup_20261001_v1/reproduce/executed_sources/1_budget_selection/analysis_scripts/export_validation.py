#!/usr/bin/env python3
"""Read-only, fail-closed export of all audited PCLA V seed/slot runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


SCHEMA = "pcla-v7-validation-export-1"
JOB_RE = re.compile(r"V_S(\d{2})_(D\d{3})_s(\d+)")
FIXED_SLOTS = set(range(1, 16)) - {3, 4}
CONDITIONAL_SLOTS = {3, 4, 16}
EXPECTED_SEEDS = {101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307}
PROTOCOL = "pcla-global-budget-20260928-v7"
# Pro V_RECIPES slots, with the sealed D035 cost branch for slots 3/4/16.
EXPECTED_RECIPE = {1: "D002", 2: "D006", 3: "D035", 4: "D042",
                   5: "D045", 6: "D054", 7: "D063", 8: "D066",
                   9: "D069", 10: "D072", 11: "D078", 12: "D086",
                   13: "D105", 14: "D107", 15: "D112", 16: "D035"}


class ExportError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def within(root: Path, value: str | Path, *, exists: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ExportError(f"input path is not absolute: {path}")
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise ExportError(f"input path escapes run root: {path}")
    if exists and (not path.is_file() or path.is_symlink()):
        raise ExportError(f"input file missing or symlinked: {path}")
    return path


def checked_hash(root: Path, value: str | Path, expected: str, hashes: dict[str, str]) -> Path:
    path = within(root, value)
    actual = digest(path)
    if not isinstance(expected, str) or actual != expected:
        raise ExportError(f"input SHA mismatch: {path}")
    previous = hashes.setdefault(str(path), actual)
    if previous != actual:
        raise ExportError(f"input changed during export: {path}")
    return path


def checked_optional(root: Path, path: Path, hashes: dict[str, str]) -> Path | None:
    if not path.exists():
        return None
    within(root, path)
    actual = digest(path)
    if hashes.setdefault(str(path), actual) != actual:
        raise ExportError(f"input changed during export: {path}")
    return path


def job_identity(item: dict, filename: str, queue_kind: str) -> tuple[int, int, str]:
    job = item.get("job", {})
    jid = job.get("job_id")
    match = JOB_RE.fullmatch(jid or "")
    if match is None or filename != f"{jid}.json":
        raise ExportError(f"queue job ID/file mismatch: {filename}")
    slot, recipe, seed = int(match[1]), match[2], int(match[3])
    expected = FIXED_SLOTS if queue_kind == "fixed" else CONDITIONAL_SLOTS
    if (slot not in expected or seed not in EXPECTED_SEEDS or
            job.get("seed") != seed or job.get("recipe_id") != recipe or
            recipe != EXPECTED_RECIPE.get(slot) or
            job.get("phase") != "V" or job.get("protocol") != PROTOCOL or
            job.get("budget_id") != "B8" or
            ("slot" in item and item["slot"] != slot)):
        raise ExportError(f"queue job scientific identity mismatch: {jid}")
    return seed, slot, recipe


def validate_roster(fixed: list[dict], conditional: list[dict]) -> dict[tuple[int, int], dict]:
    if len(fixed) != 156 or len(conditional) != 36:
        raise ExportError("full V roster needs exactly 156 fixed and 36 conditional jobs")
    roster: dict[tuple[int, int], dict] = {}
    seen_ids: set[str] = set()
    recipes: dict[int, str] = {}
    for kind, group in (("fixed", fixed), ("conditional", conditional)):
        for item in group:
            jid = item.get("job", {}).get("job_id", "")
            seed, slot, recipe = job_identity(item, f"{jid}.json", kind)
            key = (seed, slot)
            if jid in seen_ids or key in roster:
                raise ExportError(f"duplicate V ID or seed/slot: {jid}")
            if item.get("state") != "completed" or not isinstance(item.get("audit"), dict):
                raise ExportError(f"V run not independently audited completed: {jid}")
            if recipes.setdefault(slot, recipe) != recipe:
                raise ExportError(f"recipe changes within fixed slot {slot}")
            roster[key] = item
            seen_ids.add(jid)
    if set(roster) != {(seed, slot) for seed in EXPECTED_SEEDS for slot in range(1, 17)}:
        raise ExportError("V roster is not exactly 12 seeds x 16 slots")
    return roster


def load_queue(root: Path, name: str, kind: str, expected_slots: set[int],
               hashes: dict[str, str]) -> list[dict]:
    queue = root / name
    matrix_path = queue / "matrix_identity.json"
    matrix = read(matrix_path)
    hashes[str(matrix_path)] = digest(matrix_path)
    expected_count = 156 if kind == "fixed" else 36
    slots = matrix.get("fixed_slots" if kind == "fixed" else "slots")
    if (matrix.get("count") != expected_count or matrix.get("protocol") != PROTOCOL or
            set(matrix.get("seeds", [])) != EXPECTED_SEEDS or
            set(slots or []) != expected_slots):
        raise ExportError(f"{name}: matrix cardinality/identity differs")
    paths = sorted((queue / "jobs").glob("*.json"))
    if len(paths) != expected_count:
        raise ExportError(f"{name}: incomplete queue job files")
    items = []
    for path in paths:
        if path.is_symlink():
            raise ExportError(f"queue job symlink: {path}")
        item = read(path)
        job_identity(item, path.name, kind)
        hashes[str(path)] = digest(path)
        items.append(item)
    return items


def verify_source_tree(root: Path, source: Path, hashes: dict[str, str]) -> None:
    manifest_path = within(root, source / "SOURCE_MANIFEST.json")
    manifest = read(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ExportError(f"source manifest file list missing: {manifest_path}")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ExportError(f"invalid source manifest path: {name}")
        checked_hash(root, source / relative, expected, hashes)


def score(readout: dict) -> dict:
    classes = readout.get("predicted_class_counts")
    if (type(readout.get("accuracy")) not in (float, int) or
            type(readout.get("loss")) not in (float, int) or
            not math.isfinite(readout["accuracy"]) or not math.isfinite(readout["loss"]) or
            not 0 <= readout["accuracy"] <= 1 or
            not isinstance(classes, list) or len(classes) != 2 or
            any(type(value) is not int or value < 0 for value in classes) or
            sum(classes) != readout.get("samples")):
        raise ExportError("audited raw evaluation malformed")
    return {"accuracy": readout["accuracy"], "loss": readout["loss"],
            "class_0": classes[0], "class_1": classes[1], "samples": readout["samples"]}


def recovery_record(root: Path, item: dict, out: Path, hashes: dict[str, str]):
    meta = item.get("metadata_recovery")
    failures = [attempt for attempt in item.get("attempts", [])
                if type(attempt.get("exit_code")) is int and attempt["exit_code"] != 0]
    if not meta:
        if failures or item.get("exit_code") != 0:
            raise ExportError(f"unexplained failed attempt: {item['job']['job_id']}")
        return None, failures
    receipt_path = checked_hash(root, meta["receipt"], meta["receipt_sha256"], hashes)
    receipt = read(receipt_path)
    plan_path = within(root, Path(receipt["package"]) / "RECOVERY_PLAN.json")
    if (digest(plan_path) != meta.get("plan_sha256") or
            receipt.get("plan_sha256") != meta.get("plan_sha256") or
            receipt.get("job_id") != item["job"]["job_id"] or
            receipt.get("original_process_exit_code") != item.get("exit_code") or
            not receipt.get("original_attempts_preserved") or
            receipt.get("training_reexecuted") is not False or
            receipt.get("recovery_applied") is not True or
            not failures):
        raise ExportError("metadata recovery identity or original failure missing")
    hashes[str(plan_path)] = digest(plan_path)
    plan = read(plan_path)
    if (plan.get("job_id") != item["job"]["job_id"] or
            plan.get("original_run") != str(out) or
            plan.get("queue_job") != str(root / "conditional_validation_queue" / "jobs" /
                                         f"{item['job']['job_id']}.json") or
            plan.get("original_exit_code") != item.get("exit_code") or
            plan.get("original_attempt_count") != len(item.get("attempts", [])) or
            plan.get("source_manifest_sha256") != item["source_manifest_sha256"] or
            plan.get("config_sha256") != item["config_sha256"]):
        raise ExportError("metadata recovery sealed plan identity differs")
    for name, expected in receipt.get("installed_sha256", {}).items():
        if Path(name).name != name:
            raise ExportError("invalid installed metadata name")
        checked_hash(root, out / name, expected, hashes)
    failure_originals = {}
    original_keys = {"original_exception.txt": "exception",
                     "original_execution_attempts.json": "attempts",
                     "original_queue_job.json": "queue_job",
                     "original_status.json": "failed_status"}
    for name, plan_key in original_keys.items():
        path = within(root, Path(receipt["package"]) / name)
        actual = digest(path)
        if (actual != plan.get("original_sha256", {}).get(plan_key) or
                actual != plan.get("package_sha256", {}).get(name)):
            raise ExportError(f"metadata recovery original failure SHA differs: {name}")
        hashes[str(path)] = actual
        failure_originals[name] = {"path": str(path), "sha256": hashes[str(path)]}
    original_queue = read(Path(failure_originals["original_queue_job.json"]["path"]))
    if (original_queue.get("job", {}).get("job_id") != item["job"]["job_id"] or
            original_queue.get("attempts") != item.get("attempts") or
            original_queue.get("exit_code") != item.get("exit_code")):
        raise ExportError("metadata recovery original queue/failure differs")
    return {"queue_metadata_recovery": meta, "receipt": receipt,
            "receipt_path": str(receipt_path), "plan_path": str(plan_path),
            "failure_originals": failure_originals}, failures


def replay_record(root: Path, item: dict, config: dict, audit_checks: dict,
                  metrics: list[dict], roster: dict[tuple[int, int], dict],
                  hashes: dict[str, str]):
    jid = item["job"]["job_id"]
    slot = int(JOB_RE.fullmatch(jid)[1])
    if slot != 16:
        return None
    seed = item["job"]["seed"]
    donor = roster[(seed, 3)]
    donor_id = donor["job"]["job_id"]
    if item["job"].get("depends_on") != donor_id:
        raise ExportError(f"replay dependency differs: {jid}")
    binding_path = checked_hash(root, item.get("binding", ""), item.get("binding_sha256"), hashes)
    plan = config.get("quota_replay", {})
    if (plan.get("binding_path") != str(binding_path) or
            plan.get("binding_sha256") != item["binding_sha256"]):
        raise ExportError(f"replay config/binding differs: {jid}")
    binding = read(binding_path)
    expected_files = {"donor_config", "source_manifest", "data_manifest", "metrics",
                      "initial_identity", "run_identity", "status", "donor_quota_seal"}
    if binding.get("schema") != "pcla-v7-quota-replay-binding-1" or set(binding.get("files", {})) != expected_files:
        raise ExportError(f"replay donor originals incomplete: {jid}")
    for entry in binding["files"].values():
        checked_hash(root, entry["path"], entry["sha256"], hashes)
    donor_config = read(Path(binding["files"]["donor_config"]["path"]))
    donor_out = Path(donor["out"])
    if (donor_config.get("job", {}).get("job_id") != donor_id or
            binding["files"]["donor_config"]["sha256"] != donor["config_sha256"] or
            Path(binding["files"]["donor_config"]["path"]) != Path(donor["config"]) or
            Path(binding["files"]["source_manifest"]["path"]) != Path(donor["source"]) / "SOURCE_MANIFEST.json" or
            binding["files"]["source_manifest"]["sha256"] != donor["source_manifest_sha256"] or
            any(Path(binding["files"][key]["path"]) != donor_out / filename
                for key, filename in (("metrics", "metrics.json"),
                                      ("initial_identity", "initial_state_identity.json"),
                                      ("run_identity", "run_identity.json"),
                                      ("status", "status.json"),
                                      ("donor_quota_seal", "donor_quota_seal.json"))) or
            audit_checks.get("replay_masks_verified_rounds") != audit_checks.get("completed_round") or
            audit_checks.get("donor_book_sha256") !=
                read(Path(binding["files"]["donor_quota_seal"]["path"])).get("donor_book_sha256")):
        raise ExportError(f"replay donor/audit provenance differs: {jid}")
    for row in metrics:
        provenance = row.get("quota_replay", {})
        if (provenance.get("donor_job_id") != donor_id or
                provenance.get("donor_book_sha256") != audit_checks["donor_book_sha256"] or
                provenance.get("replayed_round") != row["round"]):
            raise ExportError(f"replay round provenance differs: {jid} r{row['round']}")
    return {"donor_job_id": donor_id, "donor_queue_config_sha256": donor["config_sha256"],
            "binding_path": str(binding_path), "binding_sha256": item["binding_sha256"],
            "binding_files": binding["files"], "donor_book_sha256": audit_checks["donor_book_sha256"],
            "replay_masks_verified_rounds": audit_checks["replay_masks_verified_rounds"],
            "scientific_note": "quota-only mechanism control; donor-dependent"}


def extract_one(root: Path, item: dict, roster: dict[tuple[int, int], dict],
                hashes: dict[str, str]) -> dict:
    jid = item["job"]["job_id"]
    seed, slot, recipe = int(JOB_RE.fullmatch(jid)[3]), int(JOB_RE.fullmatch(jid)[1]), item["job"]["recipe_id"]
    audit = item["audit"]
    if audit.get("exit_code") != 0:
        raise ExportError(f"independent audit failed: {jid}")
    auditor_argv = audit.get("argv")
    if not isinstance(auditor_argv, list) or len(auditor_argv) < 2:
        raise ExportError(f"independent auditor command missing: {jid}")
    auditor_name = ("audit_pcla_v7_conditional_run.py" if slot in CONDITIONAL_SLOTS
                    else "audit_pcla_v7_run.py")
    auditor_path = Path(item["source"]) / "scripts" / auditor_name
    positions = [i for i, arg in enumerate(auditor_argv) if arg == str(auditor_path)]
    if (len(positions) != 1 or positions[0] == 0 or
            not Path(auditor_argv[positions[0] - 1]).name.startswith("python")):
        raise ExportError(f"independent auditor executable path differs: {jid}")
    checked_hash(root, auditor_path, audit.get("auditor_sha256"), hashes)
    config_path = checked_hash(root, item["config"], item["config_sha256"], hashes)
    source_manifest = checked_hash(root, Path(item["source"]) / "SOURCE_MANIFEST.json",
                                   item["source_manifest_sha256"], hashes)
    if audit.get("config_sha256") != item["config_sha256"] or audit.get("source_manifest_sha256") != item["source_manifest_sha256"]:
        raise ExportError(f"audit source/config identity differs: {jid}")
    report_path = checked_hash(root, audit["report"], audit["report_sha256"], hashes)
    report = read(report_path)
    checks = report.get("checks", {})
    out = within(root, item["out"], exists=False)
    if (report.get("ok") is not True or report.get("errors") != [] or
            checks.get("planned_job_id") != jid or checks.get("phase") != "V" or
            Path(report.get("run_dir", "")).resolve() != out.resolve()):
        raise ExportError(f"independent audit report differs: {jid}")
    config = read(config_path)
    queue_job = dict(item["job"])
    released_job = dict(config.get("job", {}))
    if slot in CONDITIONAL_SLOTS:
        # The conditional queue preserves its draft job.status while the released
        # config advances only that operational field to ready/released.
        if (queue_job.pop("status", None) != "draft_blocked_dependencies" or
                released_job.pop("status", None) not in {"ready", "released"}):
            raise ExportError(f"conditional job release status differs: {jid}")
    if (queue_job != released_job or
            config.get("source_manifest") != str(source_manifest)):
        raise ExportError(f"released config identity differs: {jid}")
    if slot in CONDITIONAL_SLOTS:
        for field in ("release_receipt", "release_receipt_sha256", "technical_receipt",
                      "technical_receipt_sha256", "draft_config", "draft_config_sha256"):
            if not item.get(field):
                raise ExportError(f"conditional release evidence missing: {jid} {field}")
        matrix_path = within(root, root / "conditional_validation_queue" / "matrix_identity.json")
        matrix = read(matrix_path)
        matrix_sha = digest(matrix_path)
        if (hashes.setdefault(str(matrix_path), matrix_sha) != matrix_sha or
                matrix.get("draft_config_sha256", {}).get(jid) != item["draft_config_sha256"]):
            raise ExportError(f"conditional draft compilation SHA differs: {jid}")
    for path_field, sha_field in (("release_receipt", "release_receipt_sha256"),
                                  ("technical_receipt", "technical_receipt_sha256"),
                                  ("draft_config", "draft_config_sha256"),
                                  ("shared_profile_receipt", "shared_profile_sha256")):
        if item.get(path_field):
            checked_hash(root, item[path_field], item[sha_field], hashes)
    metrics_path = within(root, out / "metrics.json")
    status_path = within(root, out / "status.json")
    identity_path = within(root, out / "run_identity.json")
    for path in (metrics_path, status_path, identity_path):
        hashes[str(path)] = digest(path)
    journal = checked_optional(root, out / "communication_journal.json", hashes)
    initial = checked_optional(root, out / "initial_state_identity.json", hashes)
    seal = checked_optional(root, out / "donor_quota_seal.json", hashes)
    metrics, status, identity = read(metrics_path), read(status_path), read(identity_path)
    completed = checks.get("completed_round")
    budget_round = checks.get("budget_stop_round")
    half_round = checks.get("half_budget_round")
    if (type(completed) is not int or not 100 <= completed <= 200 or
            type(budget_round) is not int or not 1 <= budget_round <= completed or
            type(half_round) is not int or not 1 <= half_round <= budget_round or
            not isinstance(metrics, list) or len(metrics) != completed or
            [row.get("round") for row in metrics] != list(range(1, completed + 1)) or
            status.get("state") != "completed" or status.get("completed_round") != completed or
            status.get("budget_stop_round") != budget_round or
            checks.get("deployed_bytes") != checks.get("budget_return_bytes") or
            metrics[budget_round - 1].get("deployed_bytes") != checks["budget_return_bytes"] or
            metrics[half_round - 1].get("deployed_bytes") != checks.get("half_budget_bytes") or
            metrics[-1].get("deployed_bytes") != checks["deployed_bytes"] or
            status.get("deployed_bytes") != checks["deployed_bytes"] or
            checks["budget_return_bytes"] > checks.get("b8_cap_bytes", -1) or
            status.get("diagnostic_after_budget_bytes") != checks.get("diagnostic_after_budget_bytes") or
            metrics[-1].get("diagnostic_after_budget_bytes") != checks.get("diagnostic_after_budget_bytes")):
        raise ExportError(f"audited budget/round mapping differs: {jid}")
    readouts = checks.get("round_readouts")
    if (not isinstance(readouts, list) or len(readouts) != completed + 1 or
            [row.get("round") for row in readouts] != list(range(completed + 1))):
        raise ExportError(f"audited raw curve incomplete: {jid}")
    raw = {row["round"]: score(row) for row in readouts}
    if (identity.get("job") != config["job"] or
            identity.get("provenance", {}).get("config_sha256") != item["config_sha256"] or
            identity.get("provenance", {}).get("source_manifest_sha256") != item["source_manifest_sha256"]):
        raise ExportError(f"run identity differs: {jid}")
    expected_family = "A100" if seed in {281, 307} else "4090D"
    hardware = identity.get("provenance", {}).get("hardware", {})
    hardware_name = hardware.get("name", "")
    if (item.get("hardware_family") != expected_family or
            config["job"].get("hardware_block") != f"V_s{seed}" or
            ("A100" if expected_family == "A100" else "4090 D") not in hardware_name or
            not hardware.get("uuid")):
        raise ExportError(f"V hardware block identity differs: {jid}")
    recovery, failures = recovery_record(root, item, out, hashes)
    replay = replay_record(root, item, config, checks, metrics, roster, hashes)
    if slot == 3 and (initial is None or seal is None):
        raise ExportError(f"R-local donor original missing: {jid}")
    base = {"job_id": jid, "seed": seed, "slot": slot, "recipe_id": recipe,
            "method": item["job"]["method"], "representation": item["job"]["representation"],
            "hardware_family": item.get("hardware_family"), "hardware_uuid": hardware["uuid"],
            "source_manifest_sha256": item["source_manifest_sha256"],
            "config_sha256": item["config_sha256"], "audit_report_sha256": audit["report_sha256"]}
    endpoints = dict(base)
    endpoints.update({"completed_round": completed, "b8_cap_bytes": checks["b8_cap_bytes"],
                      "diagnostic_after_budget_bytes": checks["diagnostic_after_budget_bytes"],
                      "actual_attempted_wire_bytes": status.get("actual_attempted_wire_bytes"),
                      "failed_attempt_wire_bytes": status.get("failed_attempt_wire_bytes")})
    for label, round_id, bytes_value in (("half", half_round, checks["half_budget_bytes"]),
                                          ("b8", budget_round, checks["budget_return_bytes"]),
                                          ("r100", 100, metrics[99]["deployed_bytes"])):
        endpoints[f"{label}_round"] = round_id
        endpoints[f"{label}_deployed_bytes"] = bytes_value
        for key, value in raw[round_id].items():
            endpoints[f"{label}_{key}"] = value
    curve, ema, costs = [], [], []
    for round_id in range(completed + 1):
        met = None if round_id == 0 else metrics[round_id - 1]
        curve.append({**base, "round": round_id, **raw[round_id],
                      "deployed_bytes": None if met is None else met.get("deployed_bytes"),
                      "diagnostic_after_budget_bytes": None if met is None else met.get("diagnostic_after_budget_bytes"),
                      "actual_attempted_wire_bytes": None if met is None else met.get("actual_attempted_wire_bytes"),
                      "failed_attempt_wire_bytes": None if met is None else met.get("failed_attempt_wire_bytes")})
        if met is not None:
            if not isinstance(met.get("cost"), dict) or not isinstance(met.get("cumulative_cost"), dict):
                raise ExportError(f"per-round actual cost missing: {jid} r{round_id}")
            value = met.get("ema_evaluation")
            ema.append({**base, "round": round_id,
                        "accuracy": None if value is None else value.get("accuracy"),
                        "loss": None if value is None else value.get("loss"),
                        "samples": None if value is None else value.get("samples")})
            costs.append({**base, "round": round_id, "cost": met.get("cost"),
                          "cumulative_cost": met.get("cumulative_cost")})
    endpoint_costs = {label: {"round": round_id,
                              "cumulative_cost": metrics[round_id - 1].get("cumulative_cost")}
                      for label, round_id in (("half", half_round), ("b8", budget_round),
                                              ("r100", 100), ("final", completed))}
    provenance = {**base, "queue_state": item["state"], "queue_exit_code": item.get("exit_code"),
                  "attempts": item.get("attempts"), "metadata_recovery": recovery,
                  "failed_attempts": failures, "replay": replay,
                  "audit_report_path": str(report_path), "config_path": str(config_path),
                  "source_manifest_path": str(source_manifest), "out": str(out),
                  "input_artifact_paths": {"metrics": str(metrics_path), "status": str(status_path),
                                           "run_identity": str(identity_path),
                                           "communication_journal": None if journal is None else str(journal),
                                           "initial_state_identity": None if initial is None else str(initial),
                                           "donor_quota_seal": None if seal is None else str(seal)}}
    return {"endpoint": endpoints, "curve": curve, "ema": ema, "costs": costs,
            "endpoint_costs": {**base, "endpoints": endpoint_costs},
            "provenance": provenance}


def inspect_schema(root: Path) -> dict:
    examples = (("validation_queue", "V_S12_D086_s101"),
                ("validation_queue", "V_S13_D105_s101"),
                ("conditional_validation_queue", "V_S04_D042_s181"),
                ("conditional_validation_queue", "V_S03_D035_s101"),
                ("conditional_validation_queue", "V_S16_D035_s163"))
    rows = []
    for queue, jid in examples:
        path = root / queue / "jobs" / f"{jid}.json"
        if not path.is_file():
            rows.append({"job_id": jid, "missing": True})
            continue
        item = read(path)
        out = Path(item.get("out", ""))
        result = {"job_id": jid, "queue": queue, "state": item.get("state"),
                  "queue_keys": sorted(item), "audit_keys": sorted(item.get("audit", {})),
                  "has_metadata_recovery": bool(item.get("metadata_recovery")),
                  "has_binding": bool(item.get("binding"))}
        for filename in ("metrics.json", "status.json", "run_identity.json"):
            file = out / filename
            if file.is_file():
                value = read(file)
                result[filename] = {"type": type(value).__name__,
                                    "keys": sorted(value[0] if isinstance(value, list) and value else value),
                                    "length": len(value)}
        report = item.get("audit", {}).get("report")
        if report and Path(report).is_file():
            result["audit_check_keys"] = sorted(read(Path(report)).get("checks", {}))
        rows.append(result)
    return {"schema": "pcla-v7-validation-schema-inspection-1", "final_statistics": False,
            "examples": rows}


def csv_bytes(rows: list[dict]) -> str:
    from io import StringIO
    if not rows:
        raise ExportError("empty export table")
    fields = list(rows[0])
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != fields:
            raise ExportError("export row schema changes within table")
        writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                         if isinstance(value, (list, dict)) else value for key, value in row.items()})
    return stream.getvalue()


def jsonl_bytes(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def export(root: Path, out: Path) -> dict:
    root = root.resolve(strict=True)
    out = out.resolve(strict=False)
    if out.exists() or not out.parent.is_dir() or out.is_relative_to(root / "V") or out.is_relative_to(root / "validation_queue") or out.is_relative_to(root / "conditional_validation_queue"):
        raise ExportError("output must be a new directory outside source queues/results")
    hashes: dict[str, str] = {}
    fixed = load_queue(root, "validation_queue", "fixed", FIXED_SLOTS, hashes)
    conditional = load_queue(root, "conditional_validation_queue", "conditional", CONDITIONAL_SLOTS, hashes)
    roster = validate_roster(fixed, conditional)
    for source in {Path(item["source"]) for item in roster.values()}:
        verify_source_tree(root, source, hashes)
    extracted = [extract_one(root, roster[key], roster, hashes) for key in sorted(roster)]
    endpoints = [run["endpoint"] for run in extracted]
    curves = [row for run in extracted for row in run["curve"]]
    ema = [row for run in extracted for row in run["ema"]]
    costs = [row for run in extracted for row in run["costs"]]
    endpoint_costs = [run["endpoint_costs"] for run in extracted]
    provenance = [run["provenance"] for run in extracted]
    failures = [{"job_id": row["job_id"], "attempt": attempt,
                 "metadata_recovery": row["metadata_recovery"]}
                for row in provenance for attempt in row["failed_attempts"]]
    replay = [row["replay"] | {"job_id": row["job_id"], "seed": row["seed"]}
              for row in provenance if row["replay"] is not None]
    if len(endpoints) != 192 or len(replay) != 12:
        raise ExportError("full V endpoint/replay cardinality differs")
    for path, original in hashes.items():
        if digest(Path(path)) != original:
            raise ExportError(f"input changed before publishing: {path}")
    outputs = {
        "v_endpoints.csv": csv_bytes(endpoints),
        "v_round_curves.csv": csv_bytes(curves),
        "v_ema_rounds.csv": csv_bytes(ema),
        "v_cost_rounds.jsonl": jsonl_bytes(costs),
        "v_cost_endpoints.jsonl": jsonl_bytes(endpoint_costs),
        "v_provenance.jsonl": jsonl_bytes(provenance),
        "v_failed_attempts.jsonl": jsonl_bytes(failures),
        "v_replay_provenance.jsonl": jsonl_bytes(replay),
        "input_hashes.json": json.dumps({"schema": SCHEMA, "inputs": hashes}, indent=2,
                                        sort_keys=True, ensure_ascii=False) + "\n",
    }
    parent = out.parent
    # mkdir is atomic and refuses even an empty pre-existing destination.
    out.mkdir(exist_ok=False)
    try:
        output_hashes = {}
        for name, content in outputs.items():
            path = out / name
            path.write_text(content)
            output_hashes[name] = digest(path)
        manifest = {"schema": SCHEMA, "complete": True, "fixed_count": 156,
                    "conditional_count": 36, "job_count": 192, "replay_count": 12,
                    "raw_curve_rows": len(curves), "ema_rows": len(ema),
                    "failed_attempt_rows": len(failures), "root": str(root),
                    "exporter_sha256": digest(Path(__file__)), "output_sha256": output_hashes,
                    "statistics_or_winner_selected": False}
        (out / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except BaseException:
        shutil.rmtree(out)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-schema", action="store_true")
    args = parser.parse_args()
    if args.inspect_schema:
        print(json.dumps(inspect_schema(args.root.resolve(strict=True)), indent=2, sort_keys=True))
    else:
        if args.out is None:
            parser.error("--out is required for a complete export")
        print(json.dumps(export(args.root, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
