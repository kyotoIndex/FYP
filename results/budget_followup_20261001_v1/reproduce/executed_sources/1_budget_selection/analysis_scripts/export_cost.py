#!/usr/bin/env python3
"""Read-only, fail-closed export of all 117 root-accepted COST1 receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


SCHEMA = "pcla-v7-cost-closeout-export-1"
JOB_RE = re.compile(r"COST1_(D\d{3})_s401")
EXPECTED_RECIPES = {f"D{n:03d}" for n in range(1, 118)}
TIMING = ("task_initialization", "state_and_optimizer_setup", "client_training",
          "scoring_and_shared_information", "coordinate_and_server_decision",
          "message_processing", "budget_and_return_control")
CHECKS = {"exclusive", "identity", "original_quality_retained", "prefix_cache",
          "queue_state", "returned_parameters", "scientific_prefix", "timing"}


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


def object_digest(obj, accepted_sha: str) -> str:
    # Historical accepted_03 used compact separators; both accepted forms hash
    # the same parsed object. Preserve the digest recorded by root acceptance.
    candidates = {hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                         **kwargs).encode()).hexdigest()
                  for kwargs in ({}, {"separators": (",", ":")})}
    if accepted_sha not in candidates:
        raise ExportError("accepted COST receipt object SHA differs")
    return accepted_sha


def inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.resolve(strict=True).is_relative_to(root):
        raise ExportError(f"input path escapes COST root: {path}")
    if not path.is_file() or path.is_symlink():
        raise ExportError(f"input missing or symlinked: {path}")
    return path


def checked(root: Path, value: str | Path, expected: str, hashes: dict[str, str]) -> Path:
    path = inside(root, value)
    actual = digest(path)
    if not isinstance(expected, str) or actual != expected:
        raise ExportError(f"input SHA mismatch: {path}")
    if hashes.setdefault(str(path), actual) != actual:
        raise ExportError(f"input changed during validation: {path}")
    return path


def record(root: Path, value: str | Path, hashes: dict[str, str]) -> Path:
    path = inside(root, value)
    actual = digest(path)
    if hashes.setdefault(str(path), actual) != actual:
        raise ExportError(f"input changed during validation: {path}")
    return path


def verify_source_tree(root: Path, manifest_path: Path, hashes: dict[str, str]) -> None:
    source = manifest_path.parent
    files = read(manifest_path).get("files")
    if not isinstance(files, dict) or not files:
        raise ExportError(f"source manifest has no file closure: {manifest_path}")
    for relative, expected in files.items():
        part = Path(relative)
        if part.is_absolute() or ".." in part.parts or not part.parts:
            raise ExportError(f"unsafe source path: {relative}")
        checked(root, source / part, expected, hashes)


def roster(root: Path, hashes: dict[str, str]) -> list[dict]:
    matrix_path = record(root, root / "cost_queue/matrix_identity.json", hashes)
    matrix = read(matrix_path)
    if matrix.get("schema") != "pcla-v7-cost-queue-1" or matrix.get("count") != 117:
        raise ExportError("COST queue matrix is not the frozen 117-job roster")
    plan_path = checked(root, matrix["plan"], matrix["plan_sha256"], hashes)
    plan = read(plan_path)
    if (plan.get("schema") != "pcla-cost1-bound-plan-1" or
            plan.get("quality_results_replaced") is not False or
            not isinstance(plan.get("jobs"), list) or len(plan["jobs"]) != 117):
        raise ExportError("COST bound plan differs")
    expected = {}
    for job in plan["jobs"]:
        name = job.get("replay_id", "")
        match = JOB_RE.fullmatch(name)
        if (match is None or job.get("recipe_id") != match[1] or
                job.get("seed") != 401 or match[1] in expected):
            raise ExportError(f"COST bound plan has duplicate/invalid job: {name}")
        expected[match[1]] = job
    if set(expected) != EXPECTED_RECIPES:
        raise ExportError("COST bound plan does not cover D001-D117 once")
    paths = sorted((root / "cost_queue/jobs").glob("*.json"))
    if len(paths) != 117:
        raise ExportError("COST queue is incomplete or has extra job files")
    items = []
    for path in paths:
        item = read(record(root, path, hashes))
        name = item.get("job", {}).get("replay_id", "")
        match = JOB_RE.fullmatch(name)
        if (match is None or path.name != f"{name}.json" or
                item["job"] != expected[match[1]] or item.get("state") != "completed" or
                not isinstance(item.get("cost_acceptance"), dict) or
                not item["cost_acceptance"]):
            raise ExportError(f"COST job missing, duplicate, or not root-accepted: {path}")
        items.append(item)
    if {x["job"]["recipe_id"] for x in items} != EXPECTED_RECIPES:
        raise ExportError("COST queue recipe cardinality differs")
    expected_path = checked(root, items[0]["expected_identities"],
                            matrix["expected_identities_sha256"], hashes)
    identities = read(expected_path)
    if set(identities) != EXPECTED_RECIPES:
        raise ExportError("COST expected identities not 117 recipes")
    for item in items:
        if (item.get("expected_identities") != str(expected_path) or
                item.get("expected_identities_sha256") != matrix["expected_identities_sha256"]):
            raise ExportError("COST job expected-identity binding differs")
    return items


def extract_one(root: Path, item: dict, hashes: dict[str, str]) -> dict:
    job = item["job"]
    jid = job["replay_id"]
    recipe = job["recipe_id"]
    # The bound plan and queue are separate from the root acceptance chain.
    checked(root, item["config"], item["config_sha256"], hashes)
    source_manifest = checked(root, Path(item["source"]) / "SOURCE_MANIFEST.json",
                              item["source_manifest_sha256"], hashes)
    old_source = checked(root, item["old_source_manifest"],
                         item["old_source_manifest_sha256"], hashes)
    checked(root, item["gate"], item["gate_sha256"], hashes)
    checked(root, item["expected_identities"], item["expected_identities_sha256"], hashes)
    original_paths = {
        "config": checked(root, job["original_config"], job["original_config_sha256"], hashes),
        "source_manifest": checked(root, job["original_source_manifest"],
                                   job["original_source_manifest_sha256"], hashes),
        "audit_report": checked(root, job["original_audit_report"],
                                job["original_audit_report_sha256"], hashes),
        "metrics": checked(root, Path(job["original_run"]) / "metrics.json",
                           job["original_metrics_sha256"], hashes),
        "run_identity": checked(root, Path(job["original_run"]) / "run_identity.json",
                                job["original_run_identity_sha256"], hashes),
    }
    if (source_manifest != Path(item["source"]) / "SOURCE_MANIFEST.json" or
            old_source != original_paths["source_manifest"] or
            item["config_sha256"] != job["original_config_sha256"] or
            item["old_source_manifest_sha256"] != job["original_source_manifest_sha256"] or
            read(original_paths["audit_report"]).get("ok") is not True):
        raise ExportError(f"COST original quality/source identity differs: {jid}")
    acceptance = item["cost_acceptance"]
    acceptance_path = checked(root, acceptance["acceptance"], acceptance["acceptance_sha256"], hashes)
    report_path = checked(root, acceptance["report"], acceptance["report_sha256"], hashes)
    receipt_path = record(root, acceptance["receipt"], hashes)
    if receipt_path != root / "cost_queue/accepted_receipts" / f"{jid}.json":
        raise ExportError(f"COST receipt path differs: {jid}")
    main, report, receipt = read(acceptance_path), read(report_path), read(receipt_path)
    evidence = main.get("evidence", {}).get(recipe, {})
    receipt_sha = object_digest(receipt, evidence.get("receipt_sha256"))
    if (main.get("decision") != "accepted_under_COST_REPLAY_section4" or
            main.get("original_quality_retained") is not True or
            main.get("frozen_source_unchanged") is not True or
            evidence.get("audit_sha256") != acceptance["report_sha256"] or
            evidence.get("receipt_sha256") != receipt_sha or
            acceptance.get("receipt_object_sha256") != receipt_sha or
            report.get("accepted") is not True or report.get("job_id") != jid or
            report.get("original_results_modified") is not False or
            report.get("receipt") != receipt or
            set(report.get("checks", {})) - {"pair_lane"} != CHECKS or
            any(value.get("status") != "pass" for value in report["checks"].values())):
        raise ExportError(f"COST root acceptance/audit/receipt chain differs: {jid}")
    breakdown = receipt.get("required_breakdown_seconds")
    total = receipt.get("required_seconds")
    if (not isinstance(breakdown, dict) or set(breakdown) != set(TIMING) or
            any(type(breakdown[key]) not in (int, float) or
                not math.isfinite(breakdown[key]) or breakdown[key] < 0 for key in TIMING) or
            type(total) not in (int, float) or not math.isfinite(total) or total < 0 or
            abs(sum(breakdown.values()) - total) > 1e-8 or
            report["checks"]["timing"].get("required_breakdown_seconds") != breakdown or
            report["checks"]["timing"].get("required_seconds") != total or
            report["checks"]["timing"].get("coverage", {}).get("passes") is not True or
            receipt.get("interval_coverage_passed") is not True or
            receipt.get("exclusive_gpu") is not True or
            receipt.get("original_quality_retained") is not True or
            receipt.get("parity_passed") is not True):
        raise ExportError(f"COST seven-part actual timing differs: {jid}")
    if (receipt.get("recipe_id") != recipe or receipt.get("seed") != 401 or
            receipt.get("budget_return_round") != job["expected_budget_return_round"] or
            receipt.get("deployed_bytes") != job["expected_deployed_bytes"] or
            receipt.get("deployed_bytes") > job["budget_cap_bytes"] or
            receipt.get("original_config_sha256") != job["original_config_sha256"] or
            receipt.get("original_source_manifest_sha256") != job["original_source_manifest_sha256"] or
            receipt.get("instrumented_source_manifest_sha256") != item["source_manifest_sha256"]):
        raise ExportError(f"COST budget/source endpoint differs: {jid}")
    out = Path(item["out"])
    status_path = record(root, out / "status.json", hashes)
    run_identity_path = record(root, out / "run_identity.json", hashes)
    runner_attempt_path = record(root, out / "execution_attempts.json", hashes)
    status, run_identity, runner_attempts = read(status_path), read(run_identity_path), read(runner_attempt_path)
    interval_path = out / ("exclusive_timing.json" if (out / "exclusive_timing.json").is_file()
                           else "timing_events.json")
    checked(root, interval_path, receipt["interval_file_sha256"], hashes)
    draft_path = out / ("cost_prefix_draft.json" if (out / "cost_prefix_draft.json").is_file()
                        else "cost_timing_unaccepted.json")
    record(root, draft_path, hashes)
    status_round = status.get("budget_return_round", status.get("budget_stop_round"))
    if (status_round != receipt["budget_return_round"] or
            status.get("deployed_bytes") != receipt["deployed_bytes"] or
            ("original_endpoint_match" in status and status["original_endpoint_match"] is not True) or
            not isinstance(runner_attempts, list) or not runner_attempts):
        raise ExportError(f"COST executed endpoint/attempts differ: {jid}")
    attempts = item.get("attempts")
    if not isinstance(attempts, list) or not attempts or item.get("exit_code") != 0:
        raise ExportError(f"COST successful queue attempts missing: {jid}")
    attempt_rows = []
    for number, attempt in enumerate(attempts, 1):
        worker_log = attempt.get("worker_log")
        log_path = record(root, worker_log, hashes) if worker_log else None
        raw = attempt.get("raw_output")
        if raw is not None and raw != str(out):
            raise ExportError(f"COST queue attempt output differs: {jid}")
        attempt_rows.append({"job_id": jid, "attempt_number": number,
                             "queue_attempt": attempt, "runner_attempts": runner_attempts,
                             "worker_log_path": None if log_path is None else str(log_path),
                             "worker_log_sha256": None if log_path is None else hashes[str(log_path)],
                             "failed": attempt.get("exit_code") not in (None, 0)})
    last = attempts[-1]
    if (last.get("exit_code") != 0 or
            last.get("gpu_uuid") != item.get("gpu_uuid") or
            not any(str(run.get("slurm_job_id")) == str(last.get("slurm_job")) and
                    run.get("gpu_sharing") == "exclusive" for run in runner_attempts)):
        raise ExportError(f"COST accepted physical attempt differs: {jid}")
    cache = report["checks"]["prefix_cache"]
    hardware = report["checks"]["exclusive"]
    endpoint = {"job_id": jid, "recipe_id": recipe, "seed": 401,
                "budget_return_round": receipt["budget_return_round"],
                "deployed_bytes": receipt["deployed_bytes"],
                "budget_cap_bytes": job["budget_cap_bytes"],
                "hardware_model": receipt.get("gpu_model"), "hardware_uuid": hardware.get("gpu_uuid"),
                "required_seconds": total,
                **{f"seconds_{key}": breakdown[key] for key in TIMING},
                "digest_verified_clients": cache.get("digest_verified_clients"),
                "digest_unverifiable_clients": len(cache.get("digest_unverifiable_clients", [])),
                "rng_status": cache.get("rng_status"),
                "continuous_process_occupancy": hardware.get("continuous_process_occupancy"),
                "original_config_sha256": job["original_config_sha256"],
                "original_source_manifest_sha256": job["original_source_manifest_sha256"],
                "original_audit_sha256": job["original_audit_report_sha256"],
                "original_metrics_sha256": job["original_metrics_sha256"],
                "original_run_identity_sha256": job["original_run_identity_sha256"],
                "cost_audit_sha256": acceptance["report_sha256"],
                "cost_receipt_object_sha256": receipt_sha}
    provenance = {"job_id": jid, "queue_state": item["state"], "acceptance": acceptance,
                  "acceptance_evidence": evidence, "queue_attempts": attempts,
                  "runner_attempts": runner_attempts, "prefix_cache_check": cache,
                  "exclusive_check": hardware, "original_paths": {k: str(v) for k, v in original_paths.items()},
                  "instrumented_source_manifest": str(source_manifest), "status_path": str(status_path),
                  "run_identity_path": str(run_identity_path), "runner_attempt_path": str(runner_attempt_path),
                  "interval_path": str(interval_path), "unaccepted_draft_path": str(draft_path),
                  "run_identity": run_identity}
    return {"endpoint": endpoint, "receipt": receipt, "provenance": provenance,
            "attempts": attempt_rows}


def inspect_schema(root: Path) -> dict:
    rows = []
    for recipe in ("D011", "D097", "D109"):
        jid = f"COST1_{recipe}_s401"
        path = root / "cost_queue/jobs" / f"{jid}.json"
        if not path.is_file():
            rows.append({"job_id": jid, "missing": True})
            continue
        item = read(path)
        result = {"job_id": jid, "state": item.get("state"), "queue_keys": sorted(item),
                  "acceptance_keys": sorted(item.get("cost_acceptance", {})),
                  "queue_attempt_count": len(item.get("attempts", []))}
        report_path = item.get("cost_acceptance", {}).get("report")
        if report_path and Path(report_path).is_file():
            report = read(Path(report_path))
            result["audit_check_keys"] = sorted(report.get("checks", {}))
            result["receipt_keys"] = sorted(report.get("receipt") or {})
        rows.append(result)
    return {"schema": "pcla-v7-cost-schema-inspection-1", "final_statistics": False,
            "examples": rows}


def csv_text(rows: list[dict]) -> str:
    if not rows:
        raise ExportError("empty COST export")
    fields = list(rows[0])
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != fields:
            raise ExportError("COST row schema drift")
        writer.writerow(row)
    return stream.getvalue()


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def validate(root: Path) -> dict:
    root = root.resolve(strict=True)
    hashes: dict[str, str] = {}
    items = roster(root, hashes)
    for source in {Path(x["source"]) / "SOURCE_MANIFEST.json" for x in items} | {
            Path(x["job"]["original_source_manifest"]) for x in items}:
        verify_source_tree(root, source, hashes)
    extracted = [extract_one(root, item, hashes) for item in items]
    for path, original in hashes.items():
        if digest(Path(path)) != original:
            raise ExportError(f"input changed before publish: {path}")
    return {"root": root, "hashes": hashes, "rows": extracted}


def export(root: Path, out: Path) -> dict:
    out = out.resolve(strict=False)
    if out.exists() or not out.parent.is_dir():
        raise ExportError("COST output must be a new directory")
    verified = validate(root)
    root = verified["root"]
    if out.is_relative_to(root / "COST1") or out.is_relative_to(root / "cost_queue"):
        raise ExportError("COST output cannot be inside source queue/results")
    rows = verified["rows"]
    endpoints = [entry["endpoint"] for entry in rows]
    receipts = [{"job_id": entry["endpoint"]["job_id"], "receipt": entry["receipt"]} for entry in rows]
    provenance = [entry["provenance"] for entry in rows]
    attempts = [attempt for entry in rows for attempt in entry["attempts"]]
    failures = [attempt for attempt in attempts if attempt["failed"]]
    outputs = {"cost_endpoints.csv": csv_text(endpoints),
               "cost_receipts.jsonl": jsonl(receipts),
               "cost_provenance.jsonl": jsonl(provenance),
               "cost_attempts.jsonl": jsonl(attempts),
               "cost_failed_attempts.jsonl": jsonl(failures),
               "input_hashes.json": json.dumps({"schema": SCHEMA, "inputs": verified["hashes"]},
                                                indent=2, sort_keys=True) + "\n"}
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".staging.", dir=out.parent))
    try:
        output_hashes = {}
        for name, content in outputs.items():
            path = staging / name
            path.write_text(content)
            output_hashes[name] = digest(path)
        manifest = {"schema": SCHEMA, "complete": True, "job_count": 117,
                    "queue_attempt_count": len(attempts), "failed_queue_attempt_count": len(failures),
                    "root": str(root), "exporter_sha256": digest(Path(__file__)),
                    "output_sha256": output_hashes, "statistics_or_winner_selected": False}
        (staging / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Atomic exclusive reservation prevents overwriting an existing empty dir.
        out.mkdir(exist_ok=False)
        try:
            for path in staging.iterdir():
                if path.name != "EXPORT_MANIFEST.json":
                    os.replace(path, out / path.name)
            os.replace(staging / "EXPORT_MANIFEST.json", out / "EXPORT_MANIFEST.json")
        except BaseException:
            shutil.rmtree(out)
            raise
    finally:
        shutil.rmtree(staging)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-schema", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_schema:
        print(json.dumps(inspect_schema(args.root.resolve(strict=True)), indent=2, sort_keys=True))
    elif args.validate_only:
        result = validate(args.root)
        print(json.dumps({"schema": SCHEMA, "validated_jobs": len(result["rows"]),
                          "input_file_count": len(result["hashes"]), "export_created": False},
                         indent=2, sort_keys=True))
    else:
        if args.out is None:
            parser.error("--out required for formal COST export")
        print(json.dumps(export(args.root, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
