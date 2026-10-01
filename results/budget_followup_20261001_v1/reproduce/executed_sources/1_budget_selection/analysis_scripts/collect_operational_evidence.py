#!/usr/bin/env python3
"""Copy bounded post-D0 operational originals and index their measured fields."""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


PHASE = Path(__file__).resolve().parent
RUN = PHASE.parents[1]
REMOTE = Path("$HPC_HOME/pcla_global_budget_20260928_v7")
OUT = PHASE / "operational_evidence01"
SACCT_IDS = ("172165.284", "173697.4", "173697.6", "173697.11",
             "173697.18", "173697.19", "173697.20", "173697.26",
             "173698.31", "173698.32", "173698.37", "173698.38")
MAX_BYTES = 262_144
SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".stderr", ".stdout",
            ".exit", ".sacct", ".md", ".sh"}
EXCLUDE_NAMES = {"metrics.json", "evaluation_labels.json", "communication_journal.json",
                 "timing_events.json", "input_hashes.json"}
EXCLUDE_PREFIXES = ("logits_",)

# These are post-D0 operational runs, candidates, and closeout records. No D0
# model/quality tree or credential/config home is traversed.
REMOTE_DIRS = {
    "core01_stop": ("checks/cost_gpu_core01_intake_20260928/attempt01",
                    "checks/cost_gpu_core01_intake_20260928/attempt02"),
    "jt_technical_cap": ("checks/cost_gpu_technical01", "checks/cost_gpu_jt_repair02"),
    "diagnostic_import": ("checks/diagnostic_gpu03",),
    "diagnostic_uuid_preclaim": ("checks/diagnostic_first01", "checks/diagnostic_first02"),
    "guard_and_router": ("checks/adaptive_scheduling01", "checks/adaptive_scheduling02",
                         "checks/diagnostic_batch_launch01"),
    "seal_recovery": ("checks/conditional_seal_repair01",),
    "candidate_audit": ("checks/cost_audit_staging/candidates_27",
                        "checks/cost_audit_staging/accepted_27"),
}
REMOTE_GLOBS = {
    "guard_and_router": (
        "checks/drain_173697_4_*.json",
        "checks/drain_173697_4_*.log",
        "checks/drain_guard_173697_4_*.log",
        "checks/drain_173697_5_04.json",
        "checks/drain_guard_173697_5_04.log",
    ),
}
LOCAL_DIRS = {
    "candidate_audit": ("checks/cost_independent_audit_20260928/candidates_27",),
}
LOCAL_GLOBS = {
    "diagnostic_import": ("checks/diagnostic_gpu03/IMPORT_REVIEW.md",
                          "checks/diagnostic_gpu03/INDEPENDENT_GPU_REVIEW.md"),
    "diagnostic_uuid_preclaim": ("checks/diagnostic_first01/INDEPENDENT_FIRST_REVIEW.md",),
    "guard_and_router": ("checks/adaptive_scheduling02/OPERATIONS.md",
                         "checks/heartbeat_1726/ROUTER_*.json",
                         "checks/heartbeat_1047/CONTINUATION_1928.md",
                         "checks/heartbeat_1047/drain_173697_4_05_reconciled.json"),
    "seal_recovery": ("checks/conditional_seal_repair01/*REVIEW*.md",),
    "candidate_audit": ("checks/heartbeat_1726/cost_audit/ADAPTIVE02_*.json",
                        "checks/heartbeat_1726/cost_audit/CPU_STEPS_SACCT.txt",
                        "checks/heartbeat_1726/cost_audit/LIVE_QUEUE_SNAPSHOT.json",
                        "checks/heartbeat_1726/cost_audit/EXECUTION_RECEIPT.json"),
    "validation_tools01_argv": ("checks/heartbeat_1726/VALIDATION_EXPORT_*",
                                "checks/heartbeat_1726/EXPORT_VALIDATION_DEPLOY_SHA.txt"),
    "cost_sync27": ("checks/heartbeat_1726/COST_SYNC27_*.jsonl",
                    "checks/heartbeat_1726/COST_SYNC27_*.sh",
                    "checks/heartbeat_1726/COST_COMPLETENESS.json"),
    "formal_export_operations": ("checks/heartbeat_1800/*EXPORT*.stdout",
                                 "checks/heartbeat_1800/*EXPORT*.stderr",
                                 "checks/heartbeat_1800/ROOT_EXPORT_DECISION.json"),
    "local_cohort_source_reconciliation": (
        "checks/heartbeat_1815/COHORT_SOURCE_RECONCILIATION.json",
        "checks/heartbeat_1815/PROGRESS.md",
        "checks/heartbeat_1815/COHORT_SUMMARY.stdout",
        "checks/phase_closeout_20260929/COHORT_READONLY_PROBE_20260929.json",
        "checks/phase_closeout_20260929/COHORT_READONLY_PROBE_20260929_v2.json",
        "checks/phase_closeout_20260929/cohort_summary01/SUMMARY_MANIFEST.json",
        "checks/phase_closeout_20260929/README_COHORT_SUMMARY.md",
    ),
}


def eligible(path: Path) -> bool:
    return (path.is_file() and not path.is_symlink()
            and path.suffix in SUFFIXES and path.name not in EXCLUDE_NAMES
            and not path.name.startswith(EXCLUDE_PREFIXES)
            and (path.suffix != ".sh" or path.name.startswith("COST_SYNC27_"))
            and path.stat().st_size <= MAX_BYTES)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def subcase(category: str, relative: str) -> str:
    name = Path(relative).name
    if category == "core01_stop":
        return "old_stop_ignored" if "/attempt01/" in relative else "explicit_stop_repair"
    if category == "jt_technical_cap":
        return "jt_cap_repair" if "/cost_gpu_jt_repair02/" in relative else "initial_cost_technical_cases"
    if category == "diagnostic_import":
        if "/tech01/direction_a/" in relative or name == "direction_a.log":
            return "direction_import_failure"
        if "/tech02/direction_a/" in relative or name == "direction_a_retry02.log":
            return "direction_import_repair"
        return "other_diagnostic_technical_cases"
    if category == "diagnostic_uuid_preclaim":
        return "first_uuid_preclaim_failure" if "/diagnostic_first01/" in relative else "uuid_repair_probe"
    if category == "guard_and_router":
        if "drain_173697_4_" in relative or "drain_guard_173697_4_" in relative:
            return "old_guard_refusal_enoent_timeout"
        if "guard" in name.lower() or "drain_173697_5_04" in relative:
            return "guard_followon"
        if any(f"lane{n}" in relative for n in (10, 13)):
            return "router_preclaim_import_failure"
        if any(f"lane{n}" in relative for n in (14, 15, 16, 17)):
            return "router_wait_timeout"
        if "/adaptive_scheduling02/" in relative:
            return "router_replacement"
        return "other_router_operations"
    if category == "seal_recovery":
        if "/s101_package/" in relative or name == "INSTALL_RESULT.json":
            return "s101_exit1_metadata_recovery"
        if "/s109_package/" in relative or name == "S109_INSTALL_RESULT.json":
            return "s109_exit1_metadata_recovery"
        return "seal_recovery_crosscheck"
    if category == "candidate_audit":
        if "/repaired_audits/" in relative:
            return "d109110_derived_audit_repair"
        if name.startswith(("D109", "D110")):
            return "d109110_initial_audit_failure"
        if "ADAPTIVE02" in name or "/adaptive02/" in relative:
            return "d109110_derived_evidence_provenance"
        return "candidate_audit_and_sync_context"
    if category == "cost_sync27":
        return "first_dry_zero_output" if name in {"COST_SYNC27_DRY.jsonl", "COST_SYNC27_DRY.sh"} else "split_dry_execute"
    if category == "local_cohort_source_reconciliation":
        if name == "COHORT_READONLY_PROBE_20260929.json":
            return "stale_probe_gate_failure_context"
        if name == "COHORT_READONLY_PROBE_20260929_v2.json":
            return "current_readonly_v2_probe"
        if name in {"COHORT_SOURCE_RECONCILIATION.json", "PROGRESS.md"}:
            return "gate_failure_and_final_review_record"
        return "accepted_summary_context"
    return category


def remote_script() -> str:
    # The remote process only reads paths under the fixed project root.
    return f"""
import base64, hashlib, json
from pathlib import Path
root = Path({str(REMOTE)!r})
groups = {REMOTE_DIRS!r}
globs = {REMOTE_GLOBS!r}
suffixes = set({sorted(SUFFIXES)!r})
excluded = set({sorted(EXCLUDE_NAMES)!r})
limit = {MAX_BYTES}
for category, directories in groups.items():
    for relative in directories:
        base = root / relative
        if not base.is_dir():
            print(json.dumps({{"category": category, "missing_directory": relative}}))
            continue
        for path in sorted(base.rglob("*")):
            if (not path.is_file() or path.is_symlink() or path.suffix not in suffixes
                    or (path.suffix == ".sh" and not path.name.startswith("COST_SYNC27_"))
                    or path.name in excluded or path.name.startswith("logits_")):
                continue
            size = path.stat().st_size
            if size > limit:
                print(json.dumps({{"category": category, "skipped_large": str(path.relative_to(root)),
                                  "size_bytes": size}}))
                continue
            data = path.read_bytes()
            print(json.dumps({{"category": category, "relative": str(path.relative_to(root)),
                              "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "data_b64": base64.b64encode(data).decode()}}))
for category, patterns in globs.items():
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if not paths:
            print(json.dumps({{"category": category, "missing_glob": pattern}}))
        for path in paths:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
                continue
            data = path.read_bytes()
            print(json.dumps({{"category": category, "relative": str(path.relative_to(root)),
                              "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "data_b64": base64.b64encode(data).decode()}}))
"""


def load_remote() -> tuple[list[dict], list[dict]]:
    proc = subprocess.run(["ssh", "hpc-bnbu", "python -"], input=remote_script(),
                          text=True, capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"read-only SSH collection failed: exit {proc.returncode}: {proc.stderr[-1000:]}")
    items, notes = [], []
    for line in proc.stdout.splitlines():
        item = json.loads(line)
        if "data_b64" not in item:
            notes.append(item)
            continue
        data = base64.b64decode(item.pop("data_b64"), validate=True)
        if len(data) != item["size_bytes"] or sha(data) != item["sha256"]:
            raise RuntimeError(f"remote transfer changed: {item['relative']}")
        item["data"] = data
        items.append(item)
    return items, notes


def load_local() -> tuple[list[dict], list[dict]]:
    items, notes = [], []
    pairs = []
    for category, directories in LOCAL_DIRS.items():
        for relative in directories:
            base = RUN / relative
            if not base.is_dir():
                notes.append({"category": category, "missing_directory": relative})
                continue
            pairs.extend((category, path) for path in sorted(base.rglob("*")))
    for category, patterns in LOCAL_GLOBS.items():
        for pattern in patterns:
            matches = sorted(RUN.glob(pattern))
            if not matches:
                notes.append({"category": category, "missing_glob": pattern})
            pairs.extend((category, path) for path in matches)
    for category, path in pairs:
        if not eligible(path):
            continue
        data = path.read_bytes()
        items.append({"category": category, "relative": str(path.relative_to(RUN)),
                      "size_bytes": len(data), "sha256": sha(data), "data": data})
    return items, notes


def measured_fields(data: bytes, filename: str) -> list[dict]:
    if not filename.endswith(".json"):
        return []
    try:
        obj = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    results = []
    names = {"elapsed_wall_seconds", "technical_gpu_seconds_one_device",
             "wall_seconds", "wall_time_seconds", "duration_seconds",
             "prepare_seconds", "exercise_seconds", "required_seconds",
             "required_seconds_technical_only", "envelope_seconds"}

    def visit(value, location: str, depth: int) -> None:
        if depth > 3 or len(results) >= 40:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                here = f"{location}.{key}" if location else key
                if key in names and type(child) in (int, float):
                    results.append({"json_path": here, "value": child, "unit": "seconds",
                                    "basis": "original_numeric_field"})
                elif key in {"attempts", "driver_attempts"} and isinstance(child, list):
                    for i, attempt in enumerate(child[:8]):
                        visit(attempt, f"{here}[{i}]", depth + 1)
                elif depth < 2 and isinstance(child, dict):
                    visit(child, here, depth + 1)
    visit(obj, "", 0)
    return results


def capture_sacct(staging: Path) -> dict:
    name = Path("captures") / "SELECTED_SLURM_STEPS.psv"
    path = staging / name
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ("sacct -n -P -j " + ",".join(SACCT_IDS)
               + " -o JobIDRaw,State,ExitCode,ElapsedRaw,Start,End")
    if not path.exists():
        proc = subprocess.run(["ssh", "hpc-bnbu", command], capture_output=True,
                              check=False)
        if proc.returncode:
            raise RuntimeError(f"read-only sacct capture failed: {proc.stderr.decode(errors='replace')[-1000:]}")
        path.write_bytes(proc.stdout)
        captured_at = datetime.now(timezone.utc).isoformat()
    else:
        captured_at = None
    return {"command": command, "copy": str(name), "sha256": sha(path.read_bytes()),
            "captured_at_utc": captured_at,
            "capture_file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "unit": "ElapsedRaw is seconds",
            "scope": "selected Slurm steps; shared step time is not per-object time"}


def capture_cohort_source_check(staging: Path) -> dict:
    """Reconcile the recorded probe identities with the accepted local source."""
    reconciliation = json.loads((RUN / "checks/heartbeat_1815/COHORT_SOURCE_RECONCILIATION.json").read_text())
    old_probe = json.loads((PHASE / "COHORT_READONLY_PROBE_20260929.json").read_text())
    new_probe = json.loads((PHASE / "COHORT_READONLY_PROBE_20260929_v2.json").read_text())
    manifest = json.loads((PHASE / "cohort_summary01/SUMMARY_MANIFEST.json").read_text())
    current_sha = sha((PHASE / "summarize_cohort.py").read_bytes())
    if not (old_probe["summarizer_sha256"] == reconciliation["previous_probe_sha"]
            and new_probe["summarizer_sha256"] == reconciliation["script_sha256"]
            == manifest["summarizer_sha256"] == current_sha):
        raise RuntimeError("cohort source/probe identity mismatch")
    output_checks = {}
    for name, expected in manifest["outputs"].items():
        path = PHASE / "cohort_summary01" / name
        data = path.read_bytes()
        actual = {"sha256": sha(data), "rows": len(data.splitlines())}
        if actual != {"sha256": expected["sha256"], "rows": expected["rows"]}:
            raise RuntimeError(f"cohort summary changed: {name}")
        output_checks[name] = actual
    test_out = staging / "captures/COHORT_FINAL_FIXTURE_TEST.stdout"
    test_exit = staging / "captures/COHORT_FINAL_FIXTURE_TEST.exit"
    test_capture = None
    if test_out.exists() and test_exit.exists():
        test_capture = {"kind": "independent_local_fixture_rerun_not_original_root_stdout",
                        "stdout_copy": str(test_out.relative_to(staging)),
                        "stdout_sha256": sha(test_out.read_bytes()),
                        "exit_copy": str(test_exit.relative_to(staging)),
                        "exit_sha256": sha(test_exit.read_bytes()),
                        "exit_code": int(test_exit.read_text().strip())}
    result = {"schema": "pcla-v7-local-cohort-source-reconciliation-check-1",
              "classification": "local source/probe gate and review; no GPU or training event",
              "old_probe_source_sha256": old_probe["summarizer_sha256"],
              "accepted_source_sha256": current_sha,
              "test_source_sha256": sha((PHASE / "test_summarize_cohort.py").read_bytes()),
              "root_six_test_result_source": "checks/heartbeat_1815/COHORT_SOURCE_RECONCILIATION.json",
              "new_probe_checks": {key: new_probe[key] for key in (
                  "nonadditivity_independent_checks", "source_h32_full_spearman_checks",
                  "cohort_objects", "cohort_object_values", "cohort_selection16_contrasts")},
              "output_checks": output_checks, "independent_fixture_test_capture": test_capture}
    path = staging / "captures/COHORT_RECONCILIATION_CHECK.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"copy": str(path.relative_to(staging)), "sha256": sha(path.read_bytes()),
            "source_status": "old probe stale; current source, v2 probe, eight outputs match"}


def main() -> None:
    remote, remote_notes = load_remote()
    local, local_notes = load_local()
    resume = OUT.exists()
    if resume:
        previous = json.loads((OUT / "SOURCE_INDEX.json").read_text())
        if previous.get("schema") != "pcla-v7-post-d0-operational-evidence-index-1":
            raise RuntimeError("existing directory is not this collector's output")
        staging = OUT
    else:
        previous = {"sources": []}
        staging = Path(tempfile.mkdtemp(prefix=OUT.name + ".staging.", dir=PHASE))
    try:
        records = list(previous["sources"])
        seen = {(x["origin"], str(Path(x["source"]).relative_to(
            REMOTE if x["origin"] == "remote" else RUN))): x for x in records}
        for origin, items in (("remote", remote), ("local", local)):
            for item in items:
                relative = Path(item["relative"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"unsafe source path: {relative}")
                key = (origin, str(relative))
                if key in seen:
                    if seen[key]["sha256"] != item["sha256"]:
                        raise RuntimeError(f"existing source changed: {relative}")
                    continue
                seen[key] = item
                copy_relative = Path("originals") / origin / relative
                copy_path = staging / copy_relative
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                copy_path.write_bytes(item["data"])
                records.append({
                    "category": item["category"],
                    "subcase": subcase(item["category"], str(relative)),
                    "origin": origin,
                    "source": str((REMOTE if origin == "remote" else RUN) / relative),
                    "copy": str(copy_relative),
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                    "measured_fields": measured_fields(item["data"], relative.name),
                })
        for record in records:
            copy_path = staging / record["copy"]
            data = copy_path.read_bytes()
            if sha(data) != record["sha256"] or len(data) != record["size_bytes"]:
                raise RuntimeError(f"copied evidence changed: {copy_path}")
            record["measured_fields"] = measured_fields(data, copy_path.name)
            record["subcase"] = subcase(record["category"], str(Path(record["copy"]).relative_to(
                Path("originals") / record["origin"])))
        records.sort(key=lambda x: (x["category"], x["origin"], x["source"]))
        missing = remote_notes + local_notes
        counts = dict(sorted(Counter(x["category"] for x in records).items()))
        subcase_counts = dict(sorted(Counter(x["subcase"] for x in records).items()))
        slurm_capture = capture_sacct(staging)
        cohort_source_check = capture_cohort_source_check(staging)
        index = {
            "schema": "pcla-v7-post-d0-operational-evidence-index-1",
            "scope": "bounded operational originals after D0; not a completed phase-cost total",
            "remote_root": str(REMOTE),
            "local_root": str(RUN),
            "file_size_limit_bytes": MAX_BYTES,
            "source_count": len(records),
            "source_count_by_category": counts,
            "source_count_by_subcase": subcase_counts,
            "collection_gaps": missing,
            "time_rule": "Only original numeric seconds fields are copied; intervals and overlapping stages are never summed.",
            "deduplication_rule": "One copied source file per origin and relative path; case/event counts are not attempt counts.",
            "slurm_capture": slurm_capture,
            "cohort_source_check": cohort_source_check,
            "sources": records,
        }
        index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
        index_temp = staging / "SOURCE_INDEX.json.tmp"
        index_temp.write_bytes(index_bytes)
        os.replace(index_temp, staging / "SOURCE_INDEX.json")
        # A small count receipt makes it easy to check the package without
        # loading the full source index; hashes remain authoritative there.
        receipt = {"schema": "pcla-v7-operational-collection-receipt-1",
                   "source_count": len(records), "source_count_by_category": counts,
                   "collection_gap_count": len(missing), "source_index_sha256": sha((staging / "SOURCE_INDEX.json").read_bytes())}
        receipt_temp = staging / "COLLECTION_RECEIPT.json.tmp"
        receipt_temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(receipt_temp, staging / "COLLECTION_RECEIPT.json")
        if not resume:
            os.rename(staging, OUT)
        print(json.dumps(receipt, sort_keys=True))
    except BaseException:
        if not resume:
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
