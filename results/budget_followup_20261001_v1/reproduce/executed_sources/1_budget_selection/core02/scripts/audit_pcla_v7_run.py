#!/usr/bin/env python3
"""Read-only, independent audit of a completed PCLA v7 run.

Example: python scripts/audit_pcla_v7_run.py --config JOB.json \
  --run RUN --source FROZEN_SOURCE --out AUDIT.json
Exit 0 means all checks passed; exit 1 emits a JSON failure with exact checks.
No training or evaluation implementation is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch


PROTOCOL = "pcla-global-budget-20260928-v7"
FORMAT = "pcla-v7-runtime-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_seed(*parts) -> int:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value, config_path: Path) -> Path:
    item = Path(value)
    return item if item.is_absolute() else (config_path.parent / item).resolve()


def tensors_equal(left, right) -> bool:
    return (isinstance(left, dict) and isinstance(right, dict) and
            left.keys() == right.keys() and
            all(torch.is_tensor(left[k]) and torch.is_tensor(right[k]) and
                torch.equal(left[k], right[k]) for k in left))


def read_logits(path: Path, labels: list[int]) -> dict:
    matrix = read(path)
    if len(matrix) != len(labels) or not matrix:
        raise ValueError("logit row count differs from ordered labels")
    classes = len(matrix[0])
    if classes < 2 or any(len(row) != classes for row in matrix):
        raise ValueError("inconsistent logit class count")
    if any(not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row)
           for row in matrix):
        raise ValueError("nonfinite or invalid logit")
    if any(label < 0 or label >= classes for label in labels):
        raise ValueError("label outside logit classes")
    pred = [max(range(classes), key=lambda k: row[k]) for row in matrix]
    losses = []
    for row, label in zip(matrix, labels):
        peak = max(row)
        losses.append(peak + math.log(sum(math.exp(x - peak) for x in row)) - row[label])
    return {"samples": len(labels),
            "accuracy": sum(a == b for a, b in zip(pred, labels)) / len(labels),
            "loss": sum(losses) / len(losses),
            "predicted_class_counts": [pred.count(k) for k in range(classes)]}


def check_readout(recomputed: dict, saved: dict, context: str, errors: list[str]):
    if saved.get("samples") != recomputed["samples"]:
        errors.append(f"{context}: sample count mismatch")
    for key, tolerance in (("accuracy", 1e-12), ("loss", 1e-4)):
        value = saved.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value - recomputed[key]) > tolerance:
            errors.append(f"{context}: {key} mismatch")
    if "predicted_class_counts" in saved and saved["predicted_class_counts"] != recomputed["predicted_class_counts"]:
        errors.append(f"{context}: predicted class counts mismatch")


def _support(record: dict, module_names: set[str], rank: int, method: str,
             factor: str, round_id: int, cid: int, errors: list[str]) -> int:
    prefix = f"round {round_id} client {cid}"
    masks = record.get("modules", {})
    if set(masks) != module_names:
        errors.append(f"{prefix}: module support keys mismatch")
        return 0
    per_factor = {"a": 0, "b": 0}
    for name, pair in masks.items():
        if set(pair) != {"a", "b"}:
            errors.append(f"{prefix}: factor support keys mismatch")
            continue
        for side in ("a", "b"):
            ids = pair[side]
            if (not isinstance(ids, list) or any(not isinstance(x, int) or x < 0 or x >= rank for x in ids)
                    or ids != sorted(set(ids))):
                errors.append(f"{prefix}: invalid rank indices for {name}/{side}")
                continue
            per_factor[side] += len(ids)
    total = sum(per_factor.values())
    quota = record.get("assigned_quota")
    if record.get("selected_positions") != total or not isinstance(quota, int) or quota < 0:
        errors.append(f"{prefix}: selected count or quota mismatch")
    expected_factor = "both" if method in {"full_both", "local_both_half", "cyclic_both_half"} else factor
    if record.get("active_factor") != expected_factor:
        errors.append(f"{prefix}: active factor mismatch")
    if method == "full_both":
        if per_factor != {"a": len(module_names) * rank, "b": len(module_names) * rank}:
            errors.append(f"{prefix}: incomplete full-both support")
    elif method == "full_alt":
        if per_factor[factor] != len(module_names) * rank or per_factor["a" if factor == "b" else "b"]:
            errors.append(f"{prefix}: incomplete full-alt support")
    elif method in {"local_both_half", "cyclic_both_half"}:
        if quota % 2 or per_factor != {"a": quota // 2, "b": quota // 2}:
            errors.append(f"{prefix}: both-half support does not split quota")
    elif method == "equal_quota_cyclic":
        if per_factor["a" if factor == "b" else "b"]:
            errors.append(f"{prefix}: equal-quota support updates inactive factor")
    elif total != quota or per_factor["a" if factor == "b" else "b"]:
        errors.append(f"{prefix}: partial support does not spend active quota")
    if method == "tied_local":
        for name in module_names:
            if name.endswith(".query"):
                other = name[:-5] + "value"
                if other not in masks or masks[name][factor] != masks[other][factor]:
                    errors.append(f"{prefix}: Q/V tied support differs")
                    break
    if method == "layer":
        if quota % (2 * rank):
            errors.append(f"{prefix}: layer quota is not a whole Q/V layer")
        for name in module_names:
            if name.endswith(".query"):
                other = name[:-5] + "value"
                if other not in masks or masks[name][factor] != masks[other][factor] or len(masks[name][factor]) not in {0, rank}:
                    errors.append(f"{prefix}: layer support is not whole paired Q/V")
                    break
    return total


def check_journal(path: Path, events: list[dict], final_round: int,
                  errors: list[str]) -> set[str]:
    """Match every durable failed/committed attempt against the final wire ledger."""
    by_id = {event.get("event_id"): event for event in events}
    failed_ids: set[str] = set()
    if not path.exists():
        if any(str(event.get("event_id", "")).startswith("failed:") for event in events):
            errors.append("failed wire events lack communication journal")
        return failed_ids  # core01 completed runs predate the journal
    journal = read(path)
    if journal.get("format") != "pcla-v7-communication-journal-1" or not isinstance(journal.get("entries"), list):
        errors.append("communication journal format invalid")
        return failed_ids
    completed_attempts = Counter()
    for ordinal, entry in enumerate(journal["entries"], 1):
        r, status = entry.get("round"), entry.get("status")
        sent = entry.get("sent_events")
        if entry.get("attempt") != ordinal or not isinstance(r, int) or r < 1 or not isinstance(sent, list):
            errors.append(f"journal attempt {ordinal}: invalid identity or events")
            continue
        if status == "pending":
            errors.append(f"journal attempt {ordinal}: unresolved pending communication")
            continue
        if status == "denied_no_send":
            if sent or r != final_round + 1:
                errors.append(f"journal attempt {ordinal}: invalid denied boundary")
            continue
        if status not in {"committed", "recovered_failed"}:
            errors.append(f"journal attempt {ordinal}: unknown status")
            continue
        if status == "committed":
            completed_attempts[r] += 1
            if r > final_round:
                errors.append(f"journal attempt {ordinal}: committed beyond completed rounds")
        elif entry.get("recovered_bytes") != sum(e.get("bytes", 0) for e in sent):
            errors.append(f"journal attempt {ordinal}: recovered byte count differs")
        for event in sent:
            event_id = event.get("event_id")
            expected_id = (event_id if status == "committed"
                           else f"failed:a{ordinal}:{event_id}")
            if status == "recovered_failed":
                failed_ids.add(expected_id)
            actual = by_id.get(expected_id)
            if actual != {**event, "event_id": expected_id}:
                errors.append(f"journal attempt {ordinal}: event {expected_id} differs from ledger")
    if any(value != 1 for value in completed_attempts.values()) or set(completed_attempts) != set(range(1, final_round + 1)):
        errors.append("journal does not contain one committed attempt per completed round")
    journal_event_ids = {e.get("event_id") for entry in journal["entries"]
                         if entry.get("status") == "committed" for e in entry.get("sent_events", [])}
    actual_ids = {e.get("event_id") for e in events}
    if actual_ids != journal_event_ids | failed_ids:
        errors.append("journal and final wire ledger have different event sets")
    return failed_ids


def audit(config_path: Path, run_dir: Path, source: Path) -> dict:
    errors: list[str] = []
    checks: dict[str, object] = {}
    config_path, run_dir, source = config_path.resolve(), run_dir.resolve(), source.resolve()
    try:
        cfg = read(config_path)
        status = read(run_dir / "status.json")
        ck = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
        saved_rows = read(run_dir / "metrics.json")
        identity = read(run_dir / "run_identity.json")
        roles_path = resolve(cfg["manifest"], config_path)
        roles = read(roles_path)
        b8_path = resolve(cfg["b8_wire"], config_path)
        b8 = read(b8_path)
        labels = read(run_dir / "evaluation_labels.json")
    except Exception as exc:
        return {"ok": False, "errors": [f"required artifact unreadable: {type(exc).__name__}: {exc}"],
                "run_dir": str(run_dir)}
    job = cfg["job"]
    cap = int(cfg["runtime"]["byte_cap"])
    final_round = status.get("completed_round")
    if (status.get("state") != "completed" or not isinstance(final_round, int)
            or not 100 <= final_round <= 200):
        errors.append("status is not a completed 100..200-round trajectory")
    if not isinstance(final_round, int) or not isinstance(saved_rows, list):
        return {"ok": False, "errors": errors, "run_dir": str(run_dir), "checks": checks}
    if (job.get("phase") != "D0" or cfg.get("technical_profile_only") or
            cfg.get("technical_stop_after") is not None or
            job.get("min_rounds") != 100 or job.get("max_rounds") != 200):
        errors.append("technical or non-D0 trajectory cannot count as formal completion")
    if ck.get("format") != FORMAT or ck.get("round") != final_round:
        errors.append("latest checkpoint format/round differs from status")
    if [r.get("round") for r in saved_rows] != list(range(1, final_round + 1)):
        errors.append("metrics do not contain every completed round exactly once")
    if json.loads(json.dumps(ck.get("metrics", []))) != saved_rows:
        errors.append("checkpoint metrics differ from metrics.json")
    if job.get("protocol") != PROTOCOL or job.get("eval_role") != "recipe_dev" or identity.get("job") != job:
        errors.append("run/job protocol or evaluation role differs from config")
    if cfg.get("expanded_sha256") != canonical_hash({k: v for k, v in cfg.items() if k != "expanded_sha256"}):
        errors.append("expanded config digest mismatch")
    provenance = identity.get("provenance", {})
    if provenance.get("config_sha256") != sha(config_path):
        errors.append("run config bytes differ from recorded identity")
    attempts = read(run_dir / "execution_attempts.json") if (run_dir / "execution_attempts.json").exists() else []
    if not attempts or any(a.get("identity") != ck.get("identity") for a in attempts):
        errors.append("execution attempt identities differ from checkpoint")
    source_manifest_path = source / "SOURCE_MANIFEST.json"
    try:
        source_manifest = read(source_manifest_path)
        if "source_manifest" in cfg and resolve(cfg["source_manifest"], config_path) != source_manifest_path:
            errors.append("config source manifest path differs from audited source")
        if provenance.get("source_manifest_sha256") != sha(source_manifest_path):
            errors.append("frozen source manifest hash differs from run identity")
        for relative, expected in source_manifest["files"].items():
            path = source / relative
            if not path.is_file() or sha(path) != expected:
                errors.append(f"frozen source changed: {relative}")
        checks["source_files"] = len(source_manifest["files"])
        for relative, expected in provenance.get("sources", {}).items():
            if source_manifest["files"].get(relative) != expected:
                errors.append(f"run source identity differs from frozen manifest: {relative}")
    except Exception as exc:
        errors.append(f"frozen source manifest unreadable: {type(exc).__name__}: {exc}")
    dispatch_path = run_dir / "dispatch_identity.json"
    if dispatch_path.exists():
        dispatch = read(dispatch_path)
        if dispatch.get("config_sha256") != sha(config_path) or dispatch.get("source_manifest_sha256") != sha(source_manifest_path):
            errors.append("dispatch config/source identity differs")
    if roles.get("protocol") != PROTOCOL or provenance.get("manifest_sha256") != sha(roles_path):
        errors.append("train role manifest identity differs")
    if provenance.get("split_sha256") != roles.get("split_sha256") or roles.get("split_sha256") != canonical_hash({k: v for k, v in roles.items() if k != "split_sha256"}):
        errors.append("train role split digest differs")
    train_path = resolve(cfg["train_file"], config_path)
    if roles.get("source_sha256") != sha(train_path) or provenance.get("source_sha256") != sha(train_path):
        errors.append("SST-2 train source identity differs")
    cache_path = resolve(cfg["token_cache"], config_path)
    token_meta = read(cache_path.with_suffix(".json"))
    if (token_meta.get("cache_sha256") != sha(cache_path) or
            provenance.get("token_cache_sha256") != token_meta.get("cache_sha256") or
            token_meta.get("identity", {}).get("split_sha256") != roles.get("split_sha256")):
        errors.append("token cache identity differs")
    if (b8.get("wire_manifest_sha256") != canonical_hash({k: v for k, v in b8.items() if k != "wire_manifest_sha256"})
            or b8.get("total_bytes") != cap or provenance.get("b8_wire_manifest_sha256") != sha(b8_path)):
        errors.append("B8 wire definition or cap differs")
    role_ids = roles.get("roles", {}).get("recipe_dev", [])
    if (len(role_ids) != 4096 or len(labels) != 4096 or
            labels != [roles.get("row_labels", {}).get(item) for item in role_ids]):
        errors.append("ordered evaluation labels differ from frozen recipe_dev rows")
    checks["round_readouts"] = []
    if final_round:
        for round_id in range(0, final_round + 1):
            path = run_dir / f"logits_round_{round_id:03d}.json"
            try:
                computed = read_logits(path, labels)
                readout = read(run_dir / "evaluation_round_000.json") if round_id == 0 else saved_rows[round_id - 1]["evaluation"]
                check_readout(computed, readout, f"round {round_id}", errors)
                checks["round_readouts"].append({"round": round_id, **computed})
                if round_id in (0, final_round):
                    checks[f"predicted_class_counts_round_{round_id}"] = computed["predicted_class_counts"]
            except Exception as exc:
                errors.append(f"round {round_id}: logits unreadable or invalid: {type(exc).__name__}: {exc}")
    events = ck.get("cache", {}).get("events", [])
    if len({e.get("event_id") for e in events}) != len(events):
        errors.append("duplicate wire event ID")
    buckets = defaultdict(int)
    cumulative = defaultdict(lambda: defaultdict(int))
    for event in events:
        bucket, r, amount = event.get("bucket"), event.get("round"), event.get("bytes")
        if bucket not in {"deployed_prefix", "diagnostic_after_budget"} or not isinstance(r, int) or not 1 <= r <= (final_round or 0) or not isinstance(amount, int) or amount < 0:
            errors.append("invalid wire event bucket/round/bytes")
            continue
        buckets[bucket] += amount
        cumulative[r][bucket] += amount
    if buckets["deployed_prefix"] != ck.get("deployed_bytes") or buckets["diagnostic_after_budget"] != ck.get("diagnostic_bytes"):
        errors.append("wire event sums differ from checkpoint ledgers")
    if ck.get("deployed_bytes", cap + 1) > cap or ck.get("deployed_bytes", -1) < 0:
        errors.append("deployed ledger exceeds B8")
    deployed_running = diagnostic_running = 0
    for row in saved_rows:
        r = row["round"]
        deployed_running += cumulative[r]["deployed_prefix"]
        diagnostic_running += cumulative[r]["diagnostic_after_budget"]
        if row.get("deployed_bytes") != deployed_running or row.get("diagnostic_after_budget_bytes") != diagnostic_running:
            errors.append(f"round {r}: cumulative wire ledger differs")
        selected = row.get("selected_clients", [])
        trace = row.get("support_trace", {})
        if (len(selected) != 10 or len(set(selected)) != 10 or
                any(not isinstance(cid, int) or cid not in range(50) for cid in selected) or
                set(trace) != {str(cid) for cid in selected}):
            errors.append(f"round {r}: selected cohort/support trace differs")
            continue
        factor = "b" if r % 2 else "a"
        module_names = {name[:-7] for name in ck.get("adapter", {}) if name.endswith(".lora_a")}
        if len(module_names) != 24:
            errors.append("checkpoint does not contain 24 Q/V adapters")
        per_client = {}
        for cid in selected:
            record = trace[str(cid)]
            per_client[cid] = _support(record, module_names, int(job["rank"]), job["method"], factor, r, cid, errors)
        if job["method"] == "equal_quota_cyclic":
            low, rem = divmod(sum(trace[str(cid)]["assigned_quota"] for cid in selected), len(selected))
            sorted_ids = sorted(selected)
            expected = {cid: low for cid in selected}
            for i in range(rem):
                expected[sorted_ids[(r - 1 + i) % len(selected)]] += 1
            if per_client != expected:
                errors.append(f"round {r}: equal cohort quota distribution differs")
        clients = row.get("client_training", [])
        if [x.get("client") for x in clients] != selected or any(x.get("optimizer_steps") != 16 for x in clients):
            errors.append(f"round {r}: client step count/order differs")
        for client in clients:
            cid = client["client"]
            if client.get("optimizer_tensor_parameters") != per_client.get(cid, -10) * 768 + 1538:
                errors.append(f"round {r} client {cid}: compact optimizer size differs")
    if status.get("deployed_bytes") != ck.get("deployed_bytes") or status.get("diagnostic_after_budget_bytes") != ck.get("diagnostic_bytes"):
        errors.append("status and checkpoint ledger totals differ")
    stop = ck.get("budget_stop_round")
    if not isinstance(stop, int) or not 0 <= stop <= final_round or status.get("budget_stop_round") != stop:
        errors.append("invalid budget stop round")
    budget_path, half_path = run_dir / "budget_return.pt", run_dir / "half_budget_return.pt"
    if not budget_path.exists() or not half_path.exists():
        errors.append("budget return objects missing")
    else:
        budget = torch.load(budget_path, map_location="cpu", weights_only=True)
        half = torch.load(half_path, map_location="cpu", weights_only=True)
        expected_budget_bytes = saved_rows[stop - 1]["deployed_bytes"] if stop else 0
        if (budget.get("identity") != ck.get("identity") or budget.get("round") != stop or
                budget.get("budget_cap_bytes") != cap or budget.get("deployed_bytes") != expected_budget_bytes or
                not tensors_equal(budget.get("adapter"), ck.get("budget_adapter"))):
            errors.append("B8 return object differs from frozen endpoint")
        half_limit = cap // 2
        eligible = [0] + [row["round"] for row in saved_rows
                          if row["round"] <= stop and row["deployed_bytes"] <= half_limit]
        expected_half = max(eligible)
        expected_half_bytes = saved_rows[expected_half - 1]["deployed_bytes"] if expected_half else 0
        if (half.get("identity") != ck.get("identity") or half.get("round") != expected_half or
                half.get("budget_cap_bytes") != half_limit or half.get("deployed_bytes") != expected_half_bytes or
                not tensors_equal(half.get("adapter"), ck.get("half_adapter"))):
            errors.append("B8/2 return object differs from last eligible model")
        expected_readout = read(run_dir / "evaluation_round_000.json") if expected_half == 0 else saved_rows[expected_half - 1]["evaluation"]
        if half.get("readout") != expected_readout:
            errors.append("B8/2 readout differs from returned model round")
        checks.update({"budget_stop_round": stop, "budget_return_bytes": expected_budget_bytes,
                       "half_budget_round": expected_half, "half_budget_bytes": expected_half_bytes})
    checks.update({"completed_round": final_round, "planned_job_id": job.get("job_id"),
                   "event_count": len(events), "deployed_bytes": ck.get("deployed_bytes"),
                   "diagnostic_after_budget_bytes": ck.get("diagnostic_bytes"),
                   "b8_cap_bytes": cap})
    return {"ok": not errors, "errors": errors, "run_dir": str(run_dir), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.run_dir, args.source)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    if args.json_out:
        path = args.json_out.resolve()
        if path == args.run_dir.resolve() or args.run_dir.resolve() in path.parents:
            parser.error("--json-out must be outside the audited run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
