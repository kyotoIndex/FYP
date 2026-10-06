#!/usr/bin/env python3
"""Read-only independent receipt for a finished PCLA v8 DEV trajectory.

This auditor never imports the runner or server controller. Its receipt is
separate from process exit and from the queue's executed_pending_audit state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import torch

B8 = 2382653750
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
SUPPORTED = {"mean_r8", "mean_r4", "fixed_scalar", "fixed_head", "scalar4", "scalar17", "block16", "separable"}
PUBLIC_GRID = {"scalar4": 4, "scalar17": 17, "block16": 16, "separable": 16}
REQUIRED_GROUPS = ("task_initialization", "state_and_optimizer_setup", "client_training",
                   "scoring_and_shared_information", "coordinate_and_server_decision",
                   "message_processing", "budget_and_return_control")
LABEL_GROUP = {**{group: group for group in REQUIRED_GROUPS},
               "local_training": "client_training", "local_setup": "state_and_optimizer_setup"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def resolved(value, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def require(condition, message: str):
    if not condition:
        raise ValueError(message)


def tensors_equal(left, right) -> bool:
    return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
            and all(torch.is_tensor(left[k]) and torch.is_tensor(right[k])
                    and torch.equal(left[k], right[k]) for k in left))


def state_hash(values: dict) -> str:
    hasher = hashlib.sha256()
    for key in sorted(values):
        value = values[key].detach().cpu().contiguous()
        hasher.update(key.encode()); hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode()); hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def baseline_tensor_digest(values: dict) -> str:
    """Independent implementation of core02 pcla_repair.core.tensor_digest."""
    hasher = hashlib.sha256()
    for key in sorted(values):
        value = values[key].detach().cpu().contiguous()
        hasher.update(key.encode())
        hasher.update(str((str(value.dtype), tuple(value.shape))).encode())
        hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def checkpoint(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(value, dict) and value.get("format") == "pcla-v7-runtime-1", f"bad checkpoint {path.name}")
    return value


def check_state(value: dict, rank: int, label: str):
    require(isinstance(value, dict) and len(value) == 50, f"{label}: expected 24 Q/V factor pairs and classifier head")
    require({"classifier.weight", "classifier.bias"} <= set(value), f"{label}: missing classifier head")
    a = [name for name in value if name.endswith(".lora_a")]
    b = [name for name in value if name.endswith(".lora_b")]
    require(len(a) == len(b) == 24 and {name[:-1] for name in a} == {name[:-1] for name in b},
            f"{label}: factor key or count differs")
    for name, tensor in value.items():
        require(torch.is_tensor(tensor) and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                f"{label}: invalid tensor {name}")
        if name.endswith(".lora_a"):
            require(tensor.ndim == 2 and tensor.shape[0] == rank, f"{label}: wrong A rank")
        elif name.endswith(".lora_b"):
            require(tensor.ndim == 2 and tensor.shape[1] == rank, f"{label}: wrong B rank")
        else:
            require(name.startswith("classifier."), f"{label}: frozen baseline weight leaked into adapter")


def check_inputs(config_path: Path, source_root: Path, cfg: dict, result: dict, checks: dict):
    job, recipe, binding = cfg["job"], cfg["recipe"], cfg["binding"]
    require(result["job_id"] == job["job_id"] and result["recipe"] == recipe, "result identity/recipe differs")
    require(result["config_sha256"] == sha(config_path), "result config hash differs")
    require(binding["status"] == "bound_unverified", "config was never input bound")
    require(job["phase"] == "DEV" and job["eval_role"] == "recipe_dev" and job["protocol"] == cfg["common"]["protocol"],
            "job role/protocol differs")
    require(recipe["server_rule"] in SUPPORTED, "method requires a separate explicit auditor extension")
    require(recipe["config_sha256"] == hashlib.sha256(json.dumps(
        {k: v for k, v in recipe.items() if k != "config_sha256"}, sort_keys=True).encode()).hexdigest(),
        "received recipe hash differs")
    require(cfg["runtime"]["byte_cap"] == cfg["common"]["budget"]["B8"] == B8,
            "B8 constant differs")
    require(binding["model_revision"] == cfg["common"]["model"]["revision"] == MODEL_REVISION,
            "model revision differs")
    require(job["hardware_family"] == "RTX4090D" and "4090" in result["gpu"]["name"],
            "hardware family differs")
    require(job["rank"] == recipe["rank"] and job["lr"] == recipe["client_lr"]
            and job["head_lr"] == recipe["head_lr"], "job/recipe training settings differ")
    require(job["method"] == "full_both" and job["representation"] == "N"
            and recipe["factor_schedule"] == "both", "supported baseline coordinate path differs")
    root = config_path.resolve().parents[2]
    scope = root / "scope_authorization.json"
    require(scope.is_file() and sha(scope) == binding["scope_authorization_sha256"], "scope receipt differs")
    require(read(scope)["resource_ceiling"]["A100_for_this_run"] == 0, "scope hardware override differs")
    for field, digest_key in (("manifest", "data_manifest_sha256"), ("token_cache", "token_cache_sha256"),
                              ("token_cache_metadata", "token_cache_metadata_sha256"),
                              ("b8_wire", "b8_wire_sha256")):
        path = resolved(cfg[field], config_path)
        require(path.is_file() and sha(path) == binding[digest_key], f"input {field} hash differs")
        checks[field + "_sha256"] = sha(path)
    train = resolved(cfg["train_file"], config_path)
    manifest = read(resolved(cfg["manifest"], config_path))
    require(train.is_file() and sha(train) == manifest["source_sha256"], "SST-2 source/roles mismatch")
    require(str(job["seed"]) in manifest["partitions"], "seed role partition absent")
    require({"client_train", "client_local_readout"} <= set(manifest["partitions"][str(job["seed"])]),
            "seed algorithm/readout roles absent")
    require({"public_training", "cohort_select", "cohort_readout", "recipe_dev"} <= set(manifest["roles"]),
            "global roles absent")
    require(len(manifest["roles"]["public_training"]) == 512
            and len(manifest["roles"]["recipe_dev"]) == 4096,
            "public/recipe-development role sizes differ")
    require(read(resolved(cfg["b8_wire"], config_path))["total_bytes"] == B8, "encoded B8 differs")
    manifest_path = source_root / "SOURCE_MANIFEST.json"
    require(manifest_path.is_file() and sha(manifest_path) == binding["source_manifest_sha256"]
            and result["source_manifest_sha256"] == sha(manifest_path), "execution source manifest differs")
    for relative, expected in read(manifest_path)["files"].items():
        path = source_root / relative
        require(path.is_file() and sha(path) == expected, f"source closure differs: {relative}")
    require(binding["source_file_hashes"] == read(manifest_path)["files"],
            "compiled source closure differs")
    checks["source_manifest_sha256"] = sha(manifest_path)
    checks["model_revision"] = MODEL_REVISION
    checks["scope_sha256"] = sha(scope)


def check_queue(config_path: Path, out: Path, cfg: dict, result: dict, checks: dict):
    root = config_path.resolve().parents[2]
    job_id = cfg["job"]["job_id"]
    path = root / "queues" / "development01" / "jobs" / f"{job_id}.json"
    require(path.is_file(), "atomic queue record absent")
    item = read(path)
    require(item["state"] == "executed_pending_audit" and item["job_id"] == job_id,
            "queue is not awaiting independent audit")
    require(Path(item["config"]).resolve() == config_path.resolve()
            and item["config_sha256"] == sha(config_path)
            and Path(item["output"]).resolve() == out.resolve()
            and Path(cfg["job"]["output_dir"]).resolve() == out.resolve()
            and Path(item["result_path"]).resolve() == (out / "result.json").resolve()
            and item["result_sha256"] == sha(out / "result.json"),
            "queue current config/output/result binding differs")
    require(item["return_code"] == 0 and item["attempts"], "queue lacks successful current execution attempt")
    prior_emitted_bytes = 0
    prelaunch_rebindings = 0
    for change in item.get("rebindings", []):
        previous = change.get("previous_record")
        require(isinstance(previous, dict) and previous.get("job_id") == job_id
                and previous.get("recipe_id") == item.get("recipe_id")
                and previous.get("seed") == item.get("seed")
                and change.get("reason"), "rebound job identity/reason differs")
        if previous.get("state") == "blocked":
            require(not previous.get("attempts") and previous.get("return_code") is None,
                    "prelaunch rebind contains an unaccounted execution")
            old_config = Path(previous["config"])
            require(old_config.is_file() and sha(old_config) == previous["config_sha256"],
                    "prelaunch rebind lost previous config")
            prelaunch_rebindings += 1
            continue
        require(previous.get("state") == "failed" and previous.get("return_code") != 0
                and previous.get("attempts") and Path(previous["output"]).resolve() != out.resolve(),
                "rebound job did not preserve failed attempt identity")
        old_out = Path(previous["output"])
        require(old_out.is_dir() and (old_out / "cost_timing_failure.json").is_file()
                and (old_out / "execution_attempts.json").is_file(),
                "failed prior attempt lacks preserved cost/execution evidence")
        journal = read(old_out / "communication_journal.json")
        require(journal["format"] == "pcla-v7-communication-journal-1",
                "failed prior attempt lacks communication journal")
        for entry in journal["entries"]:
            prior_emitted_bytes += sum(event["bytes"] for event in entry["sent_events"])
    checks["queue_record_sha256"] = sha(path)
    checks["prior_failed_rebindings"] = len(item.get("rebindings", []))
    checks["prior_failed_rebindings"] -= prelaunch_rebindings
    checks["prelaunch_rebindings"] = prelaunch_rebindings
    checks["prior_failed_emitted_bytes_separate"] = prior_emitted_bytes


def check_rounds(out: Path, cfg: dict, result: dict, checks: dict):
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line]
    require(100 <= len(rows) <= 200 and [r["round"] for r in rows] == list(range(1, len(rows)+1)),
            "round sequence/horizon differs")
    require(len(rows) == result["rounds"] and rows == read(out / "metrics.json"), "result/metrics artifacts differ")
    cap, halfcap = B8, B8 // 2
    previous_deployed = previous_diagnostic = 0
    stop = rows[-1]["budget_stop_round"]
    require(isinstance(stop, int) and 1 <= stop <= len(rows)
            and result["budget_return_round"] == stop, "B8 endpoint absent or differs")
    expected_half = 0
    for index, row in enumerate(rows, 1):
        clients = row["selected_clients"]
        require(len(clients) == len(set(clients)) == 10 and all(type(c) is int and 0 <= c < 50 for c in clients),
                f"round {index}: client cohort differs")
        training = row["client_training"]
        require(len(training) == 10 and {d["client"] for d in training} == set(clients)
                and all(d["optimizer_steps"] == 16 for d in training),
                f"round {index}: local training count differs")
        for detail in training:
            groups = detail.get("optimizer_groups")
            expected_lrs = ({cfg["job"]["lr"], cfg["job"]["head_lr"]}
                            if cfg["job"]["lr"] != cfg["job"]["head_lr"]
                            else {cfg["job"]["lr"]})
            require(isinstance(groups, list) and len(groups) == len(expected_lrs)
                    and {g["lr"] for g in groups} == expected_lrs
                    and all(g["parameter_count"] > 0 for g in groups)
                    and sum(g["parameter_count"] for g in groups) == detail["optimizer_tensor_parameters"],
                    f"round {index}: optimizer LR groups differ from applied recipe")
        deployed, diagnostic = row["deployed_bytes"], row["diagnostic_after_budget_bytes"]
        require(type(deployed) is int and type(diagnostic) is int and deployed >= previous_deployed
                and diagnostic >= previous_diagnostic, f"round {index}: byte ledger decreases")
        if index <= stop:
            require(diagnostic == 0 and deployed > previous_deployed and deployed <= cap,
                    f"round {index}: deployed byte cap/bucket differs")
            if deployed <= halfcap:
                expected_half = index
        else:
            require(deployed == previous_deployed and diagnostic > previous_diagnostic,
                    f"round {index}: post-budget byte bucket differs")
        previous_deployed, previous_diagnostic = deployed, diagnostic
    require(stop == len(rows) or rows[stop]["deployed_bytes"] == rows[stop-1]["deployed_bytes"],
            "endpoint is not the last deployed model")
    require(rows[-1]["deployed_bytes"] <= cap, "deployed bytes exceed B8")
    require(read(out / "status.json") == {"state": "completed", "completed_round": len(rows),
            "budget_stop_round": stop, "deployed_bytes": previous_deployed,
            "diagnostic_after_budget_bytes": previous_diagnostic}, "terminal status differs")
    checks.update(rounds=len(rows), budget_return_round=stop, half_round=expected_half,
                  deployed_bytes=previous_deployed, diagnostic_after_budget_bytes=previous_diagnostic)
    return rows, expected_half


def check_readouts(out: Path, rows: list[dict], checks: dict):
    labels = read(out / "evaluation_labels.json")
    require(len(labels) == 4096 and all(label in (0, 1) for label in labels),
            "recipe-development labels/size differ")
    for round_id, expected in [(0, read(out / "evaluation_round_000.json"))] + [
            (row["round"], row["evaluation"]) for row in rows]:
        logits = read(out / f"logits_round_{round_id:03d}.json")
        require(len(logits) == len(labels), f"round {round_id}: logit count differs")
        correct = 0
        total_loss = 0.0
        for pair, label in zip(logits, labels):
            require(len(pair) == 2 and all(type(v) in (float, int) and math.isfinite(v) for v in pair),
                    f"round {round_id}: invalid raw logits")
            correct += int((pair[1] > pair[0]) == bool(label))
            peak = max(pair)
            total_loss += peak + math.log(math.exp(pair[0]-peak) + math.exp(pair[1]-peak)) - pair[label]
        require(expected["samples"] == 4096 and abs(expected["accuracy"] - correct / 4096) <= 1e-12
                and abs(expected["loss"] - total_loss / 4096) <= 1e-4,
                f"round {round_id}: saved readout differs from raw logits")
    checks["raw_readouts_recomputed"] = len(rows) + 1


def check_wire(out: Path, rows: list[dict], checks: dict):
    journal = read(out / "communication_journal.json")
    require(journal["format"] == "pcla-v7-communication-journal-1", "communication journal schema differs")
    entries = journal["entries"]
    committed = [entry for entry in entries if entry["status"] == "committed"]
    require([e["round"] for e in committed] == list(range(1, len(rows)+1)), "journal committed rounds differ")
    require(all(e["status"] in {"committed", "denied_no_send"} for e in entries),
            "unsettled/failed communication journal entry")
    event_ids, totals = set(), {"deployed_prefix": 0, "diagnostic_after_budget": 0}
    for row, entry in zip(rows, committed):
        events = entry["sent_events"]
        require(len(events) == 20, f"round {row['round']}: expected ten encoded downloads and uploads")
        require(Counter(e["direction"] for e in events) == {"download": 10, "upload": 10},
                f"round {row['round']}: direction count differs")
        expected_bucket = "deployed_prefix" if row["round"] <= rows[-1]["budget_stop_round"] else "diagnostic_after_budget"
        require(entry["bucket"] == expected_bucket, f"round {row['round']}: journal bucket differs")
        for event in events:
            require(event["event_id"] not in event_ids and event["round"] == row["round"]
                    and event["bucket"] == expected_bucket and type(event["bytes"]) is int and event["bytes"] > 0,
                    f"round {row['round']}: invalid wire event")
            event_ids.add(event["event_id"])
            totals[expected_bucket] += event["bytes"]
        require(totals["deployed_prefix"] == row["deployed_bytes"]
                and totals["diagnostic_after_budget"] == row["diagnostic_after_budget_bytes"],
                f"round {row['round']}: journal/ledger bytes differ")
    latest = checkpoint(out / "latest.pt")
    cache = latest["cache"]
    require({e["event_id"] for e in cache["events"]} == event_ids, "versioned cache event set differs")
    require(sum(e["bytes"] for e in cache["events"] if e["bucket"] == "deployed_prefix") == totals["deployed_prefix"]
            and sum(e["bytes"] for e in cache["events"] if e["bucket"] == "diagnostic_after_budget") == totals["diagnostic_after_budget"],
            "versioned cache bytes differ")
    require(all(0 <= client["version"] < len(rows) for client in cache["clients"].values()),
            "cache claims future parent version")
    checks["encoded_wire_events"] = len(event_ids)


def check_checkpoints(out: Path, cfg: dict, rows: list[dict], half: int, checks: dict):
    rank, stop = cfg["job"]["rank"], rows[-1]["budget_stop_round"]
    latest = checkpoint(out / "latest.pt")
    budget = checkpoint(out / "budget_state.pt")
    half_state = checkpoint(out / "half_budget_state.pt")
    returned = checkpoint(out / "budget_return.pt")
    half_returned = checkpoint(out / "half_budget_return.pt")
    require(latest["round"] == len(rows) and latest["metrics"] == rows, "latest checkpoint/metrics differ")
    require(latest["deployed_bytes"] == rows[-1]["deployed_bytes"]
            and latest["diagnostic_bytes"] == rows[-1]["diagnostic_after_budget_bytes"],
            "latest checkpoint byte ledger differs")
    require(budget["round"] == returned["round"] == stop and half_state["round"] == half_returned["round"] == half,
            "B8/half anchor round differs")
    require(budget["budget_stop_round"] == latest["budget_stop_round"] == stop,
            "B8 checkpoint lacks exact endpoint decision")
    require(budget["deployed_bytes"] == rows[stop-1]["deployed_bytes"]
            and budget["diagnostic_bytes"] == rows[stop-1]["diagnostic_after_budget_bytes"]
            and budget["metrics"] == rows[:stop],
            "B8 checkpoint byte ledger/metrics differ")
    require(half_state["half_round"] == latest["half_round"] == half
            and half_state["deployed_bytes"] == (rows[half-1]["deployed_bytes"] if half else 0)
            and half_state["metrics"] == rows[:half]
            and half_state["diagnostic_bytes"] == 0,
            "half-B8 checkpoint ledger/metrics differ")
    for label, obj in (("latest", latest), ("budget", budget), ("half", half_state),
                       ("budget return", returned), ("half return", half_returned)):
        check_state(obj["adapter"], rank, label)
        require(obj["identity"] == latest["identity"], f"{label}: replay identity differs")
    require(tensors_equal(budget["adapter"], returned["adapter"])
            and tensors_equal(latest["budget_adapter"], returned["adapter"]),
            "B8 returned model differs from frozen checkpoint")
    require(tensors_equal(half_state["adapter"], half_returned["adapter"])
            and tensors_equal(latest["half_adapter"], half_returned["adapter"]),
            "B8/2 returned model differs from frozen checkpoint")
    require(half_returned["readout"] == half_state["half_readout"] == latest["half_readout"],
            "B8/2 returned readout differs from frozen checkpoint")
    require(tensors_equal(budget["budget_adapter"], returned["adapter"]),
            "B8 checkpoint return pointer differs")
    for anchor, previous in (("budget", budget), ("half", half_state)):
        require(torch.equal(previous["torch_rng"], latest["torch_rng"]) if previous["round"] == latest["round"] else True,
                f"{anchor}: terminal torch RNG differs")
        require(previous["python_rng"] == latest["python_rng"] if previous["round"] == latest["round"] else True,
                f"{anchor}: terminal Python RNG differs")
        require(previous["cache"] == latest["cache"] if previous["round"] == latest["round"] else True,
                f"{anchor}: terminal version cache differs")
    require(returned["deployed_bytes"] == rows[-1]["deployed_bytes"]
            and returned["budget_cap_bytes"] == B8,
            "B8 return byte cap/ledger differs")
    require(half_returned["deployed_bytes"] == (rows[half-1]["deployed_bytes"] if half else 0)
            and half_returned["budget_cap_bytes"] == B8 // 2,
            "B8/2 return byte cap/ledger differs")
    checks["budget_state_sha256"] = sha(out / "budget_state.pt")
    checks["half_budget_state_sha256"] = sha(out / "half_budget_state.pt")
    return latest


def check_controller(out: Path, cfg: dict, rows: list[dict], latest: dict, checks: dict):
    recipe, job = cfg["recipe"], cfg["job"]
    rule = recipe["server_rule"]
    control_rows = [json.loads(line) for line in (out / "controller.jsonl").read_text().splitlines() if line]
    files = sorted(out.glob("controller_round_*.json"))
    if rule in {"mean_r8", "mean_r4"}:
        require(len(control_rows) == len(files) == len(rows), "baseline controller record count differs")
        prior = checkpoint(out / "boundary_round_000.pt")["adapter"]
        previous_hash = baseline_tensor_digest(prior)
        for row, line, path in zip(rows, control_rows, files):
            record = read(path)
            round_id = row["round"]
            require(path.name == f"controller_round_{round_id:03d}.json"
                    and line == {"round": round_id, **record}
                    and row["server_controller"] == record,
                    f"round {round_id}: baseline controller artifacts differ")
            require(record["rule"] == rule and record["alpha"] == record["beta"] == 1.0
                    and record["parent_hash"] == previous_hash
                    and record["proposal_hash"] == record["accepted_hash"]
                    and all(isinstance(record.get(k), str) and HEX64.fullmatch(record[k])
                            for k in ("parent_hash", "proposal_hash", "accepted_hash"))
                    and record["features_count"] == record["encoder_batches"] == 0
                    and record["public_dataset_passes"] == 0 and record["grid_public_NLL"] == [],
                    f"round {round_id}: baseline mean state/role differs")
            previous_hash = record["accepted_hash"]
            anchor = out / f"adapter_round_{round_id:03d}.pt"
            if anchor.is_file():
                saved = torch.load(anchor, map_location="cpu", weights_only=True)
                require(saved["round"] == round_id and baseline_tensor_digest(saved["adapter"]) == previous_hash,
                        f"round {round_id}: baseline accepted state differs from saved adapter")
        require(previous_hash == baseline_tensor_digest(latest["adapter"]),
                "last baseline accepted state differs from checkpoint")
        checks["controller_rounds"] = len(control_rows)
        checks["controller_hash_algorithm"] = "pcla_repair.core.tensor_digest"
        return
    require(len(control_rows) == len(files) == len(rows), "controller round record count differs")
    previous = checkpoint(out / "boundary_round_000.pt")["adapter"]
    previous_hash = state_hash(previous)
    for row, line, path in zip(rows, control_rows, files):
        record = read(path)
        require(path.name == f"controller_round_{row['round']:03d}.json"
                and line == {"round": row["round"], **record}
                and row["server_controller"] == record, f"round {row['round']}: controller artifacts differ")
        require(record["rule"] == rule and record["recipe_id"] == job["recipe_id"]
                and record["round_id"] == row["round"] and record["seed"] == job["seed"]
                and record["configuration_sha256"] == recipe["config_sha256"],
                f"round {row['round']}: controller identity differs")
        for key in ("parent_hash", "proposal_hash", "committed_state_hash"):
            require(isinstance(record.get(key), str) and HEX64.fullmatch(record[key]),
                    f"round {row['round']}: missing {key}")
        require(record["parent_hash"] == previous_hash, f"round {row['round']}: parent hash chain differs")
        previous_hash = record["committed_state_hash"]
        require(set(record["head_key_manifest"]) == {"classifier.weight", "classifier.bias"}
                and len(record["factor_key_manifest"]) == 48,
                f"round {row['round']}: controller acted on wrong coordinates")
        role = "train_public512" if recipe["public_training"] else None
        require(record["public_role"] == role, f"round {row['round']}: wrong public role")
        if rule in {"fixed_scalar", "fixed_head"}:
            require(record["alpha"] == recipe["alpha"] and record["beta"] == recipe["beta"]
                    and record["features_count"] == record["public_dataset_passes"] == record["encoder_batches"] == 0,
                    f"round {row['round']}: fixed coefficients or public counts differ")
        elif rule in PUBLIC_GRID:
            count = PUBLIC_GRID[rule]
            grid = record["grid_public_NLL"]
            require(len(grid) == count and record["head_evals"] == count
                    and record["features_count"] == 512 * record["unique_alpha_count"]
                    and record["encoder_batches"] == 8 * record["unique_alpha_count"]
                    and record["public_dataset_passes"] == record["unique_alpha_count"]
                    and HEX64.fullmatch(record["public_ordered_content_sha256"]),
                    f"round {row['round']}: public candidate accounting differs")
            require(any(g["alpha"] == record["alpha"] and g["beta"] == record["beta"]
                        and g["loss"] == record["chosen_public_NLL"] for g in grid),
                    f"round {row['round']}: chosen grid candidate absent")
            require(all(math.isfinite(g["loss"]) for g in grid)
                    and len({(g["alpha"], g["beta"]) for g in grid}) == count,
                    f"round {row['round']}: duplicate/nonfinite grid candidate")
            if rule != "separable":
                low = min(g["loss"] for g in grid)
                tied = [g for g in grid if g["loss"] <= low + 1e-12]
                chosen = min(tied, key=lambda g: ((1-g["alpha"])**2 + (1-g["beta"])**2,
                                                    -g["alpha"], -g["beta"]))
                require((record["alpha"], record["beta"]) == (chosen["alpha"], chosen["beta"]),
                        f"round {row['round']}: public-grid selection differs")
            require(0 <= record["alpha"] <= 1 and 0 <= record["beta"] <= 1,
                    f"round {row['round']}: coefficients out of range")
    require(previous_hash == state_hash(latest["adapter"]), "last controller state differs from checkpoint")
    checks["controller_rounds"] = len(control_rows)


def check_timing(out: Path, result: dict, attempts: list, checks: dict):
    files = sorted(out.glob("timing_attempt_*.json"))
    require(files and len(files) <= len(attempts), "timing attempt records absent/inconsistent")
    if len(attempts) > len(files):
        require((out / "cost_timing_failure.json").is_file(), "failed/resumed attempt has no partial timing record")
    for path in files:
        timing = read(path)
        require(timing["schema"] == "pcla-exclusive-host-timing-2" and not timing.get("partial"),
                f"{path.name}: timing capture partial or unknown")
        buckets = timing["bucket_label_seconds"]
        require(set(buckets) == {"deployed_prefix", "diagnostic_after_budget"},
                f"{path.name}: timing buckets differ")
        for bucket, labels in buckets.items():
            require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in labels.values()),
                    f"{path.name}: negative/nonfinite time")
        event_totals = {(bucket, label): 0.0 for bucket, labels in buckets.items() for label in labels}
        for event in timing["events"]:
            key = (event["bucket"], event["label"])
            require(key in event_totals and type(event["seconds"]) in (float, int)
                    and math.isfinite(event["seconds"]) and event["seconds"] >= 0,
                    f"{path.name}: invalid timing event")
            event_totals[key] += event["seconds"]
        require(all(math.isclose(value, buckets[bucket][label], rel_tol=1e-9, abs_tol=1e-7)
                    for (bucket, label), value in event_totals.items()),
                f"{path.name}: timing event/label totals differ")
        deployed = buckets["deployed_prefix"]
        groups = {group: sum(deployed.get(label, 0) for label, assigned in LABEL_GROUP.items() if assigned == group)
                  for group in REQUIRED_GROUPS}
        require(set(result["required_cost_breakdown"]) == set(groups)
                and all(math.isclose(result["required_cost_breakdown"][k], v, rel_tol=1e-9, abs_tol=1e-7)
                        for k, v in groups.items()) if path == files[-1] else True,
                "result required-time breakdown differs from final attempt")
        require(len(timing["events"]) > 0 and math.isclose(sum(timing["label_seconds"].values()),
                timing["total_seconds"], rel_tol=1e-12, abs_tol=1e-7),
                f"{path.name}: timing events/total differ")
    checks["timing_attempts"] = len(files)
    checks["execution_attempts"] = len(attempts)


def audit(config_path: Path, out: Path, source_root: Path) -> dict:
    checks = {}
    result = read(out / "result.json")
    cfg = read(config_path)
    require(result["state"] == "executed_pending_audit", "process result is not a complete execution")
    require(not (out / "exception.txt").is_file(), "execution left an exception record")
    check_inputs(config_path, source_root, cfg, result, checks)
    check_queue(config_path, out, cfg, result, checks)
    rows, half = check_rounds(out, cfg, result, checks)
    check_readouts(out, rows, checks)
    check_wire(out, rows, checks)
    latest = check_checkpoints(out, cfg, rows, half, checks)
    check_controller(out, cfg, rows, latest, checks)
    attempts = read(out / "execution_attempts.json")
    require(attempts and attempts[0]["resume_round"] == 1
            and all(1 <= attempt["resume_round"] <= len(rows) + 1 for attempt in attempts)
            and [a["resume_round"] for a in attempts] == sorted(a["resume_round"] for a in attempts)
            and all(a["identity"] == latest["identity"] for a in attempts),
            "failure/resume attempt identity or sequence differs")
    check_timing(out, result, attempts, checks)
    return checks


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args(argv)
    issues = []
    checks = {}
    try:
        checks = audit(a.config, a.out, a.source_root)
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, IndexError, json.JSONDecodeError) as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
    job_id = None
    try:
        job_id = read(a.config)["job"]["job_id"]
    except Exception:
        pass
    result_path = a.out / "result.json"
    receipt = {"accepted": not issues, "job_id": job_id,
               "result_sha256": sha(result_path) if result_path.is_file() else None,
               "auditor": {"script": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
               "checks": checks, "issue": issues[0] if issues else None, "issues": issues}
    if a.receipt.exists():
        raise FileExistsError(a.receipt)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.receipt.with_name(a.receipt.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, a.receipt)
    print(json.dumps({"accepted": receipt["accepted"], "job_id": job_id, "issues": issues}), flush=True)
    return 0 if receipt["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
