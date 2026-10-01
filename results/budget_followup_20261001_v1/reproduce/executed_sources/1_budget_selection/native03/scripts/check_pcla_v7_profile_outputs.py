#!/usr/bin/env python3
"""Read-only numerical and resource gate for one-vs-two PCLA v7 profiles.

The profiler supplies identical round counts and a wall-clock throughput.
This checker compares every completed trajectory, including checkpoints and
per-round scientific outputs. GPU memory here covers the profiler's own PIDs;
device-wide occupancy still needs the scheduler's allocation record.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


DETERMINISTIC_CHECKPOINT_KEYS = (
    "format", "identity", "round", "adapter", "budget_adapter",
    "half_adapter", "half_round", "half_readout", "ema_effective", "cache",
    "deployed_bytes", "diagnostic_bytes", "budget_stop_round",
    "torch_rng", "python_rng", "cuda_rng",
)
NONDETERMINISTIC_ROW_KEYS = {"cost", "cumulative_cost", "gpu_sharing"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_same(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _scientific_rows(rows: list[dict]) -> list[dict]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in NONDETERMINISTIC_ROW_KEYS and key != "train_seconds"
                    and not key.endswith("_seconds")}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(rows)


def _trajectory(directory: Path, rounds: int) -> dict:
    identity_path, checkpoint_path = directory / "run_identity.json", directory / "latest.pt"
    if not identity_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"missing identity or latest checkpoint: {directory}")
    identity = _read_json(identity_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if checkpoint.get("round") != rounds or len(checkpoint.get("metrics", [])) != rounds:
        raise ValueError(f"incomplete checkpoint trajectory: {directory}")
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file() and not _same(_scientific_rows(_read_json(metrics_path)),
                                            _scientific_rows(checkpoint["metrics"])):
        raise ValueError(f"metrics file differs from checkpoint: {directory}")
    return {"directory": directory, "identity": identity, "checkpoint": checkpoint}


def _identity_signature(identity: dict) -> dict:
    provenance = identity["provenance"]
    hardware = provenance["hardware"]
    return {"job": identity["job"], "recipe": identity.get("recipe"),
            "config_sha256": provenance.get("config_sha256"),
            "source_manifest_sha256": provenance.get("source_manifest_sha256"),
            "sources": provenance.get("sources"),
            "native_sources": provenance.get("native_sources"),
            "manifest_sha256": provenance.get("manifest_sha256"),
            "token_cache_sha256": provenance.get("token_cache_sha256"),
            "pretrained_model_sha256": provenance.get("pretrained_model_sha256"),
            "hardware_name": hardware.get("name"),
            "hardware_total_memory": hardware.get("total_memory")}


def _gpu_uuid(value: Any) -> str | None:
    """Normalize PyTorch's bare UUID and nvidia-smi's GPU-prefixed UUID."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid GPU UUID")
    value = value.strip().lower()
    return value[4:] if value.startswith("gpu-") else value


def _compare_trajectory(reference: dict, candidate: dict) -> dict[str, bool]:
    left, right = reference["checkpoint"], candidate["checkpoint"]
    checks = {key: key in left and key in right and _same(left[key], right[key])
              for key in DETERMINISTIC_CHECKPOINT_KEYS}
    checks["scientific_metrics"] = _same(_scientific_rows(left["metrics"]),
                                          _scientific_rows(right["metrics"]))
    checks["run_identity"] = _same(_identity_signature(reference["identity"]),
                                   _identity_signature(candidate["identity"]))
    # Logits are not in the round checkpoint, so compare every saved readout.
    for name in ("evaluation_labels.json", *
                 (f"logits_round_{round_id:03d}.json" for round_id in range(left["round"] + 1))):
        first = reference["directory"] / name
        second = candidate["directory"] / name
        checks[name] = first.is_file() and second.is_file() and first.read_bytes() == second.read_bytes()
    first_round_files = {path.name for path in reference["directory"].glob("round_metrics_*.json")}
    second_round_files = {path.name for path in candidate["directory"].glob("round_metrics_*.json")}
    checks["round_metrics_file_set"] = first_round_files == second_round_files
    for name in first_round_files & second_round_files:
        checks[name] = _same(_scientific_rows(_read_json(reference["directory"] / name)),
                             _scientific_rows(_read_json(candidate["directory"] / name)))
    return checks


def check(single: Path, dual: Path) -> dict:
    profiles = [_read_json(path / "profile.json") for path in (single, dual)]
    one, two = profiles
    rounds = one.get("rounds_each")
    structure_ok = (one.get("processes") == 1 and two.get("processes") == 2
                    and isinstance(rounds, int) and rounds >= 1
                    and two.get("rounds_each") == rounds)
    if not structure_ok:
        raise ValueError("expected matching one-process and two-process profiles")
    trajectories = [_trajectory(single / "process_0", rounds),
                    _trajectory(dual / "process_0", rounds),
                    _trajectory(dual / "process_1", rounds)]
    statuses_ok = (one.get("exit_codes") == [0] and two.get("exit_codes") == [0, 0]
                   and all(status.get("completed_round") == rounds
                           and status.get("state") in {"paused", "completed"}
                           for profile in profiles for status in profile.get("statuses", []))
                   and len(one.get("statuses", [])) == 1 and len(two.get("statuses", [])) == 2)
    numerical = {f"dual_process_{i}": _compare_trajectory(trajectories[0], trajectories[i + 1])
                 for i in range(2)}
    numerical_ok = all(all(checks.values()) for checks in numerical.values())
    single_wall, dual_wall = float(one["wall_seconds"]), float(two["wall_seconds"])
    if not all(math.isfinite(x) and x > 0 for x in (single_wall, dual_wall)):
        raise ValueError("profile wall time must be positive and finite")
    ratio = 2 * single_wall / dual_wall
    throughput_ok = ratio >= 1.25
    total_bytes = int(trajectories[0]["identity"]["provenance"]["hardware"]["total_memory"])
    total_mib = total_bytes / 2**20
    samples = two.get("samples", [])
    seen_pids = {record["pid"] for sample in samples for record in sample.get("processes", [])}
    seen_gpus = {_gpu_uuid(record["gpu"]) for sample in samples
                 for record in sample.get("processes", [])}
    if len(seen_pids) != 2 or len(seen_gpus) != 1:
        memory_valid = False
    else:
        expected_uuids = {_gpu_uuid(trajectory["identity"]["provenance"]["hardware"].get("uuid"))
                          for trajectory in trajectories[1:]}
        memory_valid = len(expected_uuids) == 1 and (None in expected_uuids or expected_uuids == seen_gpus)
    max_by_pid = {pid: max((record["memory_mib"] for sample in samples
                            for record in sample.get("processes", []) if record["pid"] == pid),
                           default=0) for pid in seen_pids}
    # Sum of per-PID sampled maxima is conservative with respect to observed
    # simultaneous use. PyTorch reserved peaks are reported separately.
    observed_mib = max(float(two.get("peak_total_process_memory_mib", 0)),
                       sum(max_by_pid.values()))
    reserved_mib = sum(max(float(row.get("cost", {}).get("cuda_peak_reserved_bytes", 0))
                           for row in trajectory["checkpoint"]["metrics"]) / 2**20
                       for trajectory in trajectories[1:])
    headroom = 1 - max(observed_mib, reserved_mib) / total_mib
    memory_ok = memory_valid and headroom >= .20
    return {"passed": bool(statuses_ok and numerical_ok and throughput_ok and memory_ok),
            "checks": {"completed": statuses_ok, "numerical_bitwise": numerical_ok,
                       "throughput_at_least_1_25": throughput_ok,
                       "observed_memory_headroom_at_least_20_percent": memory_ok},
            "numerical": numerical,
            "throughput": {"unit": "identical_rounds_per_second", "single_wall_seconds": single_wall,
                           "dual_wall_seconds": dual_wall, "dual_over_single": ratio},
            "memory": {"device_total_mib": total_mib,
                       "sampled_sum_of_pid_maxima_mib": observed_mib,
                       "sum_cuda_peak_reserved_mib": reserved_mib,
                       "headroom_fraction": headroom,
                       "scope": "profiled PIDs only; external device occupancy requires allocation audit"},
            "profile_dirs": {"single": str(single), "dual": str(dual)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.single, args.dual)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        result = {"passed": False, "error": str(exc),
                  "profile_dirs": {"single": str(args.single), "dual": str(args.dual)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"],
                      "checks": result.get("checks"), "error": result.get("error"),
                      "throughput_ratio": result.get("throughput", {}).get("dual_over_single")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
