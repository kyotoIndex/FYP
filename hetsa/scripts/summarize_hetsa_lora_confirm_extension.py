"""Recompute the frozen HetSA-LoRA confirmatory analysis using only stdlib."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIRM_ROOT = ROOT / "results" / "hetsa_lora_confirmatory_v1"
EXTENSION_ROOT = ROOT / "results" / "hetsa_lora_confirm_extension_v1"
OUTPUT_ROOT = EXTENSION_ROOT / "analysis_v1"
SEEDS = (202, 303, 404, 505, 606)
EXTENSION_SEEDS = (404, 505, 606)
METHODS = ("resource_depth", "hetsa_lora", "fedsa_lora", "fedhello", "fedlora")
WINDOW = range(26, 31)
T_CRIT_DF4_95 = 2.7764451051977987
EXPECTED_HASHES = {
    "configs/hetsa_lora_confirm_extension_v1.json": "AA86A7C3B22C8F530B8542A85EA6A2636AC768A9E22AD49A997453FF6449FE19",
    "scripts/run_full_experiments.py": "07E4DB30019F054923F5E39EDB56C69ACDED202964F1AE6682EF7F40BC618D84",
    "src/fedhello_sst2.py": "D22D5A3E58C6CE38E3484F907038BCC722CD41899B339CDF145919AC287D1CAF",
    "data/AG_NEWS/train.tsv": "02EAEA97DDFAA1BA0FA169C33A18BF9C8B280A0209AB8DAB526DFFB80248076F",
    "data/AG_NEWS/dev.tsv": "419184C5E817E3A450FCCDE842EA872CD745C0FF0CF146F8F7255B18A6671B02",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def seed_root(seed: int) -> Path:
    root = EXTENSION_ROOT if seed in EXTENSION_SEEDS else CONFIRM_ROOT
    return root / f"seed_{seed}"


def validate_hashes() -> list[dict[str, str]]:
    rows = []
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        rows.append({"file": relative, "expected_sha256": expected, "actual_sha256": actual, "match": str(actual == expected).lower()})
        if actual != expected:
            raise RuntimeError(f"Frozen input/source hash mismatch: {relative}: {actual}")
    return rows


def validate_extension_config() -> None:
    config_path = ROOT / "configs" / "hetsa_lora_confirm_extension_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["federated"]["seeds"] != list(EXTENSION_SEEDS):
        raise RuntimeError("Extension config seed list differs from the frozen protocol")
    if config["baselines"] != list(METHODS):
        raise RuntimeError("Extension config method list differs from the frozen protocol")
    if config["federated"]["rounds"] != 30:
        raise RuntimeError("Extension config round count differs from the frozen protocol")


def validate_seed(seed: int) -> dict:
    root = seed_root(seed)
    metrics_path = root / "fedhello_metrics.csv"
    allocation_path = root / "allocation_map.csv"
    if not metrics_path.is_file() or not allocation_path.is_file():
        raise RuntimeError(f"Seed {seed} is missing metrics or allocation output")
    metrics = read_csv(metrics_path)
    method_rows: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in metrics:
        method = row["method"]
        round_id = int(row["round"])
        if round_id in method_rows[method]:
            raise RuntimeError(f"Duplicate metrics row for seed {seed}, {method}, round {round_id}")
        method_rows[method][round_id] = row
    if set(method_rows) != set(METHODS):
        raise RuntimeError(f"Seed {seed} method set mismatch: {sorted(method_rows)}")
    for method in METHODS:
        if set(method_rows[method]) != set(range(1, 31)):
            raise RuntimeError(f"Seed {seed}, {method} does not have exactly rounds 1-30")

    allocations = read_csv(allocation_path)
    allocation_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in allocations:
        allocation_sets[(row["method"], int(row["round"]))].add(row["client_id"])
    if len(allocations) != 1500:
        raise RuntimeError(f"Seed {seed} allocation row count is {len(allocations)}, expected 1500")
    bad_schedule = []
    for round_id in range(1, 31):
        selections = {tuple(sorted(allocation_sets[(method, round_id)])) for method in METHODS}
        if len(selections) != 1 or len(next(iter(selections))) != 10:
            bad_schedule.append(round_id)
    if bad_schedule:
        raise RuntimeError(f"Seed {seed} participant schedule mismatch at rounds {bad_schedule}")

    holdout_path = root / "client_holdout.csv"
    holdout_rows = read_csv(holdout_path) if holdout_path.is_file() else []
    holdout_ok = None
    max_holdout_delta = None
    if seed in EXTENSION_SEEDS:
        if len(holdout_rows) != 3000:
            raise RuntimeError(f"Seed {seed} holdout row count is {len(holdout_rows)}, expected 3000")
        grouped_holdout: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in holdout_rows:
            grouped_holdout[(row["method"], int(row["round"]))].append(row)
        for method in METHODS:
            for round_id in range(1, 31):
                rows = grouped_holdout[(method, round_id)]
                if len(rows) != 20 or len({row["client_id"] for row in rows}) != 20:
                    raise RuntimeError(f"Seed {seed} holdout coverage mismatch: {method}, round {round_id}")
        deltas = []
        for method in METHODS:
            for round_id in WINDOW:
                values = [float(row["holdout_accuracy"]) for row in grouped_holdout[(method, round_id)]]
                expected = float(method_rows[method][round_id]["local_accuracy"])
                deltas.append(abs(statistics.fmean(values) - expected))
        max_holdout_delta = max(deltas)
        holdout_ok = max_holdout_delta < 1e-10
        if not holdout_ok:
            raise RuntimeError(f"Seed {seed} client_holdout does not reproduce local_accuracy; max delta={max_holdout_delta}")

    return {
        "seed": seed,
        "metric_rows": len(metrics),
        "methods": len(method_rows),
        "rounds_per_method": 30,
        "allocation_rows": len(allocations),
        "selected_per_round": 10,
        "paired_schedule_rounds": 30,
        "holdout_rows": len(holdout_rows) if holdout_path.is_file() else "legacy_not_saved",
        "holdout_reproduces_local_accuracy": "not_saved" if holdout_ok is None else str(holdout_ok).lower(),
        "holdout_max_abs_delta": max_holdout_delta,
    }, method_rows, holdout_rows


def t_summary(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    avg = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = T_CRIT_DF4_95 * sd / math.sqrt(len(values)) if len(values) == 5 else None
    return avg, sd, None if half is None else avg - half, None if half is None else avg + half


def main() -> None:
    validate_extension_config()
    hash_rows = validate_hashes()
    integrity_rows = []
    metric_rows_by_seed = {}
    holdout_rows_by_seed = {}
    for seed in SEEDS:
        integrity, metric_rows, holdout_rows = validate_seed(seed)
        integrity_rows.append(integrity)
        metric_rows_by_seed[seed] = metric_rows
        holdout_rows_by_seed[seed] = holdout_rows

    summaries = []
    summary_index = {}
    window_metrics = (
        "local_accuracy",
        "local_low",
        "local_medium",
        "local_high",
        "macro_f1",
    )
    endpoint_metrics = (
        "accuracy",
        "deployable_server_accuracy",
        "server_accuracy",
        "diagnostic_global_accuracy",
        "evaluation_samples",
        "cumulative_time",
        "cumulative_comm_mb",
    )
    for seed in SEEDS:
        per_method = metric_rows_by_seed[seed]
        for method in METHODS:
            method_rows = per_method[method]
            row = {"seed": seed, "method": method}
            for metric in window_metrics:
                values = [as_float(method_rows[r].get(metric)) for r in WINDOW]
                clean = [value for value in values if value is not None]
                row[f"{metric}_r26_30_mean"] = mean(clean)
            endpoint = method_rows[30]
            for metric in endpoint_metrics:
                row[f"{metric}_r30"] = as_float(endpoint.get(metric))
            accuracy = as_float(endpoint.get("accuracy"))
            if method in ("hetsa_lora", "fedsa_lora"):
                deployable = as_float(endpoint.get("server_accuracy"))
                diagnostic = as_float(endpoint.get("diagnostic_global_accuracy"))
                if diagnostic is None:
                    diagnostic = accuracy
            else:
                deployable = as_float(endpoint.get("deployable_server_accuracy"))
                if deployable is None:
                    deployable = accuracy
                diagnostic = as_float(endpoint.get("diagnostic_global_accuracy"))
            row["effective_deployable_accuracy_r30"] = deployable
            row["effective_diagnostic_global_accuracy_r30"] = diagnostic
            summaries.append(row)
            summary_index[(seed, method)] = row

    comparisons = (
        ("hetsa_lora", "resource_depth"),
        ("hetsa_lora", "fedsa_lora"),
        ("hetsa_lora", "fedhello"),
        ("hetsa_lora", "fedlora"),
    )
    paired_rows = []
    stat_specs = []
    for left, right in comparisons:
        deltas = []
        for seed in SEEDS:
            left_value = summary_index[(seed, left)]["local_accuracy_r26_30_mean"]
            right_value = summary_index[(seed, right)]["local_accuracy_r26_30_mean"]
            delta = left_value - right_value
            deltas.append(delta)
            paired_rows.append({
                "seed": seed,
                "comparison": f"{left}-{right}",
                "left_local_accuracy": left_value,
                "right_local_accuracy": right_value,
                "difference_pp": delta * 100,
                "threshold_pp": 1.0 if right == "resource_depth" else "",
                "passes_1pp_threshold": ("true" if delta >= 0.01 else "false") if right == "resource_depth" else "",
            })
        avg, sd, low, high = t_summary(deltas)
        stat_specs.append({
            "comparison": f"{left}-{right}",
            "metric": "local_accuracy_r26_30_mean",
            "n_seeds": len(deltas),
            "mean_difference_pp": avg * 100,
            "sd_difference_pp": sd * 100,
            "ci95_lower_pp": None if low is None else low * 100,
            "ci95_upper_pp": None if high is None else high * 100,
            "all_seeds_pass_1pp_threshold": str(all(delta >= 0.01 for delta in deltas)).lower() if right == "resource_depth" else "",
        })

    # Legacy rows used accuracy for the diagnostic A+mean(B) score and
    # server_accuracy for the deployable A-only score on personalized methods.
    server_rows = []
    for method in METHODS:
        for metric in ("effective_deployable_accuracy_r30", "effective_diagnostic_global_accuracy_r30"):
            values = [summary_index[(seed, method)][metric] for seed in SEEDS]
            clean = [value for value in values if value is not None]
            server_rows.append({
                "method": method,
                "metric": metric,
                "n_seeds": len(clean),
                "mean": mean(clean),
                "sd": statistics.stdev(clean) if len(clean) > 1 else None,
                "scope": "round30_full_evaluation; legacy fields normalized by method semantics",
            })

    cost_rows = []
    for method in METHODS:
        times = [summary_index[(seed, method)]["cumulative_time_r30"] for seed in SEEDS]
        comms = [summary_index[(seed, method)]["cumulative_comm_mb_r30"] for seed in SEEDS]
        base_times = [summary_index[(seed, "resource_depth")]["cumulative_time_r30"] for seed in SEEDS]
        base_comms = [summary_index[(seed, "resource_depth")]["cumulative_comm_mb_r30"] for seed in SEEDS]
        time_ratios = [value / base for value, base in zip(times, base_times)]
        comm_ratios = [value / base for value, base in zip(comms, base_comms)]
        cost_rows.append({
            "method": method,
            "n_seeds": len(times),
            "mean_runtime_seconds": statistics.fmean(times),
            "sd_runtime_seconds": statistics.stdev(times),
            "mean_cumulative_communication_mb": statistics.fmean(comms),
            "sd_cumulative_communication_mb": statistics.stdev(comms),
            "mean_paired_runtime_ratio_vs_resource_depth": statistics.fmean(time_ratios),
            "runtime_reduction_vs_resource_depth_pct": (1 - statistics.fmean(time_ratios)) * 100,
            "mean_paired_communication_ratio_vs_resource_depth": statistics.fmean(comm_ratios),
            "communication_reduction_vs_resource_depth_pct": (1 - statistics.fmean(comm_ratios)) * 100,
        })

    capability_rows = []
    for seed in EXTENSION_SEEDS:
        metrics = metric_rows_by_seed[seed]
        for method in METHODS:
            for capability in ("low", "medium", "high"):
                values = [as_float(metrics[method][round_id].get(f"local_{capability}")) for round_id in WINDOW]
                capability_rows.append({
                    "seed": seed,
                    "method": method,
                    "capability": capability,
                    "local_accuracy_r26_30_mean": mean([v for v in values if v is not None]),
                    "n_seeds_scope": 3,
                })

    capability_differences = []
    for capability, client_count in (("low", 12), ("medium", 6), ("high", 2)):
        seed_differences = {}
        for seed in EXTENSION_SEEDS:
            hetsa = summary_index[(seed, "hetsa_lora")][f"local_{capability}_r26_30_mean"]
            resource = summary_index[(seed, "resource_depth")][f"local_{capability}_r26_30_mean"]
            seed_differences[seed] = (hetsa - resource) * 100
        capability_differences.append({
            "capability": capability,
            "clients_in_tier": client_count,
            "seed_404_difference_pp": seed_differences[404],
            "seed_505_difference_pp": seed_differences[505],
            "seed_606_difference_pp": seed_differences[606],
            "mean_difference_pp_n3": statistics.fmean(seed_differences.values()),
            "interpretation": "exploratory_only",
        })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "integrity.csv", integrity_rows, list(integrity_rows[0]))
    write_csv(OUTPUT_ROOT / "input_hashes.csv", hash_rows, list(hash_rows[0]))
    summary_fields = list(summaries[0])
    write_csv(OUTPUT_ROOT / "seed_method_summary.csv", summaries, summary_fields)
    write_csv(OUTPUT_ROOT / "paired_differences.csv", paired_rows, list(paired_rows[0]))
    write_csv(OUTPUT_ROOT / "paired_statistics.csv", stat_specs, list(stat_specs[0]))
    write_csv(OUTPUT_ROOT / "round30_server_summary.csv", server_rows, list(server_rows[0]))
    write_csv(OUTPUT_ROOT / "cost_summary.csv", cost_rows, list(cost_rows[0]))
    write_csv(OUTPUT_ROOT / "capability_summary.csv", capability_rows, list(capability_rows[0]))
    write_csv(OUTPUT_ROOT / "capability_paired_summary.csv", capability_differences, list(capability_differences[0]))

    result = {
        "protocol": "hetsa_lora_confirm_extension_v1",
        "seed_ids": list(SEEDS),
        "primary_window": [26, 30],
        "primary_metric": "local_accuracy",
        "stats_unit": "seed",
        "hashes_match": all(row["match"] == "true" for row in hash_rows),
        "all_integrity_checks_pass": True,
        "paired_statistics": stat_specs,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Wrote analysis outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
