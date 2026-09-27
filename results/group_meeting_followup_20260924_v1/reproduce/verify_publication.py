#!/usr/bin/env python3
"""Check the published tables against the published raw rows; no training."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from verify_tables import main as verify_original_tables


ROOT = Path(__file__).resolve().parents[1]
SEEDS = {7, 13, 21, 42, 87}


def rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0):
        raise ValueError(f"{label}: {actual} != {expected}")


def check_s1() -> dict[str, float]:
    endpoints = rows("data/s1/S1_PER_SEED_ENDPOINTS.csv")
    rounds = rows("data/s1/S1_ROUND_TRAJECTORIES.csv")
    if len(endpoints) != 10 or len(rounds) != 300:
        raise ValueError("S1 expected 10 endpoints and 300 round rows")
    grouped = defaultdict(list)
    for row in rounds:
        grouped[(row["mode"], int(row["seed"]))].append(row)
    summary = {}
    for endpoint in endpoints:
        key = endpoint["mode"], int(endpoint["seed"])
        series = sorted(grouped[key], key=lambda row: int(row["round"]))
        if [int(row["round"]) for row in series] != list(range(1, 31)):
            raise ValueError(f"S1 incomplete: {key}")
        tail = statistics.mean(float(row["accuracy_fraction"]) for row in series[-5:])
        near(tail, float(endpoint["last5_accuracy_fraction"]), f"S1 {key}")
        summary.setdefault(key[0], []).append(tail)
    for mode, values in summary.items():
        if len(values) != 5:
            raise ValueError(f"S1 missing seed: {mode}")
    near(statistics.mean(summary["central_iid_both"]), .8078899082568807, "S1 IID mean")
    near(statistics.mean(summary["fed_ffa_b"]), .7980275229357797, "S1 FFA mean")
    collapsed = sorted(grouped[("central_iid_both", 13)], key=lambda row: int(row["round"]))
    for row in collapsed[20:]:
        counts = (int(row["predicted_class_0"]), int(row["predicted_class_1"]))
        if sorted(counts) != [0, 872]:
            raise ValueError("S1 seed 13 collapse record differs")
    return {mode: statistics.mean(values) for mode, values in summary.items()}


def check_curves() -> dict[str, int]:
    curve = rows("data/curves/round_curves_public.csv")
    if len(curve) != 11600:
        raise ValueError(f"Expected 11,600 curve rows, got {len(curve)}")
    grouped = defaultdict(list)
    for row in curve:
        key = row["batch"], row["method"], row["job"], int(row["seed"]), row["variant"]
        grouped[key].append(int(row["round"]))
        for field in ("accuracy", "loss", "samples", "predicted_class_counts"):
            if not row[field]:
                raise ValueError(f"Missing curve field {field}: {key}")
        counts = json.loads(row["predicted_class_counts"])
        if sum(counts) != int(row["samples"]):
            raise ValueError(f"Predicted class total mismatch: {key}")
    for key, round_numbers in grouped.items():
        if sorted(round_numbers) not in (list(range(1, 31)), list(range(1, 101))):
            raise ValueError(f"Incomplete curve: {key}")
    raw = [key for key in grouped if key[-1] == "raw"]
    if len(raw) != 225 or sum(len(grouped[key]) for key in raw) != 8500:
        raise ValueError("Raw training-trajectory coverage differs")
    return {"curve_rows": len(curve), "raw_training_series": len(raw)}


def check_costs() -> dict[str, int]:
    jobs = rows("data/costs/PER_JOB_COSTS.csv")
    summary = rows("data/costs/COST_SUMMARY_BY_CONDITION.csv")
    if len(jobs) != 115 or sum(int(row["rounds"]) for row in jobs) != 4500:
        raise ValueError("Cost job/round coverage differs")
    grouped = defaultdict(list)
    for row in jobs:
        grouped[(row["cohort"], row["hardware"], row["condition"])].append(row)
    if len(summary) != len(grouped):
        raise ValueError("Cost summary condition coverage differs")
    fields = ("client_train_seconds", "scoring_seconds", "round_wall_seconds",
              "upload_bytes", "download_bytes", "payload_total_bytes_computed",
              "peak_cuda_allocated_bytes")
    for row in summary:
        key = row["cohort"], row["hardware"], row["condition"]
        group = grouped[key]
        if len(group) != 5 or {int(x["seed"]) for x in group} != SEEDS:
            raise ValueError(f"Cost group missing jobs: {key}")
        for field in fields:
            values = [float(x[field]) for x in group if x[field] != "NA"]
            actual = row["mean_" + field]
            if values:
                near(float(actual), statistics.mean(values), f"cost {key}/{field}", 1e-5)
            elif actual != "NA":
                raise ValueError(f"Cost missingness differs: {key}/{field}")
    return {"cost_jobs": len(jobs), "cost_conditions": len(summary)}


def check_aggregation() -> int:
    comparisons = json.loads((ROOT / "data/aggregation_all_five.json").read_text())
    for row in comparisons:
        if row["status"] != "complete" or row["seeds"] != [7, 13, 21, 42, 87]:
            raise ValueError(f"Incomplete aggregation comparison: {row['comparison']}")
        near(statistics.mean(row["per_seed_difference_pp"]),
             float(row["mean_difference_pp"]), row["comparison"])
    return len(comparisons)


def main() -> None:
    verify_original_tables()
    result = {"s1_mean_fraction": check_s1(), "aggregation_comparisons": check_aggregation()}
    result.update(check_curves())
    result.update(check_costs())
    result["passed"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
