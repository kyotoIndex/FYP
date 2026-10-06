"""Create a compact report table from the audited cross-seed summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = resolve(args.results)
    with (root / "aggregate_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["method"]: row for row in rows}
    fedlora = by_method["fedlora"]
    baseline_accuracy = float(fedlora["final_accuracy_mean"])
    baseline_time = float(fedlora["final_time_mean"])
    baseline_comm = float(fedlora["final_comm_mb_mean"])
    output = []
    for row in rows:
        accuracy = float(row["final_accuracy_mean"])
        elapsed = float(row["final_time_mean"])
        communication = float(row["final_comm_mb_mean"])
        output.append(
            {
                "method": row["method"],
                "seeds": row["num_seeds"],
                "final_accuracy_mean": accuracy,
                "final_accuracy_std": float(row["final_accuracy_std"]),
                "accuracy_delta_vs_fedlora": accuracy - baseline_accuracy,
                "final_time_mean_seconds": elapsed,
                "time_saving_vs_fedlora_percent": 100 * (baseline_time - elapsed) / baseline_time,
                "final_comm_mean_mb": communication,
                "comm_saving_vs_fedlora_percent": 100
                * (baseline_comm - communication)
                / baseline_comm,
                "importance_overhead_mean_seconds": float(row["fim_overhead_seconds_mean"]),
            }
        )
    output.sort(key=lambda row: float(row["final_accuracy_mean"]), reverse=True)
    path = root / "formal_comparison_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
