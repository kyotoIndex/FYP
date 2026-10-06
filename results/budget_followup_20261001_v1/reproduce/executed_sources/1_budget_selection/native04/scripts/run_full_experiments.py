"""Run and aggregate the complete multi-seed controlled Fed-HeLLo protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "controlled_sst2.json"
TRAINER = ROOT / "src" / "fedhello_sst2.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(root: Path, seeds: list[int]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for seed in seeds:
        for row in read_csv(root / f"seed_{seed}" / "summary_to_target.csv"):
            grouped.setdefault(row["method"], []).append(row)
    output = []
    fields = (
        "final_accuracy",
        "final_macro_f1",
        "final_time",
        "final_comm_mb",
        "avg_waiting_time",
        "time_to_target",
        "comm_to_target_mb",
        "fim_overhead_seconds",
    )
    for method, rows in sorted(grouped.items()):
        result: dict[str, object] = {
            "method": method,
            "num_seeds": len(rows),
            "reached_target_count": sum(row["reached_target"].lower() == "true" for row in rows),
        }
        for field in fields:
            values = [
                float(row[field])
                for row in rows
                if row.get(field, "") != ""
            ]
            result[f"{field}_mean"], result[f"{field}_std"] = mean_std(values)
        output.append(result)
    path = root / "aggregate_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--methods", default="")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    seeds = config["federated"]["seeds"]
    root_out = Path(args.out_dir) if args.out_dir else ROOT / config["output_dir"]
    if not root_out.is_absolute():
        root_out = ROOT / root_out
    root_out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        command = [
            sys.executable,
            str(TRAINER),
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--out-dir",
            str(root_out / f"seed_{seed}"),
        ]
        if args.methods:
            command.extend(["--methods", args.methods])
        print(f"\n=== complete seed {seed} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    aggregate(root_out, seeds)
    print(f"\nwrote {root_out / 'aggregate_summary.csv'}")


if __name__ == "__main__":
    main()
