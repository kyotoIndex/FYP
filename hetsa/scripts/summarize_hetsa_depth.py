"""Summarize HetSA serving modes, Pareto points, and capability splits."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hetsa_lora_analysis_v1"

EXISTING_RUNS = [
    (ROOT / "results" / "hetsa_lora_development_v1" / "seed_101", 101),
    (ROOT / "results" / "hetsa_lora_confirmatory_v1" / "seed_202", 202),
    (ROOT / "results" / "hetsa_lora_confirmatory_v1" / "seed_303", 303),
]

DEPTH_RUNS = [
    (ROOT / "results" / "hetsa_lora_depth_v1" / f"seed_{seed}", seed)
    for seed in (101, 202, 303)
]
SST2_RUNS = [
    (ROOT / "results" / "hetsa_lora_sst2_v1" / f"seed_{seed}", seed)
    for seed in (101, 202, 303)
]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def last_round(rows: list[dict], method: str) -> dict | None:
    matched = [row for row in rows if row["method"] == method]
    if not matched:
        return None
    return max(matched, key=lambda row: int(row["round"]))


def fnum(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def existing_pareto_and_serving() -> tuple[list[dict], list[dict]]:
    pareto: list[dict] = []
    serving: list[dict] = []
    for path, seed in EXISTING_RUNS:
        metrics = read_csv(path / "fedhello_metrics.csv")
        methods = sorted({row["method"] for row in metrics})
        for method in methods:
            final = last_round(metrics, method)
            if final is None:
                continue
            local = fnum(final.get("local_accuracy"))
            global_acc = fnum(final.get("accuracy"))
            server = fnum(final.get("server_accuracy"))
            comm = fnum(final.get("cumulative_comm_mb"))
            if method in {"hetsa_lora", "fedsa_lora"}:
                deployable = server
                diagnostic = global_acc
            else:
                deployable = global_acc
                diagnostic = global_acc
            pareto.append(
                {
                    "dataset": "AG_NEWS",
                    "seed": seed,
                    "method": method,
                    "local_accuracy": "" if local is None else f"{local:.6f}",
                    "deployable_server_accuracy": ""
                    if deployable is None
                    else f"{deployable:.6f}",
                    "diagnostic_global_accuracy": ""
                    if diagnostic is None
                    else f"{diagnostic:.6f}",
                    "cumulative_comm_mb": "" if comm is None else f"{comm:.4f}",
                }
            )
            serving.append(
                {
                    "dataset": "AG_NEWS",
                    "seed": seed,
                    "method": method,
                    "personalized_client": "" if local is None else f"{local:.6f}",
                    "deployable_server": ""
                    if deployable is None
                    else f"{deployable:.6f}",
                    "diagnostic_mean_b": ""
                    if method not in {"hetsa_lora", "fedsa_lora"} or diagnostic is None
                    else f"{diagnostic:.6f}",
                    "note": (
                        "client=A_avg+B_k; server=A_avg,B=0; diagnostic=A_avg+mean(B)"
                        if method in {"hetsa_lora", "fedsa_lora"}
                        else "client and server use the same aggregated adapter"
                    ),
                }
            )
    return pareto, serving


def capability_from_holdout(runs: list[tuple[Path, int]], dataset: str) -> list[dict]:
    rows: list[dict] = []
    for path, seed in runs:
        holdout = read_csv(path / "client_holdout.csv")
        if not holdout:
            continue
        grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        for row in holdout:
            key = (row["method"], int(row["round"]), row["capability"])
            grouped[key].append(float(row["holdout_accuracy"]))
        last_by_method: dict[str, int] = {}
        for method, round_id, _capability in grouped:
            last_by_method[method] = max(round_id, last_by_method.get(method, 0))
        for method, last in last_by_method.items():
            for capability in ("low", "medium", "high"):
                values = grouped.get((method, last, capability), [])
                if not values:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "round": last,
                        "capability": capability,
                        "n_clients": len(values),
                        "holdout_accuracy": f"{sum(values) / len(values):.6f}",
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.is_file():
            path.unlink()
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pareto, serving = existing_pareto_and_serving()
    write_csv(OUT / "pareto_points.csv", pareto)
    write_csv(OUT / "serving_modes.csv", serving)
    capability = capability_from_holdout(DEPTH_RUNS, "AG_NEWS")
    capability.extend(capability_from_holdout(SST2_RUNS, "SST-2"))
    write_csv(OUT / "capability_last_round.csv", capability)
    print(f"wrote {OUT / 'pareto_points.csv'} rows={len(pareto)}")
    print(f"wrote {OUT / 'serving_modes.csv'} rows={len(serving)}")
    print(f"wrote {OUT / 'capability_last_round.csv'} rows={len(capability)}")


if __name__ == "__main__":
    main()
