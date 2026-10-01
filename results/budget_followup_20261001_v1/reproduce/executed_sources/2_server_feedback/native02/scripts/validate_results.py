"""Audit completeness and allocation invariants of a finished experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedhello_sst2 import (
    largest_remainder_counts,
    prepare_data_splits,
    read_text_rows,
    row_label,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with resolve(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    root = resolve(args.results)
    methods = config["baselines"]
    rounds = config["federated"]["rounds"]
    clients = config["federated"]["selected_clients"]
    layers = config["lora"]["num_encoder_layers"]
    rgd_rounds = config["fedhello"]["rgd_hla_rounds"]
    full_layer_methods = set(config.get("full_layer_methods", ["fedlora"]))
    errors: list[str] = []
    warnings: list[str] = []
    expected_partition_class_counts: list[int] | None = None
    if (
        config.get("data_heterogeneity", {}).get("mode")
        == "constrained_dirichlet_label_skew"
    ):
        dataset = config["dataset"]
        train_rows = read_text_rows(ROOT / dataset["train_file"], dataset)
        dev_rows = read_text_rows(ROOT / dataset["dev_file"], dataset)
        client_pool, _, _ = prepare_data_splits(
            train_rows,
            dev_rows,
            config,
            int(config["federated"]["seeds"][0]),
        )
        num_labels = int(dataset["num_labels"])
        partition_samples = (
            config["federated"]["total_clients"]
            * config["federated"]["samples_per_client"]
        )
        expected_partition_class_counts = largest_remainder_counts(
            [
                sum(row_label(row) == label for row in client_pool)
                / len(client_pool)
                for label in range(num_labels)
            ],
            partition_samples,
        )

    for seed in config["federated"]["seeds"]:
        seed_dir = root / f"seed_{seed}"
        required = (
            "run_config.json",
            "fedhello_metrics.csv",
            "allocation_map.csv",
            "fim_importance.csv",
            "summary_to_target.csv",
            "checkpoint.json",
            "client_partition.csv",
        )
        missing = [name for name in required if not (seed_dir / name).is_file()]
        if missing:
            errors.append(f"seed {seed}: missing {','.join(missing)}")
            continue

        metrics = read_csv(seed_dir / "fedhello_metrics.csv")
        allocations = read_csv(seed_dir / "allocation_map.csv")
        importance = read_csv(seed_dir / "fim_importance.csv")
        partition = read_csv(seed_dir / "client_partition.csv")
        if len(partition) != config["federated"]["total_clients"]:
            errors.append(f"seed {seed}: client partition row mismatch")
        if any(int(row["samples"]) != config["federated"]["samples_per_client"] for row in partition):
            errors.append(f"seed {seed}: client partition sample count mismatch")
        if expected_partition_class_counts is not None:
            actual_class_counts = [0] * len(expected_partition_class_counts)
            for row in partition:
                if row.get("class_counts"):
                    for value in row["class_counts"].split():
                        label, count = value.split(":", 1)
                        actual_class_counts[int(label)] += int(count)
                else:
                    actual_class_counts[1] += int(row["positive_samples"])
                    actual_class_counts[0] += int(row["negative_samples"])
            if actual_class_counts != expected_partition_class_counts:
                errors.append(
                    f"seed {seed}: constrained partition class counts "
                    f"{actual_class_counts} != {expected_partition_class_counts}"
                )
        metric_counts = Counter(row["method"] for row in metrics)
        allocation_counts = Counter(row["method"] for row in allocations)
        importance_counts = Counter(row["method"] for row in importance)

        for method in methods:
            if metric_counts[method] != rounds:
                errors.append(f"seed {seed} {method}: {metric_counts[method]} metric rows")
            if allocation_counts[method] != rounds * clients:
                errors.append(
                    f"seed {seed} {method}: {allocation_counts[method]} allocation rows"
                )

        for row in allocations:
            active = row["active_layers"].split()
            expected = layers if row["method"] in full_layer_methods else int(row["capacity"])
            valid_count = (
                1 <= len(active) <= expected
                if row["method"] == "legend"
                else len(active) == expected
            )
            if not valid_count or len(set(active)) != len(active):
                errors.append(
                    f"seed {seed} {row['method']} round {row['round']} "
                    f"client {row['client_id']}: invalid active layers"
                )
            if row.get("download_mb") and row.get("upload_mb"):
                expected_traffic = float(row["download_mb"]) + float(
                    row["upload_mb"]
                )
                if abs(expected_traffic - float(row["communication_mb"])) > 1e-8:
                    errors.append(
                        f"seed {seed} {row['method']} round {row['round']} "
                        f"client {row['client_id']}: communication total mismatch"
                    )

        if "fedhello" in methods:
            phases = Counter(
                row["phase"] for row in allocations if row["method"] == "fedhello"
            )
            expected_phases = {
                "rgd_warm_start": rgd_rounds * clients,
                "periodic_fim": (rounds - rgd_rounds) * clients,
            }
            if phases != expected_phases:
                errors.append(f"seed {seed}: Fed-HeLLo phases {dict(phases)}")

        expected_fim_hla = rounds * layers
        first_fim_round = rgd_rounds + 1
        refresh_rounds = list(
            range(
                first_fim_round,
                rounds + 1,
                config["fedhello"]["fim_update_interval"],
            )
        )
        expected_fedhello_fim = len(refresh_rounds) * layers
        if "fim_hla" in methods and importance_counts["fim_hla"] != expected_fim_hla:
            errors.append(f"seed {seed}: FIM-HLA importance row mismatch")
        if "fedhello" in methods and importance_counts["fedhello"] != expected_fedhello_fim:
            errors.append(f"seed {seed}: Fed-HeLLo importance row mismatch")
        formal_methods = {
            "legend",
            "importance_only",
            "importance_resource_lora",
            "personalized_cost_aware_lora",
        }.intersection(methods)
        for method in formal_methods:
            if importance_counts[method] != layers:
                errors.append(f"seed {seed}: {method} rank-plan row mismatch")
            method_rows = [row for row in allocations if row["method"] == method]
            if any(not row.get("active_ranks") for row in method_rows):
                errors.append(f"seed {seed}: {method} missing active ranks")
            rank_rows = [row for row in importance if row["method"] == method]
            ranks = [int(row["allocated_rank"]) for row in rank_rows if row.get("allocated_rank")]
            expected_budget = config["lora"].get(
                "total_rank_budget", layers * config["lora"]["rank"]
            )
            invalid_rank_budget = (
                sum(ranks) > expected_budget
                if method == "legend"
                else sum(ranks) != expected_budget
            )
            if len(ranks) == layers and invalid_rank_budget:
                errors.append(
                    f"seed {seed}: {method} rank budget {sum(ranks)} != {expected_budget}"
                )
            rank_by_layer = {
                int(row["layer"]): int(row["allocated_rank"])
                for row in rank_rows
                if row.get("allocated_rank")
            }
            for row in method_rows:
                active_layers = [int(value) for value in row["active_layers"].split()]
                active_ranks = [int(value) for value in row["active_ranks"].split()]
                expected_ranks = [rank_by_layer[layer] for layer in active_layers]
                if active_ranks != expected_ranks:
                    errors.append(
                        f"seed {seed} {method} round {row['round']} client "
                        f"{row['client_id']}: active ranks do not match rank plan"
                    )
                    break

        if "personalized_cost_aware_lora" in methods:
            pcal_round_one = [
                row
                for row in allocations
                if row["method"] == "personalized_cost_aware_lora" and row["round"] == "1"
            ]
            clusters = {row["client_cluster"] for row in pcal_round_one}
            if len(clusters) < 2:
                warnings.append(
                    f"seed {seed}: PCAL client clustering collapsed to one cluster ({next(iter(clusters), '')})"
                )
        online_methods = [
            method
            for method in methods
            if method == "pcla_online_layer"
            or method.startswith("pcla_online_layer_")
        ]
        for online_method in online_methods:
            online_rows = [
                row for row in importance if row["method"] == online_method
            ]
            expected_online_rows = rounds * clients * layers
            if len(online_rows) != expected_online_rows:
                errors.append(
                    f"seed {seed} {online_method}: online importance rows "
                    f"{len(online_rows)} != {expected_online_rows}"
                )
            online_allocations = [
                row for row in allocations if row["method"] == online_method
            ]
            if any(row["phase"] != "online_ema_ucb" for row in online_allocations):
                errors.append(
                    f"seed {seed} {online_method}: invalid online allocation phase"
                )
            required_online_fields = (
                "client_id",
                "ema_score",
                "selection_score",
                "selected_count_before",
                "layer_age",
                "importance_refreshed",
            )
            if any(
                any(row.get(field, "") == "" for field in required_online_fields)
                for row in online_rows
            ):
                errors.append(
                    f"seed {seed} {online_method}: incomplete online diagnostics"
                )
        for pcla_component_method in (
            "pcla_component_alternating",
            "pcla_component_warmstart",
        ):
            if pcla_component_method not in methods:
                continue
            component_rows = [
                row
                for row in importance
                if row["method"] == pcla_component_method
            ]
            warm_start_rounds = (
                config["pcla_component"].get("warm_start_rounds", 0)
                if pcla_component_method == "pcla_component_warmstart"
                else 0
            )
            expected_component_rows = (
                (rounds - warm_start_rounds)
                * clients
                * layers
                * config["lora"]["rank"]
            )
            if len(component_rows) != expected_component_rows:
                errors.append(
                    f"seed {seed} {pcla_component_method}: component importance rows "
                    f"{len(component_rows)} != {expected_component_rows}"
                )
        for component_method in (
            "random_component_alternating",
            "pcla_component_alternating",
            "random_component_warmstart",
            "pcla_component_warmstart",
        ):
            if component_method not in methods:
                continue
            warm_start_rounds = (
                config["pcla_component"].get("warm_start_rounds", 0)
                if component_method in {
                    "random_component_warmstart",
                    "pcla_component_warmstart",
                }
                else 0
            )
            component_allocations = [
                row
                for row in allocations
                if row["method"] == component_method
            ]
            for row in component_allocations:
                ranks = [int(value) for value in row["active_ranks"].split()]
                round_id = int(row["round"])
                is_warm_start = round_id <= warm_start_rounds
                expected_components = int(row["capacity"]) * (
                    config["lora"]["rank"]
                    if is_warm_start
                    else config["pcla_component"]["components_per_layer"]
                )
                if sum(ranks) != expected_components:
                    errors.append(
                        f"seed {seed} {component_method} round {row['round']} "
                        f"client {row['client_id']}: component budget mismatch"
                    )
                    break
                sparse_round = round_id - warm_start_rounds
                expected_factor = (
                    "both"
                    if is_warm_start
                    else ("b" if sparse_round % 2 == 1 else "a")
                )
                if row.get("trained_factor") != expected_factor:
                    errors.append(
                        f"seed {seed} {component_method}: alternating factor mismatch"
                    )
                    break
                if is_warm_start and row.get("phase") != (
                    "component_warm_start_full_rank"
                ):
                    errors.append(
                        f"seed {seed} {component_method}: invalid warm-start phase"
                    )
                    break
        matched_methods = {
            "random_component_alternating",
            "pcla_component_alternating",
        }
        if matched_methods.issubset(methods):
            random_rows = {
                (row["round"], row["client_id"]): row
                for row in allocations
                if row["method"] == "random_component_alternating"
            }
            for row in (
                item
                for item in allocations
                if item["method"] == "pcla_component_alternating"
            ):
                paired = random_rows.get((row["round"], row["client_id"]))
                if paired is None:
                    errors.append(
                        f"seed {seed}: missing matched random component row"
                    )
                    break
                pcla_components = sum(
                    int(value) for value in row["active_ranks"].split()
                )
                random_components = sum(
                    int(value) for value in paired["active_ranks"].split()
                )
                if (
                    row["capacity"] != paired["capacity"]
                    or row["trained_factor"] != paired["trained_factor"]
                    or row["trainable_parameters"]
                    != paired["trainable_parameters"]
                    or pcla_components != random_components
                ):
                    errors.append(
                        f"seed {seed} round {row['round']} client "
                        f"{row['client_id']}: matched component budget mismatch"
                    )
                    break
        print(
            f"seed={seed} metrics={len(metrics)} allocations={len(allocations)} "
            f"importance={len(importance)}"
        )

    aggregate = root / "aggregate_summary.csv"
    if not aggregate.is_file():
        errors.append("missing aggregate_summary.csv")
    elif len(read_csv(aggregate)) != len(methods):
        errors.append("aggregate_summary.csv method count mismatch")

    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(f"- {warning}")
    print("VALID: complete experiment outputs and allocation invariants")


if __name__ == "__main__":
    main()
