"""Validate Fed-HeLLo reproduction configuration and local controlled inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedhello_sst2 import (
    build_client_rows,
    encoder_layers,
    make_clients,
    prepare_data_splits,
    read_text_rows,
    row_label,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_controlled(config: dict) -> list[str]:
    errors: list[str] = []
    dataset = config["dataset"]
    for key in ("train_file", "dev_file"):
        path = ROOT / dataset[key]
        if not path.is_file():
            errors.append(f"missing {key}: {path}")

    train_path = ROOT / dataset["train_file"]
    dev_path = ROOT / dataset["dev_file"]
    train_rows = (
        read_text_rows(train_path, dataset) if train_path.is_file() else []
    )
    dev_rows = read_text_rows(dev_path, dataset) if dev_path.is_file() else []
    if dev_rows:
        eval_samples = int(dataset.get("eval_samples", len(dev_rows)))
        if eval_samples <= 0:
            eval_samples = len(dev_rows)
        proxy_samples = int(config["fedhello"]["proxy_samples"])
        proxy_source = dataset.get("proxy_source", "dev")
        required_dev = (
            eval_samples + proxy_samples
            if proxy_source == "dev"
            else eval_samples
        )
        if required_dev > len(dev_rows):
            errors.append(
                f"held-out requirement ({required_dev}) exceeds dev size "
                f"({len(dev_rows)})"
            )
        if proxy_source == "train" and proxy_samples >= len(train_rows):
            errors.append("proxy_samples must leave a nonempty client training pool")
        round_eval_samples = int(
            dataset.get("round_eval_samples", eval_samples)
        )
        if not 0 < round_eval_samples <= eval_samples:
            errors.append(
                "round_eval_samples must be positive and no larger than "
                "eval_samples"
            )
    if train_rows:
        observed_labels = sorted({row_label(row) for row in train_rows})
        if observed_labels != list(range(int(dataset["num_labels"]))):
            errors.append(
                f"dataset labels {observed_labels} do not match configured "
                f"range 0..{int(dataset['num_labels']) - 1}"
            )

    fed = config["federated"]
    if fed["selected_clients"] > fed["total_clients"]:
        errors.append("selected_clients cannot exceed total_clients")
    if len(config["federated"]["seeds"]) < 3:
        errors.append("controlled protocol requires at least three seeds")
    if fed["rounds"] <= 0 or fed["local_epochs"] <= 0:
        errors.append("rounds and local_epochs must be positive")

    resources = config["resource_heterogeneity"]
    if not (
        len(resources["layer_fractions"])
        == len(resources["client_ratio"])
        == len(resources["compute_factors"])
        == len(resources["communication_factors"])
    ):
        errors.append("resource capability arrays must have the same length")
    if abs(sum(resources["client_ratio"]) - 1.0) > 1e-6:
        errors.append("client_ratio must sum to one")
    if resources.get("communication_time_per_mb", 0) < 0:
        errors.append("communication_time_per_mb cannot be negative")

    lora = config["lora"]
    if lora["num_encoder_layers"] <= 0 or lora["rank"] <= 0:
        errors.append("LoRA layer count and rank must be positive")
    schedule = config["fedhello"]
    if schedule["rgd_hla_rounds"] < 0 or schedule["fim_update_interval"] <= 0:
        errors.append("RGD rounds must be nonnegative and FIM interval must be positive")
    heterogeneity = config.get("data_heterogeneity", {"mode": "linear_label_skew"})
    if heterogeneity.get("mode") not in {
        "iid",
        "linear_label_skew",
        "dirichlet_label_skew",
        "constrained_dirichlet_label_skew",
    }:
        errors.append("unsupported data heterogeneity mode")
    if (
        heterogeneity.get("mode")
        in {"dirichlet_label_skew", "constrained_dirichlet_label_skew"}
        and heterogeneity.get("alpha", 0) <= 0
    ):
        errors.append("Dirichlet alpha must be positive")
    if heterogeneity.get("mode") == "constrained_dirichlet_label_skew":
        if heterogeneity.get("preserve_global_label_distribution") is not True:
            errors.append(
                "constrained Dirichlet must preserve the global label distribution"
            )
        if heterogeneity.get("sample_reuse") is not False:
            errors.append("constrained Dirichlet must disable sample reuse")
    stage = config.get("stage")
    if stage == "algorithm_screening":
        required_methods = {
            "fedra_random_partial",
            "fedhello",
            "personalized_cost_aware_lora",
            "pcla_online_layer",
        }
    elif stage == "component_warmstart_screening":
        required_methods = {
            "fedra_random_partial",
            "fedhello",
            "random_component_warmstart",
            "pcla_component_warmstart",
        }
    elif stage in {"component_screening", "partition_corrected_screening"}:
        required_methods = {
            "fedra_random_partial",
            "fedhello",
            "random_component_alternating",
            "pcla_component_alternating",
        }
    elif stage in {
        "large_scale_corrected_formal",
        "cross_dataset_model_formal",
    }:
        required_methods = {
            "fedlora",
            "fedhello",
            "random_component_alternating",
            "pcla_component_alternating",
        }
    elif stage == "heterogeneity_formal_ablation":
        required_methods = {
            "random_component_alternating",
            "pcla_component_alternating",
        }
    elif stage == "algorithm_ablation":
        required_methods = {"fedra_random_partial"}
    else:
        required_methods = {"fim_hla", "rgd_hla", "fedhello"}
    missing_methods = sorted(required_methods.difference(config["baselines"]))
    if missing_methods:
        errors.append("missing required baselines: " + ",".join(missing_methods))
    online_methods = [
        method
        for method in config["baselines"]
        if method == "pcla_online_layer"
        or method.startswith("pcla_online_layer_")
    ]
    if config.get("stage") == "algorithm_ablation" and not online_methods:
        errors.append("algorithm ablation requires at least one online PCLA method")
    if online_methods:
        online = config.get("pcla_online_layer", {})
        required_online = {
            "importance_refresh_interval",
            "importance_ema",
            "importance_batch_size",
            "importance_batches_per_client",
            "exploration_beta",
            "freshness_gamma",
            "stability_delta",
        }
        missing_online = sorted(required_online.difference(online))
        if missing_online:
            errors.append("missing online PCLA settings: " + ",".join(missing_online))
        elif (
            online["importance_refresh_interval"] <= 0
            or online["importance_batch_size"] <= 0
            or online["importance_batches_per_client"] <= 0
        ):
            errors.append("online PCLA intervals and batch settings must be positive")
        elif not 0 <= online["importance_ema"] < 1:
            errors.append("online PCLA EMA must be in [0, 1)")
        elif any(
            online[key] < 0
            for key in ("exploration_beta", "freshness_gamma", "stability_delta")
        ):
            errors.append("online PCLA score coefficients cannot be negative")
        variants = config.get("pcla_online_layer_variants", {})
        unknown_variants = sorted(set(variants).difference(online_methods))
        if unknown_variants:
            errors.append(
                "online PCLA variants are not configured baselines: "
                + ",".join(unknown_variants)
            )
        for method in online_methods:
            resolved = dict(online)
            resolved.update(variants.get(method, {}))
            if any(
                resolved[key] < 0
                for key in ("exploration_beta", "freshness_gamma", "stability_delta")
            ):
                errors.append(f"{method}: online score coefficients cannot be negative")
    component_methods = {
        "random_component_alternating",
        "pcla_component_alternating",
        "random_component_warmstart",
        "pcla_component_warmstart",
    }
    if component_methods.intersection(config["baselines"]):
        component = config.get("pcla_component", {})
        required_component = {
            "components_per_layer",
            "importance_refresh_interval",
            "importance_ema",
            "importance_batch_size",
            "importance_batches_per_client",
            "exploration_beta",
            "freshness_gamma",
            "stability_delta",
        }
        missing_component = sorted(required_component.difference(component))
        if missing_component:
            errors.append(
                "missing component PCLA settings: "
                + ",".join(missing_component)
            )
        elif not 0 < component["components_per_layer"] <= lora["rank"]:
            errors.append("components_per_layer must be within the LoRA rank")
        elif (
            component["importance_refresh_interval"] <= 0
            or component["importance_batch_size"] <= 0
            or component["importance_batches_per_client"] <= 0
        ):
            errors.append("component PCLA intervals and batches must be positive")
        elif not 0 <= component["importance_ema"] < 1:
            errors.append("component PCLA EMA must be in [0, 1)")
        elif any(
            method in config["baselines"]
            for method in (
                "random_component_warmstart",
                "pcla_component_warmstart",
            )
        ):
            warm_start_rounds = component.get("warm_start_rounds")
            if (
                not isinstance(warm_start_rounds, int)
                or not 0 < warm_start_rounds < fed["rounds"]
            ):
                errors.append(
                    "warm_start_rounds must be an integer within the run"
                )
            if component.get("warm_start_pattern") not in {
                "triangle",
                "inverted_triangle",
                "bottleneck",
                "uniform",
            }:
                errors.append("unsupported component warm-start pattern")

    try:
        AutoTokenizer.from_pretrained(
            dataset["model"],
            local_files_only=dataset.get("local_files_only", False),
        )
        probe = AutoModelForSequenceClassification.from_pretrained(
            dataset["model"],
            num_labels=dataset["num_labels"],
            local_files_only=dataset.get("local_files_only", False),
        )
        actual_layers = len(encoder_layers(probe))
        if actual_layers != lora["num_encoder_layers"]:
            errors.append(
                f"configured encoder layers ({lora['num_encoder_layers']}) "
                f"do not match model ({actual_layers})"
            )
        if train_rows and dev_rows:
            for seed in fed["seeds"]:
                client_pool, _, _ = prepare_data_splits(
                    train_rows, dev_rows, config, int(seed)
                )
                label_ratios = [
                    sum(row_label(row) == label for row in client_pool)
                    / len(client_pool)
                    for label in range(int(dataset["num_labels"]))
                ]
                clients = make_clients(
                    config,
                    actual_layers,
                    label_ratios[1],
                    int(seed),
                    label_ratios,
                )
                partitions = build_client_rows(
                    client_pool, clients, int(seed)
                )
                if any(
                    len(partitions[client.client_id])
                    != fed["samples_per_client"]
                    for client in clients
                ):
                    errors.append(
                        f"seed {seed}: client partition size mismatch"
                    )
                    break
        del probe
    except Exception as error:
        errors.append(f"model cannot be loaded safely: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    errors = []
    if config.get("stage") in (
        "mechanism_reproduction",
        "formal_comparison",
        "algorithm_screening",
        "algorithm_ablation",
        "component_screening",
        "component_warmstart_screening",
        "partition_corrected_screening",
        "large_scale_corrected_formal",
        "cross_dataset_model_formal",
        "heterogeneity_formal_ablation",
    ):
        errors = validate_controlled(config)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VALID")
    print(f"name={config['name']}")
    if config.get("stage") in (
        "mechanism_reproduction",
        "formal_comparison",
        "algorithm_screening",
        "algorithm_ablation",
        "component_screening",
        "component_warmstart_screening",
        "partition_corrected_screening",
            "large_scale_corrected_formal",
            "cross_dataset_model_formal",
            "heterogeneity_formal_ablation",
    ):
        print(f"model={config['dataset']['model']}")
        print(f"seeds={','.join(map(str, config['federated']['seeds']))}")
        print(f"baselines={','.join(config['baselines'])}")
        print(f"output_dir={config['output_dir']}")
    else:
        print("paper protocol metadata only; original-scale inputs are not validated locally")


if __name__ == "__main__":
    main()
