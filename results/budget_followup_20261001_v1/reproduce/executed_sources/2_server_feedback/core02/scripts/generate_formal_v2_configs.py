"""Generate the immutable formal-v2 experiment matrix configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "formal_v2"
DEV_CONFIG_DIR = ROOT / "configs" / "development"
SEEDS = [7, 13, 21, 42, 87]


DATASETS = {
    "sst2": {
        "name": "SST-2",
        "format": "sst2_tsv",
        "train_file": "data/SST-2/train.tsv",
        "dev_file": "data/SST-2/dev.tsv",
        "num_labels": 2,
        "max_length": 64,
        "eval_samples": 872,
        "target_accuracy": 0.8,
    },
    "qnli": {
        "name": "QNLI",
        "format": "glue_qnli_tsv",
        "train_file": "data/QNLI/train.tsv",
        "dev_file": "data/QNLI/dev.tsv",
        "text_columns": ["question", "sentence"],
        "label_column": "label",
        "label_mapping": {
            "entailment": 0,
            "not_entailment": 1,
        },
        "num_labels": 2,
        "max_length": 128,
        "eval_samples": 5463,
        "target_accuracy": 0.75,
    },
    "ag_news": {
        "name": "AG News",
        "format": "named_tsv",
        "train_file": "data/AG_NEWS/train.tsv",
        "dev_file": "data/AG_NEWS/dev.tsv",
        "text_columns": ["title", "description"],
        "label_column": "label",
        "num_labels": 4,
        "max_length": 128,
        "eval_samples": 7600,
        "target_accuracy": 0.85,
    },
}


MODELS = {
    "bert_base": "bert-base-uncased",
    "roberta_base": "models/roberta-base",
}


def base_config(dataset_key: str, model_key: str, alpha: float) -> dict:
    dataset = copy.deepcopy(DATASETS[dataset_key])
    target_accuracy = dataset.pop("target_accuracy")
    dataset.update(
        {
            "model": MODELS[model_key],
            "local_files_only": True,
            "proxy_source": "train",
            "proxy_split_seed": 20_260_728,
            "round_eval_samples": min(
                1024, int(dataset["eval_samples"])
            ),
            "eval_monitor_seed": 20_260_728,
            "metric": "accuracy",
            "secondary_metric": "macro_f1",
        }
    )
    slug = f"{dataset_key}_{model_key}_alpha_{str(alpha).replace('.', '_')}"
    return {
        "name": f"pcla_formal_v2_{slug}",
        "stage": "cross_dataset_model_formal",
        "protocol_version": "formal_v2_explicit_sync",
        "dataset": dataset,
        "data_heterogeneity": {
            "mode": "constrained_dirichlet_label_skew",
            "alpha": alpha,
            "samples_per_client_fixed": True,
            "preserve_global_label_distribution": True,
            "sample_reuse": False,
        },
        "federated": {
            "seeds": SEEDS,
            "total_clients": 50,
            "selected_clients": 10,
            "samples_per_client": 256,
            "rounds": 30,
            "local_epochs": 1,
            "batch_size": 16,
            "eval_batch_size": 64,
            "learning_rate": 0.001,
            "client_selection": "uniform_random_without_replacement",
        },
        "lora": {
            "target_modules": ["query", "value"],
            "rank": 16,
            "alpha": 16.0,
            "dropout": 0.1,
            "num_encoder_layers": 12,
            "total_rank_budget": 192,
        },
        "resource_heterogeneity": {
            "capability_names": ["low", "medium", "high"],
            "layer_fractions": [0.5, 0.75, 1.0],
            "client_ratio": [0.6, 0.3, 0.1],
            "compute_factors": [1.8, 1.2, 0.7],
            "communication_factors": [1.3, 1.0, 0.8],
            "communication_time_per_mb": 1.5,
        },
        "communication_protocol": {
            "version": "explicit_component_sync_v2",
            "backbone_cached_on_client": True,
            "layer_method_download": "active_lora_and_classifier",
            "layer_method_upload": "active_lora_and_classifier",
            "component_method_download": (
                "selected_A_and_B_components_and_classifier"
            ),
            "component_method_upload": (
                "trained_factor_components_and_classifier"
            ),
            "importance_refresh_download": (
                "full_current_lora_adapter_and_classifier"
            ),
        },
        "fedhello": {
            "importance_score": "fim_gradient_norm",
            "importance_source": "server_proxy",
            "proxy_samples": 512,
            "proxy_batch_size": 16,
            "proxy_batches": 8,
            "allocation": "probability_clustered_sampling_without_replacement",
            "random_seeded_allocation": True,
            "resource_levels_sorted_ascending": True,
            "rgd_pattern": "bottleneck",
            "rgd_hla_rounds": 5,
            "fim_update_interval": 5,
            "combined_schedule": "rgd_then_periodic_fim",
        },
        "pcla_component": {
            "components_per_layer": 8,
            "importance_refresh_interval": 3,
            "importance_ema": 0.8,
            "importance_batch_size": 16,
            "importance_batches_per_client": 1,
            "exploration_beta": 0.5,
            "freshness_gamma": 0.05,
            "stability_delta": 0.0,
            "importance_overhead_in_time": True,
            "frozen_factor_cached_on_client": False,
        },
        "baselines": [
            "fedlora",
            "fedhello",
            "random_component_alternating",
            "pcla_component_alternating",
        ],
        "full_layer_methods": ["fedlora"],
        "metrics": [
            "accuracy",
            "macro_f1",
            "round_time",
            "cumulative_time",
            "round_comm_mb",
            "cumulative_comm_mb",
            "avg_waiting_time",
            "trainable_parameters_per_client",
            "download_parameters",
            "upload_parameters",
            "importance_sync_parameters",
            "importance_overhead_seconds",
            "selection_coverage",
            "selection_stability",
        ],
        "target_accuracy": target_accuracy,
        "output_dir": f"results/formal_v2/{slug}",
    }


def write_config(config: dict, filename: str) -> None:
    path = CONFIG_DIR / filename
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_key in DATASETS:
        for model_key in MODELS:
            config = base_config(dataset_key, model_key, 0.3)
            write_config(
                config,
                f"{dataset_key}_{model_key}_alpha_0_3.json",
            )
    for alpha in (0.1, 1.0):
        config = base_config("ag_news", "bert_base", alpha)
        config["stage"] = "heterogeneity_formal_ablation"
        config["baselines"] = [
            "random_component_alternating",
            "pcla_component_alternating",
        ]
        write_config(
            config,
            f"ag_news_bert_base_alpha_{str(alpha).replace('.', '_')}.json",
        )
    DEV_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_key, model_key in (
        ("sst2", "bert_base"),
        ("qnli", "bert_base"),
        ("ag_news", "roberta_base"),
    ):
        config = base_config(dataset_key, model_key, 0.3)
        config["stage"] = "development_check"
        config["federated"].update(
            {
                "seeds": [7],
                "total_clients": 10,
                "selected_clients": 5,
                "samples_per_client": 64,
                "rounds": 2,
                "batch_size": 16,
                "eval_batch_size": 64,
            }
        )
        config["dataset"]["round_eval_samples"] = min(
            int(config["dataset"]["eval_samples"]), 256
        )
        config["fedhello"].update(
            {
                "proxy_samples": 128,
                "proxy_batches": 1,
                "rgd_hla_rounds": 1,
                "fim_update_interval": 1,
            }
        )
        config["output_dir"] = (
            "results/_development/formal_v2_"
            f"{dataset_key}_{model_key}_v2"
        )
        path = DEV_CONFIG_DIR / (
            f"formal_v2_{dataset_key}_{model_key}.json"
        )
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
