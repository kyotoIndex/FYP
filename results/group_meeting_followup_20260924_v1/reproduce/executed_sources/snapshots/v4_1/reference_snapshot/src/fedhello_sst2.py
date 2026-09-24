"""Controlled Fed-HeLLo reproduction on SST-2 with BERT LoRA.

The HLA implementations follow arXiv:2506.12213v1. This is a mechanism
reproduction: the dataset/model scale differs from the paper's LEDGAR/BERT-base
protocol, while the allocation algorithms and scheduling rules are preserved.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "controlled_sst2.json"


@dataclass(frozen=True)
class ClientSpec:
    client_id: int
    capability: str
    capacity: int
    compute_factor: float
    communication_factor: float
    positive_ratio: float
    samples: int
    label_counts: tuple[int, ...] | None = None


@dataclass
class MethodPlan:
    ranks: list[int]
    global_importance: list[float] | None = None
    client_importance: dict[int, list[float]] | None = None
    client_clusters: dict[int, int] | None = None
    cluster_importance: dict[int, list[float]] | None = None
    overhead_seconds: float = 0.0


TextRow = tuple[str, int] | tuple[str, str | None, int]


def row_label(row: TextRow) -> int:
    return int(row[-1])


def row_texts(row: TextRow) -> tuple[str, str | None]:
    if len(row) == 2:
        return str(row[0]), None
    return str(row[0]), None if row[1] is None else str(row[1])


class TextClassificationDataset(Dataset):
    def __init__(self, rows: Sequence[TextRow], tokenizer, max_length: int):
        first_texts = []
        second_texts = []
        has_pairs = False
        for row in rows:
            first, second = row_texts(row)
            first_texts.append(first)
            second_texts.append(second)
            has_pairs = has_pairs or second is not None
        self.encodings = tokenizer(
            first_texts,
            second_texts if has_pairs else None,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([row_label(row) for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


SST2Dataset = TextClassificationDataset


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.enabled = True
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.register_buffer("component_mask", torch.ones(rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.enabled:
            hidden = F.linear(self.dropout(inputs), self.lora_a)
            hidden = hidden * self.component_mask
            update = F.linear(hidden, self.lora_b)
            output = output + update * self.scaling
        return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_torch_rng():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def normalize(values: Sequence[float]) -> list[float]:
    nonnegative = [max(float(value), 0.0) for value in values]
    total = sum(nonnegative)
    if total <= 0:
        return [1.0 / len(nonnegative)] * len(nonnegative)
    return [value / total for value in nonnegative]


def standardize(values: Sequence[float]) -> list[float]:
    numeric = [float(value) for value in values]
    mean = sum(numeric) / max(len(numeric), 1)
    variance = sum((value - mean) ** 2 for value in numeric) / max(len(numeric), 1)
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return [0.0] * len(numeric)
    return [(value - mean) / scale for value in numeric]


def ema_update(
    previous: Sequence[float] | None,
    current: Sequence[float],
    decay: float,
) -> list[float]:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    normalized_current = normalize(current)
    if previous is None:
        return normalized_current
    if len(previous) != len(normalized_current):
        raise ValueError("EMA vectors must have equal length")
    return normalize(
        [
            decay * float(old) + (1.0 - decay) * new
            for old, new in zip(previous, normalized_current)
        ]
    )


def online_layer_scores(
    importance: Sequence[float],
    selection_counts: Sequence[int],
    last_selected_rounds: Sequence[int],
    previous_layers: Iterable[int],
    round_id: int,
    exploration_beta: float,
    freshness_gamma: float,
    stability_delta: float,
) -> list[float]:
    if not (
        len(importance) == len(selection_counts) == len(last_selected_rounds)
    ):
        raise ValueError("online allocation state vectors must have equal length")
    previous = set(previous_layers)
    standardized_importance = standardize(importance)
    denominator = max(round_id - 1, 1)
    scores = []
    for layer_id, base_score in enumerate(standardized_importance):
        exploration = exploration_beta * math.sqrt(
            math.log(round_id + 1.0) / (selection_counts[layer_id] + 1.0)
        )
        last_round = last_selected_rounds[layer_id]
        age = round_id if last_round < 0 else max(round_id - last_round, 0)
        freshness = freshness_gamma * min(age / denominator, 1.0)
        stability = stability_delta if layer_id in previous else 0.0
        scores.append(base_score + exploration + freshness + stability)
    return scores


def select_online_layers(
    importance: Sequence[float],
    selection_counts: Sequence[int],
    last_selected_rounds: Sequence[int],
    previous_layers: Iterable[int],
    capacity: int,
    round_id: int,
    exploration_beta: float,
    freshness_gamma: float,
    stability_delta: float,
) -> tuple[list[int], list[float]]:
    if capacity <= 0 or capacity > len(importance):
        raise ValueError("online layer capacity must be within the model depth")
    scores = online_layer_scores(
        importance,
        selection_counts,
        last_selected_rounds,
        previous_layers,
        round_id,
        exploration_beta,
        freshness_gamma,
        stability_delta,
    )
    ranked = sorted(range(len(scores)), key=lambda layer: (-scores[layer], layer))
    return sorted(ranked[:capacity]), scores


def is_online_layer_method(method: str) -> bool:
    return method == "pcla_online_layer" or method.startswith("pcla_online_layer_")


def resolve_online_layer_config(method: str, config: dict) -> dict:
    resolved = dict(config.get("pcla_online_layer", {}))
    variants = config.get("pcla_online_layer_variants", {})
    if method in variants:
        resolved.update(variants[method])
    return resolved


def is_component_method(method: str) -> bool:
    return method in {
        "random_component_alternating",
        "pcla_component_alternating",
        "random_component_warmstart",
        "pcla_component_warmstart",
    }


def is_pcla_component_method(method: str) -> bool:
    return method in {
        "pcla_component_alternating",
        "pcla_component_warmstart",
    }


def component_round_schedule(
    method: str, round_id: int, config: dict
) -> tuple[bool, str, int]:
    """Return (is_warm_start, trained_factor, sparse_round)."""
    if not is_component_method(method):
        raise ValueError(f"{method} is not a component method")
    warm_start_rounds = (
        int(config.get("pcla_component", {}).get("warm_start_rounds", 0))
        if method in {
            "random_component_warmstart",
            "pcla_component_warmstart",
        }
        else 0
    )
    if round_id <= warm_start_rounds:
        return True, "both", 0
    sparse_round = round_id - warm_start_rounds
    factor = "b" if sparse_round % 2 == 1 else "a"
    return False, factor, sparse_round


def should_refresh_client_importance(
    last_refresh_round: int | None,
    current_round: int,
    refresh_interval: int,
) -> bool:
    if current_round <= 0 or refresh_interval <= 0:
        raise ValueError("importance rounds and intervals must be positive")
    return (
        last_refresh_round is None
        or current_round - last_refresh_round >= refresh_interval
    )


def flatten_component_values(values: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for layer in values for value in layer]


def select_component_allocation(
    importance: Sequence[Sequence[float]],
    selection_counts: Sequence[int],
    last_selected_rounds: Sequence[int],
    previous_components: Iterable[int],
    layer_capacity: int,
    components_per_layer: int,
    round_id: int,
    exploration_beta: float,
    freshness_gamma: float,
    stability_delta: float,
) -> tuple[dict[int, list[int]], list[float]]:
    if not importance or not importance[0]:
        raise ValueError("component importance cannot be empty")
    num_layers = len(importance)
    rank = len(importance[0])
    if any(len(layer) != rank for layer in importance):
        raise ValueError("all layers must use the same component rank")
    if not 0 < layer_capacity <= num_layers:
        raise ValueError("component layer capacity is invalid")
    if not 0 < components_per_layer <= rank:
        raise ValueError("components_per_layer is invalid")
    flat_importance = flatten_component_values(importance)
    component_scores = online_layer_scores(
        flat_importance,
        selection_counts,
        last_selected_rounds,
        previous_components,
        round_id,
        exploration_beta,
        freshness_gamma,
        stability_delta,
    )
    layer_values = []
    for layer_id in range(num_layers):
        start = layer_id * rank
        scores = component_scores[start : start + rank]
        best = sorted(scores, reverse=True)[:components_per_layer]
        layer_values.append(sum(best) / len(best))
    selected_layers = sorted(
        sorted(
            range(num_layers),
            key=lambda layer_id: (-layer_values[layer_id], layer_id),
        )[:layer_capacity]
    )
    total_budget = layer_capacity * components_per_layer
    selected_flat: set[int] = set()
    for layer_id in selected_layers:
        start = layer_id * rank
        best_component = max(
            range(rank),
            key=lambda component: (
                component_scores[start + component],
                -component,
            ),
        )
        selected_flat.add(start + best_component)
    remaining = sorted(
        (
            flat_id
            for layer_id in selected_layers
            for flat_id in range(layer_id * rank, (layer_id + 1) * rank)
            if flat_id not in selected_flat
        ),
        key=lambda flat_id: (-component_scores[flat_id], flat_id),
    )
    selected_flat.update(remaining[: total_budget - len(selected_flat)])
    allocation = {
        layer_id: sorted(
            flat_id % rank
            for flat_id in selected_flat
            if flat_id // rank == layer_id
        )
        for layer_id in selected_layers
    }
    return allocation, component_scores


def random_component_allocation(
    num_layers: int,
    rank: int,
    layer_capacity: int,
    components_per_layer: int,
    generator: torch.Generator,
) -> dict[int, list[int]]:
    layers = torch.randperm(num_layers, generator=generator)[:layer_capacity].tolist()
    return {
        layer_id: sorted(
            torch.randperm(rank, generator=generator)[:components_per_layer].tolist()
        )
        for layer_id in sorted(layers)
    }


def largest_remainder_counts(ratios: Sequence[float], total: int) -> list[int]:
    raw = [ratio * total for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    for index in sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True):
        if sum(counts) >= total:
            break
        counts[index] += 1
    return counts


def controlled_capability_counts(ratios: Sequence[float], total: int) -> list[int]:
    counts = largest_remainder_counts(ratios, total)
    if total >= len(ratios):
        for level in range(len(counts) - 1, -1, -1):
            if counts[level] == 0:
                raw = [ratio * total for ratio in ratios]
                eligible = [idx for idx, count in enumerate(counts) if count > 1]
                donor = max(eligible, key=lambda idx: counts[idx] - raw[idx])
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[level] += 1
    return counts


def constrained_dirichlet_positive_counts(
    total_clients: int,
    samples_per_client: int,
    global_positive_ratio: float,
    alpha: float,
    seed: int,
) -> list[int]:
    """Draw heterogeneous client label odds while preserving both margins."""
    if total_clients <= 0 or samples_per_client <= 0:
        raise ValueError("client and sample counts must be positive")
    if not 0.0 <= global_positive_ratio <= 1.0:
        raise ValueError("global positive ratio must be within [0, 1]")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    total_samples = total_clients * samples_per_client
    target_positive = round(total_samples * global_positive_ratio)
    if target_positive == 0:
        return [0] * total_clients
    if target_positive == total_samples:
        return [samples_per_client] * total_clients

    rng = random.Random(seed + 70_001)
    log_odds = []
    for _ in range(total_clients):
        negative_mass = rng.gammavariate(alpha, 1.0)
        positive_mass = rng.gammavariate(alpha, 1.0)
        ratio = positive_mass / max(negative_mass + positive_mass, 1e-12)
        ratio = min(max(ratio, 1e-9), 1.0 - 1e-9)
        log_odds.append(math.log(ratio / (1.0 - ratio)))

    def expected_counts(shift: float) -> list[float]:
        return [
            samples_per_client
            / (1.0 + math.exp(-min(max(value + shift, -60.0), 60.0)))
            for value in log_odds
        ]

    low, high = -60.0, 60.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if sum(expected_counts(middle)) < target_positive:
            low = middle
        else:
            high = middle
    expected = expected_counts((low + high) / 2.0)
    counts = [min(samples_per_client, int(math.floor(value))) for value in expected]
    remainder = target_positive - sum(counts)
    candidates = sorted(
        range(total_clients),
        key=lambda client_id: (
            -(expected[client_id] - counts[client_id]),
            client_id,
        ),
    )
    for client_id in candidates:
        if remainder <= 0:
            break
        if counts[client_id] < samples_per_client:
            counts[client_id] += 1
            remainder -= 1
    if remainder != 0 or sum(counts) != target_positive:
        raise RuntimeError("failed to construct constrained Dirichlet margins")
    return counts


def constrained_dirichlet_label_counts(
    total_clients: int,
    samples_per_client: int,
    global_label_ratios: Sequence[float],
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """Construct heterogeneous client class counts with exact row/column margins."""
    if len(global_label_ratios) < 2:
        raise ValueError("at least two label classes are required")
    if any(ratio < 0 for ratio in global_label_ratios):
        raise ValueError("global label ratios cannot be negative")
    ratio_total = sum(global_label_ratios)
    if ratio_total <= 0:
        raise ValueError("global label ratios must have positive mass")
    normalized = [ratio / ratio_total for ratio in global_label_ratios]
    if len(normalized) == 2:
        positives = constrained_dirichlet_positive_counts(
            total_clients,
            samples_per_client,
            normalized[1],
            alpha,
            seed,
        )
        return [
            [samples_per_client - positive_count, positive_count]
            for positive_count in positives
        ]
    if total_clients <= 0 or samples_per_client <= 0:
        raise ValueError("client and sample counts must be positive")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    total_samples = total_clients * samples_per_client
    target_columns = largest_remainder_counts(normalized, total_samples)
    rng = random.Random(seed + 70_001)
    expected = [
        [max(rng.gammavariate(alpha, 1.0), 1e-12) for _ in normalized]
        for _ in range(total_clients)
    ]
    for _ in range(500):
        for row in expected:
            scale = samples_per_client / max(sum(row), 1e-12)
            for label in range(len(row)):
                row[label] *= scale
        column_sums = [
            sum(expected[client][label] for client in range(total_clients))
            for label in range(len(normalized))
        ]
        for label, target in enumerate(target_columns):
            scale = target / max(column_sums[label], 1e-12)
            for client in range(total_clients):
                expected[client][label] *= scale
    counts = [
        [int(math.floor(value)) for value in row]
        for row in expected
    ]
    row_remaining = [
        samples_per_client - sum(row)
        for row in counts
    ]
    column_remaining = [
        target_columns[label]
        - sum(counts[client][label] for client in range(total_clients))
        for label in range(len(normalized))
    ]
    while sum(row_remaining) > 0:
        candidates = [
            (
                expected[client][label] - counts[client][label],
                -client,
                -label,
                client,
                label,
            )
            for client in range(total_clients)
            for label in range(len(normalized))
            if row_remaining[client] > 0 and column_remaining[label] > 0
        ]
        if not candidates:
            raise RuntimeError("failed to round constrained Dirichlet margins")
        _, _, _, client, label = max(candidates)
        counts[client][label] += 1
        row_remaining[client] -= 1
        column_remaining[label] -= 1
    if any(column_remaining):
        raise RuntimeError("constrained Dirichlet column margins do not match")
    return counts


def client_positive_ratios(
    config: dict,
    total_clients: int,
    global_positive_ratio: float,
    seed: int,
    samples_per_client: int = 1,
) -> list[float]:
    heterogeneity = config.get("data_heterogeneity", {"mode": "linear_label_skew"})
    mode = heterogeneity.get("mode", "linear_label_skew")
    if mode == "iid":
        return [global_positive_ratio] * total_clients
    if mode == "linear_label_skew":
        return [
            0.25 + 0.5 * (client_id / max(total_clients - 1, 1))
            for client_id in range(total_clients)
        ]
    if mode == "dirichlet_label_skew":
        alpha = float(heterogeneity["alpha"])
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        rng = random.Random(seed + 70_001)
        ratios = []
        for _ in range(total_clients):
            negative_mass = rng.gammavariate(alpha, 1.0)
            positive_mass = rng.gammavariate(alpha, 1.0)
            ratios.append(positive_mass / max(negative_mass + positive_mass, 1e-12))
        return ratios
    if mode == "constrained_dirichlet_label_skew":
        counts = constrained_dirichlet_positive_counts(
            total_clients,
            samples_per_client,
            global_positive_ratio,
            float(heterogeneity["alpha"]),
            seed,
        )
        return [count / samples_per_client for count in counts]
    raise ValueError(f"unknown data heterogeneity mode: {mode}")


def client_label_counts(
    config: dict,
    total_clients: int,
    samples_per_client: int,
    global_label_ratios: Sequence[float],
    seed: int,
) -> list[list[int]]:
    mode = config.get("data_heterogeneity", {}).get(
        "mode", "linear_label_skew"
    )
    if len(global_label_ratios) == 2:
        positive_ratios = client_positive_ratios(
            config,
            total_clients,
            float(global_label_ratios[1]),
            seed,
            samples_per_client,
        )
        return [
            [samples_per_client - round(samples_per_client * ratio),
             round(samples_per_client * ratio)]
            for ratio in positive_ratios
        ]
    if mode == "constrained_dirichlet_label_skew":
        return constrained_dirichlet_label_counts(
            total_clients,
            samples_per_client,
            global_label_ratios,
            float(config["data_heterogeneity"]["alpha"]),
            seed,
        )
    if mode == "iid":
        counts = largest_remainder_counts(global_label_ratios, samples_per_client)
        return [list(counts) for _ in range(total_clients)]
    if mode == "dirichlet_label_skew":
        alpha = float(config["data_heterogeneity"]["alpha"])
        rng = random.Random(seed + 70_001)
        output = []
        for _ in range(total_clients):
            masses = [
                max(rng.gammavariate(alpha, 1.0), 1e-12)
                for _ in global_label_ratios
            ]
            total = sum(masses)
            output.append(
                largest_remainder_counts(
                    [mass / total for mass in masses],
                    samples_per_client,
                )
            )
        return output
    raise ValueError(
        f"{mode} is not supported for {len(global_label_ratios)} labels"
    )


def make_clients(
    config: dict,
    num_layers: int,
    global_positive_ratio: float = 0.5,
    seed: int = 0,
    global_label_ratios: Sequence[float] | None = None,
) -> list[ClientSpec]:
    fed = config["federated"]
    resources = config["resource_heterogeneity"]
    counts = controlled_capability_counts(resources["client_ratio"], fed["total_clients"])
    label_ratios = (
        list(global_label_ratios)
        if global_label_ratios is not None
        else [1.0 - global_positive_ratio, global_positive_ratio]
    )
    per_client_label_counts = client_label_counts(
        config,
        fed["total_clients"],
        fed["samples_per_client"],
        label_ratios,
        seed,
    )
    clients: list[ClientSpec] = []
    client_id = 0
    for level, count in enumerate(counts):
        capacity = max(1, min(num_layers, math.ceil(resources["layer_fractions"][level] * num_layers)))
        for _ in range(count):
            clients.append(
                ClientSpec(
                    client_id=client_id,
                    capability=resources["capability_names"][level],
                    capacity=capacity,
                    compute_factor=resources["compute_factors"][level],
                    communication_factor=resources["communication_factors"][level],
                    positive_ratio=(
                        per_client_label_counts[client_id][1]
                        / fed["samples_per_client"]
                    ),
                    samples=fed["samples_per_client"],
                    label_counts=tuple(per_client_label_counts[client_id]),
                )
            )
            client_id += 1
    return clients


def read_sst2(path: Path) -> list[TextRow]:
    rows: list[TextRow] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            line = line.rstrip("\n")
            if line:
                text, label = line.rsplit("\t", 1)
                rows.append((text, None, int(label)))
    return rows


def read_text_rows(path: Path, dataset_config: dict) -> list[TextRow]:
    data_format = dataset_config.get("format", "sst2_tsv")
    if data_format == "sst2_tsv":
        return read_sst2(path)
    if data_format == "glue_qnli_tsv":
        label_mapping = {
            str(key): int(value)
            for key, value in dataset_config["label_mapping"].items()
        }
        rows: list[TextRow] = []
        with path.open("r", encoding="utf-8") as handle:
            next(handle)
            for line_number, line in enumerate(handle, start=2):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    raise ValueError(
                        f"{path}:{line_number} has fewer than four columns"
                    )
                raw_label = parts[-1]
                if raw_label not in label_mapping:
                    raise ValueError(
                        f"{path}:{line_number} has unknown label {raw_label!r}"
                    )
                rows.append(
                    (
                        parts[1],
                        "\t".join(parts[2:-1]),
                        label_mapping[raw_label],
                    )
                )
        return rows
    if data_format != "named_tsv":
        raise ValueError(f"unsupported dataset format: {data_format}")
    text_columns = dataset_config.get("text_columns", [])
    if not 1 <= len(text_columns) <= 2:
        raise ValueError("named_tsv requires one or two text_columns")
    label_column = dataset_config.get("label_column", "label")
    label_mapping = {
        str(key): int(value)
        for key, value in dataset_config.get("label_mapping", {}).items()
    }
    rows: list[TextRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [
            column
            for column in [*text_columns, label_column]
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path} is missing configured columns: {','.join(missing)}"
            )
        for record in reader:
            raw_label = record[label_column]
            label = (
                label_mapping[raw_label]
                if label_mapping
                else int(raw_label)
            )
            second = record[text_columns[1]] if len(text_columns) == 2 else None
            rows.append((record[text_columns[0]], second, label))
    return rows


def build_client_rows(
    train_rows: Sequence[TextRow], clients: list[ClientSpec], seed: int
) -> dict[int, list[TextRow]]:
    rng = random.Random(seed)
    observed_labels = {row_label(row) for row in train_rows}
    configured_label_count = max(
        (
            len(client.label_counts)
            for client in clients
            if client.label_counts is not None
        ),
        default=2,
    )
    labels = list(
        range(max(max(observed_labels, default=-1) + 1, configured_label_count))
    )
    by_label = {
        label: [row for row in train_rows if row_label(row) == label]
        for label in labels
    }
    for rows in by_label.values():
        rng.shuffle(rows)
    label_indices = {label: 0 for label in labels}
    result: dict[int, list[TextRow]] = {}
    requested_by_label = {label: 0 for label in labels}
    for client in clients:
        counts = (
            list(client.label_counts)
            if client.label_counts is not None
            else [
                client.samples - round(client.samples * client.positive_ratio),
                round(client.samples * client.positive_ratio),
            ]
        )
        if len(counts) != len(labels):
            raise ValueError("client label counts do not match dataset labels")
        for label, count in zip(labels, counts):
            requested_by_label[label] += count
    for label in labels:
        if requested_by_label[label] > len(by_label[label]):
            raise ValueError(
                f"client partition requests {requested_by_label[label]} samples "
                f"of label {label}, but only {len(by_label[label])} are available "
                "without reuse"
            )
    for client in clients:
        counts = (
            list(client.label_counts)
            if client.label_counts is not None
            else [
                client.samples - round(client.samples * client.positive_ratio),
                round(client.samples * client.positive_ratio),
            ]
        )
        rows: list[TextRow] = []
        for label, count in zip(labels, counts):
            start = label_indices[label]
            rows.extend(by_label[label][start : start + count])
            label_indices[label] += count
        rng.shuffle(rows)
        result[client.client_id] = rows
    return result


def partition_rows(
    clients: Sequence[ClientSpec], client_rows: dict[int, list[TextRow]]
) -> list[dict]:
    rows = []
    for client in clients:
        assigned = client_rows[client.client_id]
        label_counts = {
            label: sum(row_label(row) == label for row in assigned)
            for label in sorted({row_label(row) for row in assigned})
        }
        positives = label_counts.get(1, 0)
        rows.append(
            {
                "client_id": client.client_id,
                "capability": client.capability,
                "capacity": client.capacity,
                "samples": len(assigned),
                "positive_samples": positives,
                "negative_samples": len(assigned) - positives,
                "positive_ratio": positives / max(len(assigned), 1),
                "class_counts": " ".join(
                    f"{label}:{count}"
                    for label, count in label_counts.items()
                ),
                "compute_factor": client.compute_factor,
                "communication_factor": client.communication_factor,
            }
        )
    return rows


def split_proxy_eval(
    dev_rows: Sequence[TextRow], proxy_samples: int, eval_samples: int, seed: int
) -> tuple[list[TextRow], list[TextRow]]:
    if proxy_samples + eval_samples > len(dev_rows):
        raise ValueError("proxy_samples + eval_samples exceeds the SST-2 dev set")
    rows = list(dev_rows)
    random.Random(seed + 10_000).shuffle(rows)
    proxy = rows[:proxy_samples]
    evaluation = rows[proxy_samples : proxy_samples + eval_samples]
    return proxy, evaluation


def split_fixed_proxy_from_train(
    train_rows: Sequence[TextRow],
    proxy_samples: int,
    split_seed: int,
) -> tuple[list[TextRow], list[TextRow]]:
    if not 0 < proxy_samples < len(train_rows):
        raise ValueError("proxy_samples must leave a nonempty client training pool")
    by_label: dict[int, list[TextRow]] = {}
    for row in train_rows:
        by_label.setdefault(row_label(row), []).append(row)
    rng = random.Random(split_seed)
    for rows in by_label.values():
        rng.shuffle(rows)
    label_counts = largest_remainder_counts(
        [len(by_label[label]) / len(train_rows) for label in sorted(by_label)],
        proxy_samples,
    )
    proxy: list[TextRow] = []
    client_pool: list[TextRow] = []
    for label, count in zip(sorted(by_label), label_counts):
        rows = by_label[label]
        proxy.extend(rows[:count])
        client_pool.extend(rows[count:])
    rng.shuffle(proxy)
    rng.shuffle(client_pool)
    return client_pool, proxy


def prepare_data_splits(
    train_rows: Sequence[TextRow],
    dev_rows: Sequence[TextRow],
    config: dict,
    seed: int,
) -> tuple[list[TextRow], list[TextRow], list[TextRow]]:
    dataset = config["dataset"]
    proxy_samples = int(config["fedhello"]["proxy_samples"])
    eval_samples = int(dataset.get("eval_samples", len(dev_rows)))
    if eval_samples <= 0:
        eval_samples = len(dev_rows)
    if eval_samples > len(dev_rows):
        raise ValueError(
            f"eval_samples ({eval_samples}) exceeds dev size ({len(dev_rows)})"
        )
    proxy_source = dataset.get("proxy_source", "dev")
    if proxy_source == "train":
        split_seed = int(dataset.get("proxy_split_seed", 20_260_728))
        client_pool, proxy = split_fixed_proxy_from_train(
            train_rows, proxy_samples, split_seed
        )
        return client_pool, proxy, list(dev_rows[:eval_samples])
    if proxy_source == "dev":
        proxy, evaluation = split_proxy_eval(
            dev_rows, proxy_samples, eval_samples, seed
        )
        return list(train_rows), proxy, evaluation
    raise ValueError(f"unsupported proxy_source: {proxy_source}")


def stratified_subset(
    rows: Sequence[TextRow],
    samples: int,
    seed: int,
) -> list[TextRow]:
    if samples >= len(rows):
        return list(rows)
    if samples <= 0:
        raise ValueError("stratified subset size must be positive")
    by_label: dict[int, list[TextRow]] = {}
    for row in rows:
        by_label.setdefault(row_label(row), []).append(row)
    rng = random.Random(seed)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
    labels = sorted(by_label)
    counts = largest_remainder_counts(
        [len(by_label[label]) / len(rows) for label in labels],
        samples,
    )
    subset = [
        row
        for label, count in zip(labels, counts)
        for row in by_label[label][:count]
    ]
    rng.shuffle(subset)
    return subset


def encoder_layers(model) -> nn.ModuleList:
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    raise TypeError(
        "this runner currently supports BERT and RoBERTa sequence classifiers"
    )


def inject_lora(model, rank: int, alpha: float, dropout: float) -> None:
    inject_lora_ranks(model, [rank] * len(encoder_layers(model)), alpha, dropout)


def inject_lora_ranks(model, ranks: Sequence[int], alpha: float, dropout: float) -> None:
    if len(ranks) != len(encoder_layers(model)):
        raise ValueError("one LoRA rank is required for every encoder layer")
    for layer, rank in zip(encoder_layers(model), ranks):
        attention = layer.attention.self
        attention.query = LoRALinear(attention.query, rank, alpha, dropout)
        attention.value = LoRALinear(attention.value, rank, alpha, dropout)


def set_active_layers(model, active_layers: Iterable[int]) -> None:
    active = set(active_layers)
    for layer_id, layer in enumerate(encoder_layers(model)):
        enabled = layer_id in active
        layer.attention.self.query.enabled = enabled
        layer.attention.self.value.enabled = enabled


def set_trainable_layers(model, active_layers: Iterable[int]) -> None:
    active = set(active_layers)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer_id in active:
        for name, parameter in encoder_layers(model)[layer_id].named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_component_masks(
    model, components_by_layer: dict[int, Sequence[int]]
) -> None:
    for layer_id, layer in enumerate(encoder_layers(model)):
        selected = set(components_by_layer.get(layer_id, []))
        for module_name in ("query", "value"):
            module = getattr(layer.attention.self, module_name)
            mask = torch.zeros_like(module.component_mask)
            if selected:
                mask[list(sorted(selected))] = 1.0
            module.component_mask.copy_(mask)
            module.enabled = bool(selected)


def set_trainable_components(
    model,
    components_by_layer: dict[int, Sequence[int]],
    factor: str,
) -> None:
    if factor not in {"a", "b"}:
        raise ValueError("LoRA factor must be 'a' or 'b'")
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameter_name = f"lora_{factor}"
    for layer_id, components in components_by_layer.items():
        if not components:
            continue
        layer = encoder_layers(model)[layer_id]
        for module_name in ("query", "value"):
            getattr(
                getattr(layer.attention.self, module_name), parameter_name
            ).requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def component_trainable_parameter_count(
    model,
    components_by_layer: dict[int, Sequence[int]],
    factor: str,
) -> int:
    total = sum(parameter.numel() for parameter in model.classifier.parameters())
    for layer_id, components in components_by_layer.items():
        component_count = len(components)
        for module_name in ("query", "value"):
            module = getattr(
                encoder_layers(model)[layer_id].attention.self, module_name
            )
            width = (
                module.lora_a.shape[1]
                if factor == "a"
                else module.lora_b.shape[0]
            )
            total += component_count * width
    return total


def classifier_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.classifier.parameters())


def component_factor_parameter_count(
    model,
    components_by_layer: dict[int, Sequence[int]],
    factor: str,
) -> int:
    if factor not in {"a", "b"}:
        raise ValueError("LoRA factor must be 'a' or 'b'")
    return (
        component_trainable_parameter_count(model, components_by_layer, factor)
        - classifier_parameter_count(model)
    )


def full_lora_parameter_count(model) -> int:
    total = 0
    for layer in encoder_layers(model):
        for module_name in ("query", "value"):
            module = getattr(layer.attention.self, module_name)
            total += module.lora_a.numel() + module.lora_b.numel()
    return total


def component_communication_parameter_counts(
    model,
    components_by_layer: dict[int, Sequence[int]],
    trained_factor: str,
    full_adapter_sync: bool,
) -> tuple[int, int, int]:
    """Return download, upload, and full-adapter sync parameter counts."""
    classifier = classifier_parameter_count(model)
    trained = component_factor_parameter_count(
        model, components_by_layer, trained_factor
    )
    frozen_factor = "a" if trained_factor == "b" else "b"
    frozen = component_factor_parameter_count(
        model, components_by_layer, frozen_factor
    )
    if full_adapter_sync:
        full_adapter = full_lora_parameter_count(model)
        return classifier + full_adapter, classifier + trained, full_adapter
    return classifier + trained + frozen, classifier + trained, 0


def trainable_parameter_count(model, active_layers: Sequence[int]) -> int:
    total = sum(parameter.numel() for parameter in model.classifier.parameters())
    for layer_id in active_layers:
        for name, parameter in encoder_layers(model)[layer_id].named_parameters():
            if "lora_" in name:
                total += parameter.numel()
    return total


def deterministic_geometric_layers(pattern: str, capacity: int, num_layers: int) -> list[int]:
    if pattern == "triangle":
        return list(range(capacity))
    if pattern == "inverted_triangle":
        return list(range(num_layers - capacity, num_layers))
    if pattern == "bottleneck":
        shallow = math.ceil(capacity / 2)
        deep = math.floor(capacity / 2)
        return list(range(shallow)) + list(range(num_layers - deep, num_layers))
    if pattern == "uniform":
        if capacity == num_layers:
            return list(range(num_layers))
        return [round(index * (num_layers - 1) / max(capacity - 1, 1)) for index in range(capacity)]
    raise ValueError(f"unknown geometric pattern: {pattern}")


def geometric_prior(pattern: str, clients: Sequence[ClientSpec], num_layers: int) -> list[float]:
    counts = [0.0] * num_layers
    for client in clients:
        for layer_id in deterministic_geometric_layers(pattern, client.capacity, num_layers):
            counts[layer_id] += 1.0
    return normalize(counts)


def sample_without_replacement(
    probabilities: Sequence[float], count: int, generator: torch.Generator
) -> list[int]:
    if count >= len(probabilities):
        return list(range(len(probabilities)))
    weights = torch.tensor(probabilities, dtype=torch.float64)
    selected = torch.multinomial(weights, count, replacement=False, generator=generator).tolist()
    return sorted(selected)


def one_dimensional_kmeans(values: Sequence[float], clusters: int, iterations: int = 50) -> list[int]:
    if clusters <= 1:
        return [0] * len(values)
    clusters = min(clusters, len(values))
    ordered = sorted(float(value) for value in values)
    centroids = [ordered[round(index * (len(ordered) - 1) / (clusters - 1))] for index in range(clusters)]
    assignments = [0] * len(values)
    for _ in range(iterations):
        updated = [min(range(clusters), key=lambda idx: abs(float(value) - centroids[idx])) for value in values]
        new_centroids = []
        for cluster_id in range(clusters):
            members = [float(values[idx]) for idx, assigned in enumerate(updated) if assigned == cluster_id]
            new_centroids.append(sum(members) / len(members) if members else centroids[cluster_id])
        if updated == assignments and all(abs(a - b) < 1e-12 for a, b in zip(centroids, new_centroids)):
            break
        assignments, centroids = updated, new_centroids
    centroid_order = sorted(range(clusters), key=lambda idx: centroids[idx], reverse=True)
    rank = {cluster_id: index for index, cluster_id in enumerate(centroid_order)}
    return [rank[cluster_id] for cluster_id in assignments]


def fim_allocation_probabilities(
    importance: Sequence[float], clients: Sequence[ClientSpec]
) -> list[float]:
    capacities = sorted({client.capacity for client in clients})
    ratios = [sum(client.capacity == capacity for client in clients) / len(clients) for capacity in capacities]
    expected_capacity = sum(capacity * ratio for capacity, ratio in zip(capacities, ratios))
    base_probabilities = [
        sum(ratio for candidate, ratio in zip(capacities, ratios) if candidate >= capacity)
        / expected_capacity
        for capacity in capacities
    ]
    groups = one_dimensional_kmeans(importance, len(capacities))
    layer_weights = [base_probabilities[min(group, len(base_probabilities) - 1)] for group in groups]
    return normalize(layer_weights)


def legend_arithmetic_ranks(
    num_layers: int,
    total_rank_budget: int,
    common_difference: int = 1,
) -> list[int]:
    """Paper-based LEGEND rank plan: increasing arithmetic sequence under budget.

    LEGEND constrains the active rank sum to be no greater than the total budget
    and uses ``r_l = r_(l-1) + lambda``.  When the requested budget cannot be
    represented exactly by an integer arithmetic sequence, the largest feasible
    sequence is returned and the small remainder is intentionally unused.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if common_difference < 0:
        raise ValueError("LEGEND common difference must be non-negative")
    staircase = common_difference * num_layers * (num_layers - 1) // 2
    first_rank = (total_rank_budget - staircase) // num_layers
    if first_rank < 1:
        raise ValueError("rank budget is too small for the requested arithmetic sequence")
    ranks = [first_rank + common_difference * layer for layer in range(num_layers)]
    if sum(ranks) > total_rank_budget:
        raise RuntimeError("LEGEND arithmetic rank plan exceeded its budget")
    return ranks


def legend_depths(
    completion_times: dict[int, float],
    num_layers: int,
    ranks: Sequence[int],
    client_rank_budgets: dict[int, int],
) -> dict[int, int]:
    """Determine per-round LoRA depths using a text-consistent LCD interpretation.

    Algorithm 1 in arXiv:2412.20004v1 prints a line-3 expression that conflicts
    with its accompanying explanation.  The paper states that the fastest device
    receives depth ``L``, the slowest ``L-k_gap``, and intermediate devices lie
    inside that interval.  We therefore linearly interpolate that stated policy,
    then greedily reduce depth until the device-specific rank budget is met.
    """
    if len(ranks) != num_layers:
        raise ValueError("LEGEND rank plan must cover every transformer layer")
    if not completion_times:
        return {}
    if set(completion_times) != set(client_rank_budgets):
        raise ValueError("completion times and rank budgets must cover the same clients")
    if any(value <= 0 for value in completion_times.values()):
        raise ValueError("LEGEND completion times must be positive")

    fastest = min(completion_times.values())
    slowest = max(completion_times.values())
    spread = slowest - fastest
    depth_gap = 0 if spread <= 1e-12 else math.floor(num_layers * spread / slowest)
    depths: dict[int, int] = {}
    for client_id, completion_time in completion_times.items():
        if spread <= 1e-12:
            depth = num_layers
        else:
            relative_slowness = (completion_time - fastest) / spread
            depth = num_layers - math.floor(depth_gap * relative_slowness)
        depth = min(max(depth, 1), num_layers)
        rank_budget = client_rank_budgets[client_id]
        while depth > 1 and sum(ranks[num_layers - depth :]) > rank_budget:
            depth -= 1
        if sum(ranks[num_layers - depth :]) > rank_budget:
            raise ValueError(f"client {client_id} cannot afford even one LEGEND layer")
        depths[client_id] = depth
    return depths


def legend_simulated_completion(
    client: ClientSpec,
    active_layers: Sequence[int],
    ranks: Sequence[int],
    uniform_rank: int,
) -> float:
    """Deterministic counterpart of LEGEND Equation 12 for a simulated device.

    The constant is the frozen-backbone forward cost, the depth term models
    per-layer backpropagation, and the rank term models LoRA upload cost.
    Client factors are fixed protocol inputs rather than noisy host timings.
    """
    rank_load = sum(ranks[layer] for layer in active_layers) / max(uniform_rank, 1)
    return (
        1.0
        + len(active_layers) * client.compute_factor
        + rank_load * client.communication_factor
    )


def allocate_ranks_by_importance(
    importance: Sequence[float], total_budget: int, min_rank: int, max_rank: int
) -> list[int]:
    if min_rank * len(importance) > total_budget or max_rank * len(importance) < total_budget:
        raise ValueError("rank bounds cannot realize the requested total budget")
    scores = normalize(importance)
    ranks = [min_rank] * len(scores)
    remaining = total_budget - sum(ranks)
    raw_extra = [score * remaining for score in scores]
    for index, value in enumerate(raw_extra):
        ranks[index] += min(max_rank - min_rank, int(math.floor(value)))
    while sum(ranks) < total_budget:
        candidates = [
            (raw_extra[index] - (ranks[index] - min_rank), scores[index], -index, index)
            for index in range(len(ranks))
            if ranks[index] < max_rank
        ]
        if not candidates:
            raise RuntimeError("rank allocation exhausted before reaching the budget")
        ranks[max(candidates)[-1]] += 1
    while sum(ranks) > total_budget:
        candidates = [
            ((ranks[index] - min_rank) - raw_extra[index], -scores[index], index)
            for index in range(len(ranks))
            if ranks[index] > min_rank
        ]
        if not candidates:
            raise RuntimeError("rank allocation cannot be reduced to the budget")
        ranks[max(candidates)[-1]] -= 1
    return ranks


def assign_personalized_clusters(
    clients: Sequence[ClientSpec], client_importance: dict[int, list[float]], clusters: int
) -> dict[int, int]:
    if clusters <= 1:
        return {client.client_id: 0 for client in clients}
    result: dict[int, int] = {}
    for client in clients:
        importance = client_importance[client.client_id]
        center = sum(layer * score for layer, score in enumerate(importance)) / max(
            len(importance) - 1, 1
        )
        resource_load = (client.compute_factor + client.communication_factor) / 2.0
        raw = 0.55 * center + 0.45 * min(resource_load / 2.5, 1.0)
        result[client.client_id] = min(clusters - 1, max(0, int(raw * clusters)))
    return result


def cluster_mean_importance(
    clients: Sequence[ClientSpec],
    client_importance: dict[int, list[float]],
    client_clusters: dict[int, int],
    global_importance: list[float],
) -> dict[int, list[float]]:
    grouped: dict[int, list[list[float]]] = {}
    for client in clients:
        grouped.setdefault(client_clusters[client.client_id], []).append(
            client_importance[client.client_id]
        )
    result = {
        cluster: normalize(
            [sum(row[layer] for row in rows) / len(rows) for layer in range(len(global_importance))]
        )
        for cluster, rows in grouped.items()
    }
    return result


def personalized_layer_cost(
    client: ClientSpec, layer_id: int, rank: int, num_layers: int
) -> float:
    normalized_rank = max(rank, 1) / max(num_layers, 1)
    backprop_path = (num_layers - layer_id) / max(num_layers, 1)
    return (
        client.compute_factor * normalized_rank * backprop_path
        + client.communication_factor * normalized_rank
        + 1e-8
    )


def formal_active_layers(
    method: str, client: ClientSpec, plan: MethodPlan, num_layers: int
) -> tuple[list[int], str]:
    if method == "importance_only":
        return list(range(num_layers)), "importance_rank_all_layers"
    if method == "importance_resource_lora":
        if plan.global_importance is None:
            raise RuntimeError("global importance is missing")
        ranked = sorted(
            (
                (plan.global_importance[layer] / max(plan.ranks[layer], 1), layer)
                for layer in range(num_layers)
            ),
            reverse=True,
        )
        return sorted(layer for _, layer in ranked[: client.capacity]), "importance_resource"
    if method == "personalized_cost_aware_lora":
        if plan.client_clusters is None or plan.cluster_importance is None:
            raise RuntimeError("personalized allocation state is missing")
        cluster = plan.client_clusters[client.client_id]
        importance = plan.cluster_importance[cluster]
        ranked = sorted(
            (
                (
                    importance[layer]
                    / personalized_layer_cost(client, layer, plan.ranks[layer], num_layers),
                    layer,
                )
                for layer in range(num_layers)
            ),
            reverse=True,
        )
        return sorted(layer for _, layer in ranked[: client.capacity]), "personalized_cost_aware"
    raise ValueError(f"not a formal comparison method: {method}")


def allocate_layers(
    method: str,
    client: ClientSpec,
    num_layers: int,
    generator: torch.Generator,
    fim_probabilities: Sequence[float] | None,
    rgd_probabilities: Sequence[float],
    round_id: int,
    rgd_rounds: int,
) -> tuple[list[int], str, Sequence[float] | None]:
    if method == "fedlora":
        return list(range(num_layers)), "all", None
    if method == "resource_only":
        layers = deterministic_geometric_layers("inverted_triangle", client.capacity, num_layers)
        return layers, "resource_deep", None
    if method == "fedra_random_partial":
        probabilities = [1.0 / num_layers] * num_layers
        return sample_without_replacement(probabilities, client.capacity, generator), "uniform_random", probabilities
    if method == "fim_hla":
        if fim_probabilities is None:
            raise RuntimeError("FIM probabilities were not computed")
        return sample_without_replacement(fim_probabilities, client.capacity, generator), "fim", fim_probabilities
    if method.startswith("gd_hla_"):
        pattern = method.removeprefix("gd_hla_")
        return deterministic_geometric_layers(pattern, client.capacity, num_layers), f"gd_{pattern}", None
    if method == "rgd_hla":
        return sample_without_replacement(rgd_probabilities, client.capacity, generator), "rgd", rgd_probabilities
    if method == "fedhello":
        if round_id <= rgd_rounds:
            return sample_without_replacement(rgd_probabilities, client.capacity, generator), "rgd_warm_start", rgd_probabilities
        if fim_probabilities is None:
            raise RuntimeError("Fed-HeLLo requires cached FIM probabilities after warm start")
        return sample_without_replacement(fim_probabilities, client.capacity, generator), "periodic_fim", fim_probabilities
    raise ValueError(f"unknown method: {method}")


def should_refresh_fim(
    method: str,
    round_id: int,
    rgd_rounds: int,
    update_interval: int,
    has_cached_probabilities: bool,
) -> bool:
    if method == "fim_hla":
        return True
    if method != "fedhello" or round_id <= rgd_rounds:
        return False
    first_fim_round = rgd_rounds + 1
    return (not has_cached_probabilities) or (round_id - first_fim_round) % update_interval == 0


def estimate_fim_importance(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> list[float]:
    """Average squared LoRA-gradient norm over the server proxy batches."""
    model.to(device)
    model.train()
    all_layers = list(range(len(encoder_layers(model))))
    set_active_layers(model, all_layers)
    set_trainable_layers(model, all_layers)
    scores = [0.0] * len(all_layers)
    seen = 0
    for batch_id, batch in enumerate(loader):
        if batch_id >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        model(**batch).loss.backward()
        for layer_id, layer in enumerate(encoder_layers(model)):
            squared_norm = 0.0
            for module_name in ("query", "value"):
                module = getattr(layer.attention.self, module_name)
                for parameter in (module.lora_a, module.lora_b):
                    if parameter.grad is not None:
                        squared_norm += float(torch.sum(parameter.grad.detach() ** 2).cpu())
            scores[layer_id] += squared_norm
        seen += 1
    model.zero_grad(set_to_none=True)
    model.cpu()
    return [score / max(seen, 1) for score in scores]


def estimate_gradient_norm_importance(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> list[float]:
    """PCAL importance: average query/value LoRA gradient L2 norm per layer."""
    model.to(device)
    model.train()
    all_layers = list(range(len(encoder_layers(model))))
    set_active_layers(model, all_layers)
    set_trainable_layers(model, all_layers)
    scores = [0.0] * len(all_layers)
    seen = 0
    for batch_id, batch in enumerate(loader):
        if batch_id >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        model(**batch).loss.backward()
        for layer_id, layer in enumerate(encoder_layers(model)):
            for module_name in ("query", "value"):
                module = getattr(layer.attention.self, module_name)
                for parameter in (module.lora_a, module.lora_b):
                    if parameter.grad is not None:
                        scores[layer_id] += float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
        seen += 1
    model.zero_grad(set_to_none=True)
    model.cpu()
    return [score / max(seen, 1) for score in scores]


def estimate_component_importance(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    factor: str,
) -> list[list[float]]:
    """First-order sensitivity of aligned LoRA rank-1 components."""
    model.to(device)
    model.train()
    layers = encoder_layers(model)
    rank = layers[0].attention.self.query.rank
    all_components = {
        layer_id: list(range(rank)) for layer_id in range(len(layers))
    }
    set_component_masks(model, all_components)
    set_trainable_components(model, all_components, factor)
    scores = [[0.0] * rank for _ in layers]
    seen = 0
    for batch_id, batch in enumerate(loader):
        if batch_id >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        model(**batch).loss.backward()
        for layer_id, layer in enumerate(layers):
            for module_name in ("query", "value"):
                module = getattr(layer.attention.self, module_name)
                if factor == "b":
                    gradients = module.lora_b.grad
                    if gradients is None:
                        continue
                    for component in range(rank):
                        scores[layer_id][component] += float(
                            torch.linalg.vector_norm(gradients[:, component]).cpu()
                            * torch.linalg.vector_norm(
                                module.lora_a[component, :].detach()
                            ).cpu()
                        )
                else:
                    gradients = module.lora_a.grad
                    if gradients is None:
                        continue
                    for component in range(rank):
                        scores[layer_id][component] += float(
                            torch.linalg.vector_norm(
                                module.lora_b[:, component].detach()
                            ).cpu()
                            * torch.linalg.vector_norm(
                                gradients[component, :]
                            ).cpu()
                        )
        seen += 1
    model.zero_grad(set_to_none=True)
    model.cpu()
    return [
        [score / max(seen, 1) for score in layer_scores]
        for layer_scores in scores
    ]


def prepare_formal_method_plan(
    method: str,
    base_model,
    client_loaders: dict[int, DataLoader],
    proxy_loader: DataLoader,
    clients: Sequence[ClientSpec],
    config: dict,
    seed: int,
    device: torch.device,
) -> MethodPlan:
    num_layers = len(encoder_layers(base_model))
    uniform_rank = config["lora"]["rank"]
    uniform_ranks = [uniform_rank] * num_layers
    if method == "legend":
        legend_config = config.get("legend", {})
        return MethodPlan(
            legend_arithmetic_ranks(
                num_layers,
                int(config["lora"].get("total_rank_budget", num_layers * uniform_rank)),
                int(legend_config.get("rank_common_difference", 1)),
            )
        )

    alpha = config["lora"]["alpha"]
    dropout = config["lora"]["dropout"]
    min_rank = config["formal_comparison"]["min_rank"]
    max_rank = config["formal_comparison"]["max_rank"]
    importance_batches = config["formal_comparison"]["importance_batches"]
    inject_lora_ranks(base_model, uniform_ranks, alpha, dropout)
    start = time.perf_counter()

    if method in ("importance_only", "importance_resource_lora"):
        with preserve_torch_rng():
            raw_importance = estimate_gradient_norm_importance(
                base_model, proxy_loader, device, importance_batches
            )
        importance = normalize(raw_importance)
        ranks = allocate_ranks_by_importance(
            importance,
            num_layers * uniform_rank,
            min_rank,
            max_rank,
        )
        return MethodPlan(
            ranks=ranks,
            global_importance=importance,
            overhead_seconds=time.perf_counter() - start,
        )

    if method == "personalized_cost_aware_lora":
        client_importance = {}
        for client in clients:
            importance_loader = DataLoader(
                client_loaders[client.client_id].dataset,
                batch_size=config["formal_comparison"]["importance_batch_size"],
                shuffle=True,
                generator=torch.Generator().manual_seed(seed * 100_003 + client.client_id),
            )
            with preserve_torch_rng():
                scores = estimate_gradient_norm_importance(
                    base_model,
                    importance_loader,
                    device,
                    config["formal_comparison"]["personalized_importance_batches_per_client"],
                )
            client_importance[client.client_id] = normalize(scores)
        global_importance = normalize(
            [
                sum(client_importance[client.client_id][layer] for client in clients)
                / len(clients)
                for layer in range(num_layers)
            ]
        )
        ranks = allocate_ranks_by_importance(
            global_importance,
            num_layers * uniform_rank,
            min_rank,
            max_rank,
        )
        clusters = assign_personalized_clusters(
            clients,
            client_importance,
            config["formal_comparison"]["personalized_clusters"],
        )
        cluster_importance = cluster_mean_importance(
            clients, client_importance, clusters, global_importance
        )
        return MethodPlan(
            ranks=ranks,
            global_importance=global_importance,
            client_importance=client_importance,
            client_clusters=clusters,
            cluster_importance=cluster_importance,
            overhead_seconds=time.perf_counter() - start,
        )
    raise ValueError(f"cannot prepare plan for {method}")


def train_client(model, loader: DataLoader, active_layers: list[int], config: dict, device: torch.device):
    local_model = copy.deepcopy(model).to(device)
    set_active_layers(local_model, active_layers)
    set_trainable_layers(local_model, active_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in local_model.parameters() if parameter.requires_grad],
        lr=config["federated"]["learning_rate"],
    )
    local_model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(config["federated"]["local_epochs"]):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            local_model(**batch).loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    local_model.cpu()
    return local_model, elapsed


def train_client_components(
    model,
    loader: DataLoader,
    components_by_layer: dict[int, list[int]],
    factor: str,
    config: dict,
    device: torch.device,
):
    local_model = copy.deepcopy(model).to(device)
    set_component_masks(local_model, components_by_layer)
    set_trainable_components(local_model, components_by_layer, factor)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in local_model.parameters()
            if parameter.requires_grad
        ],
        lr=config["federated"]["learning_rate"],
    )
    local_model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(config["federated"]["local_epochs"]):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            local_model(**batch).loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    local_model.cpu()
    return local_model, elapsed


@torch.no_grad()
def evaluate_with_diagnostics(
    model, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    model.to(device)
    model.eval()
    set_active_layers(model, range(len(encoder_layers(model))))
    set_component_masks(
        model,
        {
            layer_id: list(
                range(layer.attention.self.query.rank)
            )
            for layer_id, layer in enumerate(encoder_layers(model))
        },
    )
    correct = total = predicted_positive = 0
    true_positives: list[int] | None = None
    false_positives: list[int] | None = None
    false_negatives: list[int] | None = None
    for batch in loader:
        labels = batch["labels"].to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        predictions = logits.argmax(dim=-1)
        if true_positives is None:
            num_labels = logits.shape[-1]
            true_positives = [0] * num_labels
            false_positives = [0] * num_labels
            false_negatives = [0] * num_labels
        for label in range(len(true_positives)):
            true_positives[label] += int(
                ((predictions == label) & (labels == label)).sum()
            )
            false_positives[label] += int(
                ((predictions == label) & (labels != label)).sum()
            )
            false_negatives[label] += int(
                ((predictions != label) & (labels == label)).sum()
            )
        correct += int((predictions == labels).sum())
        predicted_positive += int((predictions == 1).sum())
        total += labels.numel()
    model.cpu()
    f1_scores = []
    for true_positive, false_positive, false_negative in zip(
        true_positives or [],
        false_positives or [],
        false_negatives or [],
    ):
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return (
        correct / max(total, 1),
        predicted_positive / max(total, 1),
        sum(f1_scores) / max(len(f1_scores), 1),
    )


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    return evaluate_with_diagnostics(model, loader, device)[0]


def model_parameter_norms(model) -> tuple[float, float]:
    classifier_squared = sum(
        float(parameter.detach().float().pow(2).sum())
        for parameter in model.classifier.parameters()
    )
    lora_squared = 0.0
    for layer in encoder_layers(model):
        for module_name in ("query", "value"):
            module = getattr(layer.attention.self, module_name)
            lora_squared += float(module.lora_a.detach().float().pow(2).sum())
            lora_squared += float(module.lora_b.detach().float().pow(2).sum())
    return math.sqrt(classifier_squared), math.sqrt(lora_squared)


def aggregate_models(global_model, client_models: list[tuple[object, ClientSpec, list[int]]]) -> None:
    """Layer-wise FedAvg over contributors; absent layers retain global values."""
    with torch.no_grad():
        for layer_id, global_layer in enumerate(encoder_layers(global_model)):
            contributors = [
                (model, client)
                for model, client, active_layers in client_models
                if layer_id in active_layers
            ]
            if not contributors:
                continue
            total_weight = sum(client.samples for _, client in contributors)
            for module_name in ("query", "value"):
                global_module = getattr(global_layer.attention.self, module_name)
                for parameter_name in ("lora_a", "lora_b"):
                    average = torch.zeros_like(getattr(global_module, parameter_name))
                    for model, client in contributors:
                        local_layer = encoder_layers(model)[layer_id]
                        local_module = getattr(local_layer.attention.self, module_name)
                        average += getattr(local_module, parameter_name) * (client.samples / total_weight)
                    getattr(global_module, parameter_name).copy_(average)
        total_weight = sum(client.samples for _, client, _ in client_models)
        for name, global_parameter in global_model.classifier.named_parameters():
            average = torch.zeros_like(global_parameter)
            for model, client, _ in client_models:
                average += dict(model.classifier.named_parameters())[name] * (client.samples / total_weight)
            global_parameter.copy_(average)


def aggregate_component_models(
    global_model,
    client_models: list[
        tuple[object, ClientSpec, dict[int, list[int]]]
    ],
    factor: str,
) -> None:
    parameter_name = f"lora_{factor}"
    with torch.no_grad():
        for layer_id, global_layer in enumerate(encoder_layers(global_model)):
            rank = global_layer.attention.self.query.rank
            for component in range(rank):
                contributors = [
                    (model, client)
                    for model, client, allocation in client_models
                    if component in allocation.get(layer_id, [])
                ]
                if not contributors:
                    continue
                total_weight = sum(client.samples for _, client in contributors)
                for module_name in ("query", "value"):
                    global_parameter = getattr(
                        getattr(global_layer.attention.self, module_name),
                        parameter_name,
                    )
                    if factor == "a":
                        average = torch.zeros_like(
                            global_parameter[component, :]
                        )
                    else:
                        average = torch.zeros_like(
                            global_parameter[:, component]
                        )
                    for model, client in contributors:
                        local_parameter = getattr(
                            getattr(
                                encoder_layers(model)[layer_id].attention.self,
                                module_name,
                            ),
                            parameter_name,
                        )
                        value = (
                            local_parameter[component, :]
                            if factor == "a"
                            else local_parameter[:, component]
                        )
                        average += value * (client.samples / total_weight)
                    if factor == "a":
                        global_parameter[component, :].copy_(average)
                    else:
                        global_parameter[:, component].copy_(average)
        total_weight = sum(client.samples for _, client, _ in client_models)
        for name, global_parameter in global_model.classifier.named_parameters():
            average = torch.zeros_like(global_parameter)
            for model, client, _ in client_models:
                average += dict(model.classifier.named_parameters())[name] * (
                    client.samples / total_weight
                )
            global_parameter.copy_(average)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_checkpoint(
    path: Path,
    config: dict,
    seed: int,
    metrics: list[dict],
    allocations: list[dict],
    importance: list[dict],
) -> None:
    payload = {
        "config_fingerprint": config_fingerprint(config),
        "seed": seed,
        "completed_methods": sorted({row["method"] for row in metrics}),
        "metrics": metrics,
        "allocations": allocations,
        "importance": importance,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_checkpoint(path: Path, config: dict, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if not path.is_file():
        return [], [], []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("config_fingerprint") != config_fingerprint(config) or payload.get("seed") != seed:
        raise ValueError(
            f"checkpoint {path} belongs to a different configuration or seed; "
            "use another output directory or remove it explicitly"
        )
    return payload["metrics"], payload["allocations"], payload["importance"]


def summary_rows(metrics: list[dict], target_accuracy: float) -> list[dict]:
    rows = []
    methods = sorted({row["method"] for row in metrics})
    for method in methods:
        values = sorted((row for row in metrics if row["method"] == method), key=lambda row: row["round"])
        final = values[-1]
        target = next((row for row in values if row["accuracy"] >= target_accuracy), None)
        rows.append(
            {
                "method": method,
                "target_accuracy": target_accuracy,
                "reached_target": target is not None,
                "time_to_target": "" if target is None else target["cumulative_time"],
                "comm_to_target_mb": "" if target is None else target["cumulative_comm_mb"],
                "final_accuracy": final["accuracy"],
                "final_macro_f1": final.get("macro_f1", ""),
                "final_time": final["cumulative_time"],
                "final_comm_mb": final["cumulative_comm_mb"],
                "avg_waiting_time": final["avg_waiting_time"],
                "fim_overhead_seconds": sum(row["fim_overhead_seconds"] for row in values),
            }
        )
    return rows


def run_method(
    method: str,
    tokenizer,
    client_rows: dict[int, list[tuple[str, int]]],
    proxy_rows: list[tuple[str, int]],
    eval_rows: list[tuple[str, int]],
    clients: list[ClientSpec],
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["dataset"]["model"],
        num_labels=config["dataset"]["num_labels"],
        local_files_only=config["dataset"].get("local_files_only", False),
    )
    num_layers = len(encoder_layers(model))
    loaders = {
        client.client_id: DataLoader(
            SST2Dataset(client_rows[client.client_id], tokenizer, config["dataset"]["max_length"]),
            batch_size=config["federated"]["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(seed * 10_007 + client.client_id),
        )
        for client in clients
    }
    proxy_loader = DataLoader(
        SST2Dataset(proxy_rows, tokenizer, config["dataset"]["max_length"]),
        batch_size=config["fedhello"]["proxy_batch_size"],
        shuffle=False,
    )
    round_eval_rows = stratified_subset(
        eval_rows,
        int(config["dataset"].get("round_eval_samples", len(eval_rows))),
        int(config["dataset"].get("eval_monitor_seed", 20_260_728)),
    )
    round_eval_loader = DataLoader(
        SST2Dataset(
            round_eval_rows,
            tokenizer,
            config["dataset"]["max_length"],
        ),
        batch_size=config["federated"]["eval_batch_size"],
    )
    full_eval_loader = (
        round_eval_loader
        if len(round_eval_rows) == len(eval_rows)
        else DataLoader(
            SST2Dataset(
                eval_rows,
                tokenizer,
                config["dataset"]["max_length"],
            ),
            batch_size=config["federated"]["eval_batch_size"],
        )
    )
    formal_methods = {
        "legend",
        "importance_only",
        "importance_resource_lora",
        "personalized_cost_aware_lora",
    }
    plan: MethodPlan | None = None
    importance_rows: list[dict] = []
    if method in formal_methods:
        plan = prepare_formal_method_plan(
            method,
            model,
            loaders,
            proxy_loader,
            clients,
            config,
            seed,
            device,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            config["dataset"]["model"],
            num_labels=config["dataset"]["num_labels"],
            local_files_only=config["dataset"].get("local_files_only", False),
        )
        inject_lora_ranks(
            model,
            plan.ranks,
            config["lora"]["alpha"],
            config["lora"]["dropout"],
        )
        for layer_id, rank in enumerate(plan.ranks):
            importance_rows.append(
                {
                    "method": method,
                    "round": 0,
                    "layer": layer_id,
                    "component": "",
                    "fim_score": ""
                    if plan.global_importance is None
                    else plan.global_importance[layer_id],
                    "allocation_probability": "",
                    "allocated_rank": rank,
                    "fim_overhead_seconds": plan.overhead_seconds,
                    "client_id": "",
                    "raw_score": "",
                    "ema_score": "",
                    "selection_score": "",
                    "selected": "",
                    "selected_count_before": "",
                    "layer_age": "",
                    "importance_refreshed": "",
                }
            )
    else:
        inject_lora(
            model,
            config["lora"]["rank"],
            config["lora"]["alpha"],
            config["lora"]["dropout"],
        )
    model.cpu()
    rgd_probabilities = geometric_prior(config["fedhello"]["rgd_pattern"], clients, num_layers)
    cached_fim_probabilities: list[float] | None = None
    metrics: list[dict] = []
    allocations: list[dict] = []
    cumulative_time = cumulative_comm = 0.0
    rng = torch.Generator().manual_seed(seed * 1009 + sum(ord(char) for char in method))
    online_config = resolve_online_layer_config(method, config)
    online_importance: dict[int, list[float] | None] = {
        client.client_id: None for client in clients
    }
    online_selection_counts = {
        client.client_id: [0] * num_layers for client in clients
    }
    online_last_selected = {
        client.client_id: [-1] * num_layers for client in clients
    }
    online_previous_layers: dict[int, set[int]] = {
        client.client_id: set() for client in clients
    }
    method_previous_layers: dict[int, set[int]] = {
        client.client_id: set() for client in clients
    }
    component_config = config.get("pcla_component", {})
    component_rank = config["lora"]["rank"]
    component_importance: dict[int, list[list[float]] | None] = {
        client.client_id: None for client in clients
    }
    component_importance_last_refresh: dict[int, int | None] = {
        client.client_id: None for client in clients
    }
    component_selection_counts = {
        client.client_id: [0] * (num_layers * component_rank)
        for client in clients
    }
    component_last_selected = {
        client.client_id: [-1] * (num_layers * component_rank)
        for client in clients
    }
    component_previous: dict[int, set[int]] = {
        client.client_id: set() for client in clients
    }
    legend_config = config.get("legend", {})
    legend_status_ema = float(legend_config.get("status_ema", 0.8))
    legend_dynamic_depth = bool(legend_config.get("dynamic_depth", True))
    legend_status_source = str(
        legend_config.get("status_source", "deterministic_cost_model")
    )
    legend_completion_estimates: dict[int, float] = {}
    if method == "legend" and plan is not None:
        initial_depths = legend_depths(
            {client.client_id: 1.0 for client in clients},
            num_layers,
            plan.ranks,
            {
                client.client_id: client.capacity * config["lora"]["rank"]
                for client in clients
            },
        )
        legend_completion_estimates = {
            client.client_id: legend_simulated_completion(
                client,
                list(range(num_layers - initial_depths[client.client_id], num_layers)),
                plan.ranks,
                config["lora"]["rank"],
            )
            for client in clients
        }

    for round_id in range(1, config["federated"]["rounds"] + 1):
        fim_overhead = plan.overhead_seconds if plan is not None and round_id == 1 else 0.0
        selected_clients = clients
        if config["federated"]["selected_clients"] < len(clients):
            selected_clients = random.Random(seed * 10000 + round_id).sample(
                clients, config["federated"]["selected_clients"]
            )
        legend_round_depths: dict[int, int] = {}
        if method == "legend":
            if plan is None:
                raise RuntimeError("LEGEND LCD rank plan is missing")
            round_completion_estimates = {
                client.client_id: (
                    legend_completion_estimates[client.client_id]
                    if legend_dynamic_depth
                    else 1.0
                )
                for client in selected_clients
            }
            legend_round_depths = legend_depths(
                round_completion_estimates,
                num_layers,
                plan.ranks,
                {
                    client.client_id: client.capacity * config["lora"]["rank"]
                    for client in selected_clients
                },
            )
        refresh_fim = should_refresh_fim(
            method,
            round_id,
            config["fedhello"]["rgd_hla_rounds"],
            config["fedhello"]["fim_update_interval"],
            cached_fim_probabilities is not None,
        )
        if refresh_fim:
            start = time.perf_counter()
            with preserve_torch_rng():
                importance = estimate_fim_importance(
                    model, proxy_loader, device, config["fedhello"]["proxy_batches"]
                )
            cached_fim_probabilities = fim_allocation_probabilities(importance, clients)
            fim_overhead = time.perf_counter() - start
            for layer_id, (score, probability) in enumerate(zip(importance, cached_fim_probabilities)):
                importance_rows.append(
                    {
                        "method": method,
                        "round": round_id,
                        "layer": layer_id,
                        "component": "",
                        "fim_score": score,
                        "allocation_probability": probability,
                        "allocated_rank": config["lora"]["rank"],
                        "fim_overhead_seconds": fim_overhead,
                        "client_id": "",
                        "raw_score": "",
                        "ema_score": "",
                        "selection_score": "",
                        "selected": "",
                        "selected_count_before": "",
                        "layer_age": "",
                        "importance_refreshed": "",
                    }
                )

        online_refreshed = False
        online_raw_scores: dict[int, list[float]] = {}
        if is_online_layer_method(method):
            interval = int(online_config["importance_refresh_interval"])
            online_refreshed = round_id == 1 or (round_id - 1) % interval == 0
            if online_refreshed:
                start = time.perf_counter()
                for client in selected_clients:
                    importance_loader = DataLoader(
                        loaders[client.client_id].dataset,
                        batch_size=int(online_config["importance_batch_size"]),
                        shuffle=True,
                        generator=torch.Generator().manual_seed(
                            seed * 1_000_003 + client.client_id * 10_007 + round_id
                        ),
                    )
                    with preserve_torch_rng():
                        raw_scores = estimate_gradient_norm_importance(
                            model,
                            importance_loader,
                            device,
                            int(online_config["importance_batches_per_client"]),
                        )
                    online_raw_scores[client.client_id] = raw_scores
                    online_importance[client.client_id] = ema_update(
                        online_importance[client.client_id],
                        raw_scores,
                        float(online_config["importance_ema"]),
                    )
                measured_overhead = time.perf_counter() - start
                if online_config.get("importance_overhead_in_time", True):
                    fim_overhead = measured_overhead

        component_warm_start = False
        trained_factor = ""
        sparse_round = 0
        if is_component_method(method):
            component_warm_start, trained_factor, sparse_round = (
                component_round_schedule(method, round_id, config)
            )
        component_refreshed = False
        component_refreshed_clients: set[int] = set()
        component_raw_scores: dict[int, list[list[float]]] = {}
        if is_pcla_component_method(method) and not component_warm_start:
            interval = int(component_config["importance_refresh_interval"])
            clients_to_refresh = [
                client
                for client in selected_clients
                if should_refresh_client_importance(
                    component_importance_last_refresh[client.client_id],
                    sparse_round,
                    interval,
                )
            ]
            component_refreshed_clients = {
                client.client_id for client in clients_to_refresh
            }
            component_refreshed = bool(clients_to_refresh)
            if clients_to_refresh:
                start = time.perf_counter()
                for client in clients_to_refresh:
                    importance_loader = DataLoader(
                        loaders[client.client_id].dataset,
                        batch_size=int(component_config["importance_batch_size"]),
                        shuffle=True,
                        generator=torch.Generator().manual_seed(
                            seed * 1_000_033
                            + client.client_id * 10_009
                            + round_id
                        ),
                    )
                    with preserve_torch_rng():
                        raw_scores = estimate_component_importance(
                            model,
                            importance_loader,
                            device,
                            int(
                                component_config[
                                    "importance_batches_per_client"
                                ]
                            ),
                            trained_factor,
                        )
                    component_raw_scores[client.client_id] = raw_scores
                    previous = component_importance[client.client_id]
                    updated_flat = ema_update(
                        None
                        if previous is None
                        else flatten_component_values(previous),
                        flatten_component_values(raw_scores),
                        float(component_config["importance_ema"]),
                    )
                    component_importance[client.client_id] = [
                        updated_flat[
                            layer_id
                            * component_rank : (layer_id + 1)
                            * component_rank
                        ]
                        for layer_id in range(num_layers)
                    ]
                    component_importance_last_refresh[
                        client.client_id
                    ] = sparse_round
                measured_overhead = time.perf_counter() - start
                if component_config.get("importance_overhead_in_time", True):
                    fim_overhead = measured_overhead

        client_models = []
        client_times = []
        client_traffic = []
        round_layer_union: set[int] = set()
        round_stability: list[float] = []
        for client in selected_clients:
            online_scores: list[float] | None = None
            components_by_layer: dict[int, list[int]] | None = None
            if is_component_method(method):
                if component_warm_start:
                    active_layers = deterministic_geometric_layers(
                        component_config.get(
                            "warm_start_pattern", "bottleneck"
                        ),
                        client.capacity,
                        num_layers,
                    )
                    component_scores = [""] * (
                        num_layers * component_rank
                    )
                    phase = "component_warm_start_full_rank"
                    probabilities = None
                elif method in {
                    "random_component_alternating",
                    "random_component_warmstart",
                }:
                    components_by_layer = random_component_allocation(
                        num_layers,
                        component_rank,
                        client.capacity,
                        int(component_config["components_per_layer"]),
                        rng,
                    )
                    component_scores = [""] * (
                        num_layers * component_rank
                    )
                    phase = (
                        "random_component_warmstart_alternating"
                        if method == "random_component_warmstart"
                        else "random_component_alternating"
                    )
                else:
                    client_importance = component_importance[client.client_id]
                    if client_importance is None:
                        raise RuntimeError(
                            "component importance was not initialized"
                        )
                    counts = component_selection_counts[client.client_id]
                    last_selected = component_last_selected[client.client_id]
                    previous_components = component_previous[client.client_id]
                    components_by_layer, component_scores = (
                        select_component_allocation(
                            client_importance,
                            counts,
                            last_selected,
                            previous_components,
                            client.capacity,
                            int(component_config["components_per_layer"]),
                            sparse_round,
                            float(component_config["exploration_beta"]),
                            float(component_config["freshness_gamma"]),
                            float(component_config["stability_delta"]),
                        )
                    )
                    raw_scores = component_raw_scores.get(client.client_id)
                    selected_flat = {
                        layer_id * component_rank + component
                        for layer_id, components in components_by_layer.items()
                        for component in components
                    }
                    for layer_id in range(num_layers):
                        for component in range(component_rank):
                            flat_id = layer_id * component_rank + component
                            previous_round = last_selected[flat_id]
                            layer_age = (
                                sparse_round
                                if previous_round < 0
                                else max(sparse_round - previous_round, 0)
                            )
                            importance_rows.append(
                                {
                                    "method": method,
                                    "round": round_id,
                                    "layer": layer_id,
                                    "component": component,
                                    "fim_score": client_importance[layer_id][
                                        component
                                    ],
                                    "allocation_probability": "",
                                    "allocated_rank": len(
                                        components_by_layer.get(layer_id, [])
                                    ),
                                    "fim_overhead_seconds": fim_overhead
                                    if client.client_id
                                    in component_refreshed_clients
                                    else 0.0,
                                    "client_id": client.client_id,
                                    "raw_score": ""
                                    if raw_scores is None
                                    else raw_scores[layer_id][component],
                                    "ema_score": client_importance[layer_id][
                                        component
                                    ],
                                    "selection_score": component_scores[
                                        flat_id
                                    ],
                                    "selected": flat_id in selected_flat,
                                    "selected_count_before": counts[flat_id],
                                    "layer_age": layer_age,
                                    "importance_refreshed": client.client_id
                                    in component_refreshed_clients,
                                }
                            )
                    for flat_id in selected_flat:
                        counts[flat_id] += 1
                        last_selected[flat_id] = sparse_round
                    component_previous[client.client_id] = selected_flat
                    phase = (
                        "component_ema_ucb_warmstart_alternating"
                        if method == "pcla_component_warmstart"
                        else "component_ema_ucb_alternating"
                    )
                if components_by_layer is not None:
                    active_layers = sorted(components_by_layer)
                    probabilities = None
            elif is_online_layer_method(method):
                client_importance = online_importance[client.client_id]
                if client_importance is None:
                    raise RuntimeError("online importance was not initialized")
                counts = online_selection_counts[client.client_id]
                last_selected = online_last_selected[client.client_id]
                previous_layers = online_previous_layers[client.client_id]
                active_layers, online_scores = select_online_layers(
                    client_importance,
                    counts,
                    last_selected,
                    previous_layers,
                    client.capacity,
                    round_id,
                    float(online_config["exploration_beta"]),
                    float(online_config["freshness_gamma"]),
                    float(online_config["stability_delta"]),
                )
                phase, probabilities = "online_ema_ucb", None
                raw_scores = online_raw_scores.get(client.client_id, [""] * num_layers)
                for layer_id in range(num_layers):
                    previous_round = last_selected[layer_id]
                    layer_age = (
                        round_id
                        if previous_round < 0
                        else max(round_id - previous_round, 0)
                    )
                    importance_rows.append(
                        {
                            "method": method,
                            "round": round_id,
                            "layer": layer_id,
                            "component": "",
                            "fim_score": client_importance[layer_id],
                            "allocation_probability": "",
                            "allocated_rank": config["lora"]["rank"],
                            "fim_overhead_seconds": fim_overhead
                            if online_refreshed
                            else 0.0,
                            "client_id": client.client_id,
                            "raw_score": raw_scores[layer_id],
                            "ema_score": client_importance[layer_id],
                            "selection_score": online_scores[layer_id],
                            "selected": layer_id in active_layers,
                            "selected_count_before": counts[layer_id],
                            "layer_age": layer_age,
                            "importance_refreshed": online_refreshed,
                        }
                    )
                for layer_id in active_layers:
                    counts[layer_id] += 1
                    last_selected[layer_id] = round_id
                online_previous_layers[client.client_id] = set(active_layers)
            elif method == "legend":
                depth = legend_round_depths[client.client_id]
                active_layers = list(range(num_layers - depth, num_layers))
                phase = (
                    "legend_dynamic_depth"
                    if legend_dynamic_depth
                    else "legend_fixed_depth_diagnostic"
                )
                probabilities = None
            elif method in formal_methods:
                active_layers, phase = formal_active_layers(
                    method, client, plan, num_layers
                )
                probabilities = None
            else:
                active_layers, phase, probabilities = allocate_layers(
                    method,
                    client,
                    num_layers,
                    rng,
                    cached_fim_probabilities,
                    rgd_probabilities,
                    round_id,
                    config["fedhello"]["rgd_hla_rounds"],
                )
            previous_method_layers = method_previous_layers[client.client_id]
            if previous_method_layers:
                active_set = set(active_layers)
                round_stability.append(
                    len(previous_method_layers.intersection(active_set))
                    / max(len(previous_method_layers.union(active_set)), 1)
                )
            method_previous_layers[client.client_id] = set(active_layers)
            round_layer_union.update(active_layers)
            if components_by_layer is not None:
                local_model, elapsed = train_client_components(
                    model,
                    loaders[client.client_id],
                    components_by_layer,
                    trained_factor,
                    config,
                    device,
                )
                trainable_parameters = component_trainable_parameter_count(
                    local_model, components_by_layer, trained_factor
                )
            else:
                local_model, elapsed = train_client(
                    model,
                    loaders[client.client_id],
                    active_layers,
                    config,
                    device,
                )
                trainable_parameters = trainable_parameter_count(
                    local_model, active_layers
                )
            if components_by_layer is not None:
                (
                    download_parameters,
                    upload_parameters,
                    importance_sync_parameters,
                ) = component_communication_parameter_counts(
                    local_model,
                    components_by_layer,
                    trained_factor,
                    client.client_id in component_refreshed_clients,
                )
            else:
                download_parameters = trainable_parameters
                upload_parameters = trainable_parameters
                importance_sync_parameters = 0
            download_mb = download_parameters * 4 / (1024 * 1024)
            upload_mb = upload_parameters * 4 / (1024 * 1024)
            importance_sync_mb = (
                importance_sync_parameters * 4 / (1024 * 1024)
            )
            payload_mb = upload_mb
            traffic_mb = download_mb + upload_mb
            simulated_time = (
                elapsed * client.compute_factor
                + traffic_mb * config["resource_heterogeneity"]["communication_time_per_mb"]
                * client.communication_factor
            )
            client_models.append(
                (
                    local_model,
                    client,
                    components_by_layer
                    if components_by_layer is not None
                    else active_layers,
                )
            )
            client_times.append(simulated_time)
            client_traffic.append(traffic_mb)
            allocations.append(
                {
                    "method": method,
                    "round": round_id,
                    "client_id": client.client_id,
                    "capability": client.capability,
                    "capacity": client.capacity,
                    "phase": phase,
                    "active_layers": " ".join(map(str, active_layers)),
                    "allocation_probabilities": "" if probabilities is None else " ".join(f"{value:.10f}" for value in probabilities),
                    "active_ranks": " ".join(
                        str(
                            len(components_by_layer[layer])
                            if components_by_layer is not None
                            else (
                                config["lora"]["rank"]
                                if plan is None
                                else plan.ranks[layer]
                            )
                        )
                        for layer in active_layers
                    ),
                    "active_components": ""
                    if components_by_layer is None
                    else ";".join(
                        f"{layer}:"
                        + " ".join(
                            map(str, components_by_layer[layer])
                        )
                        for layer in active_layers
                    ),
                    "trained_factor": trained_factor
                    if is_component_method(method)
                    else "",
                    "client_cluster": ""
                    if plan is None or plan.client_clusters is None
                    else plan.client_clusters[client.client_id],
                    "trainable_parameters": trainable_parameters,
                    "download_parameters": download_parameters,
                    "upload_parameters": upload_parameters,
                    "importance_sync_parameters": importance_sync_parameters,
                    "one_way_payload_mb": payload_mb,
                    "download_mb": download_mb,
                    "upload_mb": upload_mb,
                    "importance_sync_mb": importance_sync_mb,
                    "communication_mb": traffic_mb,
                    "measured_local_seconds": elapsed,
                    "simulated_client_seconds": simulated_time,
                }
            )
            if method == "legend" and legend_dynamic_depth:
                previous_estimate = legend_completion_estimates[client.client_id]
                if legend_status_source == "measured_wall_clock":
                    observed_completion = simulated_time
                elif legend_status_source == "deterministic_cost_model":
                    observed_completion = legend_simulated_completion(
                        client,
                        active_layers,
                        plan.ranks,
                        config["lora"]["rank"],
                    )
                else:
                    raise ValueError(
                        f"unknown LEGEND status source: {legend_status_source}"
                    )
                legend_completion_estimates[client.client_id] = (
                    legend_status_ema * previous_estimate
                    + (1.0 - legend_status_ema) * observed_completion
                )
        if is_component_method(method) and not component_warm_start:
            aggregate_component_models(model, client_models, trained_factor)
        else:
            aggregate_models(model, client_models)
        round_time = max(client_times) + fim_overhead
        waiting = sum(max(client_times) - value for value in client_times) / len(client_times)
        cumulative_time += round_time
        cumulative_comm += sum(client_traffic)
        evaluation_scope = (
            "full"
            if round_id == config["federated"]["rounds"]
            else "monitor"
        )
        evaluation_loader = (
            full_eval_loader
            if evaluation_scope == "full"
            else round_eval_loader
        )
        (
            accuracy,
            predicted_positive_rate,
            macro_f1,
        ) = evaluate_with_diagnostics(model, evaluation_loader, device)
        classifier_norm, lora_norm = model_parameter_norms(model)
        metrics.append(
            {
                "method": method,
                "round": round_id,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "evaluation_scope": evaluation_scope,
                "evaluation_samples": (
                    len(eval_rows)
                    if evaluation_scope == "full"
                    else len(round_eval_rows)
                ),
                "predicted_positive_rate": predicted_positive_rate,
                "classifier_parameter_norm": classifier_norm,
                "lora_parameter_norm": lora_norm,
                "round_time": round_time,
                "cumulative_time": cumulative_time,
                "round_comm_mb": sum(client_traffic),
                "cumulative_comm_mb": cumulative_comm,
                "avg_waiting_time": waiting,
                "fim_overhead_seconds": fim_overhead,
                "selection_coverage": len(round_layer_union) / num_layers,
                "selection_stability": ""
                if not round_stability
                else sum(round_stability) / len(round_stability),
                "importance_refreshed": online_refreshed
                if is_online_layer_method(method)
                else (
                    component_refreshed
                    if is_pcla_component_method(method)
                    else ""
                ),
            }
        )
        print(
            f"{method:26s} round={round_id:02d} acc={accuracy:.4f} "
            f"time={cumulative_time:.2f}s comm={cumulative_comm:.2f}MB "
            f"wait={waiting:.2f}s fim={fim_overhead:.2f}s",
            flush=True,
        )
        del client_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics, allocations, importance_rows


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fed-HeLLo controlled SST-2 runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--legend-rank-common-difference",
        type=int,
        default=None,
        help="Diagnostic override for the LEGEND arithmetic rank difference.",
    )
    parser.add_argument(
        "--legend-fixed-depth",
        action="store_true",
        help="Diagnostic override that disables LCD depth adaptation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if args.legend_rank_common_difference is not None:
        config.setdefault("legend", {})["rank_common_difference"] = (
            args.legend_rank_common_difference
        )
    if args.legend_fixed_depth:
        config.setdefault("legend", {})["dynamic_depth"] = False
    methods = config["baselines"] if not args.methods else [item.strip() for item in args.methods.split(",") if item.strip()]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config["dataset"]["model"],
        local_files_only=config["dataset"].get("local_files_only", False),
    )
    train_rows = read_text_rows(
        resolve_path(config["dataset"]["train_file"]),
        config["dataset"],
    )
    dev_rows = read_text_rows(
        resolve_path(config["dataset"]["dev_file"]),
        config["dataset"],
    )
    train_rows, proxy_rows, eval_rows = prepare_data_splits(
        train_rows,
        dev_rows,
        config,
        args.seed,
    )
    probe = AutoModelForSequenceClassification.from_pretrained(
        config["dataset"]["model"],
        num_labels=config["dataset"]["num_labels"],
        local_files_only=config["dataset"].get("local_files_only", False),
    )
    num_layers = len(encoder_layers(probe))
    del probe
    if num_layers != config["lora"]["num_encoder_layers"]:
        raise ValueError(
            f"config expects {config['lora']['num_encoder_layers']} layers but model has {num_layers}"
        )
    num_labels = int(config["dataset"]["num_labels"])
    observed_labels = sorted({row_label(row) for row in train_rows})
    if observed_labels != list(range(num_labels)):
        raise ValueError(
            f"dataset labels {observed_labels} do not match configured "
            f"range 0..{num_labels - 1}"
        )
    global_label_ratios = [
        sum(row_label(row) == label for row in train_rows) / len(train_rows)
        for label in range(num_labels)
    ]
    clients = make_clients(
        config,
        num_layers,
        global_label_ratios[1],
        args.seed,
        global_label_ratios,
    )
    client_rows = build_client_rows(train_rows, clients, args.seed)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "client_partition.csv", partition_rows(clients, client_rows))
    checkpoint_path = out_dir / "checkpoint.json"
    if args.no_resume:
        all_metrics, all_allocations, all_importance = [], [], []
    else:
        all_metrics, all_allocations, all_importance = load_checkpoint(
            checkpoint_path, config, args.seed
        )
    completed_methods = {row["method"] for row in all_metrics}
    print(f"device={device} model={config['dataset']['model']} seed={args.seed}")
    print("clients=" + ",".join(f"{client.client_id}:{client.capability}:{client.capacity}" for client in clients))
    for method in methods:
        if method in completed_methods:
            print(f"\n=== {method} (resumed: already complete) ===", flush=True)
            continue
        print(f"\n=== {method} ===", flush=True)
        metrics, allocations, importance = run_method(
            method,
            tokenizer,
            client_rows,
            proxy_rows,
            eval_rows,
            clients,
            config,
            args.seed,
            device,
        )
        all_metrics.extend(metrics)
        all_allocations.extend(allocations)
        all_importance.extend(importance)
        write_csv(out_dir / "fedhello_metrics.csv", all_metrics)
        write_csv(out_dir / "allocation_map.csv", all_allocations)
        write_csv(out_dir / "fim_importance.csv", all_importance)
        write_csv(
            out_dir / "summary_to_target.csv",
            summary_rows(all_metrics, config["target_accuracy"]),
        )
        write_checkpoint(
            checkpoint_path,
            config,
            args.seed,
            all_metrics,
            all_allocations,
            all_importance,
        )
    if not all_metrics:
        raise RuntimeError("no methods were executed or restored")
    run_config = copy.deepcopy(config)
    run_config["executed_seed"] = args.seed
    run_config["executed_methods"] = methods
    run_config["device"] = str(device)
    run_config["effective_clients"] = [client.__dict__ for client in clients]
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
