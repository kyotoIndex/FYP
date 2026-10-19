"""Local official-file ingestion, grouped partitions, and shared task collators."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch


GENRES = ("fiction", "government", "slate", "telephone", "travel")
LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_source(task: str, source: Path) -> tuple[dict[str, list[dict]], dict]:
    splits, provenance = {}, {}
    if task == "multinli":
        filenames = {"train": "multinli_1.0_train.jsonl",
                     "final_matched": "multinli_1.0_dev_matched.jsonl",
                     "final_mismatched": "multinli_1.0_dev_mismatched.jsonl"}
        for split, filename in filenames.items():
            path = source / filename
            provenance[filename] = file_hash(path)
            rows = []
            for r in load_jsonl(path):
                if r["gold_label"] not in LABELS:
                    continue
                rows.append({"id": r["pairID"], "text": r["sentence1"],
                             "text_pair": r["sentence2"], "label": LABELS[r["gold_label"]],
                             "domain": r["genre"], "group": digest(normalized(r["sentence1"]))})
            splits[split] = rows
    elif task == "ag_news":
        for split, filename in {"train": "train.csv", "final": "test.csv"}.items():
            path = source / filename
            provenance[filename] = file_hash(path)
            rows = []
            with path.open(newline="") as f:
                for i, r in enumerate(csv.reader(f)):
                    if len(r) != 3 or r[0] not in {"1", "2", "3", "4"}:
                        raise ValueError("AG News expects original headerless label,title,description CSV")
                    text = f"{r[1]} {r[2]}"
                    rows.append({"id": f"{split}-{i}", "text": text, "label": int(r[0]) - 1,
                                 "domain": "news", "group": digest(normalized(text))})
            splits[split] = rows
    elif task == "e2e":
        for split, stem in {"train": "train-fixed.no-ol", "validation": "devel-fixed.no-ol",
                            "final": "test-fixed"}.items():
            candidates = [source / stem, source / f"{stem}.csv"]
            path = next((p for p in candidates if p.is_file()), candidates[0])
            provenance[path.name] = file_hash(path)
            rows = []
            with path.open(newline="") as f:
                for i, r in enumerate(csv.DictReader(f)):
                    mr, ref = r["mr"], r["ref"]
                    # Ignore slot order for grouping references of the same MR.
                    canonical = "|".join(sorted(normalized(x) for x in re.findall(r"\w+\[[^\]]*\]", mr)))
                    if not canonical:
                        raise ValueError(f"Cannot parse E2E MR: {mr}")
                    rows.append({"id": f"{split}-{i}", "text": mr, "target": ref,
                                 "domain": "restaurant", "group": digest(canonical)})
            splits[split] = rows
    else:
        raise ValueError(f"Unknown task: {task}")
    return splits, provenance


def holdout_groups(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    strata = defaultdict(list)
    for key, values in sorted(groups.items()):
        # Whole premise/MR groups remain intact; exact label stratification is not
        # asserted when one group contains multiple labels.
        strata[values[0]["domain"]].append(key)
    selected = set()
    for keys in strata.values():
        rng.shuffle(keys)
        n = min(len(keys) - 1, max(1, round(len(keys) * fraction)))
        selected.update(keys[:n])
    return ([r for r in rows if r["group"] not in selected],
            [r for r in rows if r["group"] in selected])


def separate_cross_genre_groups(rows: list[dict], policy: str) -> tuple[list[dict], list[dict], dict]:
    """Keep pure-genre partitions explicit without splitting identical premises."""
    if policy not in {"error", "exclude"}:
        raise ValueError("Unknown cross-genre policy")
    domains = defaultdict(set)
    for row in rows:
        domains[row["group"]].add(row["domain"])
    ambiguous = {key for key, values in domains.items() if len(values) > 1}
    excluded = [r for r in rows if r["group"] in ambiguous]
    if ambiguous and policy == "error":
        raise ValueError(f"{len(ambiguous)} premise groups ({len(excluded)} rows) span genres; "
                         "use an explicit cross-genre exclusion policy for pure-genre clients")
    audit = {"policy": policy, "groups": len(ambiguous), "rows": len(excluded),
             "by_domain": dict(Counter(r["domain"] for r in excluded)),
             "group_hashes": sorted(ambiguous)}
    return [r for r in rows if r["group"] not in ambiguous], excluded, audit


def allocate_clients(rows: list[dict], clients: int, partition: str, alpha: float,
                     seed: int, min_samples: int, retries: int = 200) -> tuple[list[list[int]], int]:
    if len(rows) < clients * min_samples:
        raise ValueError("Insufficient data for requested clients/minimum sample count")
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["group"]].append(i)
    if partition == "genre" and clients % len(GENRES):
        raise ValueError("Genre partition requires a multiple of five clients")
    if partition == "dirichlet" and alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    rng = np.random.default_rng(seed)
    for attempt in range(retries):
        result = [[] for _ in range(clients)]
        weights = {}
        keys = list(sorted(groups))
        rng.shuffle(keys)
        for key in keys:
            indices = groups[key]
            row = rows[indices[0]]
            if partition == "genre":
                domains = {rows[i]["domain"] for i in indices}
                if len(domains) != 1:
                    raise ValueError("Shared premise spans genres; resolve explicitly before genre partition")
                start = GENRES.index(row["domain"]) * (clients // len(GENRES))
                eligible = range(start, start + clients // len(GENRES))
                client = min(eligible, key=lambda j: len(result[j]))
            elif partition == "iid":
                client = min(range(clients), key=lambda j: len(result[j]))
            elif partition == "dirichlet":
                label = row["label"]
                if any(rows[i]["label"] != label for i in indices):
                    raise ValueError("Dirichlet label partition expects a single label per duplicate group")
                if label not in weights:
                    weights[label] = rng.dirichlet(np.full(clients, alpha))
                client = int(rng.choice(clients, p=weights[label]))
            else:
                raise ValueError(f"Unknown partition: {partition}")
            result[client].extend(indices)
        if min(map(len, result)) >= min_samples:
            return result, attempt + 1
    raise ValueError("Partition failed minimum-size constraint; no silent sample dropping")


def resource_roles(clients: list[list[int]], rows: list[dict], scenario: str,
                   fraction: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    if not 0 <= fraction <= 1:
        raise ValueError("Strong fraction outside [0,1]")
    roles = ["weak"] * len(clients)
    if scenario in {"R1", "R2"}:
        strong_domains = {"government", "travel"} if scenario == "R1" else {"fiction", "telephone"}
        for i, ids in enumerate(clients):
            domains = {rows[j]["domain"] for j in ids}
            if len(domains) != 1 or not domains <= set(GENRES):
                raise ValueError("R1/R2 require pure MultiNLI genre clients")
            roles[i] = "strong" if domains <= strong_domains else "weak"
        if sum(x == "strong" for x in roles) != round(len(clients) * fraction):
            raise ValueError("R1/R2 domain allocation disagrees with requested strong fraction")
    elif scenario == "R0":
        strata = defaultdict(list)
        for i, ids in enumerate(clients):
            domains = tuple(sorted({rows[j]["domain"] for j in ids}))
            strata[domains].append(i)
        # Stratify only when this preserves the requested total exactly.
        if sum(round(len(v) * fraction) for v in strata.values()) != round(len(clients) * fraction):
            strata = {"all": list(range(len(clients)))}
        for ids in strata.values():
            rng.shuffle(ids)
            for i in ids[:round(len(ids) * fraction)]:
                roles[i] = "strong"
    elif scenario == "homogeneous":
        roles = ["strong"] * len(clients)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return roles


def assign_evaluation(rows: list[dict], train: list[dict], clients: list[list[int]], seed: int) -> None:
    """Allocate held-out groups by training-domain/label profiles, without training on them."""
    profiles = [Counter((train[j]["domain"], train[j].get("label")) for j in ids) for ids in clients]
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    rng = np.random.default_rng(seed)
    for _, values in sorted(groups.items()):
        if len({r["domain"] for r in values}) != 1:
            # Preserve official evaluation rows and group integrity, but do not
            # invent one training-domain/client identity for a mixed-genre group.
            for r in values:
                r["client"] = None
            continue
        row = values[0]
        key = row["domain"], row.get("label")
        weights = np.array([p[key] for p in profiles], dtype=float)
        client = int(rng.choice(len(clients), p=weights / weights.sum())) if weights.sum() else None
        for r in values:
            r["client"] = client  # Unseen domains have no invented resource-group identity.


def prepare(task: str, source: Path, output: Path, *, partition: str, clients: int,
            alpha: float, seed: int, validation_fraction: float, min_samples: int,
            strong_fraction: float, scenario: str, cross_genre_policy: str = "error") -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite data manifest: {output}")
    if not 0 < validation_fraction < 1 or clients < 1 or min_samples < 1:
        raise ValueError("Invalid validation fraction, client count or minimum samples")
    splits, provenance = read_source(task, source)
    excluded, cross_genre_audit = [], None
    if task == "multinli" and partition == "genre":
        splits["train"], excluded, cross_genre_audit = separate_cross_genre_groups(splits["train"], cross_genre_policy)
    elif cross_genre_policy != "error":
        raise ValueError("Cross-genre policy applies only to MultiNLI genre partitions")
    if "validation" not in splits:
        splits["train"], splits["validation"] = holdout_groups(splits["train"], validation_fraction, seed)
    # Preserve official evaluation sets, remove any overlapping groups from train.
    eval_groups = set().union(*({r["group"] for r in rows} for name, rows in splits.items() if name != "train"))
    before = len(splits["train"])
    splits["train"] = [r for r in splits["train"] if r["group"] not in eval_groups]
    final_groups = set().union(*({r["group"] for r in rows} for name, rows in splits.items() if name.startswith("final")))
    before_valid = len(splits["validation"])
    splits["validation"] = [r for r in splits["validation"] if r["group"] not in final_groups]
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Empty train/validation after overlap handling")
    allocation, attempts = allocate_clients(splits["train"], clients, partition, alpha, seed, min_samples)
    roles = resource_roles(allocation, splits["train"], scenario, strong_fraction, seed + 1)
    for client, ids in enumerate(allocation):
        for index in ids:
            splits["train"][index]["client"] = client
    for i, (name, rows) in enumerate(sorted(splits.items())):
        if name != "train":
            assign_evaluation(rows, splits["train"], allocation, seed + 10 + i)
    unassigned = {name: sum(r.get("client") is None for r in rows)
                  for name, rows in splits.items() if name != "train"}
    if excluded:
        # Auditable source rows, excluded from train/validation and final scoring.
        splits["excluded_cross_genre"] = excluded
    output.mkdir(parents=True)
    files = {}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = {"path": path.name, "sha256": file_hash(path), "rows": len(rows)}
    manifest = {"version": 2, "task": task, "seed": seed, "partition": partition,
                "alpha": alpha, "scenario": scenario, "source_hashes": provenance,
                "cross_genre_audit": cross_genre_audit, "unassigned_evaluation_rows": unassigned,
                "label_mapping": LABELS if task == "multinli" else None,
                "removed_train_overlap_rows": before - len(splits["train"]),
                "removed_validation_overlap_rows": before_valid - len(splits["validation"]),
                "partition_attempts": attempts, "files": files,
                "clients": [{"id": i, "role": roles[i], "train_indices": ids,
                             "samples": len(ids),
                             "domains": dict(Counter(splits["train"][j]["domain"] for j in ids)),
                             "labels": dict(Counter(str(splits["train"][j].get("label")) for j in ids))}
                            for i, ids in enumerate(allocation)]}
    write_json(output / "manifest.json", manifest)
    return manifest


class TokenizedRows:
    def __init__(self, rows: list[dict], tokenizer, task: dict):
        self.rows, self.items = rows, []
        self.truncated = 0
        max_length = task["max_length"]
        for row in rows:
            if task["kind"] == "classification":
                encoded = tokenizer(row["text"], row.get("text_pair"), truncation=False,
                                    add_special_tokens=True)
                self.truncated += len(encoded["input_ids"]) > max_length
                encoded = tokenizer(row["text"], row.get("text_pair"), truncation=True,
                                    max_length=max_length, add_special_tokens=True)
                self.items.append({"input_ids": encoded["input_ids"], "labels": row["label"]})
            else:
                prompt = f"Describe this restaurant: {row['text']}\nDescription:"
                prefix = tokenizer.encode(prompt, add_special_tokens=False)
                target = tokenizer.encode(" " + row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(prefix) >= max_length - 1:
                    raise ValueError("Generation prompt leaves no target space; change task length explicitly")
                self.truncated += len(prefix) + len(target) > max_length
                ids = (prefix + target)[:max_length]
                self.items.append({"input_ids": ids, "labels": ([-100] * len(prefix) + target)[:max_length],
                                   "prompt_ids": prefix})
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer requires an explicit pad token")
        self.causal = task["kind"] == "generation"

    def batch(self, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
        items = [self.items[i] for i in indices]
        length = max(len(r["input_ids"]) for r in items)
        ids = [r["input_ids"] + [self.pad_id] * (length - len(r["input_ids"])) for r in items]
        masks = [[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in items]
        labels = [r["labels"] + [-100] * (length - len(r["labels"])) if self.causal else r["labels"] for r in items]
        return {"input_ids": torch.tensor(ids, device=device),
                "attention_mask": torch.tensor(masks, device=device),
                "labels": torch.tensor(labels, device=device)}
