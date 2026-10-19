"""Existing flat token cache, manifest metadata and deterministic per-client data streams."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def derived_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


class CachedRows:
    """Flat-token classification cache produced by the shared MNLI pipeline."""

    def __init__(self, value: dict):
        self.tokens = value["tokens"]
        self.offsets = value["offsets"]
        self.labels = value["labels"]
        self.pad_id = int(value["pad_id"])
        if self.offsets.numel() != self.labels.numel() + 1:
            raise ValueError("Token cache requires N+1 offsets for N labels")

    def __len__(self) -> int:
        return self.labels.numel()

    def length(self, index: int) -> int:
        return int(self.offsets[index + 1] - self.offsets[index])

    def batch(self, indices, device) -> dict[str, torch.Tensor]:
        values = [self.tokens[int(self.offsets[i]) : int(self.offsets[i + 1])].long() for i in indices]
        width = max(v.numel() for v in values)
        ids = torch.full((len(values), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, value in enumerate(values):
            ids[row, : value.numel()] = value
            mask[row, : value.numel()] = 1
        return {
            "input_ids": ids.to(device),
            "attention_mask": mask.to(device),
            "labels": self.labels[list(indices)].long().to(device),
        }


def load_cache(path: Path) -> CachedRows:
    return CachedRows(torch.load(path, map_location="cpu", weights_only=True))


def read_metadata(path: Path) -> list[dict]:
    fields = ("group", "domain", "client", "label")
    with path.open() as stream:
        return [{k: row.get(k) for k in fields} for row in map(json.loads, filter(str.strip, stream))]


class ClientStream:
    """Fresh permutation per local epoch; position is the only resumable state."""

    def __init__(self, indices: list[int], seed: int, client: int, position: int = 0):
        self.indices = np.asarray(sorted(indices), dtype=np.int64)
        self.seed, self.client, self.position = seed, client, position
        self._epoch, self._order = -1, None

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            rng = np.random.default_rng(derived_seed(self.seed, "stream", self.client, epoch))
            self._order, self._epoch = rng.permutation(self.indices), epoch
        return self._order

    def take(self, size: int) -> list[int]:
        out, n = [], len(self.indices)
        for _ in range(size):
            epoch, offset = divmod(self.position, n)
            out.append(int(self._permutation(epoch)[offset]))
            self.position += 1
        return out


def nested_strong_clients(clients: dict[int, dict], fraction: float) -> list[int]:
    """Deterministic strong subset of the manifest's strong clients, nested across fractions.

    Manifest strong clients are ordered by interleaving their domains (sorted names, ids
    ascending), so a smaller fraction is always a prefix of a larger one and the domain
    coupling of the original scenario is preserved. Independent of the training seed.
    """
    by_domain: dict[str, list[int]] = defaultdict(list)
    for cid in sorted(clients):
        if clients[cid]["role"] == "strong":
            by_domain[clients[cid]["domain"]].append(cid)
    order, columns = [], [by_domain[d] for d in sorted(by_domain)]
    for i in range(max(map(len, columns), default=0)):
        order.extend(col[i] for col in columns if i < len(col))
    count = round(fraction * len(clients))
    if not 0 < count <= len(order):
        raise ValueError(f"strong_fraction {fraction} needs {count} strong clients; manifest has {len(order)}")
    return sorted(order[:count])


def override_roles(clients: dict[int, dict], fraction: float) -> list[int]:
    strong = set(nested_strong_clients(clients, fraction))
    for cid, client in clients.items():
        client["role"] = "strong" if cid in strong else "weak"
    return sorted(strong)


class Dataset:
    """Training/validation caches joined with manifest client identities."""

    def __init__(self, cache_dir: Path, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text())
        root = manifest_path.parent
        self.train = load_cache(cache_dir / "train.pt")
        self.valid = load_cache(cache_dir / "validation.pt")
        train_meta = read_metadata(root / manifest["files"]["train"]["path"])
        valid_meta = read_metadata(root / manifest["files"]["validation"]["path"])
        if len(train_meta) != len(self.train) or len(valid_meta) != len(self.valid):
            raise ValueError("Manifest rows and token cache rows differ")
        if {r["group"] for r in train_meta} & {r["group"] for r in valid_meta}:
            raise ValueError("Training and validation share a premise group")
        self.clients: dict[int, dict] = {}
        for client in manifest["clients"]:
            cid = int(client["id"])
            for i in client["train_indices"]:
                if int(train_meta[i]["client"]) != cid:
                    raise ValueError(f"Train row {i} is not owned by client {cid}")
            self.clients[cid] = {
                "role": client["role"],
                "samples": int(client["samples"]),
                "train_indices": list(client["train_indices"]),
                "domain": Counter(client["domains"]).most_common(1)[0][0],
            }
        self.valid_by_client: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(valid_meta):
            if row["client"] is not None:
                self.valid_by_client[int(row["client"])].append(i)
        missing = sorted(set(self.clients) - set(self.valid_by_client))
        if missing:
            raise ValueError(f"Clients without validation rows: {missing}")
        self.manifest_summary = {
            "path": str(manifest_path),
            "task": manifest.get("task"),
            "scenario": manifest.get("scenario"),
            "partition": manifest.get("partition"),
            "data_seed": manifest.get("seed"),
            "train_rows": len(self.train),
            "validation_rows": len(self.valid),
            "unassigned_validation_rows": sum(r["client"] is None for r in valid_meta),
            "roles": dict(Counter(c["role"] for c in self.clients.values())),
        }
