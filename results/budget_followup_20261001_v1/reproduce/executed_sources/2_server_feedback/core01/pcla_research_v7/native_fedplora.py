"""Fed-PLoRA paper Select-N-Fold adapter and contributor aggregation.

Each global rank index is a parallel one-rank pair. A client samples indices
per projection, folds the unselected pair products into its frozen local base,
trains selected pairs, and sends only their deltas plus task head. The global
base is immutable; its local fold is never averaged or persisted.
"""
from __future__ import annotations

import hashlib
import random
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

OFFICIAL_REPO = "https://github.com/TNI-playground/Fed-PLoRA"
OFFICIAL_COMMIT = "778c69ba3087984c595411a3b1e58dc6fb710acc"
PAPER_RULE = "Section 3.3 Select-N-Fold and contributor-only aggregation"


def select_indices(rank: int, client_rank: int, *, protocol: str, seed: int,
                   round_id: int, client_id: int, module: str) -> tuple[int, ...]:
    if not 0 < client_rank <= rank or round_id < 1:
        raise ValueError("invalid Fed-PLoRA rank/round")
    identity = f"{protocol}|plora|{seed}|{round_id}|{client_id}|{module}"
    keyed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    return tuple(sorted(random.Random(keyed).sample(range(rank), client_rank)))


def folded_local_weight(base_weight: Tensor, a: Tensor, b: Tensor,
                        selected: tuple[int, ...], *, scaling=1.0) -> Tensor:
    if base_weight.ndim != 2 or a.ndim != 2 or b.ndim != 2 or (
            a.shape[0] != b.shape[1] or a.shape[1] != base_weight.shape[1] or
            b.shape[0] != base_weight.shape[0]):
        raise ValueError("Fed-PLoRA projection shape mismatch")
    if len(selected) != len(set(selected)) or any(i < 0 or i >= a.shape[0] for i in selected):
        raise ValueError("invalid selected rank IDs")
    unselected = [i for i in range(a.shape[0]) if i not in selected]
    if not unselected:
        return base_weight.detach().clone()
    return base_weight.detach().clone() + float(scaling) * (b[:, unselected] @ a[unselected, :]).detach()


class FedPLoRALocalLinear(nn.Module):
    """Client-local folded W0 and trainable selected parallel one-rank pairs."""
    def __init__(self, base: nn.Linear, a: Tensor, b: Tensor,
                 selected: tuple[int, ...], *, scaling=1.0, dropout=.1):
        super().__init__()
        folded = folded_local_weight(base.weight, a, b, selected, scaling=scaling)
        self.register_buffer("folded_weight", folded)
        if base.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base.bias.detach().clone())
        self.selected = tuple(selected)
        self.a = nn.Parameter(a[list(selected), :].detach().clone())
        self.b = nn.Parameter(b[:, list(selected)].detach().clone())
        self.scaling = float(scaling)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        return F.linear(x, self.folded_weight, self.bias) + (
            F.linear(F.linear(self.dropout(x), self.a), self.b) * self.scaling)

    def expand_into(self, parent_a: Tensor, parent_b: Tensor):
        if len(self.selected) != self.a.shape[0] or parent_a.shape[0] != parent_b.shape[1]:
            raise ValueError("local/global pair mismatch")
        next_a, next_b = parent_a.clone(), parent_b.clone()
        next_a[list(self.selected), :] = self.a.detach()
        next_b[:, list(self.selected)] = self.b.detach()
        return next_a, next_b


def aggregate_contributors(parent: Mapping[str, Tensor],
                           children: Mapping[int, Mapping[str, Tensor]],
                           selections: Mapping[int, Mapping[str, tuple[int, ...]]],
                           sample_weights: Mapping[int, int]) -> tuple[dict[str, Tensor], dict]:
    """Paper rank-wise contributor mean; head uses common sample weights."""
    if not children or set(children) != set(selections) or set(children) != set(sample_weights):
        raise ValueError("Fed-PLoRA cohort fields differ")
    if any(w <= 0 for w in sample_weights.values()):
        raise ValueError("positive sample weights required")
    modules = sorted(name[:-7] for name in parent if name.endswith(".lora_a"))
    if not modules or any(f"{module}.lora_b" not in parent for module in modules):
        raise ValueError("missing PLoRA factor pair")
    result = {name: value.clone() for name, value in parent.items()}
    counts = {}
    for cid, child in children.items():
        if set(child) != set(parent) or set(selections[cid]) != set(modules):
            raise ValueError("client state or projection masks incomplete")
    for module in modules:
        a_name, b_name = f"{module}.lora_a", f"{module}.lora_b"
        rank = parent[a_name].shape[0]
        counts[module] = []
        for j in range(rank):
            contributors = [cid for cid in children if j in selections[cid][module]]
            counts[module].append(len(contributors))
            for cid, child in children.items():
                if cid not in contributors and (not torch.equal(child[a_name][j], parent[a_name][j]) or
                                                not torch.equal(child[b_name][:, j], parent[b_name][:, j])):
                    raise ValueError("unselected PLoRA pair changed locally")
            if contributors:
                result[a_name][j] = torch.stack([children[cid][a_name][j].double()
                                                  for cid in contributors]).mean(0).to(parent[a_name].dtype)
                result[b_name][:, j] = torch.stack([children[cid][b_name][:, j].double()
                                                    for cid in contributors]).mean(0).to(parent[b_name].dtype)
    total = sum(sample_weights.values())
    for name in parent:
        if name.startswith("classifier."):
            result[name] = sum(children[cid][name].double() * (sample_weights[cid]/total)
                               for cid in children).to(parent[name].dtype)
    return result, {"contributors_per_projection_rank": counts,
                    "zero_contributor_positions": sum(v == 0 for row in counts.values() for v in row)}
