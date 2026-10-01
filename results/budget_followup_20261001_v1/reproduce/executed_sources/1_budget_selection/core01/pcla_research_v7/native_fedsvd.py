"""FedSVD adapter from the paper's Algorithm 1 and official reinit_lora.

This module operates on the v7 adapter/head state. The runtime owns training,
optimizer construction, wire transmission, and checkpointing. FedSVD trains B
only; A is reset by server SVD before the next participating cohort.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Mapping

import torch
from torch import Tensor

OFFICIAL_REPO = "https://github.com/seanie12/fed-svd"
OFFICIAL_COMMIT = "e73398ba34b6b269c3bf76ed9d74906761fc1a4d"
OFFICIAL_METHOD = "misc/utils.py:reinit_lora"


def refresh_due(round_id: int, period: int) -> bool:
    if round_id < 1 or period not in (1, 5):
        raise ValueError("FedSVD round >=1 and protocol period 1 or 5 required")
    return round_id >= 2 and (round_id - 2) % period == 0


def _pairs(state: Mapping[str, Tensor]) -> list[tuple[str, str]]:
    names = sorted(n for n in state if n.endswith(".lora_a"))
    if not names or any(n[:-1] + "b" not in state for n in names):
        raise ValueError("FedSVD requires complete LoRA A/B factor pairs")
    if len(names) != sum(n.endswith(".lora_b") for n in state):
        raise ValueError("unpaired LoRA B factor")
    return [(name, name[:-1] + "b") for name in names]


def svd_pair(b: Tensor, a: Tensor, *, relative_tol=1e-5) -> tuple[Tensor, Tensor, dict]:
    """Faithful full-matrix SVD of B@A, in the tensor's deployed dtype.

    It deliberately follows the author's full `torch.linalg.svd` rather than
    a thin QR surrogate, including its rank-deficient null-basis behavior.
    """
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("incompatible LoRA factors")
    if b.dtype != a.dtype or b.device != a.device or not b.is_floating_point():
        raise ValueError("factors must share floating dtype/device")
    if not torch.isfinite(b).all() or not torch.isfinite(a).all():
        raise ValueError("nonfinite factor")
    rank = a.shape[0]
    if rank > min(b.shape[0], a.shape[1]):
        raise ValueError("rank exceeds matrix dimensions")
    product = b @ a
    u, singular, vh = torch.linalg.svd(product, full_matrices=False)
    next_b = u[:, :rank] * singular[:rank].unsqueeze(0)
    next_a = vh[:rank, :]
    reconstructed = next_b @ next_a
    absolute_error = torch.linalg.matrix_norm(reconstructed.double()-product.double()).item()
    denominator = max(torch.linalg.matrix_norm(product.double()).item(), 1e-12)
    relative_error = absolute_error / denominator
    if relative_error > relative_tol and absolute_error > 1e-6:
        raise ArithmeticError(f"FedSVD SVD changed effective matrix: relative={relative_error}")
    identity = hashlib.sha256(product.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return next_b, next_a, {"rank": rank, "product_sha256": identity,
                            "absolute_error": absolute_error,
                            "relative_error": relative_error,
                            "singular_values": singular[:rank].detach().cpu().tolist()}


def refresh_state(state: Mapping[str, Tensor], round_id: int, period: int,
                  probe: Callable[[Mapping[str, Tensor]], Tensor] | None = None,
                  logit_tol=1e-4) -> tuple[dict[str, Tensor], dict]:
    """Return a new parent state and a per-module preservation audit."""
    if not refresh_due(round_id, period):
        return {name: value.clone() for name, value in state.items()}, {"refreshed": False}
    before_logits = probe(state).detach().double() if probe is not None else None
    result = {name: value.clone() for name, value in state.items()}
    modules = {}
    for a_name, b_name in _pairs(state):
        next_b, next_a, audit = svd_pair(state[b_name], state[a_name])
        result[a_name], result[b_name] = next_a, next_b
        modules[a_name.rsplit(".", 1)[0]] = audit
    logit_error = None
    if before_logits is not None:
        logit_error = (probe(result).detach().double()-before_logits).abs().max().item()
        if logit_error > logit_tol:
            raise ArithmeticError(f"FedSVD refresh changed fixed eval logits by {logit_error}")
    return result, {"refreshed": True, "round": round_id, "period": period,
                    "modules": modules, "max_abs_eval_logit_error": logit_error}


def trainable_names(state: Mapping[str, Tensor]) -> set[str]:
    _pairs(state)
    return {name for name in state if name.endswith(".lora_b") or name.startswith("classifier.")}


def aggregate_b_head(parent: Mapping[str, Tensor], children: list[Mapping[str, Tensor]],
                     weights: list[int]) -> dict[str, Tensor]:
    """Algorithm 1 sample-weighted mean of B; head follows shared task policy."""
    if not children or len(children) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("invalid client aggregation weights")
    mutable = trainable_names(parent)
    total = sum(weights)
    result = {name: value.clone() for name, value in parent.items()}
    for child in children:
        if set(child) != set(parent):
            raise ValueError("client state keys differ")
        for name in parent:
            if name not in mutable and not torch.equal(child[name], parent[name]):
                raise ValueError(f"FedSVD frozen factor changed locally: {name}")
    for name in mutable:
        delta = sum((child[name].double()-parent[name].double()) * (weight / total)
                    for child, weight in zip(children, weights))
        result[name] = (parent[name].double()+delta).to(parent[name].dtype)
    return result
