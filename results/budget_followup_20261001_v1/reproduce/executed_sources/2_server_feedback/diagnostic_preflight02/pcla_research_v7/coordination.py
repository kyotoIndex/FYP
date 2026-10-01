"""V7 T/JL/JQ math with explicit same-parent probe inputs.

The trainer must supply real eval-mode two-batch gradients for every active
factor. This module does not synthesize a probe or read diagnostic samples.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .selectors import cyclic_mask, module_order, normalized_scores, topk_mask


@dataclass(frozen=True)
class ClientPayload:
    client_id: int
    gradient_sketches: dict[str, Tensor]  # h x rank, float32 wire tensor
    update_sketches: dict[str, Tensor]    # h x rank, float32 wire tensor
    local_scores: dict[str, Tensor]       # rank, float32 wire tensor
    parent_version: int


@dataclass(frozen=True)
class CoordinationResult:
    masks: dict[int, dict[str, Tensor]]
    local_masks: dict[int, dict[str, Tensor]]
    cyclic_masks: dict[int, dict[str, Tensor]]
    jl_masks: dict[int, dict[str, Tensor]]
    lambda_value: float
    objective_initial: float | None
    objective_final: float | None
    choices: dict[int, str] | None


def coordination_due(round_id: int, period: int) -> bool:
    if round_id < 1 or period < 1:
        raise ValueError("round and period must be positive")
    return (round_id - 1) % period == 0


def _keyed_seed(*parts: object) -> int:
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def projection(seed: int, round_id: int, module: str, factor: str,
               width: int, h: int = 32) -> Tensor:
    """Shared Rademacher h x width matrix, independent by keyed round/module/factor."""
    if seed < 0 or round_id < 1 or not module or factor not in ("a", "b") or width < 1 or h < 1:
        raise ValueError("invalid projection key or shape")
    rng = np.random.default_rng(_keyed_seed(seed, round_id, module, factor))
    # Same basic signs for h32/h128; each h has its own 1/sqrt(h) normalization.
    signs = rng.integers(0, 2, size=(h, width), dtype=np.int8)
    return torch.from_numpy((2 * signs.astype(np.float64) - 1) / math.sqrt(h))


def sketch(matrix: Tensor, projection_matrix: Tensor, factor: str) -> Tensor:
    """B: S_out G_B. A: (G_A S_in.T).T. Returns h x r float32."""
    g = matrix.detach().to(device="cpu", dtype=torch.float64)
    s = projection_matrix.detach().to(device="cpu", dtype=torch.float64)
    if g.ndim != 2 or s.ndim != 2 or factor not in ("a", "b"):
        raise ValueError("invalid sketch inputs")
    if factor == "b":
        if s.shape[1] != g.shape[0]:
            raise ValueError("B projection width mismatch")
        out = s @ g
    else:
        if s.shape[1] != g.shape[1]:
            raise ValueError("A projection width mismatch")
        out = (g @ s.T).T
    if not torch.isfinite(out).all():
        raise ValueError("nonfinite sketch")
    return out.float()


def virtual_adamw_step(parameter: Tensor, gradient: Tensor, learning_rate: float,
                       *, eps: float = 1e-8, weight_decay: float = .01) -> Tensor:
    """Fresh AdamW step, zero moments, including decoupled selected weight decay."""
    if (parameter.shape != gradient.shape or not all(math.isfinite(x) for x in (learning_rate, eps, weight_decay))
            or learning_rate <= 0 or eps <= 0 or weight_decay < 0):
        raise ValueError("invalid virtual AdamW inputs")
    p = parameter.detach().to(device="cpu", dtype=torch.float64)
    g = gradient.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(p).all() or not torch.isfinite(g).all():
        raise ValueError("nonfinite virtual AdamW inputs")
    # m_hat=g and v_hat=g**2 after bias correction at step one, for any valid betas.
    return (-learning_rate * g / (g.abs() + eps) - learning_rate * weight_decay * p).float()


def client_payload(client_id: int, parent_version: int,
                   factors: Mapping[str, tuple[Tensor, Tensor]],
                   gradients: Mapping[str, Tensor], factor: str,
                   learning_rate: float, seed: int, round_id: int,
                   *, h: int = 32, eps: float = 1e-8,
                   weight_decay: float = .01) -> ClientPayload:
    """Build the two wire sketches and fresh scores from an actual local probe."""
    if client_id < 0 or parent_version < 0 or factors.keys() != gradients.keys() or factor not in ("a", "b"):
        raise ValueError("invalid client probe metadata")
    scores = normalized_scores(factors, gradients, factor)
    c, t = {}, {}
    for name, (b, a) in factors.items():
        parameter = a if factor == "a" else b
        g = gradients[name]
        if g.shape != parameter.shape:
            raise ValueError(f"full gradient required for {name}")
        s = projection(seed, round_id, name, factor, parameter.shape[1 if factor == "a" else 0], h)
        c[name] = sketch(g, s, factor)
        t[name] = sketch(virtual_adamw_step(parameter, g, learning_rate,
                                            eps=eps, weight_decay=weight_decay), s, factor)
    return ClientPayload(client_id, c, t,
                         {name: value.float() for name, value in scores.items()}, parent_version)


def _weighted_sum(payloads: Sequence[ClientPayload], weights: Mapping[int, float],
                  field: str, module: str) -> Tensor:
    return sum((float(weights[p.client_id]) * getattr(p, field)[module].double()
                for p in payloads), torch.zeros_like(getattr(payloads[0], field)[module], dtype=torch.float64))


def aggregate_gradient_sketches(payloads: Sequence[ClientPayload],
                                weights: Mapping[int, float],
                                *, expected_parent_version: int | None = None) -> dict[str, Tensor]:
    """T uses this same-cohort Cbar before rotation; no stale sketch reuse."""
    _validate_payloads(payloads, weights, expected_parent_version)
    return {name: _weighted_sum(payloads, weights, "gradient_sketches", name)
            for name in payloads[0].gradient_sketches}


def _validate_payloads(payloads: Sequence[ClientPayload], weights: Mapping[int, float],
                       expected_parent_version: int | None = None) -> None:
    if not payloads or len({p.client_id for p in payloads}) != len(payloads):
        raise ValueError("empty or duplicate cohort")
    ids = {p.client_id for p in payloads}
    versions = {p.parent_version for p in payloads}
    if ids != set(weights) or len(versions) != 1 or any(not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1., abs_tol=1e-12):
        raise ValueError("cohort weight or parent-version mismatch")
    if expected_parent_version is not None and versions != {expected_parent_version}:
        raise ValueError("stale parent version")
    reference = payloads[0]
    names = set(reference.gradient_sketches)
    if not names or names != set(reference.update_sketches) or names != set(reference.local_scores):
        raise ValueError("incomplete payload")
    for p in payloads:
        if names != set(p.gradient_sketches) or names != set(p.update_sketches) or names != set(p.local_scores):
            raise ValueError("module mismatch")
        for name in names:
            c, t, score = p.gradient_sketches[name], p.update_sketches[name], p.local_scores[name]
            if c.ndim != 2 or t.shape != c.shape or score.shape != (c.shape[1],):
                raise ValueError("sketch or score shape mismatch")
            if not torch.isfinite(c).all() or not torch.isfinite(t).all() or not torch.isfinite(score).all():
                raise ValueError("nonfinite sketch or score")


def _masked_update(payload: ClientPayload, mask: Mapping[str, Tensor],
                   weight: float) -> dict[str, Tensor]:
    return {name: weight * payload.update_sketches[name].double() * mask[name].double()[None, :]
            for name in mask}


def coordinate_masks(payloads: Sequence[ClientPayload], weights: Mapping[int, float],
                     budgets: Mapping[int, int], factors: Mapping[str, tuple[Tensor, Tensor]],
                     factor: str, visits: Mapping[int, int], method: str,
                     *, xi: float = 1., offsets: Mapping[str, int] | None = None,
                     offsets_by_client: Mapping[int, Mapping[str, int]] | None = None,
                     expected_parent_version: int | None = None) -> CoordinationResult:
    """Select JL or finite-menu JQ support before any actual local training.

    Only the sketches, local scores, and public parent factors reach this
    server-side function. The trainer must supply the live probe via
    ``client_payload`` and then train all selected clients from that parent.
    """
    _validate_payloads(payloads, weights, expected_parent_version)
    if method not in ("JL", "JQ") or factor not in ("a", "b") or not math.isfinite(xi) or xi < 0:
        raise ValueError("invalid coordinated method")
    ids = sorted(p.client_id for p in payloads)
    by_id = {p.client_id: p for p in payloads}
    names = module_order(payloads[0].gradient_sketches)
    if set(names) != set(factors) or set(ids) != set(budgets) or set(ids) != set(visits):
        raise ValueError("factor, budget, or visit map mismatch")
    if offsets_by_client is not None and set(offsets_by_client) != set(ids):
        raise ValueError("cyclic offset cohort mismatch")
    rank = payloads[0].gradient_sketches[names[0]].shape[1]
    cbar = aggregate_gradient_sketches(payloads, weights)
    local, cyclic, jl = {}, {}, {}
    for cid in ids:
        p = by_id[cid]
        budget = budgets[cid]
        local[cid] = topk_mask(p.local_scores, budget)
        cyclic[cid] = cyclic_mask(names, rank, budget, visits[cid], factor=factor,
                                  offsets=offsets_by_client[cid] if offsets_by_client is not None else offsets)
        jl_score = {n: -float(weights[cid]) * (cbar[n] * p.update_sketches[n].double()).sum(0)
                    for n in names}
        jl[cid] = topk_mask(jl_score, budget)
    if method == "JL":
        return CoordinationResult(jl, local, cyclic, jl, 0., None, None, None)

    # H = A A.T for B, B.T B for A. Its dimensions and units are per module.
    gram = {}
    for name, (b, a) in factors.items():
        partner = (a if factor == "b" else b).detach().cpu().double()
        hmat = partner @ partner.T if factor == "b" else partner.T @ partner
        if hmat.shape != (rank, rank):
            raise ValueError("Gram rank mismatch")
        gram[name] = hmat

    def total(masks: Mapping[int, Mapping[str, Tensor]]) -> tuple[float, float]:
        linear, quadratic = 0., 0.
        for name in names:
            summed = sum((float(weights[cid]) * by_id[cid].update_sketches[name].double()
                          * masks[cid][name].double()[None, :] for cid in ids),
                         torch.zeros_like(cbar[name]))
            linear -= float((cbar[name] * summed).sum())
            quadratic += float(torch.einsum("hr,rs,hs->", summed, gram[name], summed))
        return linear, max(0., quadratic)

    reference_linear, reference_quadratic = total(cyclic)
    numerator = xi * max(0., reference_linear)
    curvature = numerator / max(reference_quadratic, np.finfo(np.float64).tiny) if reference_quadratic else 0.
    if not math.isfinite(curvature):
        raise FloatingPointError("nonfinite JQ curvature")
    menu = {cid: {"local": local[cid], "cyclic": cyclic[cid], "JL": jl[cid]} for cid in ids}
    choices = {cid: "JL" for cid in ids}

    def objective() -> float:
        linear, quad = total({cid: menu[cid][choices[cid]] for cid in ids})
        return linear - .5 * curvature * quad

    initial = objective()
    current = initial
    for _ in range(2):
        for cid in ids:
            original = choices[cid]
            picked = original
            best = current
            for option in ("local", "cyclic", "JL"):
                if option == original:
                    continue
                choices[cid] = option
                value = objective()
                if value > best + 1e-12:
                    picked, best = option, value
            choices[cid] = picked
            current = best
    selected = {cid: menu[cid][choices[cid]] for cid in ids}
    return CoordinationResult(selected, local, cyclic, jl, curvature, initial, current, choices)
