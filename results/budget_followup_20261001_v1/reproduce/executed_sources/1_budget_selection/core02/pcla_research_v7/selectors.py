"""V7 support selection at the independent (module, rank, factor) unit."""
from __future__ import annotations

import re
from typing import Mapping

import torch
from torch import Tensor


def module_order(names) -> list[str]:
    """Stable network order, including layer.2 before layer.10."""
    def key(name: str):
        return tuple((0, int(part)) if part.isdigit() else (1, part)
                     for part in re.split(r"(\d+)", name))
    return sorted(names, key=key)


def normalized_scores(factors: Mapping[str, tuple[Tensor, Tensor]],
                      gradients: Mapping[str, Tensor], factor: str) -> dict[str, Tensor]:
    """Fresh eval-probe scores. Factor pairs are (B, A); no EMA or UCB state."""
    if factor not in ("a", "b") or factors.keys() != gradients.keys():
        raise ValueError("factor/gradient modules mismatch")
    result = {}
    for name, (b, a) in factors.items():
        g = gradients[name].detach().cpu().double()
        partner = (b if factor == "a" else a).detach().cpu().double()
        numerator = torch.linalg.vector_norm(g, dim=1 if factor == "a" else 0)
        denominator = torch.linalg.vector_norm(partner, dim=0 if factor == "a" else 1)
        if numerator.shape != denominator.shape or not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
            raise ValueError(f"invalid gradient or partner: {name}")
        nz = denominator[denominator > 0]
        score = torch.zeros_like(numerator)
        if len(nz):
            epsilon = float(torch.median(nz)) * 1e-12
            score = numerator / denominator.clamp_min(epsilon)
        result[name] = score
    return result


def energy_scores(factors: Mapping[str, tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    """Squared Frobenius energy of each rank-one effective component."""
    result = {}
    for name, (b, a) in factors.items():
        bb, aa = b.detach().cpu().double(), a.detach().cpu().double()
        if bb.ndim != 2 or aa.ndim != 2 or bb.shape[1] != aa.shape[0]:
            raise ValueError(f"invalid factor pair: {name}")
        result[name] = bb.square().sum(0) * aa.square().sum(1)
    return result


def _check(scores: Mapping[str, Tensor], budget: int) -> tuple[list[str], int]:
    names = module_order(scores)
    if not names or not isinstance(budget, int) or budget < 0:
        raise ValueError("invalid scores or budget")
    ranks = {scores[n].numel() for n in names}
    if len(ranks) != 1 or any(scores[n].ndim != 1 or not torch.isfinite(scores[n]).all() for n in names):
        raise ValueError("scores must be finite equal-rank vectors")
    rank = ranks.pop()
    if budget > len(names) * rank:
        raise ValueError("budget exceeds available positions")
    return names, rank


def topk_mask(scores: Mapping[str, Tensor], budget: int) -> dict[str, Tensor]:
    names, rank = _check(scores, budget)
    mask = {n: torch.zeros(rank, dtype=torch.bool) for n in names}
    order_index = {name: i for i, name in enumerate(names)}
    order = sorted(((float(scores[n][j]), n, j) for n in names for j in range(rank)),
                   key=lambda v: (-v[0], order_index[v[1]], v[2]))
    for _, n, j in order[:budget]:
        mask[n][j] = True
    return mask


def cyclic_mask(module_names: list[str], rank: int, budget: int, visits: int,
                *, factor: str = "b", offsets: Mapping[str, int] | None = None) -> dict[str, Tensor]:
    """Equal module quotas; positions advance by participation count.

    Q/V independence is supplied through explicit module offsets. No global
    round parity enters the rank cursor, so each participating client cycles.
    """
    names = module_order(module_names)
    if len(set(names)) != len(names) or not names or rank < 1 or visits < 0 or factor not in ("a", "b"):
        raise ValueError("invalid cyclic support arguments")
    if budget < 0 or budget > len(names) * rank:
        raise ValueError("invalid cyclic budget")
    per, remainder = divmod(budget, len(names))
    mask = {n: torch.zeros(rank, dtype=torch.bool) for n in names}
    # Rotate the remainder across modules, so unequal K does not always favor Q.
    for i, name in enumerate(names):
        count = per + int((i - visits) % len(names) < remainder)
        offset = (offsets or {}).get(name, i)
        start = (int(offset) + visits * max(count, 1)) % rank
        for j in range(count):
            mask[name][(start + j) % rank] = True
    return mask


def layer_mask(scores: Mapping[str, Tensor], budget: int,
               qv_pairs: Mapping[int, tuple[str, str]]) -> dict[str, Tensor]:
    """Select whole Q/V layers using the mean of all independent scores."""
    names, rank = _check(scores, budget)
    if budget % (2 * rank):
        raise ValueError("whole-layer selection requires a multiple of 2*rank")
    grouped = {n for pair in qv_pairs.values() for n in pair}
    if grouped != set(names) or len(grouped) != 2 * len(qv_pairs):
        raise ValueError("qv_pairs must partition modules")
    ordered = sorted(qv_pairs, key=lambda layer: (-float(torch.cat([scores[n] for n in qv_pairs[layer]]).mean()), layer))
    picked = set(ordered[:budget // (2 * rank)])
    return {n: torch.full((rank,), layer in picked, dtype=torch.bool)
            for layer, pair in qv_pairs.items() for n in pair}


def tied_local_mask(scores: Mapping[str, Tensor], budget: int,
                    qv_pairs: Mapping[int, tuple[str, str]]) -> dict[str, Tensor]:
    """Explicit Q/V-tied ablation: sum each pair's scores, spend two units."""
    names, rank = _check(scores, budget)
    if budget % 2 or {n for pair in qv_pairs.values() for n in pair} != set(names):
        raise ValueError("tied support requires paired modules and even budget")
    candidates = sorted(((float(scores[q][j] + scores[v][j]), layer, j)
                         for layer, (q, v) in qv_pairs.items() for j in range(rank)),
                        key=lambda x: (-x[0], x[1], x[2]))
    mask = {n: torch.zeros(rank, dtype=torch.bool) for n in names}
    for _, layer, j in candidates[:budget // 2]:
        q, v = qv_pairs[layer]
        mask[q][j] = mask[v][j] = True
    return mask


def both_half_masks(scores_a: Mapping[str, Tensor], scores_b: Mapping[str, Tensor],
                    budget: int, *, policy: str, visits: int = 0,
                    offsets: Mapping[str, int] | None = None) -> dict[str, dict[str, Tensor]]:
    """Same total scalar-step budget: half positions per factor."""
    if budget % 2 or scores_a.keys() != scores_b.keys():
        raise ValueError("both-half needs an even budget and identical modules")
    half = budget // 2
    if policy == "local":
        return {"a": topk_mask(scores_a, half), "b": topk_mask(scores_b, half)}
    if policy == "cyclic":
        names, rank = _check(scores_a, half)
        return {"a": cyclic_mask(names, rank, half, visits, factor="a", offsets=offsets),
                "b": cyclic_mask(names, rank, half, visits, factor="b", offsets=offsets)}
    raise ValueError("both-half supports local or cyclic")


def equalized_quotas(quotas: Mapping[int, int], round_id: int) -> dict[int, int]:
    """Preserve cohort K; rotate remainder over sorted client IDs."""
    if not quotas or round_id < 1 or any(not isinstance(v, int) or v < 0 for v in quotas.values()):
        raise ValueError("invalid quota or round")
    ids = sorted(quotas)
    low, rem = divmod(sum(quotas.values()), len(ids))
    out = {cid: low for cid in ids}
    for j in range(rem):
        out[ids[(round_id - 1 + j) % len(ids)]] += 1
    return out
