"""Torch-only numerical primitives for controlled federated LoRA experiments.

Parameter masks restrict updates, not the forward function. This is logical
sparsity: dense kernels and dense Adam state remain and are reported as such.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

BASE_COMMIT = "a88f65d437fbb893690aadcade194b1f221d46b9"
LEGACY_BLOB = "ab6931b79036721dcd368bb85ca08af476d6c639"
SCORE_KINDS = ("legacy", "normalized_projection", "gate", "gradient_norm")


def keyed_seed(*parts: object) -> int:
    """Stable across processes; Python hash() is intentionally not used."""
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def tensor_digest(items: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(items):
        t = items[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(t.dtype), tuple(t.shape))).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class BudgetLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1 or not 0 <= dropout < 1:
            raise ValueError("rank must be positive and dropout in [0,1)")
        self.base, self.rank = base, rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.register_buffer("update_mask", base.weight.new_ones(rank))
        self.factor = "none"
        self.forward_mode = "active_frozen"
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        # Detach PARAMETERS, not branch outputs: frozen weights must still
        # propagate gradients into earlier trainable Transformer layers.
        a, b = self.lora_a.detach(), self.lora_b.detach()
        if self.factor in ("a", "both"):
            a = a + (self.lora_a - a) * self.update_mask[:, None]
        if self.factor in ("b", "both"):
            b = b + (self.lora_b - b) * self.update_mask[None, :]
        hidden = F.linear(self.dropout(x), a)
        if self.forward_mode == "mask_out":
            hidden = hidden * self.update_mask
        return self.base(x) + F.linear(hidden, b) * self.scaling


def lora_modules(model: nn.Module) -> dict[str, BudgetLoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, BudgetLoRALinear)}


def adapter_state(model: nn.Module) -> dict[str, Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if ".lora_a" in n or ".lora_b" in n or n.startswith("classifier.")}


def load_adapter(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = set(adapter_state_keys(model))
    if set(state) != expected:
        raise ValueError("adapter keys do not match model")
    with torch.no_grad():
        for n, value in state.items():
            if value.shape != parameters[n].shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid adapter tensor: {n}")
            parameters[n].copy_(value)


def adapter_state_keys(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters()
            if ".lora_a" in n or ".lora_b" in n or n.startswith("classifier.")]


def check_support(support: Tensor, shape: tuple[int, int], budget: int | None = None) -> None:
    if tuple(support.shape) != shape or support.dtype != torch.bool:
        raise ValueError("support must be a boolean [layers,rank] tensor")
    if budget is not None and int(support.sum()) != budget:
        raise ValueError(f"support uses {int(support.sum())}, expected {budget} credits")


def configure(model: nn.Module, layout: Mapping[str, int], support: Tensor,
              factor: str, *, forward_mode: str = "active_frozen", head: bool = True
              ) -> dict[str, Tensor]:
    if factor not in ("a", "b", "both", "none"):
        raise ValueError("unknown trained factor")
    if forward_mode not in ("active_frozen", "mask_out"):
        raise ValueError("unknown forward semantics")
    mods = lora_modules(model)
    if set(layout) != set(mods):
        raise ValueError("layout must name every LoRA module exactly once")
    check_support(support, (max(layout.values()) + 1, next(iter(mods.values())).rank))
    for p in model.parameters():
        p.requires_grad_(False)
    masks: dict[str, Tensor] = {}
    for name, module in mods.items():
        m = support[layout[name]].to(module.lora_a.device)
        module.update_mask.copy_(m)
        module.factor, module.forward_mode = factor, forward_mode
        for f in ("a", "b"):
            p = getattr(module, "lora_" + f)
            if factor in (f, "both") and bool(m.any()):
                p.requires_grad_(True)
                masks[name + ".lora_" + f] = (m[:, None] if f == "a" else m[None, :]).expand_as(p)
    if head:
        for name, p in model.classifier.named_parameters():
            p.requires_grad_(True)
            masks["classifier." + name] = torch.ones_like(p, dtype=torch.bool)
    return masks


class FrozenSliceGuard:
    """Zero gradients alone do NOT stop AdamW decay or existing momentum.

    This guard restores inactive entries exactly after optimizer.step() and
    clears their optimizer tensors. It does not claim sparse optimizer memory.
    """
    def __init__(self, model: nn.Module, masks: Mapping[str, Tensor]):
        params = dict(model.named_parameters())
        self.items = [(params[n], m.to(params[n].device), params[n].detach().clone())
                      for n, m in masks.items() if not bool(m.all())]

    @torch.no_grad()
    def restore(self, optimizer: torch.optim.Optimizer) -> None:
        for p, mask, original in self.items:
            p[~mask] = original[~mask]
            for state in optimizer.state.get(p, {}).values():
                if torch.is_tensor(state) and state.shape == p.shape:
                    state.masked_fill_(~mask, 0)


def component_scores(model: nn.Module, layout: Mapping[str, int], factor: str,
                     kind: str) -> Tensor:
    if factor not in ("a", "b") or kind not in SCORE_KINDS:
        raise ValueError("invalid factor or score")
    mods = lora_modules(model)
    out = torch.zeros(max(layout.values()) + 1, next(iter(mods.values())).rank, dtype=torch.float64)
    for name, module in mods.items():
        p = getattr(module, "lora_" + factor)
        if p.grad is None:
            raise RuntimeError(f"missing probe gradient: {name}/{factor}")
        grad = p.grad.detach().double()
        partner = (module.lora_a if factor == "b" else module.lora_b).detach().double()
        gnorm = torch.linalg.vector_norm(grad, dim=0 if factor == "b" else 1)
        pnorm = torch.linalg.vector_norm(partner, dim=1 if factor == "b" else 0)
        if kind == "legacy":
            score = gnorm * pnorm
        elif kind == "gradient_norm":
            score = gnorm
        elif kind == "gate":
            score = (grad * p.detach().double()).sum(dim=0 if factor == "b" else 1).abs()
        else:
            # Exact zero means an unavailable tangent direction. Adding an
            # absolute epsilon would only be approximately scale invariant.
            score = torch.zeros_like(gnorm)
            nz = pnorm > 0
            score[nz] = gnorm[nz] / pnorm[nz]
        if not torch.isfinite(score).all():
            raise FloatingPointError("nonfinite sensitivity; do not silently rank it")
        # Q/V are a fixed-cost bundle. Sum their scalar scores, not their vectors.
        out[layout[name]] += score.cpu()
    return out


def rank_order(values: Tensor) -> list[int]:
    x = values.detach().cpu().flatten().tolist()
    if not all(math.isfinite(v) for v in x):
        raise ValueError("nonfinite allocation score")
    return sorted(range(len(x)), key=lambda i: (-x[i], i))


def hierarchical(scores: Tensor, q: int, m: int) -> Tensor:
    layers, rank = scores.shape
    if not 0 < q <= layers or not 0 < m <= rank:
        raise ValueError("invalid layer or rank budget")
    selected = torch.zeros_like(scores, dtype=torch.bool, device="cpu")
    layer_values = torch.tensor([sum(scores[l, rank_order(scores[l])[:m]]).item() / m
                                 for l in range(layers)])
    active = rank_order(layer_values)[:q]
    for l in active:
        selected[l, rank_order(scores[l])[0]] = True
    candidates = [(float(scores[l, j]), l, j) for l in active for j in range(rank)
                  if not selected[l, j]]
    for _, l, j in sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[:q*m-q]:
        selected[l, j] = True
    check_support(selected, (layers, rank), q*m)
    return selected


def allocate(policy: str, scores: Tensor, q: int, m: int, seed: int,
             *, donor: Tensor | None = None, visit: int = 0) -> Tensor:
    scores = scores.detach().cpu()
    layers, rank = scores.shape
    if not 0 < q <= layers or not 0 < m <= rank:
        raise ValueError("invalid budget")
    budget = q*m
    mask = torch.zeros(layers, rank, dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)
    if policy == "pcla":
        return hierarchical(scores, q, m)
    if policy in ("layer_only", "deep_layer"):
        if budget % rank:
            raise ValueError("whole-layer baseline cannot exactly spend this round's budget")
        chosen = (rank_order(scores.sum(1)) if policy == "layer_only" else list(reversed(range(layers))))
        mask[chosen[:budget//rank]] = True
    elif policy == "random_layer":
        if budget % rank:
            raise ValueError("whole-layer baseline cannot exactly spend this round's budget")
        mask[torch.randperm(layers, generator=generator)[:budget//rank]] = True
    elif policy == "uniform_cyclic":
        per_layer, remainder = divmod(budget, layers)
        for l in range(layers):
            count = per_layer + int((l - visit - seed) % layers < remainder)
            start = (seed + l + visit*max(per_layer, 1)) % rank
            mask[l, [(start+j) % rank for j in range(count)]] = True
    elif policy == "random":
        for l in torch.randperm(layers, generator=generator)[:q].tolist():
            mask[l, torch.randperm(rank, generator=generator)[:m]] = True
    elif policy == "replay_random_rank":
        if donor is None:
            raise ValueError("trajectory-conditioned control requires a donor trace")
        check_support(donor, (layers, rank), budget)
        for l in range(layers):
            count = int(donor[l].sum())
            mask[l, torch.randperm(rank, generator=generator)[:count]] = True
    elif policy == "cyclic":
        # visit is PER FACTOR and client; global round parity can starve ranks.
        for l in [(seed + visit*q + j) % layers for j in range(q)]:
            mask[l, [(seed + l + visit*m + j) % rank for j in range(m)]] = True
    elif policy == "flat_topk":
        mask.view(-1)[rank_order(scores)[:budget]] = True
    else:
        raise ValueError(f"unknown policy {policy}")
    check_support(mask, (layers, rank), budget)
    return mask


@dataclass
class UtilityState:
    """Separate A/B state; no claim of probabilistic UCB guarantees."""
    importance: Tensor | None = None
    counts: Tensor | None = None
    last_selected: Tensor | None = None
    last_refresh: int = -1
    visits: int = 0

    def refresh(self, raw: Tensor, round_id: int, decay: float) -> None:
        if not 0 <= decay < 1 or (raw < 0).any() or not torch.isfinite(raw).all():
            raise ValueError("invalid sensitivity/EMA input")
        raw = raw.detach().cpu().double()
        norm = raw / raw.sum() if float(raw.sum()) > 0 else torch.full_like(raw, 1/raw.numel())
        self.importance = norm if self.importance is None else decay*self.importance+(1-decay)*norm
        self.last_refresh = round_id
        if self.counts is None:
            self.counts = torch.zeros_like(raw)
            self.last_selected = torch.full_like(raw, -1)

    def score(self, round_id: int, beta: float, gamma: float) -> Tensor:
        if self.importance is None:
            raise RuntimeError("state has not been probed")
        x = self.importance
        std = x.std(unbiased=False)
        z = (x-x.mean())/std if float(std) > 1e-12 else torch.zeros_like(x)
        age = torch.where(self.last_selected < 0, round_id,
                          round_id-self.last_selected)
        return z + beta*torch.sqrt(math.log(round_id+1)/(self.counts+1)) + gamma*(age/max(round_id-1,1)).clamp(max=1)

    def observe(self, support: Tensor, round_id: int) -> None:
        if self.counts is not None:
            self.counts += support
            self.last_selected[support] = round_id
        self.visits += 1


class DeltaAccumulator:
    """Streaming aggregation with explicit per-coordinate denominators.

    fixed: global += sum(n_k * masked_delta_k) / sum(n_k).
    contributors: divides each coordinate by its observed contributor weight.
    Neither generally debiases data-dependent deterministic support selection.
    """
    def __init__(self, base: Mapping[str, Tensor], total_weight: int,
                 mode: str = "fixed"):
        if total_weight <= 0 or mode not in ("fixed", "contributors"):
            raise ValueError("invalid aggregation")
        self.base = {n: v.detach().cpu().clone() for n, v in base.items()}
        self.sums = {n: torch.zeros_like(v, dtype=torch.float64) for n,v in self.base.items()}
        self.coverage = {n: torch.zeros_like(v, dtype=torch.float64) for n,v in self.base.items()}
        self.total_weight, self.mode, self.seen = total_weight, mode, 0

    def add(self, local: Mapping[str, Tensor], masks: Mapping[str, Tensor], weight: int) -> None:
        if weight <= 0 or self.seen+weight > self.total_weight or set(local) != set(self.base):
            raise ValueError("inconsistent client weights or adapter keys")
        for n in self.base.keys() - masks.keys():
            if not torch.equal(local[n], self.base[n]):
                raise RuntimeError(f"inactive tensor changed: {n}")
        for n, mask in masks.items():
            if n not in local or mask.shape != local[n].shape:
                raise ValueError("update mask mismatch")
            mask = mask.detach().cpu().bool()
            delta = local[n].double()-self.base[n].double()
            if not torch.isfinite(delta).all():
                raise FloatingPointError("invalid client update")
            if torch.count_nonzero(delta[~mask]):
                raise RuntimeError(f"frozen parameter changed: {n}")
            self.sums[n] += weight*delta*mask
            self.coverage[n] += weight*mask
        self.seen += weight

    def finish(self) -> dict[str, Tensor]:
        if self.seen != self.total_weight:
            raise RuntimeError("missing clients: refusing partial-round aggregate")
        result = {}
        for n, base in self.base.items():
            denom = self.total_weight if self.mode == "fixed" else self.coverage[n].clamp(min=1)
            result[n] = (base.double()+self.sums[n]/denom).to(base.dtype)
        return result
