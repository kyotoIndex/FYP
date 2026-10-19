"""Shared pieces: component allocation, block vectors, set-masked local training, aggregation, evaluation.

Terminology (design doc §4): a *block* is one LoRA component k of one adapter j (query/value of a layer),
represented as the concatenation [b_k (d_out), a_k (d_in)]. Strong clients train every block; a weak
client trains r components per adapter, assigned by the server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import ClientStream, Dataset, derived_seed
from .model import LoRALinear, TaskModel


# ---------------------------------------------------------------- adapters and blocks
def adapter_names(model: TaskModel) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def to_blocks(state: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    """[J, R, d_out + d_in] tensor of components (b column ++ a row) for every adapter."""
    rows = []
    for name in names:
        a, b = state[f"{name}.lora_a"], state[f"{name}.lora_b"]
        rows.append(torch.cat([b.t(), a], dim=1))
    return torch.stack(rows).float()


def from_blocks(blocks: torch.Tensor, names: list[str], template: dict[str, torch.Tensor]) -> dict:
    out = dict(template)
    for j, name in enumerate(names):
        d_out = template[f"{name}.lora_b"].shape[0]
        out[f"{name}.lora_b"] = blocks[j, :, :d_out].t().contiguous().to(template[f"{name}.lora_b"].dtype)
        out[f"{name}.lora_a"] = blocks[j, :, d_out:].contiguous().to(template[f"{name}.lora_a"].dtype)
    return out


def head_names(state: dict) -> list[str]:
    return sorted(n for n in state if "lora_" not in n)


# ---------------------------------------------------------------- allocation
def balanced_allocation(weak: list[int], n_adapters: int, R: int, r: int, seed: int, round_id: int):
    """Coverage-balanced dealing (design §5.1), independently per adapter.

    Returns sets[client] = list (per adapter) of r distinct components, and counts [J, R] of weak
    contributors per block. Inclusion probability of a weak client for block (j,k) is counts[j,k]/len(weak).
    """
    w = len(weak)
    sets = {c: [] for c in weak}
    counts = torch.zeros(n_adapters, R)
    if w == 0:
        return sets, counts
    # When r * w < R (tight upload budgets), some components get no weak contributor this round:
    # their weak inclusion probability is 0 and they are aggregated from strong clients only.
    for j in range(n_adapters):
        rng = random.Random(derived_seed(seed, "alloc", round_id, j))
        base, extra = divmod(r * w, R)
        bonus = set(rng.sample(range(R), extra))
        c = [base + (1 if k in bonus else 0) for k in range(R)]
        slots = [k for k in range(R) for _ in range(c[k])]
        order = list(weak)
        rng.shuffle(order)
        per = {cid: [] for cid in weak}
        for position, k in enumerate(slots):
            per[order[position % w]].append(k)
        for cid in weak:
            if len(set(per[cid])) != r:
                raise AssertionError("dealing produced a duplicate or wrong-size set")
            sets[cid].append(sorted(per[cid]))
        counts[j] = torch.tensor(c, dtype=torch.float32)
    return sets, counts


def hetero_allocation(weak: list[int], ranks: dict[int, int], n_adapters: int, R: int, seed: int, round_id: int):
    """Coverage-balanced dealing when weak clients have different ranks r_i (batch 2026-10-18).

    Per adapter, clients in random order each take the r_i components with the fewest contributors so far
    (random tie-break), so counts differ by at most one where possible. Inclusion probability of client i
    for any component is r_i / R by symmetry of the random order and tie-breaks.
    """
    sets = {c: [] for c in weak}
    counts = torch.zeros(n_adapters, R)
    for j in range(n_adapters):
        rng = random.Random(derived_seed(seed, "hetero-alloc", round_id, j))
        order = list(weak)
        rng.shuffle(order)
        c = [0] * R
        for cid in order:
            keys = sorted(range(R), key=lambda k: (c[k], rng.random()))
            chosen = sorted(keys[:ranks[cid]])
            for k in chosen:
                c[k] += 1
            sets[cid].append(chosen)
        counts[j] = torch.tensor(c, dtype=torch.float32)
    return sets, counts


def prefix_allocation(weak: list[int], n_adapters: int, R: int, r: int):
    sets = {c: [list(range(r)) for _ in range(n_adapters)] for c in weak}
    counts = torch.zeros(n_adapters, R)
    counts[:, :r] = len(weak)
    return sets, counts


def random_allocation(weak: list[int], n_adapters: int, R: int, r: int, seed: int, round_id: int):
    """Fed-PLoRA-style independent random selection: each weak client draws r components per adapter,
    with no coordination across clients (a component may get no weak contributor in a round)."""
    sets = {c: random_pairs(n_adapters, R, r, seed, "alloc-random", round_id, c) for c in weak}
    counts = torch.zeros(n_adapters, R)
    for c in weak:
        for j, chosen in enumerate(sets[c]):
            counts[j, chosen] += 1
    return sets, counts


def topk_sparsify(delta: torch.Tensor, fraction: float) -> torch.Tensor:
    """FLASC-style upload: keep the largest-magnitude `fraction` of the entries of the whole LoRA update
    (global top-k over all adapters and components; index overhead of the sparse encoding not counted)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    flat = delta.reshape(-1)
    k = max(1, int(round(fraction * flat.numel())))
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep[flat.abs().topk(k).indices] = True
    return torch.where(keep, flat, torch.zeros_like(flat)).reshape(delta.shape)


def random_pairs(n_adapters: int, R: int, r: int, seed: int, *tags) -> list[list[int]]:
    rng = random.Random(derived_seed(seed, *tags))
    return [sorted(rng.sample(range(R), r)) for _ in range(n_adapters)]


# ---------------------------------------------------------------- local training
def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def local_train(model: TaskModel, start: dict, data: Dataset, stream: ClientStream, sets, cfg: dict,
                device: torch.device, amp_alpha: float | None = None, freeze_head: bool = False,
                sketch_alpha: float | None = None, lora_lr_scale: float | None = None,
                sketch_straight_through: bool = False) -> tuple[dict, dict]:
    """Fresh AdamW from `start`; sets=None trains every component, else sets[j] lists trainable components.

    amp_alpha: amplified restricted training (D3) -- the forward uses start + alpha * (LoRA change), so the
    function moves per step about as fast as when all R components train (alpha = R / r); the uploaded
    displacement is the raw change. freeze_head: the classifier head is not updated.
    Batch 2 controls: sketch_alpha -- FSLoRA forward (selected components x alpha as a whole, unselected
    components dropped); sketch_straight_through -- same forward, gradient through the unscaled product (as amp);
    lora_lr_scale -- LoRA parameters use lr * scale (head lr unchanged).
    """
    if amp_alpha is not None and sketch_alpha is not None:
        raise ValueError("amp_alpha and sketch_alpha are alternatives")
    model.load_task_state(start)
    model.train()
    if amp_alpha is not None:
        for adapter in model.adapters:
            adapter.amp = (float(amp_alpha), adapter.lora_a.detach().clone(), adapter.lora_b.detach().clone())
    if sketch_alpha is not None:
        if sets is None:
            raise ValueError("sketched training needs the selected components")
        for adapter, chosen in zip(model.adapters, sets):
            factor = torch.zeros(model.rank, device=adapter.lora_a.device)
            factor[list(chosen)] = float(sketch_alpha)
            adapter.sketch = factor
            adapter.sketch_st = bool(sketch_straight_through)
    try:
        return _local_steps(model, data, stream, sets, cfg, device, freeze_head, lora_lr_scale)
    finally:
        for adapter in model.adapters:
            adapter.amp = None
            adapter.sketch = None
            adapter.sketch_st = False


def _local_steps(model, data, stream, sets, cfg, device, freeze_head, lora_lr_scale=None):
    params = list(model.task_parameters().values())
    head_params = [p for n, p in model.task_parameters().items() if "lora_" not in n]
    groups = params
    if lora_lr_scale is not None:
        lora_params = [p for n, p in model.task_parameters().items() if "lora_" in n]
        groups = [{"params": lora_params, "lr": cfg["lr"] * lora_lr_scale}, {"params": head_params}]
    masks = None
    if sets is not None:
        masks = []
        for adapter, chosen in zip(model.adapters, sets):
            mask = torch.zeros(model.rank, device=device)
            mask[list(chosen)] = 1.0
            masks.append(mask)
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                            weight_decay=cfg["weight_decay"])
    losses = []
    sync(device)
    started = time.perf_counter()
    for _ in range(cfg["local_steps"]):
        batch = data.train.batch(stream.take(cfg["batch_size"]), device)
        loss = torch.nn.functional.cross_entropy(model(batch), batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if masks is not None:
            for adapter, mask in zip(model.adapters, masks):
                adapter.lora_a.grad.mul_(mask[:, None])
                adapter.lora_b.grad.mul_(mask[None, :])
        if freeze_head:
            for p in head_params:
                p.grad = None                              # AdamW skips parameters without a gradient
        if cfg["gradient_clip"]:
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
        opt.step()
        losses.append(float(loss.detach()))
    sync(device)
    model.eval()
    if not all(math.isfinite(x) for x in losses):
        raise FloatingPointError("Nonfinite local training loss")
    return model.task_state(), {"seconds": time.perf_counter() - started, "mean_loss": float(np.mean(losses))}


# ---------------------------------------------------------------- aggregation
def block_aggregates(deltas: dict[int, torch.Tensor], weights: dict[int, float], strong: list[int],
                     weak_sets: dict[int, list[list[int]]], counts: torch.Tensor,
                     inclusion: dict[int, float] | None = None) -> dict:
    """Per-block owner and Hajek aggregates of displacement blocks (design §5.2, §5.6).

    deltas[c]: [J, R, D] displacement of client c (entries of untrained blocks are ignored).
    inclusion: per-client inclusion probability for heterogeneous weak ranks (pi_i = r_i / R); None uses the
    common probability counts / n_weak of uniform-rank dealing.
    Returns owner, hajek, strong_mean, weak_mean (None-safe via masks), and weight tensors.
    """
    J, R, D = next(iter(deltas.values())).shape
    s_sum = torch.zeros(J, R, D); s_w = 0.0
    for c in strong:
        s_sum += weights[c] * deltas[c]; s_w += weights[c]
    w_sum = torch.zeros(J, R, D); w_w = torch.zeros(J, R)
    weak_total = sum(weights[c] for c in weak_sets)
    for c, sets in weak_sets.items():
        mask = torch.zeros(J, R)
        for j, chosen in enumerate(sets):
            mask[j, chosen] = 1.0
        w_sum += weights[c] * mask[..., None] * deltas[c]
        w_w += weights[c] * mask
    n_weak = len(weak_sets)
    owner = (s_sum + w_sum) / (s_w + w_w).clamp_min(1e-12)[..., None]
    if inclusion is None:
        inflate = torch.where(counts > 0, n_weak / counts.clamp_min(1), torch.zeros_like(counts))  # 1/pi_weak
        hajek = (s_sum + inflate[..., None] * w_sum) / (s_w + inflate * w_w).clamp_min(1e-12)[..., None]
    else:
        h_sum, h_w = s_sum.clone(), torch.full((J, R), float(s_w))
        for c, sets in weak_sets.items():
            mask = torch.zeros(J, R)
            for j, chosen in enumerate(sets):
                mask[j, chosen] = 1.0
            wt = weights[c] / inclusion[c]
            h_sum += wt * mask[..., None] * deltas[c]
            h_w += wt * mask
        hajek = h_sum / h_w.clamp_min(1e-12)[..., None]
    covered = (s_w + w_w) > 0
    strong_mean = s_sum / s_w if s_w > 0 else None
    weak_mean = torch.where(w_w[..., None] > 0, w_sum / w_w.clamp_min(1e-12)[..., None], torch.zeros_like(w_sum))
    # FSLoRA / FLASC: every participant counts in every block, untrained or unsent entries as zeros
    zerofill = (s_sum + w_sum) / max(s_w + weak_total, 1e-12)
    return {"owner": owner, "hajek": hajek, "zerofill": zerofill, "strong_mean": strong_mean, "weak_mean": weak_mean,
            "strong_weight": s_w, "weak_weight": w_w, "weak_total": weak_total, "covered": covered}


def rcs_coefficients(restr_ema: torch.Tensor, full_ema: torch.Tensor, n: torch.Tensor, warmup: int,
                     c_min: float, c_max: float) -> torch.Tensor:
    """Per-adapter Replay-Calibrated Shrinkage coefficient (research_two_directions §2.3).

    restr_ema / full_ema: [J, R, D] EMA of strong clients' restricted-replay and full displacements per block
    (vectors are averaged first, then compared: the systematic over-shoot only appears after averaging).
    c_j = sum_k <restr, full> / sum_k |restr|^2 over blocks with data; 1.0 until `warmup` replays per adapter.
    """
    seen = (n > 0).float()
    num = ((restr_ema * full_ema).sum(-1) * seen).sum(1)
    den = (restr_ema.pow(2).sum(-1) * seen).sum(1)
    ready = n.sum(1) >= warmup
    est = (num / den.clamp_min(1e-30)).clamp(c_min, c_max)
    return torch.where(ready & (den > 0), est, torch.ones_like(est))


def mix(aggr: dict, a: float) -> torch.Tensor:
    """A-SCRA convex mix (design §5.7): a=1 Hajek (SCRA), a=0 owner."""
    return a * aggr["hajek"] + (1 - a) * aggr["owner"]


def weighted_head_mean(states: dict[int, dict], weights: dict[int, float], start: dict) -> dict:
    total = sum(weights[c] for c in states)
    out = {}
    for name in head_names(start):
        out[name] = start[name] + sum(weights[c] * (states[c][name] - start[name]) for c in states) / total
    return out


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_global(model: TaskModel, data: Dataset, state: dict, device, batch_size: int) -> dict:
    model.load_task_state(state)
    model.eval()
    rows = sorted(i for idx in data.valid_by_client.values() for i in idx)
    order = sorted(rows, key=lambda i: (data.valid.length(i), i))
    nll, correct = {}, {}
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        batch = data.valid.batch(ids, device)
        logits = model(batch).double()
        values = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none").cpu().tolist()
        hits = (logits.argmax(-1) == batch["labels"]).cpu().tolist()
        for i, v, h in zip(ids, values, hits):
            nll[i], correct[i] = v, bool(h)
    clients = {c: [float(np.mean([correct[i] for i in idx])), float(np.mean([nll[i] for i in idx]))]
               for c, idx in data.valid_by_client.items()}
    return {"client_macro_accuracy": float(np.mean([v[0] for v in clients.values()])),
            "client_macro_nll": float(np.mean([v[1] for v in clients.values()])),
            "clients": {str(c): v for c, v in sorted(clients.items())}}


# ---------------------------------------------------------------- io
def atomic_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def participants(seed: int, round_id: int, clients: list[int], k: int) -> list[int]:
    """Identical schedule to the headroom experiments (same derived seed tags)."""
    return sorted(random.Random(derived_seed(seed, "participants", round_id)).sample(clients, k))


def load_plan(path: Path, project_root: str | None = None) -> dict:
    plan = json.loads(Path(path).read_text())
    root = Path(project_root or plan["project_root"]).expanduser()
    plan["project_root"] = str(root)
    for key in ("cache_directory", "manifest_path", "run_root"):
        value = Path(plan[key]).expanduser()
        plan[key] = str(value if value.is_absolute() else root / value)
    name = Path(plan["model"]["name"]).expanduser()
    plan["model"]["name"] = str(name if name.is_absolute() else root / name)
    return plan
