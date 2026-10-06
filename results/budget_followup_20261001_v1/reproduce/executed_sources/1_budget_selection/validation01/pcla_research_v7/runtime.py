"""Round-boundary federated LoRA simulator for the v7 training protocol.

Only tensors in ``support`` become optimizer parameters.  The frozen rank
components remain in the forward graph, so they still transmit input gradients.
The wire adapter is deliberately explicit: an unverified byte estimate cannot
be used for budget admission.
"""
from __future__ import annotations

import os
import hashlib
import json
import resource
import sys
import platform
import datetime
from contextlib import nullcontext
from pathlib import Path
import random
import time
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pcla_repair.core import BudgetLoRALinear, keyed_seed, tensor_digest, lora_modules
from pcla_repair.runtime import atomic_checkpoint, atomic_json, batches, to_device, seed_train, fingerprint
from .transport import TensorSpec, packet_length, encode_packet, decode_packet
from .coordinates import map_pair, refresh_due
from .data import PROTOCOL
from .selectors import module_order, normalized_scores, topk_mask, cyclic_mask

SUPPORTED = {"full_both", "full_alt", "local", "cyclic", "local_both_half", "cyclic_both_half",
             "JL", "JQ", "tied_local", "layer", "energy", "equal_quota_cyclic"}
CHECKPOINT_FORMAT = "pcla-v7-runtime-1"


def state(model: nn.Module) -> dict[str, Tensor]:
    """Adapter and classification head, copied to CPU without active slices."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith((".lora_a", ".lora_b")) or n.startswith("classifier.")}


@torch.no_grad()
def active_state(model: nn.Module) -> dict[str, Tensor]:
    """Copy effective factors without committing slices or changing the optimizer."""
    values = state(model)
    for name, module in lora_modules(model).items():
        if not getattr(module, "_v7_active", False):
            continue
        for factor in ("a", "b"):
            selected = getattr(module, "active_" + factor)
            if selected is not None:
                ids = getattr(module, "active_" + factor + "_ids").cpu()
                value = selected.detach().cpu()
                values[name + ".lora_" + factor].index_copy_(
                    0 if factor == "a" else 1, ids,
                    value if factor == "a" else value.T)
    return values


@torch.no_grad()
def load_state(model: nn.Module, values: Mapping[str, Tensor]) -> None:
    params = dict(model.named_parameters())
    if set(values) != set(state(model)):
        raise ValueError("adapter/head state keys differ")
    for name, value in values.items():
        if value.shape != params[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid adapter/head tensor {name}")
        params[name].copy_(value)


def _compact_forward(self: BudgetLoRALinear, x: Tensor) -> Tensor:
    # Construct one full factor per side.  index_copy is differentiable to the
    # small active Parameter; the detached complement still carries dx.
    a, b = self.lora_a.detach(), self.lora_b.detach()
    if self.active_a is not None:
        a = a.index_copy(0, self.active_a_ids, self.active_a)
    if self.active_b is not None:
        b = b.T.index_copy(0, self.active_b_ids, self.active_b).T
    dropped = self.dropout(x)
    return self.base(x) + F.linear(F.linear(dropped, a), b) * self.scaling


def _mask(value: Tensor, rank: int, name: str) -> Tensor:
    if value.dtype != torch.bool or value.ndim != 1 or len(value) != rank:
        raise ValueError(f"{name} support must be boolean rank vector")
    return value.detach().cpu()


def attach_slices(model: nn.Module, support: Mapping[str, Mapping[str, Tensor]],
                  *, train_head: bool = True) -> int:
    """Install compact active tensors. Call ``release_slices`` after training.

    A support maps each projection name to independent ``a`` and ``b`` masks.
    Empty masks retain the full frozen forward contribution.
    """
    modules = lora_modules(model)
    if set(support) != set(modules):
        raise ValueError("support must name every LoRA projection independently")
    for p in model.parameters():
        p.requires_grad_(False)
    count = 0
    for name, module in modules.items():
        if getattr(module, "_v7_active", False):
            raise RuntimeError("active slices already attached")
        for factor in ("a", "b"):
            ids = _mask(support[name][factor], module.rank, f"{name}/{factor}").to(module.lora_a.device).nonzero().flatten()
            original = getattr(module, "lora_" + factor)
            selected = original.detach().index_select(0 if factor == "a" else 1, ids)
            if factor == "b":
                selected = selected.T.contiguous()  # rows are independent rank-1 columns
            module.register_buffer("active_" + factor + "_ids", ids, persistent=False)
            module.register_parameter("active_" + factor,
                                      nn.Parameter(selected.clone()) if len(ids) else None)
            count += selected.numel()
        module._v7_original_forward = module.forward
        module.forward = _compact_forward.__get__(module, BudgetLoRALinear)
        module._v7_active = True
    if train_head:
        for p in model.classifier.parameters():
            p.requires_grad_(True)
            count += p.numel()
    return count


@torch.no_grad()
def release_slices(model: nn.Module) -> None:
    for module in lora_modules(model).values():
        if not getattr(module, "_v7_active", False):
            continue
        for factor in ("a", "b"):
            selected = getattr(module, "active_" + factor)
            if selected is not None:
                original = getattr(module, "lora_" + factor)
                ids = getattr(module, "active_" + factor + "_ids")
                original.index_copy_(0 if factor == "a" else 1, ids,
                                     selected if factor == "a" else selected.T)
            delattr(module, "active_" + factor)
            delattr(module, "active_" + factor + "_ids")
        module.forward = module._v7_original_forward
        del module._v7_original_forward
        del module._v7_active


def full_support(model: nn.Module, factor: str = "both") -> dict[str, dict[str, Tensor]]:
    if factor not in {"a", "b", "both"}:
        raise ValueError("factor must be a, b, or both")
    return {n: {f: torch.full((m.rank,), factor in (f, "both"), dtype=torch.bool)
                for f in ("a", "b")} for n, m in lora_modules(model).items()}


def _probe_scores(model: nn.Module, dataset, factor: str, seed: int,
                  *, batch_size: int = 16, probe_batches: int = 2) -> dict[str, Tensor]:
    """Fresh eval-mode average gradient on client training rows only."""
    supports = full_support(model, factor)
    attach_slices(model, supports, train_head=False)
    device = next(model.parameters()).device
    model.eval()
    grads = {n: torch.zeros_like(m.lora_a if factor == "a" else m.lora_b,
                                 dtype=torch.float64, device="cpu")
             for n, m in lora_modules(model).items()}
    count = 0
    try:
        for i, batch in enumerate(batches(dataset, batch_size, seed)):
            if i >= probe_batches:
                break
            model.zero_grad(set_to_none=True)
            loss = model(**to_device(batch, device)).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite probe loss")
            loss.backward()
            for name, module in lora_modules(model).items():
                active = getattr(module, "active_" + factor)
                if active.grad is None:
                    raise RuntimeError("missing projection probe gradient")
                value = active.grad.detach().double().cpu()
                grads[name] += value if factor == "a" else value.T
            count += 1
    finally:
        model.zero_grad(set_to_none=True)
        release_slices(model)
    if count == 0:
        raise ValueError("empty client probe")
    factors = {name: (module.lora_b.detach().cpu(), module.lora_a.detach().cpu())
               for name, module in lora_modules(model).items()}
    return normalized_scores(factors, {name: value / count for name, value in grads.items()}, factor)


def select_support(model: nn.Module, dataset, method: str, factor: str, budget: int,
                   *, seed: int, visit: int, batch_size: int = 16,
                   probe_batches: int = 2,
                   probe_seeds: Mapping[str, int] | None = None) -> dict[str, dict[str, Tensor]]:
    """Local/cyclic projection masks with an exact total position count."""
    modules = lora_modules(model)
    names = module_order(modules)
    both = method.endswith("both_half")
    factors = ("a", "b") if both else (factor,)
    if both and budget % 2:
        raise ValueError("both-half budget must be even")
    each = budget // 2 if both else budget
    if each < 0 or each > sum(m.rank for m in modules.values()):
        raise ValueError("position budget out of range")
    support = {n: {"a": torch.zeros(m.rank, dtype=torch.bool),
                   "b": torch.zeros(m.rank, dtype=torch.bool)} for n, m in modules.items()}
    for active_factor in factors:
        if method.startswith("local"):
            scores = _probe_scores(model, dataset, active_factor,
                                   (probe_seeds or {}).get(active_factor,
                                       keyed_seed(seed, "probe", active_factor)),
                                   batch_size=batch_size, probe_batches=probe_batches)
            masks = topk_mask(scores, each)
        elif method.startswith("cyclic"):
            ranks = {modules[n].rank for n in names}
            if len(ranks) != 1:
                raise ValueError("cyclic requires equal projection ranks")
            rank = ranks.pop()
            offsets = {n: keyed_seed(seed, "cyclic", n, active_factor) % rank for n in names}
            masks = cyclic_mask(names, rank, each, visit, factor=active_factor, offsets=offsets)
        else:
            raise NotImplementedError(method)
        if sum(int(value.sum()) for value in masks.values()) != each:
            raise RuntimeError("selector did not spend exact budget")
        for n in names:
            support[n][active_factor] = masks[n]
    return support


@torch.no_grad()
def refresh_coordinates(model: nn.Module, parent: Mapping[str, Tensor], kind: str,
                        verification_dataset, *, batch_size: int = 16) -> tuple[dict[str, Tensor], dict]:
    """Map D/S on CPU float64 and verify fixed train-row eval logits."""
    if kind not in {"D", "S"}:
        raise ValueError("only D/S need the direct coordinate map")
    device = next(model.parameters()).device
    load_state(model, parent)
    model.eval()
    probe_batch = next(iter(batches(verification_dataset, batch_size, 0, shuffle=False)))
    inputs = to_device(probe_batch, device)
    before = model(**inputs).logits.detach().cpu()
    updated = {n: t.clone() for n, t in parent.items()}
    diagnostics = {}
    for name in sorted(lora_modules(model)):
        mapped = map_pair(kind, parent[name + ".lora_b"], parent[name + ".lora_a"])
        updated[name + ".lora_b"] = mapped.b
        updated[name + ".lora_a"] = mapped.a
        diagnostics[name] = {"numerical_rank": mapped.numerical_rank,
                             "rank_threshold": mapped.rank_threshold,
                             "relative_error": mapped.relative_error,
                             "absolute_error": mapped.absolute_error}
    load_state(model, updated)
    after = model(**inputs).logits.detach().cpu()
    max_error = float((after - before).abs().max())
    if max_error > 1e-4:
        raise FloatingPointError(f"coordinate refresh logits differ by {max_error}")
    diagnostics["logits_max_absolute_error"] = max_error
    return updated, diagnostics


def local_train(model: nn.Module, dataset, support: Mapping[str, Mapping[str, Tensor]],
                *, seed: int, lr: float, batch_size: int = 16, epochs: int = 1,
                betas=(.9, .999), eps=1e-8, weight_decay=.01,
                capture_steps: tuple[int, ...] = (), timing=None) -> dict:
    """Fresh AdamW and one shared dropout draw per projection per forward."""
    if lr <= 0 or epochs < 1 or batch_size < 1:
        raise ValueError("invalid training configuration")
    total_steps = ((len(dataset) + batch_size - 1) // batch_size) * epochs
    if (len(set(capture_steps)) != len(capture_steps) or
            any(type(step) is not int or not 1 <= step <= total_steps
                for step in capture_steps)):
        raise ValueError("capture_steps must be distinct executed optimizer steps")
    prefixes = {}
    capture_seconds = 0.
    with timing.span("local_training") if timing is not None else nullcontext():
        with timing.span("local_setup") if timing is not None else nullcontext():
            seed_train(seed)
            device = next(model.parameters()).device
            allocated = attach_slices(model, support)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, betas=tuple(betas), eps=eps,
                                          weight_decay=weight_decay, foreach=False)
        steps = 0
        model.train()
        started = time.perf_counter()
        try:
            for epoch in range(epochs):
                for batch in batches(dataset, batch_size, keyed_seed(seed, "data", epoch)):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**to_device(batch, device)).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite local loss")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                        raise FloatingPointError("nonfinite local gradient")
                    optimizer.step()
                    steps += 1
                    if steps in capture_steps:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        capture_started = time.perf_counter()
                        with timing.span("prefix_capture") if timing is not None else nullcontext():
                            prefixes[steps] = active_state(model)
                        capture_seconds += time.perf_counter() - capture_started
        finally:
            release_slices(model)
        outcome = {"optimizer_steps": steps, "optimizer_tensor_parameters": allocated,
                   "train_seconds": time.perf_counter() - started - capture_seconds}
        if capture_steps:
            outcome.update(prefix_states=prefixes, prefix_capture_seconds=capture_seconds)
        return outcome


@torch.no_grad()
def evaluate(model: nn.Module, dataset, batch_size: int = 64,
             *, record_logits: bool = False) -> dict:
    if any(getattr(m, "_v7_active", False) for m in lora_modules(model).values()):
        raise RuntimeError("evaluation requires committed slices")
    model.eval()
    device = next(model.parameters()).device
    correct, count, loss_sum = 0, 0, 0.
    logits_rows, label_rows = [], []
    for batch in batches(dataset, batch_size, 0, shuffle=False):
        output = model(**to_device(batch, device))
        labels = batch["labels"]
        if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
            raise FloatingPointError("nonfinite evaluation")
        correct += int((output.logits.argmax(-1).cpu() == labels).sum())
        count += len(labels)
        loss_sum += float(output.loss) * len(labels)
        if record_logits:
            logits_rows.extend(output.logits.detach().cpu().tolist())
            label_rows.extend(labels.tolist())
    if count == 0:
        raise ValueError("empty evaluation dataset")
    result = {"accuracy": correct / count, "loss": loss_sum / count, "samples": count}
    if record_logits:
        result["logits"] = logits_rows
        result["labels"] = label_rows
    return result


def initial_ema(model: nn.Module, initial: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result = {name: torch.zeros_like(module.base.weight, dtype=torch.float64, device="cpu")
              for name, module in lora_modules(model).items()}
    result.update({name: value.double().clone() for name, value in initial.items()
                   if name.startswith("classifier.")})
    return result


def advance_ema(model: nn.Module, ema: Mapping[str, Tensor], current: Mapping[str, Tensor],
                beta: float = .8) -> dict[str, Tensor]:
    result = {}
    for name, module in lora_modules(model).items():
        b, a = current[name + ".lora_b"].double(), current[name + ".lora_a"].double()
        result[name] = beta * ema[name] + (1 - beta) * module.scaling * (b @ a)
    for name in current:
        if name.startswith("classifier."):
            result[name] = beta * ema[name] + (1 - beta) * current[name].double()
    return result


@torch.no_grad()
def evaluate_ema(model: nn.Module, ema: Mapping[str, Tensor], dataset,
                 batch_size: int = 64) -> dict:
    """Temporarily evaluate W0+effective EMA, then restore exact W0/adapters."""
    modules = lora_modules(model)
    base_before = {n: m.base.weight.detach().clone() for n, m in modules.items()}
    head_before = {n: p.detach().clone() for n, p in model.named_parameters()
                   if n.startswith("classifier.")}
    scales = {n: m.scaling for n, m in modules.items()}
    params = dict(model.named_parameters())
    try:
        for n, m in modules.items():
            m.base.weight.add_(ema[n].to(m.base.weight.device, m.base.weight.dtype))
            m.scaling = 0.
        for n, value in head_before.items():
            params[n].copy_(ema[n].to(value.device, value.dtype))
        result = evaluate(model, dataset, batch_size)
    finally:
        for n, m in modules.items():
            m.base.weight.copy_(base_before[n])
            m.scaling = scales[n]
        for n, value in head_before.items():
            params[n].copy_(value)
    if any(not torch.equal(m.base.weight, base_before[n]) for n, m in modules.items()):
        raise AssertionError("EMA evaluation changed W0")
    return result


def aggregate(parent: Mapping[str, Tensor], children: list[Mapping[str, Tensor]],
              weights: list[int]) -> dict[str, Tensor]:
    """Every selected participant contributes; absent component update is zero."""
    if not children or len(children) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("invalid aggregation cohort")
    denom = sum(weights)
    result = {}
    for name, base in parent.items():
        total = torch.zeros_like(base, dtype=torch.float64)
        for child, weight in zip(children, weights):
            if set(child) != set(parent):
                raise ValueError("missing participant state")
            total += (child[name].double() - base.double()) * (weight / denom)
        result[name] = (base.double() + total).to(base.dtype)
    return result


def _tensor_sha(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class RawTensorWire:
    """Actual raw-tensor packets with versioned per-client factor caches.

    The upload packet shape is known from the support before training.  Its
    actual delta values are encoded and decoded after training; a mismatch in
    predicted and actual byte length aborts the round.
    """
    def __init__(self, *, force_full_download: bool = False):
        self.force_full_download = force_full_download

    def preview(self, *, round_id: int, cohort: list[int], parent: Mapping[str, Tensor],
                supports: Mapping[int, Mapping[str, Mapping[str, Tensor]]], cache: dict) -> dict:
        clients = cache.get("clients", {})
        digests = {n: _tensor_sha(t) for n, t in parent.items()}
        names = sorted(n for n in parent if n.endswith((".lora_a", ".lora_b")))
        modules = sorted({n.rsplit(".", 1)[0] for n in names})
        rank = parent[names[0]].shape[0] if names[0].endswith("lora_a") else parent[names[0]].shape[1]
        result = {"bytes": 0, "downloads": {}, "uploads": {}, "digests": digests,
                  "round": round_id, "modules": modules, "rank": rank, "events": []}
        for cid in cohort:
            known = clients.get(str(cid), {}).get("digests", {})
            delta = {n: t for n, t in parent.items()
                     if self.force_full_download or known.get(n) != digests[n]}
            md = {"task_init": str(cid) not in clients, "parent_version": round_id - 1}
            encoded = encode_packet("model_download", round_id - 1, cid, delta, md)
            head, decoded = decode_packet(encoded)
            if set(decoded) != set(delta) or head["version"] != round_id - 1:
                raise RuntimeError("download packet failed round trip")
            result["downloads"][cid] = encoded
            result["events"].append({"event_id": f"r{round_id}:c{cid}:down", "direction": "download", "bytes": len(encoded)})
            upload_specs = {}
            mask_ids = []
            for n in names:
                module, parameter = n.rsplit(".", 1)
                factor = "a" if parameter == "lora_a" else "b"
                ids = supports[cid][module][factor].nonzero().flatten().tolist()
                if ids:
                    shape = (len(ids), parent[n].shape[1]) if factor == "a" else (parent[n].shape[0], len(ids))
                    upload_specs[n] = TensorSpec(shape)
                base = (modules.index(module) * 2 + (0 if factor == "a" else 1)) * rank
                mask_ids.extend(base + i for i in ids)
            for n, t in parent.items():
                if n.startswith("classifier."):
                    upload_specs[n] = TensorSpec(tuple(t.shape))
            meta = {"parent_version": round_id - 1, "delta": True}
            length = packet_length("delta_upload", round_id - 1, cid, upload_specs,
                                   meta, mask_ids, len(modules) * 2 * rank)
            result["uploads"][cid] = {"specs": upload_specs, "mask": mask_ids,
                                      "mask_size": len(modules) * 2 * rank,
                                      "metadata": meta, "bytes": length}
            result["events"].append({"event_id": f"r{round_id}:c{cid}:up", "direction": "upload", "bytes": length})
            result["bytes"] += len(encoded) + length
        return result

    def commit(self, preview: dict, cache: dict, parent: Mapping[str, Tensor],
               children: list[Mapping[str, Tensor]], cohort: list[int],
               supports: Mapping[int, Mapping[str, Mapping[str, Tensor]]],
               *, diagnostic: bool = False) -> dict:
        updated = {"clients": dict(cache.get("clients", {})),
                   "events": list(cache.get("events", []))}
        existing = {event["event_id"] for event in updated["events"]}
        for cid, child in zip(cohort, children):
            spec = preview["uploads"][cid]
            tensors = {}
            for name in spec["specs"]:
                diff = child[name] - parent[name]
                if name.endswith((".lora_a", ".lora_b")):
                    module, parameter = name.rsplit(".", 1)
                    factor = "a" if parameter == "lora_a" else "b"
                    ids = supports[cid][module][factor].nonzero().flatten()
                    diff = diff.index_select(0 if factor == "a" else 1, ids)
                tensors[name] = diff.contiguous()
            encoded = encode_packet("delta_upload", preview["round"] - 1, cid, tensors,
                                    spec["metadata"], spec["mask"], spec["mask_size"])
            head, decoded = decode_packet(encoded)
            if len(encoded) != spec["bytes"] or set(decoded) != set(tensors) or head["version"] != preview["round"] - 1:
                raise RuntimeError("upload wire packet mismatch")
            updated["clients"][str(cid)] = {"version": preview["round"] - 1,
                                               "digests": preview["digests"]}
        for event in preview["events"]:
            if event["event_id"] in existing:
                raise ValueError("duplicate wire event")
            updated["events"].append({**event, "round": preview["round"],
                                      "bucket": "diagnostic_after_budget" if diagnostic else "deployed_prefix"})
        return updated


def _atomic_round_checkpoint(out_dir: Path, payload: dict) -> None:
    latest, previous = out_dir / "latest.pt", out_dir / "previous.pt"
    if latest.exists():
        os.replace(latest, previous)
    atomic_checkpoint(latest, payload)


def _load_checkpoint(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("incompatible v7 checkpoint")
    return obj


def run_experiment(model: nn.Module, clients: Mapping[int, object], evaluation,
                   capacities: Mapping[int, int], job: dict, config: dict,
                   out_dir: Path, provenance: dict, wire=None,
                   *, selector=None, stop_after: int | None = None) -> list[dict]:
    """Run complete rounds and checkpoint only after ledger and model commit.

    ``wire`` defaults to the raw-tensor codec above. Its preview includes exact
    lengths; actual deltas are encoded and checked before committing the ledger.
    """
    method = job["method"]
    representation = job.get("representation", "N")
    valid_pair = (representation == "N" or
                  (representation in {"D", "S", "T"} and method in {"local", "cyclic"}) or
                  (representation == "S" and method in {"layer", "energy"}))
    if method not in SUPPORTED or not valid_pair:
        raise NotImplementedError(f"v7 method/representation not integrated: {method}/{job.get('representation')}")
    if evaluation is None or provenance.get("evaluation_role") not in {"recipe_dev", "synthetic"}:
        raise ValueError("v7 D0/V evaluation must be train-only recipe_dev")
    if provenance["evaluation_role"] == "recipe_dev":
        required = {"protocol": PROTOCOL, "eval_role": "recipe_dev",
                    "min_rounds": 100, "max_rounds": 200}
        if any(job.get(key) != value for key, value in required.items()):
            raise ValueError("formal job differs from frozen v7 protocol")
        controls = {"clients_per_round": 10, "batch_size": 16,
                    "local_epochs": 1, "probe_batch_size": 16, "probe_batches": 2,
                    "weight_decay": .01, "eps": 1e-8}
        if any(config.get(key, value) != value for key, value in controls.items()):
            raise ValueError("formal training/probe configuration differs from v7")
        if tuple(config.get("betas", (.9, .999))) != (.9, .999):
            raise ValueError("formal AdamW betas differ from v7")
    if not clients or set(clients) != set(capacities):
        raise ValueError("client/capacity mismatch")
    if wire is None:
        wire = RawTensorWire()
    identity_provenance = dict(provenance)
    actual_hardware = identity_provenance.get("hardware", {})
    identity_provenance["hardware"] = {key: actual_hardware[key]
                                        for key in ("name", "total_memory") if key in actual_hardware}
    identity = fingerprint({"job": job, "config": config, "provenance": identity_provenance,
                            "initial_state": tensor_digest(state(model))})
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.pt"
    if latest.exists():
        ck = _load_checkpoint(latest)
        if ck["identity"] != identity:
            raise ValueError("resume identity changed")
        parent, rows, cache, deployed, diagnostic = (ck[x] for x in
            ("adapter", "metrics", "cache", "deployed_bytes", "diagnostic_bytes"))
        stop_round = ck["budget_stop_round"]
        budget_adapter = ck["budget_adapter"]
        ema = ck["ema_effective"]
        half_adapter, half_round, half_readout = (ck[x] for x in
            ("half_adapter", "half_round", "half_readout"))
        start = ck["round"] + 1
        torch.set_rng_state(ck["torch_rng"])
        random.setstate(ck["python_rng"])
        if torch.cuda.is_available() and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    else:
        parent, rows, cache, deployed, diagnostic, stop_round, start = state(model), [], {}, 0, 0, None, 1
        budget_adapter = None
        ema = initial_ema(model, parent)
        half_adapter, half_round, half_readout = {n: t.clone() for n, t in parent.items()}, 0, None
        dispatch_files = {".run.lock", "dispatch_identity.json", "worker.log", "exception.txt"}
        if any(p.name not in dispatch_files for p in out_dir.iterdir()):
            raise ValueError("nonempty directory without resumable checkpoint")
    rounds_min = int(job.get("min_rounds", config.get("min_rounds", 100)))
    rounds_max = int(job.get("max_rounds", config.get("max_rounds", 200)))
    byte_cap = int(config["byte_cap"])
    if byte_cap <= 0 or rounds_min < 1 or rounds_max < rounds_min:
        raise ValueError("invalid budget or round limits")
    if start == 1:
        checkpoint_pending = True
    else:
        checkpoint_pending = False
    selected_n = int(config.get("clients_per_round", 10))
    if selected_n > len(clients):
        raise ValueError("too few clients")
    attempts_path = out_dir / "execution_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append({"identity": identity, "resume_round": start,
                     "utc_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "host": platform.node(), "pid": os.getpid(),
                     "hardware": actual_hardware,
                     "gpu_sharing": os.environ.get("PCLA_GPU_SHARING", "unspecified"),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "slurm_step_id": os.environ.get("SLURM_STEP_ID")})
    atomic_json(attempts_path, attempts)
    def checkpoint(round_id: int) -> None:
        payload = {"format": CHECKPOINT_FORMAT, "identity": identity, "round": round_id,
                   "adapter": parent, "budget_adapter": budget_adapter,
                   "half_adapter": half_adapter, "half_round": half_round,
                   "half_readout": half_readout,
                   "ema_effective": ema,
                   "metrics": rows, "cache": cache,
                   "deployed_bytes": deployed, "diagnostic_bytes": diagnostic,
                   "budget_stop_round": stop_round, "torch_rng": torch.get_rng_state(),
                   "python_rng": random.getstate(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        _atomic_round_checkpoint(out_dir, payload)
        # V diagnostics need a complete, independently resumable parent state
        # at both anchors.  The older adapter_round artifact is intentionally
        # left in its historical lightweight format.
        if job.get("phase") == "V" and round_id in {10, 60}:
            atomic_checkpoint(out_dir / f"anchor_round_{round_id:03d}.pt", payload)

    # Encoded control messages and model downloads precede local training.  A
    # failed client therefore spends real wire bytes even without a round
    # checkpoint.  Keep that expenditure durable and replay it on resume.
    journal_path = out_dir / "communication_journal.json"
    journal = (json.loads(journal_path.read_text()) if journal_path.exists()
               else {"format": "pcla-v7-communication-journal-1", "entries": []})
    if journal.get("format") != "pcla-v7-communication-journal-1":
        raise ValueError("incompatible communication journal")
    for entry in journal["entries"]:
        if entry["status"] != "pending":
            continue
        if entry["round"] <= start - 1:
            entry["status"] = "committed"
            continue
        if entry["round"] != start:
            raise ValueError("communication journal does not match checkpoint boundary")
        existing = {event["event_id"] for event in cache.get("events", [])}
        recovered = [{**event, "event_id": f"failed:a{entry['attempt']}:{event['event_id']}"}
                     for event in entry["sent_events"]]
        present = [event["event_id"] in existing for event in recovered]
        if any(present) and not all(present):
            raise ValueError("partially recovered communication ledger")
        if not all(present):
            cache = {"clients": dict(cache.get("clients", {})),
                     "events": list(cache.get("events", [])) + recovered}
        sunk = sum(int(event["bytes"]) for event in recovered)
        if not all(present):
            if entry["bucket"] == "deployed_prefix":
                deployed += sunk
            else:
                diagnostic += sunk
        entry["status"] = "recovered_failed"
        entry["recovered_bytes"] = sunk
        checkpoint(start - 1)
    atomic_json(journal_path, journal)

    if checkpoint_pending:
        zero = evaluate(model, evaluation, int(config.get("eval_batch_size", 64)), record_logits=True)
        atomic_json(out_dir / "evaluation_labels.json", zero.pop("labels"))
        atomic_json(out_dir / "logits_round_000.json", zero.pop("logits"))
        atomic_json(out_dir / "evaluation_round_000.json", zero)
        half_readout = zero
        checkpoint(0)
    atomic_json(out_dir / "status.json", {"state": "running", "completed_round": start - 1,
                                           "budget_stop_round": stop_round})

    for round_id in range(start, rounds_max + 1):
        if stop_round is not None and round_id > rounds_min:
            break
        round_started = time.perf_counter()
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        last_completed_parent = parent
        cohort = random.Random(keyed_seed(job["seed"], "participants", round_id)).sample(sorted(clients), selected_n)
        factor = "b" if round_id % 2 else "a"
        preflight_seconds = 0.
        coordinated_round = ((method in {"JL", "JQ"} and round_id >= 1 and
                              (round_id - 1) % int(job["period"]) == 0) or
                             (representation == "T" and refresh_due(round_id, int(job["period"]))))
        upper_bound = None
        if stop_round is None and coordinated_round:
            from .cohort_runtime import preflight_round_upper_bound
            load_state(model, parent)
            preflight_started = time.perf_counter()
            upper_bound = preflight_round_upper_bound(model, parent, cohort,
                {cid: int(capacities[cid]) for cid in cohort}, round_id=round_id,
                method=method, representation=representation,
                period=int(job["period"]), factor=factor, cache=cache,
                h=int(config.get("sketch_h", 32)),
                force_full_download=bool(getattr(wire, "force_full_download", False)))
            preflight_seconds = time.perf_counter() - preflight_started
            if deployed + upper_bound > byte_cap:
                stop_round = round_id - 1
                budget_adapter = {n: t.clone() for n, t in last_completed_parent.items()}
                if rows:
                    rows[-1]["budget_stop_round"] = stop_round
                if round_id > rounds_min:
                    checkpoint(round_id - 1)
                    atomic_json(out_dir / "metrics.json", rows)
                    break
        coordinate_info = None
        cohort_plan = None
        coordinate_started = time.perf_counter()
        if representation in {"D", "S"} and refresh_due(round_id, int(job["period"])):
            if provenance["evaluation_role"] == "recipe_dev":
                from .cohort_runtime import fixed_coordinate_verification_dataset
                verification_dataset = fixed_coordinate_verification_dataset(clients)
                verification_size = 32
            else:
                verification_dataset = clients[min(clients)]
                verification_size = min(32, len(verification_dataset))
            parent, coordinate_info = refresh_coordinates(model, parent, representation,
                verification_dataset, batch_size=verification_size)
        coordinate_seconds = time.perf_counter() - coordinate_started
        load_state(model, parent)
        # Start the durable attempt before prepare_cohort: J/T broadcasts are
        # encoded and issued before its client probes, which may fail.
        journal_entry = {"attempt": len(journal["entries"]) + 1, "round": round_id,
                         "bucket": "diagnostic_after_budget" if stop_round is not None else "deployed_prefix",
                         "status": "pending", "phase": "preparing",
                         "sent_events": []}
        journal["entries"].append(journal_entry)
        atomic_json(journal_path, journal)
        def record_control(packet) -> None:
            event = {"event_id": packet.event_id, "round": round_id,
                     "direction": packet.direction, "bytes": packet.nbytes,
                     "bucket": journal_entry["bucket"]}
            if any(old["event_id"] == event["event_id"] for old in journal_entry["sent_events"]):
                raise ValueError("duplicate issued control packet")
            journal_entry["sent_events"].append(event)
            journal_entry["phase"] = "control_issued"
            atomic_json(journal_path, journal)
        scoring_started = time.perf_counter()
        if representation == "T" or method in {"JL", "JQ", "tied_local", "layer", "energy", "equal_quota_cyclic"}:
            from .cohort_runtime import prepare_cohort
            visits = {cid: sum(cid in row["selected_clients"] for row in rows) for cid in cohort}
            cohort_plan = prepare_cohort(model, parent, clients, cohort,
                {cid: int(capacities[cid]) for cid in cohort}, visits,
                round_id=round_id, seed=int(job["seed"]), factor=factor,
                method=method, representation="N" if representation in {"D", "S"} else representation,
                period=max(1, int(job["period"])), learning_rate=float(job["lr"]),
                cache=cache, xi=float(job.get("xi", 1.)),
                h=int(config.get("sketch_h", 32)),
                eps=float(config.get("eps", 1e-8)),
                weight_decay=float(config.get("weight_decay", .01)),
                batch_size=int(config.get("probe_batch_size", 16)),
                on_control_packet=record_control)
            parent, supports = cohort_plan.parent, cohort_plan.supports
            coordinate_info = {"phase": cohort_plan.phase,
                               "direct_refresh": coordinate_info,
                               "representation_error_max": cohort_plan.representation_error_max,
                               "probe_examples": cohort_plan.probe_examples,
                               "decision": cohort_plan.decision}
            load_state(model, parent)
        elif method == "full_both":
            supports = {cid: full_support(model) for cid in cohort}
        elif method == "full_alt":
            supports = {cid: full_support(model, factor) for cid in cohort}
        elif selector is None:
            supports = {}
            for cid in cohort:
                load_state(model, parent)
                visits = sum(cid in row["selected_clients"] for row in rows)
                supports[cid] = select_support(model, clients[cid], method, factor,
                    int(capacities[cid]), seed=keyed_seed(job["seed"], "selection", cid),
                    visit=visits, batch_size=int(config.get("probe_batch_size", 16)),
                    probe_batches=int(config.get("probe_batches", 2)),
                    probe_seeds={f: keyed_seed(job["seed"], "v7_probe", round_id, cid, f)
                                 for f in ("a", "b")})
        else:
            supports = selector(model=model, parent=parent, clients=clients,
                                cohort=cohort, capacities=capacities, job=job,
                                round_id=round_id, factor=factor)
        scoring_seconds = time.perf_counter() - scoring_started
        encoding_started = time.perf_counter()
        preview = wire.preview(round_id=round_id, cohort=cohort, parent=parent,
                               supports=supports,
                               cache=cohort_plan.cache_after_controls if cohort_plan else cache)
        preview_seconds = time.perf_counter() - encoding_started
        charge = int(preview["bytes"]) + (cohort_plan.control_bytes if cohort_plan else 0)
        if charge < 0:
            raise ValueError("negative wire charge")
        if upper_bound is not None:
            from .cohort_runtime import assert_preflight_covers_actual
            assert_preflight_covers_actual(upper_bound, charge, round_id=round_id)
        if stop_round is None and deployed + charge > byte_cap:
            stop_round = round_id - 1
            budget_adapter = {n: t.clone() for n, t in last_completed_parent.items()}
            if rows:
                rows[-1]["budget_stop_round"] = stop_round
            # At/after round 100 the denied round performs no communication.
            if round_id > rounds_min:
                journal_entry["status"] = "denied_no_send"
                atomic_json(journal_path, journal)
                parent = last_completed_parent
                checkpoint(round_id - 1)
                atomic_json(out_dir / "metrics.json", rows)
                break
        if "events" in preview:
            bucket = "diagnostic_after_budget" if stop_round is not None else "deployed_prefix"
            if journal_entry["sent_events"] and bucket != journal_entry["bucket"]:
                raise AssertionError("sent control traffic changed budget bucket")
            journal_entry["bucket"] = bucket
            journal_entry["sent_events"].extend(
                {**event, "round": round_id, "bucket": bucket}
                for event in preview["events"] if event["direction"] == "download")
            journal_entry["phase"] = "control_and_download"
            atomic_json(journal_path, journal)
        children, weights, details = [], [], []
        for cid in cohort:
            load_state(model, parent)
            outcome = local_train(model, clients[cid], supports[cid],
                                  seed=keyed_seed(job["seed"], "train", round_id, cid),
                                  lr=float(job["lr"]), batch_size=int(config.get("batch_size", 16)),
                                  epochs=int(config.get("local_epochs", 1)),
                                  betas=config.get("betas", (.9, .999)),
                                  eps=float(config.get("eps", 1e-8)),
                                  weight_decay=float(config.get("weight_decay", .01)))
            if provenance["evaluation_role"] == "recipe_dev" and outcome["optimizer_steps"] != 16:
                raise RuntimeError("v7 SST-2 client must execute exactly 16 local AdamW steps")
            children.append(state(model))
            weights.append(len(clients[cid]))
            details.append({"client": cid, **outcome})
        parent_before = parent
        aggregation_started = time.perf_counter()
        parent = aggregate(parent, children, weights)
        aggregation_seconds = time.perf_counter() - aggregation_started
        load_state(model, parent)
        encoding_started = time.perf_counter()
        if "events" in preview:
            # The upload stage may be interrupted mid-packet.  Reserving the
            # full verified preview here is conservative and visible in the
            # journal instead of silently refunding a possibly sent upload.
            journal_entry["phase"] = "upload_started_conservative"
            journal_entry["sent_events"] += [
                {**event, "round": round_id, "bucket": journal_entry["bucket"]}
                for event in preview["events"] if event["direction"] == "upload"]
            atomic_json(journal_path, journal)
        if cohort_plan is not None:
            from .cohort_runtime import committed_control_cache
            cache = committed_control_cache(cohort_plan, diagnostic=stop_round is not None)
        cache = wire.commit(preview, cache, parent_before, children, cohort, supports,
                            diagnostic=stop_round is not None)
        encoding_seconds = preview_seconds + time.perf_counter() - encoding_started
        if stop_round is None:
            deployed += charge
        else:
            diagnostic += charge
        evaluation_started = time.perf_counter()
        readout = evaluate(model, evaluation, int(config.get("eval_batch_size", 64)),
                           record_logits=True)
        labels = readout.pop("labels")
        if labels != json.loads((out_dir / "evaluation_labels.json").read_text()):
            raise ValueError("evaluation labels or order changed")
        atomic_json(out_dir / f"logits_round_{round_id:03d}.json", readout.pop("logits"))
        evaluation_seconds = time.perf_counter() - evaluation_started
        ema_started = time.perf_counter()
        ema = advance_ema(model, ema, parent)
        ema_update_seconds = time.perf_counter() - ema_started
        ema_started = time.perf_counter()
        ema_readout = evaluate_ema(model, ema, evaluation,
                                   int(config.get("eval_batch_size", 64)))
        ema_eval_seconds = time.perf_counter() - ema_started
        if stop_round is None and deployed <= byte_cap // 2:
            half_adapter = {n: t.clone() for n, t in parent.items()}
            half_round, half_readout = round_id, dict(readout)
        cost = {"round_compute_wall_seconds": time.perf_counter() - round_started,
                "scoring_seconds": scoring_seconds if cohort_plan is None else 0.,
                "cohort_preparation_seconds": scoring_seconds if cohort_plan is not None else 0.,
                "coordinate_seconds": coordinate_seconds,
                "wire_encode_decode_seconds": encoding_seconds,
                "coordination_preflight_seconds": preflight_seconds,
                "local_training_seconds": sum(x["train_seconds"] for x in details),
                "aggregation_seconds": aggregation_seconds,
                "raw_evaluation_seconds": evaluation_seconds,
                "ema_update_seconds": ema_update_seconds,
                "ema_evaluation_seconds": ema_eval_seconds,
                "cpu_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                       (1024 if sys.platform == "darwin" else 1)),
                "cuda_peak_allocated_bytes": 0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": 0 if device.type != "cuda" else torch.cuda.max_memory_reserved(device)}
        support_trace = {str(cid): {"assigned_quota": int(capacities[cid]),
            "selected_positions": sum(int(mask.sum()) for masks in supports[cid].values()
                                      for mask in masks.values()),
            "active_factor": "both" if method in {"full_both", "local_both_half", "cyclic_both_half"} else factor,
            "modules": {name: {f: masks[f].nonzero().flatten().tolist() for f in ("a", "b")}
                        for name, masks in supports[cid].items()}}
                         for cid in cohort}
        cumulative = dict(rows[-1]["cumulative_cost"]) if rows else {}
        for key, value in cost.items():
            cumulative[key] = max(cumulative.get(key, 0), value) if "peak" in key else cumulative.get(key, 0) + value
        rows.append({"round": round_id, "selected_clients": cohort,
                     "deployed_bytes": deployed, "diagnostic_after_budget_bytes": diagnostic,
                     "budget_stop_round": stop_round, "evaluation": readout,
                     "ema_evaluation": ema_readout,
                     "client_training": details, "coordinate_refresh": coordinate_info,
                     "support_trace": support_trace,
                     "gpu_sharing": os.environ.get("PCLA_GPU_SHARING", "unspecified"),
                     "cost": cost, "cumulative_cost": cumulative})
        artifact_started = time.perf_counter()
        checkpoint(round_id)
        journal_entry["status"] = "committed"
        atomic_json(journal_path, journal)
        if (round_id % 10 == 0 or round_id in {10, 60, stop_round}
                or rounds_min - 4 <= round_id <= rounds_min
                or round_id >= rounds_max - 4):
            atomic_checkpoint(out_dir / f"adapter_round_{round_id:03d}.pt",
                              {"round": round_id, "adapter": parent,
                               "deployed_bytes": deployed, "diagnostic_bytes": diagnostic})
        atomic_json(out_dir / "metrics.json", rows)
        atomic_json(out_dir / f"round_wall_{round_id:03d}.json",
                    {"round": round_id, "total_wall_seconds": time.perf_counter() - round_started,
                     "checkpoint_and_artifact_seconds": time.perf_counter() - artifact_started})
        atomic_json(out_dir / "status.json", {"state": "running", "completed_round": round_id,
                                               "budget_stop_round": stop_round})
        if stop_after is not None and round_id >= stop_after:
            atomic_json(out_dir / "status.json", {"state": "paused", "completed_round": round_id,
                                                   "budget_stop_round": stop_round})
            break
        if round_id >= rounds_min and stop_round is not None:
            break
    if rows and rows[-1]["round"] == rounds_max and stop_round is None:
        stop_round = rounds_max
        rows[-1]["budget_stop_round"] = stop_round
        budget_adapter = {n: t.clone() for n, t in parent.items()}
        checkpoint(rounds_max)
        atomic_json(out_dir / "metrics.json", rows)
    if budget_adapter is not None:
        atomic_checkpoint(out_dir / "budget_return.pt", {"format": CHECKPOINT_FORMAT,
            "round": stop_round, "adapter": budget_adapter, "deployed_bytes": deployed,
            "budget_cap_bytes": byte_cap, "identity": identity})
    atomic_checkpoint(out_dir / "half_budget_return.pt", {"format": CHECKPOINT_FORMAT,
        "round": half_round, "adapter": half_adapter, "deployed_bytes":
        rows[half_round - 1]["deployed_bytes"] if half_round else 0,
        "budget_cap_bytes": byte_cap // 2, "readout": half_readout,
        "identity": identity})
    if stop_after is None and rows:
        atomic_json(out_dir / "status.json", {"state": "completed", "completed_round": rows[-1]["round"],
                                               "budget_stop_round": stop_round,
                                               "deployed_bytes": deployed,
                                               "diagnostic_after_budget_bytes": diagnostic})
    return rows
