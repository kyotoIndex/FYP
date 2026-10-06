"""Server rules for actual, decoded PCLA adapter/head states.

Public examples are *training* examples.  This module never caches features
between calls: a feature is valid only for this model, parent, proposal and
ordered public batch sequence.  All returned states live on CPU.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from contextlib import contextmanager
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pcla_research_v7.runtime import load_state, state
from pcla_repair.core import lora_modules

GRID4 = (0.0, 0.25, 0.5, 1.0)
HEAD_KEYS = ("classifier.weight", "classifier.bias")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hash_state(values):
    h = hashlib.sha256()
    for key in sorted(values):
        v = values[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def _validate(parent, proposal, children):
    keys = set(parent)
    if keys != set(proposal) or not set(HEAD_KEYS) <= keys:
        raise ValueError("parent/proposal keys or classifier head differ")
    if not all(k.startswith("classifier.") or k.endswith((".lora_a", ".lora_b")) for k in keys):
        raise ValueError("unexpected trainable coordinate")
    for item in (parent, proposal, *children):
        if set(item) != keys:
            raise ValueError("child state keys differ")
        for key, value in item.items():
            if value.device.type != "cpu" or value.shape != parent[key].shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid CPU state tensor {key}")


def interpolate(parent, proposal, alpha: float, beta: float):
    """Factor-space interpolation, with bitwise-exact endpoints."""
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (alpha, beta)):
        raise ValueError("interpolation scales must be in [0,1]")
    result = {}
    for key, old in parent.items():
        new = proposal[key]
        scale = beta if key in HEAD_KEYS else alpha
        value = old.clone() if scale == 0 else new.clone() if scale == 1 else old + scale * (new - old)
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite interpolated {key}")
        result[key] = value
    return result


def _nll(logits: Tensor, labels: Tensor):
    if logits.dtype != torch.float32:
        logits = logits.float()
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("logits/labels shape mismatch")
    if not torch.isfinite(logits).all() or (labels < 0).any() or (labels >= logits.shape[-1]).any():
        raise FloatingPointError("invalid public logits or labels")
    return F.cross_entropy(logits.double(), labels.long(), reduction="sum").item()


def _choose(rows):
    if not rows or any(not math.isfinite(row["loss"]) for row in rows):
        raise FloatingPointError("empty or nonfinite candidate losses")
    minimum = min(row["loss"] for row in rows)
    tied = [r for r in rows if r["loss"] <= minimum + 1e-12]
    return min(tied, key=lambda r: ((1-r["alpha"])**2 + (1-r["beta"])**2,
                                    -r["alpha"], -r["beta"]))


@contextmanager
def _preserve_caller(model):
    original = state(model)
    modes = {m: m.training for m in model.modules()}
    grad_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    py_rng = random.getstate()
    np_rng = None
    try:
        import numpy as np
        np_rng = np.random.get_state()
    except ImportError:
        pass
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        with torch.no_grad():
            load_state(model, original)
            for n, b in model.named_buffers():
                b.copy_(buffers[n])
        for m, mode in modes.items():
            m.training = mode
        for n, p in model.named_parameters():
            p.requires_grad_(grad_flags[n])
            p.grad = grads[n]
        random.setstate(py_rng)
        if np_rng is not None:
            np.random.set_state(np_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _public_batches(dataset, batch_size):
    if dataset is None or len(dataset) == 0:
        raise ValueError("public training dataset required")
    batches = list(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))
    h = hashlib.sha256()
    for batch in batches:
        if "labels" not in batch:
            raise ValueError("public batch lacks labels")
        for key in sorted(batch):
            v = batch[key].detach().cpu().contiguous()
            h.update(key.encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return batches, h.hexdigest()


def _features(model, batches, device):
    features, labels = [], []
    def capture(_module, inputs):
        features.append(inputs[0].detach())
    hook = model.classifier.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for batch in batches:
                kwargs = {k: v.to(device) for k, v in batch.items()}
                output = model(**kwargs)
                if not torch.isfinite(output.logits).all() or not torch.isfinite(features[-1]).all():
                    raise FloatingPointError("nonfinite public forward")
                labels.append(batch["labels"].to(device))
    finally:
        hook.remove()
    return torch.cat(features), torch.cat(labels)


def _head_logits(head_module, features, head_state, device):
    # The current BERT head is a single Linear; this is its actual functional
    # forward, including the model's classifier module implementation.
    params = {k.removeprefix("classifier."): v.to(device) for k, v in head_state.items() if k in HEAD_KEYS}
    return functional_call(head_module, params, (features,)).float()


def _grid(model, parent, proposal, batches, alphas, betas_for_alpha, record, device):
    rows = []
    model.eval()
    for alpha in alphas:
        load_state(model, interpolate(parent, proposal, alpha, 1.0))
        _sync(device); t0 = time.perf_counter()
        features, labels = _features(model, batches, device)
        _sync(device); record["timers"]["public_feature_seconds"] += time.perf_counter() - t0
        record["features_count"] += len(labels)
        record["encoder_batches"] += len(batches)
        record["public_dataset_passes"] += 1
        _sync(device); t0 = time.perf_counter()
        for beta in betas_for_alpha(alpha):
            candidate = interpolate(parent, proposal, alpha, beta)
            logits = _head_logits(model.classifier, features, candidate, device)
            loss = _nll(logits, labels) / len(labels)
            rows.append({"alpha": float(alpha), "beta": float(beta), "loss": loss})
            record["head_evals"] += 1
        _sync(device); record["timers"]["head_grid_seconds"] += time.perf_counter() - t0
    return rows


def _fedexp(parent, proposal, children, weights, epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("FedExP epsilon must be positive")
    denom = float(sum(weights))
    a = [w / denom for w in weights]
    numerator = 0.0
    denominator = 0.0
    for key in parent:
        base = parent[key].double()
        avg_delta = torch.zeros_like(base)
        for child, weight in zip(children, a):
            d = child[key].double() - base
            numerator += weight * float((d * d).sum())
            avg_delta += weight * d
        denominator += float((avg_delta * avg_delta).sum())
    eta = max(1.0, numerator / (2 * (denominator + epsilon)))
    result = {k: (parent[k].double() + eta * (proposal[k].double() - parent[k].double())).to(parent[k].dtype)
              for k in parent}
    if not math.isfinite(eta) or any(not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("nonfinite FedExP extrapolation")
    return result, eta


def _fedlaw(model, children, weights, batches, recipe, seed, record, device):
    if not children:
        raise ValueError("FedLAW needs client states")
    lr = float(recipe["server_lr"])
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid FedLAW server LR")
    if int(recipe.get("server_epochs", 1)) != 1:
        raise ValueError("FedLAW-PEFT protocol requires one public epoch")
    base = dict(model.named_parameters())
    keys = sorted(children[0])
    child_device = [{k: child[k].to(device) for k in keys} for child in children]
    initial = torch.tensor(weights, dtype=torch.float32, device=device)
    raw = nn.Parameter(torch.cat((initial.log(), torch.zeros(1, device=device))))
    opt = torch.optim.Adam([raw], lr=lr, betas=(0.5, 0.999))
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(sum(len(b["labels"]) for b in batches), generator=generator).tolist()
    samples = [dict(zip(b.keys(), values)) for b in batches for values in zip(*[b[k] for k in b])]
    size = int(recipe.get("server_batch_size", 64))
    if size <= 0:
        raise ValueError("invalid FedLAW batch size")
    model.train()
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    _sync(device); t0 = time.perf_counter()
    for offset in range(0, len(order), size):
        selected = [samples[i] for i in order[offset:offset+size]]
        batch = {k: torch.stack([sample[k] for sample in selected]).to(device) for k in selected[0]}
        lam = raw[:-1].softmax(0)
        gamma = raw[-1].exp()
        fused = {k: gamma * sum(lam[i] * child_device[i][k] for i in range(len(children))) for k in keys}
        params = dict(base); params.update(fused)
        opt.zero_grad(set_to_none=True)
        output = functional_call(model, params, (), batch)
        logits = output.logits.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite FedLAW logits")
        loss = F.cross_entropy(logits, batch["labels"].long())
        loss.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all():
            raise FloatingPointError("nonfinite FedLAW gradient")
        opt.step()
        record["server_gradient_batches"] += 1
        record["encoder_batches"] += 1
        record["head_evals"] += 1
        record["features_count"] += len(selected)
    record["public_dataset_passes"] = 1
    _sync(device); record["timers"]["native_server_gradient_seconds"] += time.perf_counter() - t0
    lam = raw[:-1].softmax(0).detach()
    gamma = raw[-1].exp().detach()
    result = {k: (gamma * sum(lam[i] * child_device[i][k] for i in range(len(children)))).detach().cpu()
              for k in keys}
    if not torch.isfinite(gamma) or any(not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("nonfinite FedLAW result")
    return result, float(gamma), lam.cpu().tolist()


def apply_server_rule(model: nn.Module, old_parent: Mapping[str, Tensor], proposal: Mapping[str, Tensor],
                      children: list[Mapping[str, Tensor]], weights: list[float], public_dataset,
                      recipe: Mapping, seed: int, round_id: int):
    """Return accepted CPU state and a per-round controller record.

    The caller owns wire accounting. `proposal` is the already decoded ordinary
    aggregate; each child is an actual decoded post-local state.
    """
    rule = recipe["server_rule"]
    if len(children) != len(weights) or any(not math.isfinite(float(w)) or w <= 0 for w in weights):
        raise ValueError("invalid cohort weights")
    _validate(old_parent, proposal, children)
    if set(state(model)) != set(old_parent) or any(getattr(m, "_v7_active", False) for m in lora_modules(model).values()):
        raise ValueError("model/state mismatch or active slices attached")
    device = next(model.parameters()).device
    record = {"rule": rule, "recipe_id": recipe.get("recipe_id"), "seed": int(seed),
              "round_id": int(round_id), "public_role": "train_public512" if public_dataset is not None else None,
              "parent_hash": _hash_state(old_parent), "proposal_hash": _hash_state(proposal),
              "configuration_sha256": recipe.get("config_sha256"),
              "head_key_manifest": list(HEAD_KEYS),
              "factor_key_manifest": sorted(k for k in old_parent if k not in HEAD_KEYS),
              "features_count": 0, "encoder_batches": 0, "head_evals": 0,
              "public_dataset_passes": 0, "server_gradient_batches": 0,
              "grid_public_NLL": [], "timers": {"public_feature_seconds": 0.0,
                  "head_grid_seconds": 0.0, "native_server_gradient_seconds": 0.0}}
    with _preserve_caller(model):
        if rule in ("mean_r8", "mean_r4", "fedsb_r32"):
            accepted = {k: v.clone() for k, v in proposal.items()}
            record.update(alpha=1.0, beta=1.0)
        elif rule in ("fixed_scalar", "fixed_head"):
            alpha, beta = float(recipe["alpha"]), float(recipe["beta"])
            accepted = interpolate(old_parent, proposal, alpha, beta)
            record.update(alpha=alpha, beta=beta)
        elif rule == "fedexp_peft":
            accepted, eta = _fedexp(old_parent, proposal, children, weights, float(recipe["server_epsilon"]))
            record.update(alpha=eta, beta=eta, eta=eta)
        elif rule in ("scalar4", "scalar17", "block16", "separable", "headfit", "fedlaw_peft"):
            batches, public_hash = _public_batches(public_dataset, int(recipe.get("server_batch_size", 64)) if rule == "fedlaw_peft" else 64)
            record["public_ordered_content_sha256"] = public_hash
            if rule == "fedlaw_peft":
                accepted, gamma, lam = _fedlaw(model, children, weights, batches, recipe, seed + round_id, record, device)
                record.update(gamma=gamma, lambda_weights=lam, alpha=None, beta=None)
            elif rule == "headfit":
                load_state(model, proposal)
                model.eval()
                _sync(device); t0 = time.perf_counter()
                features, labels = _features(model, batches, device)
                _sync(device); record["timers"]["public_feature_seconds"] += time.perf_counter() - t0
                record["features_count"] = len(labels); record["encoder_batches"] = len(batches)
                record["public_dataset_passes"] = 1
                head = {k: nn.Parameter(proposal[k].to(device).clone()) for k in HEAD_KEYS}
                opt = torch.optim.AdamW(head.values(), lr=float(recipe["server_head_lr"]),
                                         weight_decay=float(recipe.get("server_head_weight_decay", .01)))
                generator = torch.Generator().manual_seed(int(seed + round_id))
                _sync(device); t0 = time.perf_counter()
                head_steps = int(recipe.get("server_head_steps", 32))
                head_batch = int(recipe.get("server_batch_size", 16))
                if head_steps <= 0 or head_batch <= 0:
                    raise ValueError("invalid headfit schedule")
                order = torch.randperm(len(labels), generator=generator)
                cursor = 0
                for _ in range(head_steps):
                    if cursor + head_batch > len(order):
                        order = torch.randperm(len(labels), generator=generator)
                        cursor = 0
                    ids = order[cursor:cursor+head_batch].to(device)
                    cursor += head_batch
                    params = {k.removeprefix("classifier."): v for k, v in head.items()}
                    logits = functional_call(model.classifier, params, (features[ids],)).float()
                    loss = F.cross_entropy(logits, labels[ids].long())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite headfit loss")
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    record["head_evals"] += 1
                _sync(device); record["timers"]["native_server_gradient_seconds"] += time.perf_counter() - t0
                accepted = {k: v.clone() for k, v in proposal.items()}
                accepted.update({k: v.detach().cpu().clone() for k, v in head.items()})
                with torch.no_grad():
                    initial_logits = functional_call(model.classifier,
                        {k.removeprefix("classifier."): proposal[k].to(device) for k in HEAD_KEYS}, (features,)).float()
                    final_logits = functional_call(model.classifier,
                        {k.removeprefix("classifier."): v for k, v in head.items()}, (features,)).float()
                    record["mean_update_public_NLL"] = _nll(initial_logits, labels) / len(labels)
                    record["chosen_public_NLL"] = _nll(final_logits, labels) / len(labels)
                    record["head_evals"] += 2
                record.update(alpha=1.0, beta=None, headfit_steps=head_steps)
            else:
                alphas = GRID4 if rule != "scalar17" else tuple(i / 16 for i in range(17))
                betas = (lambda a: alphas if rule in ("scalar4", "scalar17") else GRID4)
                rows = _grid(model, old_parent, proposal, batches, alphas,
                             lambda a: (a,) if rule in ("scalar4", "scalar17") else betas(a), record, device)
                chosen = _choose(rows)
                if rule == "separable":
                    a = _choose([r for r in rows if r["beta"] == 0])["alpha"]
                    b = _choose([r for r in rows if r["alpha"] == 0])["beta"]
                    chosen = next(r for r in rows if r["alpha"] == a and r["beta"] == b)
                accepted = interpolate(old_parent, proposal, chosen["alpha"], chosen["beta"])
                record.update(alpha=chosen["alpha"], beta=chosen["beta"],
                              chosen_public_NLL=chosen["loss"], grid_public_NLL=rows,
                              unique_alpha_count=len(alphas))
                mean = next((r for r in rows if (r["alpha"],r["beta"]) == (1.,1.)), None)
                parent = next((r for r in rows if (r["alpha"],r["beta"]) == (0.,0.)), None)
                record["mean_update_public_NLL"] = None if mean is None else mean["loss"]
                record["parent_public_NLL"] = None if parent is None else parent["loss"]
        else:
            raise ValueError(f"unsupported server rule {rule}")
        if any(not torch.isfinite(v).all() for v in accepted.values()):
            raise FloatingPointError("nonfinite accepted state")
        record["committed_state_hash"] = _hash_state(accepted)
        record["no_op_reason"] = "chosen_parent" if record["committed_state_hash"] == record["parent_hash"] else None
    return accepted, record
