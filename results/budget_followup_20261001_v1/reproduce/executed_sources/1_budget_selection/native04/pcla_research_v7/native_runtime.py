"""Independent v7 execution path for three pinned native FL algorithms.

No native recipe passes through the core method switch. The shared parts are
only frozen data/model preparation, eval, codec, and atomic persistence.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import Subset

from pcla_repair.core import BudgetLoRALinear, keyed_seed, lora_modules, tensor_digest
from pcla_repair.runtime import atomic_checkpoint, atomic_json, batches, fingerprint, seed_train, to_device
from .data import TokenCache, PROTOCOL
from .native_fedsb import FedSBLinear, estimate_signed_steps, initialize_pair, aggregate_r_head
from .native_fedsvd import refresh_state, aggregate_b_head
from .native_fedplora import FedPLoRALocalLinear, select_indices, aggregate_contributors
from .runtime import evaluate
from .transport import TensorSpec, packet_length, encode_packet, decode_packet

METHODS = {"fedsb", "fedsvd", "fedplora"}
CHECKPOINT_FORMAT = "pcla-v7-native-1"


def _sha_tensor(t: Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def native_modules(model: nn.Module, method: str) -> dict[str, nn.Module]:
    if method == "fedsb":
        return {n: m for n, m in model.named_modules() if isinstance(m, FedSBLinear)}
    return lora_modules(model)


def _replace_module(model: nn.Module, name: str, module: nn.Module) -> None:
    if "." in name:
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, module)
    else:
        setattr(model, name, module)


def native_state(model: nn.Module, method: str) -> dict[str, Tensor]:
    suffixes = (".a", ".b", ".r") if method == "fedsb" else (".lora_a", ".lora_b")
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith(suffixes) or n.startswith("classifier.")}


@torch.no_grad()
def load_native_state(model: nn.Module, method: str, state: Mapping[str, Tensor]) -> None:
    params = dict(model.named_parameters())
    if set(state) != set(native_state(model, method)):
        raise ValueError("native state keys differ")
    for name, value in state.items():
        if params[name].shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid native tensor {name}")
        params[name].copy_(value.to(params[name].device))


def initialize_fedsb(model: nn.Module, public_dataset, rank: int, lr: float,
                     *, batch_size: int = 16) -> dict:
    """Calculate full-model task gradients before installing any adapter."""
    targets = [(name, module) for name, module in model.named_modules()
               if name.endswith((".attention.self.query", ".attention.self.value"))
               and isinstance(module, nn.Linear)]
    if len(targets) != 24:
        raise ValueError("Fed-SB requires 24 BERT Q/V base projections")
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    begun = time.perf_counter()
    cpu_peak_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                          (1024 if sys.platform == "darwin" else 1))
    cuda_allocated_before = 0 if device.type != "cuda" else torch.cuda.memory_allocated(device)
    steps = estimate_signed_steps(model, public_dataset,
        [name + ".weight" for name, _ in targets], lr,
        device=device, batch_size=batch_size,
        local_optimizer_steps=16)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gradient_seconds = time.perf_counter() - begun
    svd_started = time.perf_counter()
    audits = {}
    for name, base in targets:
        a, b, r, audit = initialize_pair(steps[name + ".weight"], rank)
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf,
                FedSBLinear(base, a, b, r, scaling=1., dropout=.1))
        audits[name] = audit
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cost = {"total_wall_seconds": time.perf_counter() - begun,
            "public_gradient_seconds": gradient_seconds,
            "svd_and_install_seconds": time.perf_counter() - svd_started,
            "cpu_peak_rss_before_kb": cpu_peak_before,
            "cpu_peak_rss_after_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                         (1024 if sys.platform == "darwin" else 1)),
            "cuda_allocated_before_bytes": cuda_allocated_before,
            "cuda_allocated_after_bytes": 0 if device.type != "cuda" else torch.cuda.memory_allocated(device),
            "cuda_peak_allocated_bytes": 0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": 0 if device.type != "cuda" else torch.cuda.max_memory_reserved(device)}
    return {"public_rows": 512, "full_weight_gradient_targets": len(targets),
            "effective_lr": lr / 16, "initialization_seed": 42,
            "svd_dtype": "float32", "projections": audits, "cost": cost}


def _selected(method: str, parent: Mapping[str, Tensor], cohort: list[int],
              capacities: Mapping[int, int], job: dict) -> dict[int, dict[str, tuple[int, ...]]]:
    modules = sorted(n[:-7] for n in parent if n.endswith(".lora_a")) if method != "fedsb" else sorted(n[:-2] for n in parent if n.endswith(".r"))
    if method != "fedplora":
        return {cid: {name: tuple(range(int(job["rank"]))) for name in modules} for cid in cohort}
    tiers = job.get("native_ranks")
    if not tiers or len(tiers) != 3:
        raise ValueError("Fed-PLoRA needs three native tier ranks")
    return {cid: {name: select_indices(int(job["rank"]), int(tiers[{96: 0, 128: 1, 192: 2}[capacities[cid]]]),
                protocol=PROTOCOL, seed=int(job["seed"]), round_id=int(job["_round"]), client_id=cid, module=name)
                  for name in modules} for cid in cohort}


def _upload_tensors(method: str, parent: Mapping[str, Tensor], child: Mapping[str, Tensor],
                    selection: Mapping[str, tuple[int, ...]]) -> tuple[dict[str, Tensor], list[int], int]:
    tensors = {}
    mask = []
    rank = next((t.shape[0] for n, t in parent.items() if n.endswith(".lora_a")), 0)
    modules = sorted(selection)
    for i, module in enumerate(modules):
        ids = list(selection[module])
        if method == "fedsb":
            name = module + ".r"
            tensors[name] = child[name] - parent[name]
        elif method == "fedsvd":
            name = module + ".lora_b"
            tensors[name] = child[name] - parent[name]
        else:
            for suffix, axis in ((".lora_a", 0), (".lora_b", 1)):
                name = module + suffix
                tensors[name] = (child[name] - parent[name]).index_select(axis, torch.tensor(ids)).contiguous()
            mask.extend(i * rank + j for j in ids)
    for name in sorted(parent):
        if name.startswith("classifier."):
            tensors[name] = child[name] - parent[name]
    return tensors, mask, len(modules) * rank if method == "fedplora" else 0


class NativeWire:
    """Versioned real packets and exact preflight lengths for native factors."""
    def preview(self, method: str, round_id: int, cohort: list[int], parent: Mapping[str, Tensor],
                selections: Mapping[int, Mapping[str, tuple[int, ...]]], cache: dict) -> dict:
        known_clients = cache.get("clients", {})
        digests = {name: _sha_tensor(t) for name, t in parent.items()}
        result = {"round": round_id, "bytes": 0, "digests": digests, "uploads": {}, "events": []}
        for cid in cohort:
            known = known_clients.get(str(cid), {}).get("digests", {})
            delta = {name: t for name, t in parent.items() if known.get(name) != digests[name]}
            down_meta = {"task_init": str(cid) not in known_clients, "parent_version": round_id - 1,
                         "native_method": method}
            raw = encode_packet("model_download", round_id - 1, cid, delta, down_meta)
            header, decoded = decode_packet(raw)
            if header["version"] != round_id - 1 or set(decoded) != set(delta):
                raise RuntimeError("native download failed decode")
            sample, mask, mask_size = _upload_tensors(method, parent, parent, selections[cid])
            specs = {name: TensorSpec(tuple(t.shape)) for name, t in sample.items()}
            up_meta = {"parent_version": round_id - 1, "delta": True,
                       "native_method": method}
            length = packet_length("delta_upload", round_id - 1, cid, specs, up_meta,
                                   mask if mask_size else None, mask_size or None)
            result["uploads"][cid] = {"specs": specs, "mask": mask, "mask_size": mask_size,
                                      "metadata": up_meta, "bytes": length}
            result["bytes"] += len(raw) + length
            result["events"].extend([
                {"event_id": f"r{round_id}:c{cid}:down", "direction": "download", "bytes": len(raw)},
                {"event_id": f"r{round_id}:c{cid}:up", "direction": "upload", "bytes": length}])
        return result

    def validate_upload(self, method: str, preview: dict, cid: int,
                        parent: Mapping[str, Tensor], child: Mapping[str, Tensor],
                        selection: Mapping[str, tuple[int, ...]]) -> int:
        """Physically encode the just-trained client update before journaling it."""
        tensors, mask, size = _upload_tensors(method, parent, child, selection)
        spec = preview["uploads"][cid]
        raw = encode_packet("delta_upload", preview["round"] - 1, cid, tensors,
                            spec["metadata"], mask if size else None, size or None)
        header, decoded = decode_packet(raw)
        if len(raw) != spec["bytes"] or header["version"] != preview["round"] - 1 or set(decoded) != set(tensors):
            raise RuntimeError("native upload differs from preflight")
        for name, value in decoded.items():
            if not torch.equal(torch.from_numpy(value), tensors[name].detach().cpu()):
                raise RuntimeError(f"native wire tensor changed: {name}")
        return len(raw)

    def commit(self, method: str, preview: dict, cache: dict, parent: Mapping[str, Tensor],
               children: Mapping[int, Mapping[str, Tensor]],
               selections: Mapping[int, Mapping[str, tuple[int, ...]]],
               *, diagnostic: bool = False) -> dict:
        result = {"clients": dict(cache.get("clients", {})), "events": list(cache.get("events", []))}
        existing = {x["event_id"] for x in result["events"]}
        for cid, child in children.items():
            self.validate_upload(method, preview, cid, parent, child, selections[cid])
            result["clients"][str(cid)] = {"version": preview["round"] - 1,
                                           "digests": preview["digests"]}
        for event in preview["events"]:
            if event["event_id"] in existing:
                raise ValueError("duplicate native wire event")
            result["events"].append({**event, "round": preview["round"],
                                     "bucket": "diagnostic_after_budget" if diagnostic else "deployed_prefix"})
        return result


def _optimizer(params, recipe: dict, *, head=False):
    lr = float(recipe["head_lr"] if head else recipe["lr"])
    if not head and recipe["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=float(recipe["adapter_weight_decay"]))
    return torch.optim.AdamW(params, lr=lr, betas=(.9, .999), eps=1e-8,
                             weight_decay=.01 if head else float(recipe["adapter_weight_decay"]),
                             foreach=False)


def local_train_native(model: nn.Module, method: str, dataset, parent: Mapping[str, Tensor],
                       selection: Mapping[str, tuple[int, ...]], recipe: dict,
                       *, seed: int, batch_size: int = 16) -> tuple[dict[str, Tensor], dict]:
    seed_train(seed)
    device = next(model.parameters()).device
    load_native_state(model, method, parent)
    original = {}
    selected_layers = {}
    started = time.perf_counter()
    try:
        if method == "fedplora":
            for name, ids in selection.items():
                module = model.get_submodule(name)
                if not isinstance(module, BudgetLoRALinear):
                    raise ValueError("Fed-PLoRA needs global LoRA factor pairs")
                original[name] = module
                local = FedPLoRALocalLinear(module.base, module.lora_a, module.lora_b,
                                           ids, scaling=module.scaling,
                                           dropout=module.dropout.p)
                _replace_module(model, name, local)
                selected_layers[name] = local
        for p in model.parameters():
            p.requires_grad_(False)
        adapter_params = []
        if method == "fedsb":
            adapter_params = [m.r for m in native_modules(model, method).values()]
        elif method == "fedsvd":
            for m in native_modules(model, method).values():
                m.factor = "b"
                adapter_params.append(m.lora_b)
        else:
            for m in selected_layers.values():
                adapter_params.extend([m.a, m.b])
        for p in adapter_params:
            p.requires_grad_(True)
        head_params = list(model.classifier.parameters())
        for p in head_params:
            p.requires_grad_(True)
        adapter_opt, head_opt = _optimizer(adapter_params, recipe), _optimizer(head_params, recipe, head=True)
        steps = 0
        model.train()
        for batch in batches(dataset, batch_size, keyed_seed(seed, "data", 0)):
            adapter_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            loss = model(**to_device(batch, device)).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite native local loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter_params + head_params):
                raise FloatingPointError("nonfinite native local gradient")
            if recipe.get("adapter_clip_norm") is not None:
                torch.nn.utils.clip_grad_norm_(adapter_params, float(recipe["adapter_clip_norm"]))
            adapter_opt.step()
            head_opt.step()
            steps += 1
        if method == "fedplora":
            child = {name: value.clone() for name, value in parent.items()}
            for name, local in selected_layers.items():
                child[name + ".lora_a"], child[name + ".lora_b"] = local.expand_into(
                    parent[name + ".lora_a"].to(device), parent[name + ".lora_b"].to(device))
                child[name + ".lora_a"] = child[name + ".lora_a"].detach().cpu()
                child[name + ".lora_b"] = child[name + ".lora_b"].detach().cpu()
            for name, p in model.classifier.named_parameters():
                child["classifier." + name] = p.detach().cpu().clone()
        else:
            child = native_state(model, method)
        return child, {"optimizer_steps": steps,
                       "optimizer_tensor_parameters": sum(p.numel() for p in adapter_params + head_params),
                       "train_seconds": time.perf_counter() - started,
                       "native_selected_positions": sum(map(len, selection.values()))}
    finally:
        if method == "fedplora":
            for name, module in original.items():
                _replace_module(model, name, module)
        if method == "fedsvd":
            for module in native_modules(model, method).values():
                module.factor = "none"


def _effective(model: nn.Module, method: str, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result = {}
    for name, module in native_modules(model, method).items():
        if method == "fedsb":
            product = values[name + ".b"].double() @ values[name + ".r"].double() @ values[name + ".a"].double()
        else:
            product = values[name + ".lora_b"].double() @ values[name + ".lora_a"].double()
        result[name] = module.scaling * product
    result.update({name: value.double().clone() for name, value in values.items()
                   if name.startswith("classifier.")})
    return result


def initial_ema_native(model: nn.Module, method: str, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    # CODEX_NEXT.md fixes E_0=0 even when Fed-SB's deployed initializer is
    # nonzero. The classifier starts from its initial weights separately.
    result = {name: torch.zeros_like(module.base.weight, dtype=torch.float64, device="cpu")
              for name, module in native_modules(model, method).items()}
    result.update({name: value.double().clone() for name, value in values.items()
                   if name.startswith("classifier.")})
    return result


def advance_ema_native(model: nn.Module, method: str, ema: Mapping[str, Tensor],
                       values: Mapping[str, Tensor], beta: float = .8) -> dict[str, Tensor]:
    now = _effective(model, method, values)
    return {name: beta * ema[name] + (1 - beta) * value for name, value in now.items()}


@torch.no_grad()
def evaluate_ema_native(model: nn.Module, method: str, ema: Mapping[str, Tensor],
                        dataset, batch_size: int = 64) -> dict:
    modules = native_modules(model, method)
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
        return evaluate(model, dataset, batch_size)
    finally:
        for n, m in modules.items():
            m.base.weight.copy_(base_before[n])
            m.scaling = scales[n]
        for n, value in head_before.items():
            params[n].copy_(value)


def _checkpoint(out: Path, payload: dict) -> None:
    latest, previous = out / "latest.pt", out / "previous.pt"
    if latest.exists():
        # Keep latest readable until the new checkpoint is fully fsynced and
        # atomically installed. A failed write must not erase the last round.
        pending_previous = out / f"previous.{os.getpid()}.tmp"
        pending_previous.unlink(missing_ok=True)
        os.link(latest, pending_previous)
        os.replace(pending_previous, previous)
    atomic_checkpoint(latest, payload)


def run_native(model: nn.Module, clients: Mapping[int, object], evaluation,
               capacities: Mapping[int, int], job: dict, recipe: dict, config: dict,
               out: Path, provenance: dict, *, stop_after: int | None = None,
               wire: NativeWire | None = None) -> list[dict]:
    method = job["method"]
    if method not in METHODS or recipe["required_adapter"] != method or recipe["method"] != method:
        raise ValueError("native method/adapter mismatch")
    if provenance.get("evaluation_role") not in {"recipe_dev", "synthetic"} or set(clients) != set(capacities):
        raise ValueError("native data roles or clients invalid")
    if provenance["evaluation_role"] == "recipe_dev" and (job.get("protocol") != PROTOCOL or
            job.get("eval_role") != "recipe_dev" or job.get("min_rounds") != 100 or job.get("max_rounds") != 200 or
            len(clients) != 50 or len(evaluation) != 4096):
        raise ValueError("formal native job differs from protocol")
    if provenance["evaluation_role"] == "recipe_dev":
        required_identity = {"config_sha256", "source_manifest_sha256", "manifest_sha256",
                             "token_cache_sha256", "pretrained_model_sha256",
                             "b8_wire_manifest_sha256", "native_sources"}
        if any(not provenance.get(key) for key in required_identity):
            raise ValueError("formal native source/data/config identity incomplete")
    if wire is None:
        wire = NativeWire()
    actual_hardware = provenance.get("hardware", {})
    identity_provenance = dict(provenance)
    identity_provenance["hardware"] = {k: actual_hardware[k] for k in ("name", "total_memory") if k in actual_hardware}
    if isinstance(identity_provenance.get("native_initialization"), dict):
        identity_provenance["native_initialization"] = {
            k: v for k, v in identity_provenance["native_initialization"].items()
            if k not in {"cost", "projections"}}
    initial = native_state(model, method)
    identity = fingerprint({"job": job, "recipe": recipe, "config": config,
                            "provenance": identity_provenance,
                            "initial_state": tensor_digest(initial)})
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        if ck.get("format") != CHECKPOINT_FORMAT or ck["identity"] != identity:
            raise ValueError("native resume identity changed")
        parent, rows, cache, deployed, diagnostic = (ck[k] for k in
            ("adapter", "metrics", "cache", "deployed_bytes", "diagnostic_bytes"))
        stop_round, budget_adapter, ema = ck["budget_stop_round"], ck["budget_adapter"], ck["ema_effective"]
        half_adapter, half_round, half_readout = ck["half_adapter"], ck["half_round"], ck["half_readout"]
        start = ck["round"] + 1
        torch.set_rng_state(ck["torch_rng"])
        random.setstate(ck["python_rng"])
        if torch.cuda.is_available() and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    else:
        allowed = {".run.lock", "dispatch_identity.json", "worker.log", "exception.txt",
                   "native_initialization.json", "native_initialization_attempts.json",
                   "execution_attempts.json", "wire_attempt_journal.json",
                   "evaluation_labels.json", "logits_round_000.json",
                   "evaluation_round_000.json", "status.json"}
        if any(p.name not in allowed for p in out.iterdir()):
            raise ValueError("nonempty native output without checkpoint")
        parent, rows, cache, deployed, diagnostic, stop_round, budget_adapter, start = initial, [], {}, 0, 0, None, None, 1
        ema = initial_ema_native(model, method, parent)
        half_adapter, half_round, half_readout = {k: v.clone() for k, v in parent.items()}, 0, None
    min_rounds, max_rounds, cap = int(job["min_rounds"]), int(job["max_rounds"]), int(config["byte_cap"])
    if cap <= 0 or max_rounds < min_rounds or min_rounds < 1:
        raise ValueError("invalid native budget/horizon")
    attempts_file = out / "execution_attempts.json"
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
    if any(item.get("identity") != identity for item in attempts):
        raise ValueError("native prior attempt identity differs")
    journal_file = out / "wire_attempt_journal.json"
    journal = json.loads(journal_file.read_text()) if journal_file.exists() else {
        "format": "pcla-v7-native-wire-attempt-1", "identity": identity, "events": []}
    if journal.get("format") != "pcla-v7-native-wire-attempt-1" or journal.get("identity") != identity:
        raise ValueError("native wire attempt journal identity differs")
    if not isinstance(journal.get("events"), list):
        raise ValueError("native wire attempt journal malformed")
    if latest.exists():
        committed_count = ck.get("journal_committed_event_count")
        committed_digest = ck.get("journal_committed_sha256")
        if (not isinstance(committed_count, int) or committed_count < 0 or
                committed_count > len(journal["events"]) or
                fingerprint(journal["events"][:committed_count]) != committed_digest):
            raise ValueError("native committed journal prefix differs from checkpoint")
    attempts.append({"identity": identity, "resume_round": start,
                     "utc_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "host": platform.node(), "pid": os.getpid(), "hardware": actual_hardware,
                     "gpu_sharing": os.environ.get("PCLA_GPU_SHARING", "unspecified"),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    atomic_json(attempts_file, attempts)
    attempt_number = len(attempts)

    def journal_events(round_id: int, events: list[dict], bucket: str):
        seen = {(e["attempt"], e["event_id"]) for e in journal["events"]}
        if any((attempt_number, event["event_id"]) in seen for event in events):
            raise ValueError("duplicate native attempted message")
        journal["events"].extend({**event, "attempt": attempt_number,
                                  "round": round_id, "bucket": bucket} for event in events)
        atomic_json(journal_file, journal)

    def attempted_cost() -> tuple[int, int]:
        attempted = sum(int(e["bytes"]) for e in journal["events"])
        failed = attempted - deployed - diagnostic
        if failed < 0:
            raise ValueError("native attempt journal below committed ledger")
        return attempted, failed

    def save(round_id: int):
        _checkpoint(out, {"format": CHECKPOINT_FORMAT, "identity": identity,
            "round": round_id, "adapter": parent, "budget_adapter": budget_adapter,
            "half_adapter": half_adapter, "half_round": half_round,
            "half_readout": half_readout, "ema_effective": ema, "metrics": rows,
            "cache": cache, "deployed_bytes": deployed, "diagnostic_bytes": diagnostic,
            "journal_committed_event_count": len(journal["events"]),
            "journal_committed_sha256": fingerprint(journal["events"]),
            "budget_stop_round": stop_round, "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})

    if start == 1:
        load_native_state(model, method, parent)
        zero = evaluate(model, evaluation, int(config.get("eval_batch_size", 64)), record_logits=True)
        atomic_json(out / "evaluation_labels.json", zero.pop("labels"))
        atomic_json(out / "logits_round_000.json", zero.pop("logits"))
        atomic_json(out / "evaluation_round_000.json", zero)
        half_readout = zero
        save(0)
    if method == "fedsvd":
        # A single fixed training-only probe for every refresh and recipe of
        # this partition. Cohort membership must not alter the tolerance test.
        anchor = clients[min(clients)]
        count = min(32, len(anchor))
        if provenance["evaluation_role"] == "recipe_dev" and count != 32:
            raise ValueError("FedSVD requires fixed 32 client-training probes")
        probe_rows = Subset(anchor, range(count))
        fixed_probe_batch = next(iter(batches(probe_rows, count, 0, shuffle=False)))
    else:
        fixed_probe_batch = None
    atomic_json(out / "status.json", {"state": "running", "completed_round": start - 1,
                                       "budget_stop_round": stop_round})
    for round_id in range(start, max_rounds + 1):
        if stop_round is not None and round_id > min_rounds:
            break
        begun = time.perf_counter()
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        previous = parent
        cohort = random.Random(keyed_seed(job["seed"], "participants", round_id)).sample(sorted(clients),
                               int(config.get("clients_per_round", 10)))
        coordinate = {"refreshed": False}
        coordinate_started = time.perf_counter()
        if method == "fedsvd":
            def probe(candidate):
                load_native_state(model, method, candidate)
                model.eval()
                return model(**to_device(fixed_probe_batch,
                                         next(model.parameters()).device)).logits.detach().cpu()
            parent, coordinate = refresh_state(parent, round_id, int(recipe["period"]),
                                               probe=probe,
                                               compute_device=next(model.parameters()).device)
        coordinate_seconds = time.perf_counter() - coordinate_started
        load_native_state(model, method, parent)
        selection_started = time.perf_counter()
        selection_job = dict(job, native_ranks=recipe.get("native_ranks"), _round=round_id)
        selections = _selected(method, parent, cohort, capacities, selection_job)
        scoring_seconds = time.perf_counter() - selection_started
        wire_started = time.perf_counter()
        preview = wire.preview(method, round_id, cohort, parent, selections, cache)
        encoding_seconds = time.perf_counter() - wire_started
        charge = preview["bytes"]
        if stop_round is None and deployed + charge > cap:
            stop_round = round_id - 1
            budget_adapter = {k: v.clone() for k, v in previous.items()}
            if rows:
                rows[-1]["budget_stop_round"] = stop_round
            if round_id > min_rounds:
                parent = previous
                save(round_id - 1)
                atomic_json(out / "metrics.json", rows)
                break
        bucket = "diagnostic_after_budget" if stop_round is not None else "deployed_prefix"
        journal_events(round_id, [event for event in preview["events"]
                                  if event["direction"] == "download"], bucket)
        children, details = {}, []
        for cid in cohort:
            child, detail = local_train_native(model, method, clients[cid], parent,
                selections[cid], recipe, seed=keyed_seed(job["seed"], "train", round_id, cid),
                batch_size=int(config.get("batch_size", 16)))
            if provenance["evaluation_role"] == "recipe_dev" and detail["optimizer_steps"] != 16:
                raise RuntimeError("native client must execute 16 steps")
            wire.validate_upload(method, preview, cid, parent, child, selections[cid])
            journal_events(round_id, [event for event in preview["events"]
                                      if event["event_id"] == f"r{round_id}:c{cid}:up"], bucket)
            children[cid] = child
            details.append({"client": cid, **detail})
        aggregation_started = time.perf_counter()
        if method == "fedsb":
            next_parent = aggregate_r_head(parent, list(children.values()), [len(clients[cid]) for cid in cohort])
            agg_info = {}
        elif method == "fedsvd":
            next_parent = aggregate_b_head(parent, list(children.values()), [len(clients[cid]) for cid in cohort])
            agg_info = {}
        else:
            next_parent, agg_info = aggregate_contributors(parent, children, selections,
                                                            {cid: len(clients[cid]) for cid in cohort})
        aggregation_seconds = time.perf_counter() - aggregation_started
        wire_started = time.perf_counter()
        cache = wire.commit(method, preview, cache, parent, children, selections,
                            diagnostic=stop_round is not None)
        encoding_seconds += time.perf_counter() - wire_started
        parent = next_parent
        deployed += 0 if stop_round is not None else charge
        diagnostic += charge if stop_round is not None else 0
        load_native_state(model, method, parent)
        eval_started = time.perf_counter()
        readout = evaluate(model, evaluation, int(config.get("eval_batch_size", 64)), record_logits=True)
        labels = readout.pop("labels")
        if labels != json.loads((out / "evaluation_labels.json").read_text()):
            raise ValueError("native evaluation labels changed")
        atomic_json(out / f"logits_round_{round_id:03d}.json", readout.pop("logits"))
        evaluation_seconds = time.perf_counter() - eval_started
        ema_started = time.perf_counter()
        ema = advance_ema_native(model, method, ema, parent)
        ema_update_seconds = time.perf_counter() - ema_started
        ema_started = time.perf_counter()
        ema_readout = evaluate_ema_native(model, method, ema, evaluation,
                                          int(config.get("eval_batch_size", 64)))
        ema_eval_seconds = time.perf_counter() - ema_started
        if stop_round is None and deployed <= cap // 2:
            half_adapter = {k: v.clone() for k, v in parent.items()}
            half_round, half_readout = round_id, dict(readout)
        attempted_bytes, failed_bytes = attempted_cost()
        row = {"round": round_id, "selected_clients": cohort, "deployed_bytes": deployed,
               "diagnostic_after_budget_bytes": diagnostic, "budget_stop_round": stop_round,
               "actual_attempted_wire_bytes": attempted_bytes,
               "failed_attempt_wire_bytes": failed_bytes,
               "evaluation": readout, "ema_evaluation": ema_readout, "client_training": details,
               "coordinate_refresh": coordinate, "native_aggregation": agg_info,
               "support_trace": {str(cid): {"modules": {n: list(ids) for n, ids in selections[cid].items()},
                                                "selected_positions": sum(map(len, selections[cid].values()))}
                                 for cid in cohort},
               "cost": {"round_compute_wall_seconds": time.perf_counter() - begun,
                        "scoring_seconds": scoring_seconds,
                        "cohort_preparation_seconds": 0.,
                        "coordinate_seconds": coordinate_seconds,
                        "wire_encode_decode_seconds": encoding_seconds,
                        "coordination_preflight_seconds": 0.,
                        "local_training_seconds": sum(d["train_seconds"] for d in details),
                        "aggregation_seconds": aggregation_seconds,
                        "raw_evaluation_seconds": evaluation_seconds,
                        "ema_update_seconds": ema_update_seconds,
                        "ema_evaluation_seconds": ema_eval_seconds,
                        "cpu_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                               (1024 if sys.platform == "darwin" else 1)),
                        "cuda_peak_allocated_bytes": 0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device),
                        "cuda_peak_reserved_bytes": 0 if device.type != "cuda" else torch.cuda.max_memory_reserved(device)},
               "gpu_sharing": os.environ.get("PCLA_GPU_SHARING", "unspecified")}
        cumulative = dict(rows[-1]["cumulative_cost"]) if rows else {}
        for key, value in row["cost"].items():
            cumulative[key] = max(cumulative.get(key, 0), value) if "peak" in key else cumulative.get(key, 0) + value
        row["cumulative_cost"] = cumulative
        rows.append(row)
        save(round_id)
        if round_id % 10 == 0 or round_id in {60, stop_round} or min_rounds - 4 <= round_id <= min_rounds or round_id >= max_rounds - 4:
            atomic_checkpoint(out / f"adapter_round_{round_id:03d}.pt",
                              {"round": round_id, "adapter": parent, "deployed_bytes": deployed,
                               "diagnostic_bytes": diagnostic})
        atomic_json(out / "metrics.json", rows)
        atomic_json(out / f"round_wall_{round_id:03d}.json",
                    {"round": round_id, "total_wall_seconds": time.perf_counter() - begun})
        atomic_json(out / "status.json", {"state": "running", "completed_round": round_id,
                                           "budget_stop_round": stop_round,
                                           "actual_attempted_wire_bytes": attempted_bytes,
                                           "failed_attempt_wire_bytes": failed_bytes})
        if stop_after is not None and round_id >= stop_after:
            atomic_json(out / "status.json", {"state": "paused", "completed_round": round_id,
                                               "budget_stop_round": stop_round,
                                               "actual_attempted_wire_bytes": attempted_bytes,
                                               "failed_attempt_wire_bytes": failed_bytes})
            break
        if stop_round is not None and round_id >= min_rounds:
            break
    if rows and rows[-1]["round"] == max_rounds and stop_round is None:
        stop_round = max_rounds
        budget_adapter = {k: v.clone() for k, v in parent.items()}
        rows[-1]["budget_stop_round"] = stop_round
        save(max_rounds)
        atomic_json(out / "metrics.json", rows)
    if budget_adapter is not None:
        atomic_checkpoint(out / "budget_return.pt", {"format": CHECKPOINT_FORMAT,
            "round": stop_round, "adapter": budget_adapter, "deployed_bytes": deployed,
            "budget_cap_bytes": cap, "identity": identity})
    atomic_checkpoint(out / "half_budget_return.pt", {"format": CHECKPOINT_FORMAT,
        "round": half_round, "adapter": half_adapter,
        "deployed_bytes": rows[half_round - 1]["deployed_bytes"] if half_round else 0,
        "budget_cap_bytes": cap // 2, "readout": half_readout, "identity": identity})
    if stop_after is None and rows:
        attempted_bytes, failed_bytes = attempted_cost()
        atomic_json(out / "status.json", {"state": "completed", "completed_round": rows[-1]["round"],
            "budget_stop_round": stop_round, "deployed_bytes": deployed,
            "diagnostic_after_budget_bytes": diagnostic,
            "actual_attempted_wire_bytes": attempted_bytes,
            "failed_attempt_wire_bytes": failed_bytes})
    return rows
