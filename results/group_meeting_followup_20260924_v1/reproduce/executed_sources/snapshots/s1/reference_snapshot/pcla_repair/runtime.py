"""Single-device simulator; parallelize independent jobs, never client state.

The same model object is reused sequentially for clients. Only adapter/head
states and accumulated deltas are retained on CPU, not ten full backbones.
"""
from __future__ import annotations

from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .optimization_dev import (make_optimizer, TrainReadout, classification_readout,
                               adapter_readout, constant_references)

from .core import (UtilityState, DeltaAccumulator, FrozenSliceGuard,
                   adapter_state, load_adapter, configure, component_scores,
                   allocate, keyed_seed, tensor_digest, lora_modules)

POLICIES = ("pcla", "layer_only", "random", "replay_random_rank", "cyclic",
            "deep_layer", "random_layer", "uniform_cyclic", "flat_topk", "full_alt", "full_both", "head_only")


def fingerprint(value: object) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        torch.save(value, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def load_checkpoint(path: Path) -> dict:
    if path.suffix != ".pt":
        raise ValueError("Legacy checkpoint.json stores logs, not model weights. Generate new probe checkpoints.")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "pcla-repair-v1" or "adapter" not in value:
        raise ValueError("not a versioned PCLA model checkpoint")
    return value


def seed_train(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batches(dataset: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k:v.to(device) for k,v in batch.items()}


def probe(model: nn.Module, layout: dict[str, int], dataset: Dataset,
          factor: str, kind: str, seed: int, cfg: dict, device: torch.device) -> tuple[Tensor, float]:
    rank = next(iter(lora_modules(model).values())).rank
    full = torch.ones(max(layout.values())+1, rank, dtype=torch.bool)
    configure(model, layout, full, factor, head=False)
    model.eval()
    synchronize(device); started = time.perf_counter()
    total, count = torch.zeros_like(full, dtype=torch.float64), 0
    for i,batch in enumerate(batches(dataset, cfg["probe_batch_size"], seed)):
        if i >= cfg["probe_batches"]:
            break
        model.zero_grad(set_to_none=True)
        loss = model(**to_device(batch, device)).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite probe loss")
        loss.backward()
        total += component_scores(model, layout, factor, kind)
        count += 1
    model.zero_grad(set_to_none=True)
    synchronize(device)
    if count == 0:
        raise ValueError("empty probe dataset")
    return total/count, time.perf_counter()-started


def local_train(model: nn.Module, layout: dict[str,int], dataset: Dataset,
                support: Tensor, factor: str, seed: int, cfg: dict,
                device: torch.device, *, diagnostic: dict | None = None
                ) -> tuple[dict[str,Tensor], float, int]:
    masks = configure(model, layout, support, factor,
                      forward_mode=cfg["forward_mode"], head=cfg["train_head"])
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    optimizer = make_optimizer(model, cfg)
    readout = TrainReadout(model) if diagnostic is not None else None
    guard = FrozenSliceGuard(model, masks)
    seed_train(seed)
    model.train()
    synchronize(device); started = time.perf_counter()
    for epoch in range(cfg["local_epochs"]):
        loader = batches(dataset, cfg["batch_size"], keyed_seed(seed, "data", epoch))
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**to_device(batch, device)).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("nonfinite training gradient")
            if readout is not None:
                readout.observe(loss, batch["labels"])
            if cfg.get("clip_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg["clip_grad_norm"])
            optimizer.step()
            guard.restore(optimizer)
    synchronize(device)
    elapsed = time.perf_counter()-started
    allocated = sum(p.numel() for p in params)
    model.zero_grad(set_to_none=True)
    if diagnostic is not None:
        diagnostic.update(readout.result())
        diagnostic["effective_head_learning_rate"] = cfg.get("head_learning_rate", cfg["learning_rate"])
        diagnostic["effective_lora_learning_rate"] = cfg.get("lora_learning_rate", cfg["learning_rate"])
    return masks, elapsed, allocated


@torch.no_grad()
def evaluate(model: nn.Module, layout: dict[str,int], dataset: Dataset,
             batch_size: int, device: torch.device, *, record_logits: bool = False) -> dict:
    rank = next(iter(lora_modules(model).values())).rank
    full = torch.ones(max(layout.values())+1, rank, dtype=torch.bool)
    configure(model, layout, full, "none", head=False)
    model.eval()
    labels, predictions, loss_sum, count = [], [], 0., 0
    saved_logits = []
    for batch in batches(dataset, batch_size, 0, False):
        b = to_device(batch, device)
        out = model(**b)
        if not torch.isfinite(out.logits).all() or not torch.isfinite(out.loss):
            raise FloatingPointError("nonfinite evaluation")
        y = b["labels"]
        labels.extend(y.cpu().tolist())
        predictions.extend(out.logits.argmax(-1).cpu().tolist())
        if record_logits:
            saved_logits.append(out.logits.detach().cpu())
        loss_sum += float(out.loss)*y.numel(); count += y.numel()
    if not count:
        raise ValueError("empty evaluation dataset")
    accuracy = sum(a==b for a,b in zip(labels,predictions))/count
    result = {"accuracy":accuracy,"loss":loss_sum/count,"samples":count,
              "predictions":predictions}
    if record_logits:
        logits = torch.cat(saved_logits)
        result.update(classification_readout(logits, torch.tensor(labels)))
        result["logits"] = logits.tolist()
        result["labels"] = labels
    return result


def payload_size(payload: dict) -> int:
    buffer = io.BytesIO(); torch.save(payload, buffer)
    return buffer.tell()


def upload_payload(base: Mapping[str,Tensor], local: Mapping[str,Tensor],
                   layout: dict[str,int], support: Tensor, masks: Mapping[str,Tensor],
                   round_id: int, client: int, factor: str) -> dict:
    tensors = {}
    for n in masks:
        delta = local[n]-base[n]
        if n.startswith("classifier."):
            tensors[n] = delta.contiguous()
        else:
            module_name, parameter = n.rsplit(".",1)
            ids = support[layout[module_name]].nonzero().flatten()
            tensors[n] = (delta[ids,:] if parameter == "lora_a" else delta[:,ids]).contiguous()
    return {"round":round_id,"client":client,"factor":factor,"layout":layout,
            "rank_ids":{str(l):support[l].nonzero().flatten() for l in range(len(support))
                        if support[l].any()},"delta":tensors}


def run_experiment(model: nn.Module, layout: dict[str,int],
                   clients: Mapping[int,Dataset], capacities: Mapping[int,int],
                   evaluation: Dataset, cfg: dict, seed: int, policy: str,
                   out_dir: Path, provenance: dict, *, donor_path: Path | None = None,
                   stop_after: int | None = None, partition: dict | None = None) -> list[dict]:
    """Caller owns an exclusive out_dir lock. stop_after supports resume tests."""
    if policy not in POLICIES or not clients or set(clients) != set(capacities):
        raise ValueError("invalid experiment policy/clients")
    if cfg["selected_clients"] > len(clients) or cfg["selected_clients"] < 1:
        raise ValueError("invalid participation")
    development = bool(cfg.get("record_optimization_diagnostics", False))
    if development and not (provenance.get("evaluation_split") == "tuning" or provenance.get("synthetic")):
        raise ValueError("optimization development must use train-derived tuning data")
    device = next(model.parameters()).device
    layers, rank = max(layout.values())+1, next(iter(lora_modules(model).values())).rank
    shape = (layers,rank)
    initial = adapter_state(model)
    identity = {"seed":seed,"config":cfg,"provenance":provenance,
                "initial_adapter_sha256":tensor_digest(initial)}
    identity_hash = fingerprint(identity)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir/"checkpoint.pt"
    states = {f"{cid}/{factor}":UtilityState() for cid in clients for factor in ("a","b")}
    global_state, rows, trace, start = initial, [], {}, 1
    if checkpoint_path.exists():
        restored = load_checkpoint(checkpoint_path)
        if restored["identity_hash"] != identity_hash or restored["policy"] != policy:
            raise ValueError("resume identity changed: use a new output directory")
        global_state,rows,trace = restored["adapter"],restored["metrics"],restored["trace"]
        states = {k:UtilityState(**v) for k,v in restored["client_states"].items()}
        start = restored["round"]+1
    elif any(out_dir.iterdir()):
        # CLI creates the lock before calling us.
        extras = [p for p in out_dir.iterdir() if p.name != ".run.lock"]
        if extras:
            raise ValueError("nonempty output lacks resumable checkpoint; use a new directory")
    donor = None
    if policy == "replay_random_rank":
        if donor_path is None:
            raise ValueError("missing donor checkpoint")
        donor = load_checkpoint(donor_path)
        if donor["identity_hash"] != identity_hash or donor["policy"] != "pcla" or donor["round"] != cfg["rounds"]:
            raise ValueError("donor must be a complete PCLA run with the exact same identity")
    atomic_json(out_dir/"identity.json", {**identity,"identity_hash":identity_hash,"policy":policy})
    def snapshot(round_id: int) -> dict:
        return {"format":"pcla-repair-v1","policy":policy,"identity_hash":identity_hash,
                "identity":identity,"round":round_id,"adapter":global_state,
                "metrics":rows,"trace":trace,"client_states":{k:asdict(v) for k,v in states.items()},
                "torch_rng_state":torch.get_rng_state()}
    if development:
        atomic_json(out_dir/"diagnostic_references.json", constant_references(clients, evaluation))
        if start == 1:
            zero = evaluate(model, layout, evaluation, cfg["eval_batch_size"], device, record_logits=True)
            atomic_json(out_dir/"evaluation_labels.json", zero.pop("labels"))
            atomic_json(out_dir/"logits_round_000.json", zero.pop("logits"))
            atomic_json(out_dir/"evaluation_round_000.json", zero)
    if start == 1:
        atomic_checkpoint(checkpoint_path, snapshot(0))
        atomic_checkpoint(out_dir/"probe_round_000.pt", snapshot(0))
    if partition is not None:
        # Persist IDs before training so even an interrupted run is traceable.
        if fingerprint(partition) != provenance.get("partition_sha256"):
            raise ValueError("partition IDs do not match their recorded fingerprint")
        atomic_json(out_dir/"partition_ids.json", partition)
    import platform, datetime
    attempts_path=out_dir/"execution_attempts.json"
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append({"host":platform.node(),"python":platform.python_version(),"device":str(device),
                     "gpu_name":None if device.type!="cuda" else torch.cuda.get_device_name(device),
                     "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"pid":os.getpid(),
                     "utc_started":datetime.datetime.now(datetime.timezone.utc).isoformat(),"resume_round":start})
    atomic_json(attempts_path,attempts)
    try:
        for r in range(start,cfg["rounds"]+1):
            atomic_json(out_dir/"status.json", {"state":"running","round":r,"completed_round":r-1})
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            round_start=time.perf_counter()
            selected = random.Random(keyed_seed(seed,"participants",r)).sample(sorted(clients), cfg["selected_clients"])
            weight = sum(len(clients[c]) for c in selected)
            aggregator = DeltaAccumulator(global_state, weight, cfg["aggregation"])
            train_seconds,probe_seconds,traffic,details = 0.,0.,0,[]
            factor = "b" if r%2 else "a"
            for cid in selected:
                load_adapter(model,global_state)
                state = states[f"{cid}/{factor}"]
                scores = torch.zeros(shape,dtype=torch.float64)
                measured_probe = 0.
                if policy in ("pcla","layer_only","flat_topk"):
                    if state.importance is None or r-state.last_refresh >= cfg["refresh_rounds"]:
                        raw,measured_probe = probe(model,layout,clients[cid],factor,cfg["score"],
                                                  keyed_seed(seed,"probe",r,cid),cfg,device)
                        state.refresh(raw,r,cfg["ema"])
                    scores=state.score(r,cfg["beta"],cfg["gamma"])
                train_factor=factor
                if policy in ("full_alt","full_both"):
                    support=torch.ones(shape,dtype=torch.bool)
                    if policy=="full_both": train_factor="both"
                elif policy=="head_only":
                    support=torch.zeros(shape,dtype=torch.bool); train_factor="none"
                else:
                    donor_support=None if donor is None else donor["trace"][f"{r}/{cid}"]
                    support=allocate(policy,scores,capacities[cid],cfg["credits_per_layer"],
                                     keyed_seed(seed,"allocation",policy,r,cid),donor=donor_support,
                                     visit=state.visits)
                    if policy in ("cyclic", "uniform_cyclic"):
                        # A fixed offset is needed; a round-dependent random seed
                        # would destroy cyclic coverage.
                        support=allocate(policy,scores,capacities[cid],cfg["credits_per_layer"],
                                         keyed_seed(seed,"cyclic",cid,factor),visit=state.visits)
                local_diagnostic = {} if development else None
                masks,elapsed,allocated=local_train(model,layout,clients[cid],support,train_factor,
                                                    keyed_seed(seed,"train",r,cid),cfg,device,
                                                    diagnostic=local_diagnostic)
                local=adapter_state(model)
                aggregator.add(local,masks,len(clients[cid]))
                upload=upload_payload(global_state,local,layout,support,masks,r,cid,train_factor)
                download_bytes=payload_size({"round":r,"adapter":global_state})
                upload_bytes=payload_size(upload)
                logical=sum(int(mask.sum()) for mask in masks.values())
                details.append({"client":cid,"support":support.tolist(),"factor":train_factor,
                                "sample_weight":len(clients[cid]),"credits":int(support.sum()),
                                "logical_update_parameters":logical,"optimizer_tensor_parameters":allocated,
                                "download_bytes":download_bytes,"upload_bytes":upload_bytes,
                                "probe_seconds":measured_probe,"train_seconds":elapsed,
                                "score_age":None if state.last_refresh<0 else r-state.last_refresh})
                if development:
                    details[-1]["optimization"] = local_diagnostic
                    details[-1]["adapter_update"] = adapter_readout(local, global_state)
                trace[f"{r}/{cid}"]=support.clone()
                state.observe(support,r)
                probe_seconds+=measured_probe; train_seconds+=elapsed; traffic+=download_bytes+upload_bytes
                del local,upload,masks
            previous_global_state = global_state
            global_state=aggregator.finish()
            load_adapter(model,global_state)
            eval_start=time.perf_counter()
            evaluation_result=evaluate(model,layout,evaluation,cfg["eval_batch_size"],device,
                                       record_logits=development)
            if development:
                labels = evaluation_result.pop("labels")
                if labels != json.loads((out_dir/"evaluation_labels.json").read_text()):
                    raise ValueError("evaluation labels/order changed")
                atomic_json(out_dir/f"logits_round_{r:03d}.json", evaluation_result.pop("logits"))
                evaluation_result["adapter_diagnostics"] = adapter_readout(global_state, previous_global_state)
            eval_seconds=time.perf_counter()-eval_start
            predictions=evaluation_result.pop("predictions")
            rows.append({"round":r,**evaluation_result,"train_seconds":train_seconds,
                         "probe_seconds":probe_seconds,"eval_seconds":eval_seconds,
                         "simulator_round_wall_seconds":time.perf_counter()-round_start,
                         "serialized_traffic_bytes":traffic,"selected_clients":selected,
                         "cuda_peak_allocated":0 if device.type!="cuda" else torch.cuda.max_memory_allocated(device),
                         "cuda_peak_reserved":0 if device.type!="cuda" else torch.cuda.max_memory_reserved(device)})
            atomic_json(out_dir/f"clients_round_{r:03d}.json",details)
            atomic_json(out_dir/f"predictions_round_{r:03d}.json",predictions)
            atomic_json(out_dir/"metrics.json",rows)
            atomic_checkpoint(checkpoint_path,snapshot(r))
            if r in cfg["probe_rounds"]:
                atomic_checkpoint(out_dir/f"probe_round_{r:03d}.pt",snapshot(r))
            print(json.dumps({"policy":policy,"seed":seed,**rows[-1]},ensure_ascii=False),flush=True)
            if stop_after is not None and r>=stop_after and r<cfg["rounds"]:
                atomic_json(out_dir/"status.json",{"state":"paused","completed_round":r})
                return rows
        atomic_json(out_dir/"status.json",{"state":"completed","completed_round":cfg["rounds"]})
    except BaseException as exc:
        atomic_json(out_dir/"status.json",{"state":"failed","exception":type(exc).__name__,"message":str(exc),
                                           "last_saved_round":load_checkpoint(checkpoint_path)["round"]})
        raise
    return rows
