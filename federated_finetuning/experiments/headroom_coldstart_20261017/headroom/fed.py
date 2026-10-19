"""Federated rounds, component-wise aggregation, evaluation and resumable checkpoints."""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from . import METHODS
from .data import ClientStream, Dataset, derived_seed, override_roles
from .model import TaskModel, is_lora_a, is_lora_b, upload_parameter_count


# ---------------------------------------------------------------- method spec
def client_rank(method: str, role: str, full_rank: int, weak_rank: int) -> int:
    if method == "fedavg_r8":
        return full_rank
    if method == "fedavg_r2":
        return weak_rank
    if method in {"hetero_r8r2", "local_only"}:
        return full_rank if role == "strong" else weak_rank
    raise ValueError(f"Unknown method {method}; expected one of {METHODS}")


def is_federated(method: str) -> bool:
    return method != "local_only"


# ---------------------------------------------------------------- aggregation
def aggregate(updates: list[tuple[dict, int, float]], previous: dict) -> dict:
    """Sample-weighted FedAvg of factors, per LoRA component over clients that trained it.

    updates: (state, train_rank, weight). Component k of A (row) and B (column) averages
    only clients with train_rank > k; a component nobody trained keeps the server value.
    The classifier head averages all participants.
    """
    weights = torch.tensor([w for _, _, w in updates], dtype=torch.float64)
    ranks = torch.tensor([r for _, r, _ in updates])
    result = {}
    for name, old in previous.items():
        stacked = torch.stack([s[name].double() for s, _, _ in updates])
        if is_lora_a(name) or is_lora_b(name):
            full = old.shape[0] if is_lora_a(name) else old.shape[1]
            # (clients, components) weight matrix: w_i if client i trained component k.
            mask = (torch.arange(full)[None, :] < ranks[:, None]).double() * weights[:, None]
            denom = mask.sum(0)
            if is_lora_a(name):
                summed = torch.einsum("ck,ckd->kd", mask, stacked)
                value = torch.where(denom[:, None] > 0, summed / denom.clamp_min(1e-300)[:, None], old.double())
            else:
                summed = torch.einsum("ck,cdk->dk", mask, stacked)
                value = torch.where(denom[None, :] > 0, summed / denom.clamp_min(1e-300)[None, :], old.double())
        else:
            value = torch.einsum("c,c...->...", weights, stacked) / weights.sum()
        result[name] = value.to(old.dtype)
    return result


# ---------------------------------------------------------------- local training
def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def local_train(model: TaskModel, start: dict, data: Dataset, stream: ClientStream, rank: int,
                cfg: dict, device: torch.device) -> tuple[dict, dict]:
    model.load_task_state(start)
    model.train()
    params = list(model.task_parameters().values())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                            weight_decay=cfg["weight_decay"])
    losses = []
    sync(device)
    started = time.perf_counter()
    for _ in range(cfg["local_steps"]):
        batch = data.train.batch(stream.take(cfg["batch_size"]), device)
        loss = torch.nn.functional.cross_entropy(model(batch), batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.mask_gradients(rank)
        if cfg["gradient_clip"]:
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
        opt.step()
        losses.append(float(loss.detach()))
    sync(device)
    elapsed = time.perf_counter() - started
    model.eval()
    if not all(math.isfinite(x) for x in losses):
        raise FloatingPointError("Nonfinite local training loss")
    return model.task_state(), {"seconds": elapsed, "mean_loss": float(np.mean(losses)),
                                "last_loss": losses[-1]}


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def row_scores(model: TaskModel, data: Dataset, indices: list[int], device, batch_size: int):
    """Per-row NLL and correctness for the given validation indices (length-sorted batches)."""
    model.eval()
    order = sorted(indices, key=lambda i: (data.valid.length(i), i))
    nll, correct = {}, {}
    for start in range(0, len(order), batch_size):
        ids = order[start : start + batch_size]
        batch = data.valid.batch(ids, device)
        logits = model(batch).double()
        values = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none")
        hits = logits.argmax(-1) == batch["labels"]
        for i, v, h in zip(ids, values.cpu().tolist(), hits.cpu().tolist()):
            nll[i], correct[i] = v, bool(h)
    if not all(math.isfinite(v) for v in nll.values()):
        raise FloatingPointError("Nonfinite evaluation NLL")
    return nll, correct


def per_client(data: Dataset, nll: dict, correct: dict) -> dict[int, dict]:
    out = {}
    for cid, rows in data.valid_by_client.items():
        out[cid] = {"nll": float(np.mean([nll[i] for i in rows])),
                    "accuracy": float(np.mean([correct[i] for i in rows])),
                    "rows": len(rows)}
    return out


def summarize_clients(data: Dataset, clients: dict[int, dict], ranks: dict[int, int]) -> dict:
    def macro(ids, key):
        return float(np.mean([clients[c][key] for c in ids])) if ids else None

    all_ids = sorted(clients)
    rows = sum(clients[c]["rows"] for c in all_ids)
    summary = {
        "client_macro_accuracy": macro(all_ids, "accuracy"),
        "client_macro_nll": macro(all_ids, "nll"),
        "sample_accuracy": sum(clients[c]["accuracy"] * clients[c]["rows"] for c in all_ids) / rows,
        "sample_nll": sum(clients[c]["nll"] * clients[c]["rows"] for c in all_ids) / rows,
        "by_role": {}, "by_domain": {}, "by_train_rank": {},
    }
    for key, groups in (("by_role", lambda c: data.clients[c]["role"]),
                        ("by_domain", lambda c: data.clients[c]["domain"]),
                        ("by_train_rank", lambda c: str(ranks[c]))):
        buckets: dict[str, list[int]] = {}
        for c in all_ids:
            buckets.setdefault(groups(c), []).append(c)
        summary[key] = {g: {"client_macro_accuracy": macro(ids, "accuracy"),
                            "client_macro_nll": macro(ids, "nll"), "clients": len(ids)}
                        for g, ids in sorted(buckets.items())}
    return summary


def evaluate(model, data, device, cfg, method, global_state, local_states, ranks) -> tuple[dict, dict]:
    started = time.perf_counter()
    if is_federated(method):
        model.load_task_state(global_state)
        everything = sorted(i for rows in data.valid_by_client.values() for i in rows)
        nll, correct = row_scores(model, data, everything, device, cfg["eval_batch_size"])
        clients = per_client(data, nll, correct)
    else:
        clients = {}
        for cid, rows in sorted(data.valid_by_client.items()):
            model.load_task_state(local_states[cid])
            nll, correct = row_scores(model, data, rows, device, cfg["eval_batch_size"])
            clients[cid] = {"nll": float(np.mean(list(nll.values()))),
                            "accuracy": float(np.mean(list(correct.values()))), "rows": len(rows)}
    sync(device)
    summary = summarize_clients(data, clients, ranks)
    summary["seconds"] = time.perf_counter() - started
    return summary, clients


# ---------------------------------------------------------------- checkpoint io
def atomic_save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def participants(seed: int, round_id: int, clients: list[int], k: int) -> list[int]:
    return sorted(random.Random(derived_seed(seed, "participants", round_id)).sample(clients, k))


# ---------------------------------------------------------------- main loop
def client_curve(clients: dict[int, dict]) -> dict[str, list[float]]:
    return {str(c): [v["accuracy"], v["nll"]] for c, v in sorted(clients.items())}


def run(plan: dict, method: str, seed: int, output: Path, device: torch.device,
        resume: bool = False, stop_after_round: int | None = None,
        strong_fraction: float | None = None) -> dict:
    cfg = plan["train"]
    output.mkdir(parents=True, exist_ok=True)
    ckpt_path = output / "checkpoint.pt"
    if (output / "result.json").exists():
        raise FileExistsError(f"{output} already has result.json; refusing to overwrite")
    if ckpt_path.exists() and not resume:
        raise FileExistsError(f"{ckpt_path} exists; pass --resume to continue")

    torch.set_num_threads(plan.get("cpu_threads", 4))
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    strong_clients = (override_roles(data.clients, strong_fraction) if strong_fraction is not None
                      else sorted(c for c, v in data.clients.items() if v["role"] == "strong"))
    full_rank, weak_rank = plan["lora"]["full_rank"], plan["lora"]["weak_rank"]
    model = TaskModel.build(plan["model"], plan["num_labels"], full_rank, plan["lora"]["scale"], seed).to(device)
    client_ids = sorted(data.clients)
    ranks = {c: client_rank(method, data.clients[c]["role"], full_rank, weak_rank) for c in client_ids}
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    identity = {"method": method, "seed": seed, "plan_id": plan["id"], "client_ranks": ranks,
                "strong_fraction": strong_fraction, "strong_clients": strong_clients}

    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity differs from requested run")
        print(f"resuming {method} seed={seed} after round {state['round']}", flush=True)
    else:
        initial = model.task_state()
        state = {
            "identity": identity, "round": 0, "global": initial,
            "local": {c: {k: v.clone() for k, v in initial.items()} for c in client_ids}
            if not is_federated(method) else None,
            "positions": {c: 0 for c in client_ids},
            "history": [], "rounds": [], "elapsed_seconds": 0.0,
        }
        summary, clients = evaluate(model, data, device, cfg, method, state["global"], state["local"], ranks)
        state["history"].append({"round": 0, **summary, "clients": client_curve(clients)})
        atomic_json(output / "identity.json", {**identity, "dataset": data.manifest_summary,
                                                "plan": plan, "torch": torch.__version__,
                                                "device": str(device), "host": platform.node(),
                                                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    streams = {c: ClientStream(data.clients[c]["train_indices"], seed, c, state["positions"][c]) for c in client_ids}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rounds = plan["federation"]["rounds"]
    for round_id in range(state["round"] + 1, rounds + 1):
        started = time.perf_counter()
        chosen = participants(seed, round_id, client_ids, plan["federation"]["clients_per_round"])
        updates, records = [], []
        for c in chosen:
            start = state["global"] if is_federated(method) else state["local"][c]
            new_state, record = local_train(model, start, data, streams[c], ranks[c], cfg, device)
            if is_federated(method):
                updates.append((new_state, ranks[c], weights[c]))
            else:
                state["local"][c] = new_state
            records.append({"client": c, "rank": ranks[c],
                            "upload_params": upload_parameter_count(new_state, ranks[c])
                            if is_federated(method) else 0, **record})
        if is_federated(method):
            state["global"] = aggregate(updates, state["global"])
        state["positions"] = {c: s.position for c, s in streams.items()}
        round_seconds = time.perf_counter() - started
        state["rounds"].append({"round": round_id, "clients": chosen, "seconds": round_seconds,
                                "train_seconds": sum(r["seconds"] for r in records),
                                "mean_loss": float(np.mean([r["mean_loss"] for r in records])),
                                "upload_params": sum(r["upload_params"] for r in records),
                                "client_records": records})
        if round_id % plan["evaluation"]["every_rounds"] == 0 or round_id == rounds:
            summary, clients = evaluate(model, data, device, cfg, method, state["global"], state["local"], ranks)
            state["history"].append({"round": round_id, **summary, "clients": client_curve(clients)})
            print(f"[{method} s{seed}] round {round_id}: macro_acc={summary['client_macro_accuracy']:.4f} "
                  f"macro_nll={summary['client_macro_nll']:.4f} loss={state['rounds'][-1]['mean_loss']:.4f}",
                  flush=True)
        state["round"] = round_id
        state["elapsed_seconds"] += time.perf_counter() - started
        if round_id % plan["checkpoint_every"] == 0 or round_id == rounds:
            atomic_save(ckpt_path, state)
            atomic_json(output / "status.json", {"round": round_id, "rounds": rounds,
                                                 "elapsed_seconds": state["elapsed_seconds"]})
        if stop_after_round is not None and round_id >= stop_after_round:
            atomic_save(ckpt_path, state)
            return {"stopped_after_round": round_id}

    final, final_clients = evaluate(model, data, device, cfg, method, state["global"], state["local"], ranks)
    result = {
        **identity, "status": "completed", "rounds": rounds,
        "final": final, "final_clients": {str(c): v for c, v in final_clients.items()},
        "history": state["history"],
        "cost": {"elapsed_seconds": state["elapsed_seconds"],
                 "train_seconds": sum(r["train_seconds"] for r in state["rounds"]),
                 "optimizer_steps": rounds * plan["federation"]["clients_per_round"] * cfg["local_steps"],
                 "upload_params_total": sum(r["upload_params"] for r in state["rounds"]),
                 "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None},
        "round_loss": [r["mean_loss"] for r in state["rounds"]],
    }
    if is_federated(method) and plan["personalize"]["steps"] > 0:
        result["personalized"] = personalize(model, data, device, plan, state, streams, ranks)
    torch.save({"global": state["global"], "local": state["local"]}, output / "final_state.pt")
    atomic_json(output / "result.json", result)
    return result


def personalize(model, data, device, plan, state, streams, ranks) -> dict:
    """FedAvg+FT reference: each client fine-tunes the final global model on its own data.

    Uses fresh streams positioned after training so no training batch order changes.
    """
    cfg = dict(plan["train"], local_steps=plan["personalize"]["steps"])
    started = time.perf_counter()
    clients = {}
    for cid, rows in sorted(data.valid_by_client.items()):
        stream = ClientStream(data.clients[cid]["train_indices"], streams[cid].seed, cid, streams[cid].position)
        tuned, _ = local_train(model, state["global"], data, stream, ranks[cid], cfg, device)
        model.load_task_state(tuned)
        nll, correct = row_scores(model, data, rows, device, cfg["eval_batch_size"])
        clients[cid] = {"nll": float(np.mean(list(nll.values()))),
                        "accuracy": float(np.mean(list(correct.values()))), "rows": len(rows)}
    summary = summarize_clients(data, clients, ranks)
    summary["seconds"] = time.perf_counter() - started
    summary["steps_per_client"] = cfg["local_steps"]
    summary["clients"] = {str(c): v for c, v in clients.items()}
    return summary
