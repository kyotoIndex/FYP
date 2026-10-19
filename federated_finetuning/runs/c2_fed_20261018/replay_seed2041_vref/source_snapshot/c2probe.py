"""C2-0: one-step mechanism probe on a decoder LLM (docs/designs/c2_0_probe_design_2026-10-18.md).

Stage `c2check`: builds / verifies the prompt cache, loads the model, evaluates the base model on the
evaluation subset and runs one local step; writes <run_root>/c2check/check.json (handoff gate).
Stage `c2snap`: a short homogeneous rank-R FedAvg run (every participant trains all components) that saves
neutral global states at c2.snapshot_rounds (round 0 = cold start, B = 0). Resumable (checkpoint every
c2.checkpoint_every rounds); snapshot hashes go to snapshots/hashes.json and are verified by the probe.
Stage `c2probe`: from one snapshot, simulate `draws` rounds on the real schedule of the following rounds.
Every participant trains full; weak clients also train restricted (coverage-balanced r of R components) and
amplified-restricted (alpha = R/r) on the same batches. Rules applied once to the snapshot:
  r8 (all full, all blocks), vref (full train, assigned blocks, Hajek), L2 (restricted, owner),
  hajek (restricted, Hajek), amp (amplified restricted, Hajek).
Measured per weak client (on its assigned components, vs its own full run) and per aggregate (vs r8 on the
same draw): exact product-space step / cosine of the LoRA update, function-space step on the label-word
log-probabilities of a fixed probe batch, and one-step core-weak accuracy (paired difference to r8).
Resumable: a partial file is written after every draw.
Stage `c2report`: pre-registered reading M0-M2 (design §5); M2 is read only on snapshots that pass the
learnability gate (core-weak accuracy >= c2.learnable_min_core_acc before the update).

Scope: this probe does NOT measure resource savings. Unassigned components stay trainable parameters with
zeroed gradients, so peak memory here says nothing about the folded implementation estimated in C1.
The model has no newly initialised classification head; the frozen tied output embedding still computes the
label-word logits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .core import adapter_names, atomic_json, atomic_save, balanced_allocation, local_train, participants
from .d2 import core_weak_clients, lora_update
from .data import ClientStream, derived_seed, override_roles
from .llm import (Groups, LLMData, LLMTaskModel, build_prompt_cache, evaluate_llm, label_logprobs, product_gram,
                  sha256_file)

RULES = ["r8", "vref", "L2", "hajek", "amp"]


def _setup(plan: dict, device: torch.device):
    cache = Path(plan["cache_directory"])
    build_prompt_cache(Path(plan["manifest_path"]), plan["model"]["name"], cache, plan["c2"]["max_len"])
    data = LLMData(cache, Path(plan["manifest_path"]))
    strong = override_roles(data.clients, plan["d0"]["strong_fraction"])
    R = plan["lora"]["full_rank"]
    model = LLMTaskModel.build(plan["model"], R, plan["lora"]["scale"], plan["d0"]["seed"], device)
    names = adapter_names(model)
    return data, strong, model, names


def identity(plan: dict, data: LLMData, device: torch.device) -> dict:
    """Code, plan, model, tokenizer, data and cache versions of a C2 stage (written next to its outputs)."""
    import transformers
    here = Path(__file__).resolve().parent
    mdir = Path(plan["model"]["name"])
    revision = (mdir / "REVISION.txt").read_text().strip() if (mdir / "REVISION.txt").exists() else None
    return {"plan": plan,
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "code_hashes": {f"scra/{p.name}": sha256_file(p) for p in sorted(here.glob("*.py"))},
            "model_dir": str(mdir), "model_revision": revision,
            "model_config_sha256": sha256_file(mdir / "config.json") if (mdir / "config.json").exists() else None,
            "cache_fingerprint": data.meta["fingerprint"], "cache_sha256": data.meta["cache_sha256"],
            "label_words": data.meta["label_words"], "label_token_ids": data.meta["label_token_ids"],
            "torch": torch.__version__, "transformers": transformers.__version__, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def _probe_rows(data: LLMData, core: list[int], n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(derived_seed(seed, "c2-probe-batch"))
    pool = sorted(i for c in core for i in data.valid_by_client[c])
    return sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())


def restrict_to_sets(state: dict, base: dict, names: list[str], sets_c: list[list[int]]) -> dict:
    """`state` with every component outside the client's assigned set reset to `base` (the round start), so
    a full run can be compared with a restricted one on the same components (as D3 did per block)."""
    out = dict(state)
    for n, chosen in zip(names, sets_c):
        keep = torch.zeros(state[f"{n}.lora_a"].shape[0], dtype=torch.bool)
        keep[list(chosen)] = True
        out[f"{n}.lora_a"] = torch.where(keep[:, None], state[f"{n}.lora_a"], base[f"{n}.lora_a"])
        out[f"{n}.lora_b"] = torch.where(keep[None, :], state[f"{n}.lora_b"], base[f"{n}.lora_b"])
    return out


def _fstep(dx: torch.Tensor, dref: torch.Tensor) -> dict:
    ref2 = float(dref @ dref)
    return {"step": float(dx @ dref) / ref2 if ref2 > 0 else None,
            "cos": float(dx @ dref) / float(dx.norm() * dref.norm()) if ref2 > 0 and dx.norm() > 0 else None}


def _pstep(gram: np.ndarray, x: int, ref: int) -> dict:
    g = gram[ref, ref]
    return {"step": float(gram[x, ref] / g) if g > 0 else None,
            "cos": float(gram[x, ref] / np.sqrt(gram[x, x] * g)) if g > 0 and gram[x, x] > 0 else None,
            "paired_rel_error": float((gram[x, x] + g - 2 * gram[x, ref]) / g) if g > 0 else None}


def _core_acc(ev: dict, core: list[int]) -> float:
    return float(np.mean([ev["clients"][str(c)][0] for c in core]))


# ---------------------------------------------------------------- asset check (handoff)
def run_c2_check(plan: dict, output: Path, device: torch.device) -> dict:
    """Builds or verifies the prompt cache, loads the model, evaluates the base model and runs one local step.
    Writes check.json with what the handoff must confirm."""
    data, strong, model, names = _setup(plan, device)
    core = core_weak_clients(data, strong)
    batch = data.train.batch(list(range(min(4, len(data.train)))), device)
    with torch.no_grad():
        logits = model(batch)
    state0 = model.task_state()
    base = evaluate_llm(model, data, state0, data.eval_subset(plan["c2"]["eval_per_client"], plan["d0"]["seed"]),
                        device, plan["train"]["eval_batch_size"])
    cfg = dict(plan["train"], local_steps=1)
    first = sorted(data.clients)[0]
    _, rec = local_train(model, state0, data, ClientStream(data.clients[first]["train_indices"], 0, first),
                         None, cfg, device)
    info = {"label_words": data.meta["label_words"], "label_token_ids": data.meta["label_token_ids"],
            "label_words_single_token": len(data.meta["label_token_ids"]) == 3,
            "cache_splits": data.meta["splits"], "clients": len(data.clients), "strong_clients": strong,
            "core_weak_clients": len(core), "adapters": len(names),
            "lora_params_millions": sum(p.numel() for p in model.task_parameters().values()) / 1e6,
            "logits_shape": list(logits.shape), "one_step_loss": rec["mean_loss"],
            "one_step_loss_finite": bool(np.isfinite(rec["mean_loss"])),
            "base_model_accuracy": {"all": base["client_macro_accuracy"], "core": _core_acc(base, core),
                                    "chance": 1 / 3},
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "note": "peak memory here is not resource evidence (unassigned components are not folded)"}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "check.json", info)
    atomic_json(output / "identity.json", identity(plan, data, device))
    print(json.dumps(info, indent=2), flush=True)
    return info


# ---------------------------------------------------------------- snapshots
def run_c2_snapshots(plan: dict, output: Path, device: torch.device, stop_after_round: int | None = None) -> Path:
    c2, cfg = plan["c2"], plan["train"]
    out_json = output / "snapshot_run.json"
    if out_json.exists():
        print(f"{out_json} exists; skipping", flush=True)
        return out_json
    torch.set_num_threads(plan.get("cpu_threads", 4))
    data, strong, model, names = _setup(plan, device)
    ident = identity(plan, data, device)
    core = core_weak_clients(data, strong)
    seed = plan["d0"]["seed"]
    client_ids = sorted(data.clients)
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    subset = data.eval_subset(c2["eval_per_client"], seed)
    ckpt = output / "checkpoint.pt"
    hashes_path = output / "snapshots" / "hashes.json"
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st["plan_sha256"] != ident["plan_sha256"] or st["cache_sha256"] != ident["cache_sha256"]:
            raise ValueError("checkpoint was written for another plan or cache")
        if st.get("code_hashes") != ident["code_hashes"]:
            raise ValueError("source code changed since the checkpoint was written; restore the code or delete "
                             f"{ckpt} to restart")
        print(f"resuming snapshot run after round {st['round']}", flush=True)
    else:
        st = {"round": 0, "g_state": model.task_state(), "positions": {c: 0 for c in client_ids},
              "eval": [], "rounds": [], "elapsed": 0.0, "hashes": {},
              "plan_sha256": ident["plan_sha256"], "cache_sha256": ident["cache_sha256"],
              "code_hashes": ident["code_hashes"]}
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "identity.json", ident)
        ev = evaluate_llm(model, data, st["g_state"], subset, device, cfg["eval_batch_size"])
        st["eval"].append({"round": 0, "core_accuracy": _core_acc(ev, core), **ev})
        if 0 in c2["snapshot_rounds"]:
            path = output / "snapshots" / "round_0000.pt"
            atomic_save(path, st["g_state"])
            st["hashes"][path.name] = sha256_file(path)
            atomic_json(hashes_path, st["hashes"])
    g_state = st["g_state"]
    groups = Groups(names, g_state)
    streams = {c: ClientStream(data.clients[c]["train_indices"], seed, c, st["positions"][c]) for c in client_ids}
    for t in range(st["round"] + 1, c2["snapshot_run_rounds"] + 1):
        started = time.perf_counter()
        chosen = participants(seed, t, client_ids, plan["federation"]["clients_per_round"])
        g_blocks = groups.to(g_state)
        acc = {w: torch.zeros_like(b) for w, b in g_blocks.items()}
        total, losses = 0.0, []
        for c in chosen:
            state, rec = local_train(model, g_state, data, streams[c], None, cfg, device)
            new = groups.to(state)
            for w in acc:
                acc[w] += weights[c] * (new[w] - g_blocks[w])
            total += weights[c]
            losses.append(rec["mean_loss"])
        g_state = groups.apply({w: g_blocks[w] + acc[w] / total for w in acc}, g_state)
        st["rounds"].append({"round": t, "loss": float(np.mean(losses)), "seconds": time.perf_counter() - started})
        if t in c2["snapshot_rounds"]:
            path = output / "snapshots" / f"round_{t:04d}.pt"
            atomic_save(path, g_state)
            st["hashes"][path.name] = sha256_file(path)
            atomic_json(hashes_path, st["hashes"])
        if t % 10 == 0 or t == c2["snapshot_run_rounds"]:
            ev = evaluate_llm(model, data, g_state, subset, device, cfg["eval_batch_size"])
            st["eval"].append({"round": t, "core_accuracy": _core_acc(ev, core), **ev})
            print(f"[C2 snap] round {t}: acc={ev['client_macro_accuracy']:.4f} core={st['eval'][-1]['core_accuracy']:.4f} "
                  f"loss={st['rounds'][-1]['loss']:.4f}", flush=True)
        st.update({"round": t, "g_state": g_state, "positions": {c: s.position for c, s in streams.items()}})
        st["elapsed"] += time.perf_counter() - started
        if t % c2.get("checkpoint_every", 5) == 0 or t == c2["snapshot_run_rounds"]:
            atomic_save(ckpt, st)
        if stop_after_round is not None and t >= stop_after_round:
            atomic_save(ckpt, st)
            return ckpt
    atomic_json(out_json, {"strong_clients": strong, "core_weak_clients": core, "rounds": st["rounds"],
                           "eval": st["eval"], "snapshot_sha256": st["hashes"], "elapsed_seconds": st["elapsed"],
                           "peak_memory_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                                               if device.type == "cuda" else None),
                           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    return out_json


# ---------------------------------------------------------------- probe
def run_c2_probe(plan: dict, snapshot_round: int, output: Path, device: torch.device,
                 stop_after_draws: int | None = None) -> Path:
    c2, cfg = plan["c2"], plan["train"]
    out_path = output / f"round_{snapshot_round:04d}.json"
    partial_path = output / f"round_{snapshot_round:04d}.partial.json"
    if out_path.exists():
        print(f"{out_path} exists; skipping", flush=True)
        return out_path
    torch.set_num_threads(plan.get("cpu_threads", 4))
    started_all = time.perf_counter()
    data, strong_all, model, names = _setup(plan, device)
    ident = identity(plan, data, device)
    seed = plan["d0"]["seed"]
    R, r = plan["lora"]["full_rank"], plan["lora"]["weak_rank"]
    alpha = R / r if c2.get("amp_alpha", "auto") == "auto" else float(c2["amp_alpha"])
    core = core_weak_clients(data, strong_all)
    client_ids = sorted(data.clients)
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    subset = data.eval_subset(c2["eval_per_client"], seed)
    probe = _probe_rows(data, core, c2["probe_examples"], seed)
    snap_dir = Path(plan["run_root"]) / "c2snap" / "snapshots"
    snap = snap_dir / f"round_{snapshot_round:04d}.pt"
    hashes = json.loads((snap_dir / "hashes.json").read_text())
    snap_hash = sha256_file(snap)
    if hashes.get(snap.name) != snap_hash:
        raise ValueError(f"{snap} does not match snapshots/hashes.json")
    g_state = torch.load(snap, map_location="cpu", weights_only=True)
    groups = Groups(names, g_state)
    g_blocks = groups.to(g_state)
    J = len(names)
    base_lp = label_logprobs(model, data, g_state, probe, device)

    def evaluate(state):
        ev = evaluate_llm(model, data, state, subset, device, cfg["eval_batch_size"])
        return {"acc": ev["client_macro_accuracy"], "nll": ev["client_macro_nll"], "core": _core_acc(ev, core)}

    rows, elapsed_before = [], 0.0
    if partial_path.exists():
        part = json.loads(partial_path.read_text())
        if part["snapshot_sha256"] != snap_hash or part["plan_sha256"] != ident["plan_sha256"]:
            raise ValueError(f"{partial_path} belongs to another snapshot or plan")
        if part.get("code_hashes") != ident["code_hashes"]:
            raise ValueError(f"source code changed since {partial_path} was written; restore the code or delete it")
        rows, base_eval, elapsed_before = part["rows"], part["baseline"], part["elapsed_seconds"]
        print(f"resuming probe r{snapshot_round} after draw {len(rows)}", flush=True)
    else:
        base_eval = evaluate(g_state)
        output.mkdir(parents=True, exist_ok=True)
    for d in range(len(rows) + 1, c2["draws"] + 1):
        t = snapshot_round + d
        started = time.perf_counter()
        chosen = participants(seed, t, client_ids, plan["federation"]["clients_per_round"])
        strong = [c for c in chosen if c in strong_all]
        weak = [c for c in chosen if c not in strong_all]
        sets, counts = balanced_allocation(weak, J, R, r, seed, t)
        full, restr, amp, client_rows = {}, {}, {}, []
        losses = {"full": [], "restr": [], "amp": []}
        for c in chosen:
            def stream():
                return ClientStream(data.clients[c]["train_indices"], c2["seed"] + t, c, 0)
            st_full, rec = local_train(model, g_state, data, stream(), None, cfg, device)
            full[c] = groups.to(st_full)
            if c in strong:
                continue
            losses["full"].append(rec["mean_loss"])
            st_restr, rec = local_train(model, g_state, data, stream(), sets[c], cfg, device)
            losses["restr"].append(rec["mean_loss"])
            st_amp, rec = local_train(model, g_state, data, stream(), sets[c], cfg, device, amp_alpha=alpha)
            losses["amp"].append(rec["mean_loss"])
            restr[c], amp[c] = groups.to(st_restr), groups.to(st_amp)
            # product space on the assigned components only: the full run restricted to S is the reference
            st_full_s = restrict_to_sets(st_full, g_state, names, sets[c])
            gram = product_gram([st_full_s, st_restr, st_amp], g_state, names)
            # function space: the raw (un-amplified) client models against the full run; a client that matches
            # full training component by component moves about r/R as far, an overshooting one further
            lp = {k: label_logprobs(model, data, s, probe, device) - base_lp
                  for k, s in (("full", st_full), ("restr", st_restr), ("amp", st_amp))}
            client_rows.append({"client": c, "product": {"restr": _pstep(gram, 1, 0), "amp": _pstep(gram, 2, 0)},
                                "function": {"restr": _fstep(lp["restr"], lp["full"]),
                                             "amp": _fstep(lp["amp"], lp["full"])},
                                "update_norm_full_assigned": float(np.sqrt(gram[0, 0]))})
        disp = {k: {} for k in RULES}
        for w in g_blocks:
            idx = groups.index[w]
            cnt = counts[idx]
            ws = {c: groups.sets(sets[c], w) for c in weak}
            f_w = {c: full[c][w] - g_blocks[w] for c in chosen}
            mixed = {c: ((restr[c][w] - g_blocks[w]) if c in weak else f_w[c]) for c in chosen}
            amped = {c: ((amp[c][w] - g_blocks[w]) if c in weak else f_w[c]) for c in chosen}
            disp["r8"][w] = lora_update(f_w, weights, chosen, {}, torch.zeros_like(cnt), "owner")
            disp["vref"][w] = lora_update(f_w, weights, strong, ws, cnt, "hajek")
            disp["L2"][w] = lora_update(mixed, weights, strong, ws, cnt, "owner")
            disp["hajek"][w] = lora_update(mixed, weights, strong, ws, cnt, "hajek")
            disp["amp"][w] = lora_update(amped, weights, strong, ws, cnt, "hajek")
        del full, restr, amp
        states = {k: groups.apply({w: g_blocks[w] + disp[k][w] for w in g_blocks}, g_state) for k in RULES}
        gram = product_gram([states[k] for k in RULES], g_state, names)
        lps = {k: label_logprobs(model, data, states[k], probe, device) - base_lp for k in RULES}
        row = {"round": t, "strong": len(strong), "weak": len(weak), "clients": client_rows,
               "local_loss": {k: (float(np.mean(v)) if v else None) for k, v in losses.items()},
               "aggregate": {k: {"product": _pstep(gram, i, 0), "function": _fstep(lps[k], lps["r8"])}
                             for i, k in enumerate(RULES)}}
        for k in RULES:
            row[k] = evaluate(states[k])
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        atomic_json(partial_path, {"snapshot_round": snapshot_round, "snapshot_sha256": snap_hash,
                                   "plan_sha256": ident["plan_sha256"], "code_hashes": ident["code_hashes"],
                                   "baseline": base_eval, "rows": rows,
                                   "elapsed_seconds": elapsed_before + time.perf_counter() - started_all})
        print(f"[C2 probe r{snapshot_round}] draw {d}/{c2['draws']} (round {t}): "
              + " ".join(f"{k}: core={row[k]['core']:.4f} step={row['aggregate'][k]['product']['step']:.2f}"
                         for k in RULES) + f" ({row['seconds']:.0f}s)", flush=True)
        if stop_after_draws is not None and len(rows) >= stop_after_draws and len(rows) < c2["draws"]:
            return partial_path
    out = {"snapshot_round": snapshot_round, "snapshot_sha256": snap_hash, "draws": c2["draws"], "R": R, "r": r,
           "amp_alpha": alpha, "rules": RULES, "core_weak_clients": core, "probe_rows": probe,
           "baseline": base_eval, "rows": rows, "identity": ident,
           "elapsed_seconds": elapsed_before + time.perf_counter() - started_all,
           "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    atomic_json(out_path, out)
    partial_path.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------- report (pre-registered reading, design §5)
def _med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _paired(diffs: list[float]) -> dict:
    d = np.array(diffs)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n": int(len(d)),
            "all_same_sign": bool((d > 0).all() or (d < 0).all())}


def c2_snapshot_summary(blob: dict, gate: float) -> dict:
    rows = blob["rows"]
    agg = {}
    for k in RULES:
        core_d = [r[k]["core"] - r["r8"]["core"] for r in rows]
        agg[k] = {"product_step": _med([r["aggregate"][k]["product"]["step"] for r in rows]),
                  "function_step": _med([r["aggregate"][k]["function"]["step"] for r in rows]),
                  "paired_rel_error": _med([r["aggregate"][k]["product"]["paired_rel_error"] for r in rows]),
                  "core_diff": _paired(core_d),
                  "acc_diff_mean": float(np.mean([r[k]["acc"] - r["r8"]["acc"] for r in rows]))}
    amp_vs_vref = _paired([r["amp"]["core"] - r["vref"]["core"] for r in rows])
    cl = [c for r in rows for c in r["clients"]]
    client = {kind: {"product_step": _med([c["product"][kind]["step"] for c in cl]),
                     "product_cos": _med([c["product"][kind]["cos"] for c in cl]),
                     "function_step": _med([c["function"][kind]["step"] for c in cl]),
                     "function_cos": _med([c["function"][kind]["cos"] for c in cl])} for kind in ("restr", "amp")}
    loss = {k: _med([r["local_loss"][k] for r in rows]) for k in ("full", "restr", "amp")}
    return {"aggregate": agg, "amp_minus_vref_core": amp_vs_vref, "client": client, "local_loss": loss,
            "baseline_core": blob["baseline"]["core"], "learnable": bool(blob["baseline"]["core"] >= gate)}


def c2_verdicts(summaries: list[dict]) -> dict:
    need = 2 if len(summaries) >= 3 else len(summaries)

    def count(pred, pool):
        n = 0
        for s in pool:
            try:
                n += bool(pred(s))
            except TypeError:
                pass
        return n

    m0 = count(lambda s: s["aggregate"]["hajek"]["product_step"] >= 1.2
               or s["aggregate"]["hajek"]["function_step"] >= 1.2, summaries) >= need
    m1 = count(lambda s: 0.9 <= s["aggregate"]["amp"]["product_step"] <= 1.1
               and 0.95 <= s["client"]["amp"]["product_step"] <= 1.05
               and s["client"]["amp"]["product_cos"] >= 0.95, summaries) >= need
    gated = [s for s in summaries if s["learnable"]]
    if not gated:
        m2 = "not interpretable (no snapshot passes the learnability gate)"
    else:
        need2 = 2 if len(gated) >= 3 else len(gated)
        m2 = count(lambda s: min(s["aggregate"]["L2"]["core_diff"]["mean"],
                                 s["aggregate"]["hajek"]["core_diff"]["mean"]) <= -0.005
                   and abs(s["amp_minus_vref_core"]["mean"]) <= 0.002, gated) >= need2
    return {"M0_restricted_overshoots_without_new_head": m0, "M1_amplification_corrects": m1,
            "M2_one_step_accuracy_signal": m2, "M2_snapshots_used": len(gated)}


def write_c2_report(run_root: Path, gate: float = 0.45) -> dict:
    files = sorted(p for p in (run_root / "c2probe").glob("round_*.json") if not p.name.endswith(".partial.json"))
    if not files:
        raise FileNotFoundError(f"no C2 probe results under {run_root / 'c2probe'}")
    blobs = [json.loads(f.read_text()) for f in files]
    sums = [c2_snapshot_summary(b, gate) for b in blobs]
    v = c2_verdicts(sums)
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    pd = lambda p: f"{100 * p['mean']:+.2f} ± {100 * p['sd']:.2f} ({p['n_neg']}−/{p['n_pos']}+)"  # noqa: E731
    snap_run = run_root / "c2snap" / "snapshot_run.json"
    lines = [f"# C2-0 decoder-LLM mechanism probe ({run_root.name})", "",
             "One update from each snapshot of a homogeneous rank-R run, over the real schedule of the following "
             "rounds (same protocol as D2 / D3). Steps are projections on the full-training reference: per weak "
             "client against its own full run on its assigned components, per aggregate against r8 on the same "
             "draw. `product` = exact LoRA ΔW (all adapters), `function` = label-word log-probabilities on a fixed "
             "probe batch. Accuracy diffs are core-weak, one step, paired per draw: mean ± sd (negative/positive "
             "draws).", "",
             "**Scope.** This probe does not measure resource savings (unassigned components are not folded; peak "
             "memory here is not evidence for C1). The model has no newly initialised classification head, but the "
             "frozen tied output embedding still computes the label-word logits. M1 checks update scale, not task "
             "gain; task gain is C2-1 / C2-2.", "",
             f"**Learnability gate.** M2 is read only on snapshots whose core-weak accuracy before the update is "
             f"≥ {100 * gate:.0f}% (chance 33.3%).", ""]
    if snap_run.exists():
        sr = json.loads(snap_run.read_text())
        lines += ["Snapshot run (all / core accuracy): " + ", ".join(
            f"r{e['round']} {100 * e['client_macro_accuracy']:.2f} / {100 * e.get('core_accuracy', float('nan')):.2f}"
            for e in sr["eval"]) + ".", ""]
    for b, s in zip(blobs, sums):
        lines += [f"## Snapshot round {b['snapshot_round']} ({b['draws']} draws, R={b['R']}, r={b['r']}, "
                  f"α={b['amp_alpha']}; before update: acc {100 * b['baseline']['acc']:.2f}, "
                  f"core {100 * b['baseline']['core']:.2f} — gate {'passed' if s['learnable'] else 'NOT passed'})", "",
                  "| rule | product step vs r8 | function step vs r8 | paired rel. error | core diff vs r8 (pp) | all diff (pp) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for k in RULES:
            a = s["aggregate"][k]
            lines.append(f"| {k} | {f(a['product_step'])} | {f(a['function_step'])} | {f(a['paired_rel_error'], 3)} | "
                         f"{pd(a['core_diff'])} | {100 * a['acc_diff_mean']:+.2f} |")
        c, lo = s["client"], s["local_loss"]
        lines += ["", f"amp − vref (core, pp): {pd(s['amp_minus_vref_core'])}.",
                  f"Per weak client (medians, assigned components): restricted product step {f(c['restr']['product_step'])}, "
                  f"cos {f(c['restr']['product_cos'])}; amplified product step {f(c['amp']['product_step'])}, cos "
                  f"{f(c['amp']['product_cos'])}. Function step vs own full run (≈ r/R = {b['r'] / b['R']:.3f} when "
                  f"matched): restricted {f(c['restr']['function_step'], 3)}, amplified {f(c['amp']['function_step'], 3)}. "
                  f"Local loss: full {f(lo['full'], 4)}, restricted {f(lo['restr'], 4)}, amplified {f(lo['amp'], 4)}.", ""]
    lines += ["## Pre-registered reading (≥ 2 of 3 snapshots)", "", "| question | criterion | result |", "|---|---|---|",
              f"| M0 restricted training overshoots without a new head | hajek aggregate product or function step ≥ 1.2 | "
              f"{v['M0_restricted_overshoots_without_new_head']} |",
              f"| M1 amplification corrects the update scale | amp aggregate step in [0.9, 1.1], per-client amp step in "
              f"[0.95, 1.05], cos ≥ 0.95 | {v['M1_amplification_corrects']} |",
              f"| M2 one-step accuracy signal (gated snapshots only: {v['M2_snapshots_used']}) | L2 or hajek core diff "
              f"≤ −0.5 pp and \\|amp − vref\\| ≤ 0.2 pp | {v['M2_one_step_accuracy_signal']} |", "",
              "Next step: M0 and M1 → C2-1 (headroom run). M0 but not M1 → inspect the per-client probe for the failing "
              "assumption before changing the method. Not M0 → state the scope (settings with co-trained new "
              "parameters); C2-1 only checks that restricted training costs nothing here. M0 and M1 without M2 → "
              "the mechanism is present but one step does not show it in accuracy; go to C2-1. M2 not interpretable → "
              "the task was not learned; fix the prompt / labels or lengthen the snapshot run before reading M2. "
              "A marginal M2 (few draws in the same direction) is not a strong result."]
    out_dir = run_root / "diagnostics_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT_C2_0.md").write_text("\n".join(lines) + "\n")
    report = {"summaries": sums, "verdicts": v, "snapshots": [b["snapshot_round"] for b in blobs], "gate": gate}
    atomic_json(out_dir / "report_c2_0.json", report)
    return report
