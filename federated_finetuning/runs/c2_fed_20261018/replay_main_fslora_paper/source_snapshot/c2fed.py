"""C2-1: multi-round federated runs on the decoder LLM (docs/designs/c2_1_design_2026-10-18.md).

One run = one method on the Qwen2.5-0.5B / MNLI next-token setup of C2-0, 10% strong clients, R = 64, r = 8:
  r8         every participant trains all R components, weighted mean (homogeneous reference)
  L2         weak clients train their assigned r components (coverage-balanced), owner averaging
  vref       weak clients train all components but upload only their assigned ones, Hajek (variance-matched reference)
  amp        weak clients train their assigned components with amplified forward, Hajek; alpha = c2fed.amp_alpha
             (default R/r; the alpha sweep uses 4 / 16 / 32 as well)
  fslora     FSLoRA's forward (selected components x R/r as a whole, unselected ones dropped), TRUE gradient, Hajek
  fslora_st  FSLoRA's forward value with amp's gradient path (straight-through), Hajek
Same allocation and aggregation for amp / fslora / fslora_st, so amp vs fslora_st differs only in sketching the
state, and fslora_st vs fslora only in the gradient path (FIX_BATCH2_CONTROLS §2 on RoBERTa).
External positioning (design §11; several factors differ, not mechanism evidence):
  fslora_paper  FSLoRA as published, under our AdamW protocol: FSLoRA forward with true gradient, independent random
                selection of r components per client, zero-filled average over all participants
  fedplora      Fed-PLoRA style: independent random selection, plain restricted training, owner averaging
Aggregation controls (design §13): coverage-balanced allocation with zero-filled averaging,
  amp_zf        amplified restricted training (alpha = R/r), balanced, zero-fill (vs amp: aggregation only)
  fslora_zf     FSLoRA forward with true gradient, balanced, zero-fill (vs fslora: aggregation only;
                vs fslora_paper: allocation only)
  vref_zf       weak clients train fully, upload assigned components, zero-fill (vs vref: aggregation only, with no
                restricted local training at all)
Server step (c2fed.server_step, default 1): the aggregated LoRA update is multiplied by this factor before it is
applied; amp with server_step 0.25 separates "zero-fill helps because the step is smaller" from zero-fill itself.

Stage A (learning-rate check) runs r8 for a few rounds at several learning rates on a development seed; the lr is
chosen by core accuracy (design §3). As a diagnostic, the first `diag_restricted_rounds` rounds replay restricted
training for every weak participant on the same batches (not aggregated) and record both local losses.

Mechanism diagnostics (`diag_every`, restricted methods only): in round 1 and every `diag_every` rounds, every
weak participant also replays full training on the same batches (not aggregated). Recorded per weak client: its
update on its assigned components against its own full run (exact product-space step / cosine) and the
function-space step on the label-word log-probabilities of a fixed probe batch; per round: the method's
aggregate against this round's r8 aggregate (all participants full), in product and function space.

Resumable (checkpoint every c2fed.checkpoint_every rounds). Output: <run_root>/fed.json and identity.json.
Scope as in C2-0: no resource measurement (unassigned components are not folded).
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .c2probe import _core_acc, _fstep, _probe_rows, _pstep, _setup, identity, restrict_to_sets
from .core import atomic_json, atomic_save, balanced_allocation, local_train, participants, random_allocation
from .d2 import core_weak_clients, lora_update
from .data import ClientStream
from .llm import Groups, evaluate_llm, label_logprobs, product_gram

METHODS = ("r8", "L2", "vref", "amp", "fslora", "fslora_st", "fslora_paper", "fedplora", "amp_zf", "fslora_zf",
           "vref_zf")
RESTRICTED = ("L2", "amp", "fslora", "fslora_st", "fslora_paper", "fedplora", "amp_zf", "fslora_zf")
# method -> (weak local training, allocation, aggregation)
SPEC = {"r8": ("full", None, "all"), "vref": ("full", "balanced", "hajek"),
        "L2": ("plain", "balanced", "owner"), "amp": ("amp", "balanced", "hajek"),
        "fslora": ("sketch", "balanced", "hajek"), "fslora_st": ("sketch_st", "balanced", "hajek"),
        "fslora_paper": ("sketch", "random", "zerofill"), "fedplora": ("plain", "random", "owner"),
        "amp_zf": ("amp", "balanced", "zerofill"), "fslora_zf": ("sketch", "balanced", "zerofill"),
        "vref_zf": ("full", "balanced", "zerofill")}


def save_source_snapshot(dest: Path) -> None:
    """Exact copy of the code that produced a run (scra/*.py and run.py), next to its identity.json."""
    here = Path(__file__).resolve().parent
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(here.glob("*.py")) + [here.parent / "run.py"]:
        if f.exists():
            shutil.copy2(f, dest / f.name)


# Fields that must agree between runs that are compared (everything except seed, method and method knobs).
# Paths are not compared (runs were made under different HPC accounts); the model, data and prompt cache are
# compared by content (revision and hashes recorded in identity.json) instead.
COMPARABLE = (("train",), ("lora",), ("federation",), ("d0", "strong_fraction"), ("c2", "eval_seed"),
              ("c2", "eval_per_client"), ("c2", "max_len"), ("c2", "probe_examples"), ("c2fed", "rounds"),
              ("c2fed", "eval_every"), ("c2fed", "diag_every"))
CONTENT = ("model_revision", "model_config_sha256", "cache_fingerprint", "cache_sha256", "label_token_ids")


def comparable_fields(ident: dict) -> dict:
    out = {}
    for path in COMPARABLE:
        x = ident["plan"]
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        out[".".join(path)] = x
    out.update({k: ident.get(k) for k in CONTENT})
    return out


def _aggregate(method, deltas, weights, chosen, strong, weak, sets, counts, groups, g_blocks):
    kind = SPEC[method][2]
    disp = {}
    for w in g_blocks:
        cnt = counts[groups.index[w]]
        d_w = {c: deltas[c][w] for c in chosen}
        if kind == "all":
            disp[w] = lora_update(d_w, weights, chosen, {}, torch.zeros_like(cnt), "owner")
        else:
            ws = {c: groups.sets(sets[c], w) for c in weak}
            disp[w] = lora_update(d_w, weights, strong, ws, cnt, kind)
    return disp


def run_c2_fed(plan: dict, output: Path, device: torch.device, stop_after_round: int | None = None) -> Path:
    cfg, cf = plan["train"], plan["c2fed"]
    method = cf["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    out_json = output / "fed.json"
    if out_json.exists():
        print(f"{out_json} exists; skipping", flush=True)
        return out_json
    torch.set_num_threads(plan.get("cpu_threads", 4))
    data, strong_all, model, names = _setup(plan, device)
    ident = identity(plan, data, device)
    core = core_weak_clients(data, strong_all)
    seed = plan["d0"]["seed"]
    R, r = plan["lora"]["full_rank"], plan["lora"]["weak_rank"]
    alpha = float(cf.get("amp_alpha", R / r))          # amp only
    sketch = R / r                                      # FSLoRA's factor
    server_step = float(cf.get("server_step", 1.0))
    client_ids = sorted(data.clients)
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    eval_seed = plan["c2"].get("eval_seed", seed)
    subset = data.eval_subset(plan["c2"]["eval_per_client"], eval_seed)
    diag_every = cf.get("diag_every") if method in RESTRICTED else None
    probe = _probe_rows(data, core, plan["c2"]["probe_examples"], eval_seed) if diag_every else None
    ckpt = output / "checkpoint.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st["plan_sha256"] != ident["plan_sha256"] or st["cache_sha256"] != ident["cache_sha256"]:
            raise ValueError("checkpoint was written for another plan or cache")
        if st.get("code_hashes") != ident["code_hashes"]:
            raise ValueError("source code changed since the checkpoint was written; restore the code or delete "
                             f"{ckpt} to restart")
        print(f"resuming {method} after round {st['round']}", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "identity.json", ident)
        save_source_snapshot(output / "source_snapshot")
        g0 = model.task_state()
        ev = evaluate_llm(model, data, g0, subset, device, cfg["eval_batch_size"])
        st = {"round": 0, "g_state": g0, "positions": {c: 0 for c in client_ids},
              "eval": [{"round": 0, "core_accuracy": _core_acc(ev, core), **ev}], "rounds": [], "elapsed": 0.0,
              "plan_sha256": ident["plan_sha256"], "cache_sha256": ident["cache_sha256"],
              "code_hashes": ident["code_hashes"]}
    g_state = st["g_state"]
    groups = Groups(names, g_state)
    J = len(names)
    streams = {c: ClientStream(data.clients[c]["train_indices"], seed, c, st["positions"][c]) for c in client_ids}
    for t in range(st["round"] + 1, cf["rounds"] + 1):
        started = time.perf_counter()
        chosen = participants(seed, t, client_ids, plan["federation"]["clients_per_round"])
        strong = [c for c in chosen if c in strong_all]
        weak = [c for c in chosen if c not in strong_all]
        if SPEC[method][1] == "random":
            sets, counts = random_allocation(weak, J, R, r, seed, t)
        else:
            sets, counts = balanced_allocation(weak, J, R, r, seed, t)
        g_blocks = groups.to(g_state)
        deltas, losses, loss_diag = {}, [], {"full": [], "restricted": []}
        replay_restricted = t <= cf.get("diag_restricted_rounds", 0)
        mech = bool(diag_every) and (t == 1 or t % diag_every == 0)
        base_lp = label_logprobs(model, data, g_state, probe, device) if mech else None
        full_deltas, client_diag = {}, []
        for c in chosen:
            restricted = method in RESTRICTED and c in weak
            position = streams[c].position
            state, rec = local_train(
                model, g_state, data, streams[c], sets[c] if restricted else None, cfg, device,
                amp_alpha=alpha if (restricted and SPEC[method][0] == "amp") else None,
                sketch_alpha=sketch if (restricted and SPEC[method][0] in ("sketch", "sketch_st")) else None,
                sketch_straight_through=restricted and SPEC[method][0] == "sketch_st")
            new = groups.to(state)
            deltas[c] = {w: new[w] - g_blocks[w] for w in g_blocks}
            losses.append(rec["mean_loss"])
            if replay_restricted and c in weak and not restricted:      # stage-A diagnostic, not aggregated
                loss_diag["full"].append(rec["mean_loss"])
                stream = ClientStream(data.clients[c]["train_indices"], seed, c, position)
                _, rrec = local_train(model, g_state, data, stream, sets[c], cfg, device)
                loss_diag["restricted"].append(rrec["mean_loss"])
            if mech and restricted:                                      # mechanism diagnostic, not aggregated
                stream = ClientStream(data.clients[c]["train_indices"], seed, c, position)
                st_full, frec = local_train(model, g_state, data, stream, None, cfg, device)
                fnew = groups.to(st_full)
                full_deltas[c] = {w: fnew[w] - g_blocks[w] for w in g_blocks}
                gram = product_gram([restrict_to_sets(st_full, g_state, names, sets[c]), state], g_state, names)
                lp_full = label_logprobs(model, data, st_full, probe, device) - base_lp
                lp_m = label_logprobs(model, data, state, probe, device) - base_lp
                client_diag.append({"client": c, "product": _pstep(gram, 1, 0), "function": _fstep(lp_m, lp_full),
                                    "local_loss": rec["mean_loss"], "full_local_loss": frec["mean_loss"]})
        disp = _aggregate(method, deltas, weights, chosen, strong, weak, sets, counts, groups, g_blocks)
        if server_step != 1.0:
            disp = {w: server_step * d for w, d in disp.items()}
        new_state = groups.apply({w: g_blocks[w] + disp[w] for w in g_blocks}, g_state)
        diag_out = None
        if mech and client_diag:
            ref = {c: (full_deltas[c] if c in full_deltas else deltas[c]) for c in chosen}
            r8_disp = _aggregate("r8", ref, weights, chosen, strong, weak, sets, counts, groups, g_blocks)
            r8_state = groups.apply({w: g_blocks[w] + r8_disp[w] for w in g_blocks}, g_state)
            gram = product_gram([r8_state, new_state], g_state, names)
            lp_r8 = label_logprobs(model, data, r8_state, probe, device) - base_lp
            lp_m = label_logprobs(model, data, new_state, probe, device) - base_lp
            diag_out = {"clients": client_diag,
                        "aggregate": {"product": _pstep(gram, 1, 0), "function": _fstep(lp_m, lp_r8)}}
            if cf.get("save_diag_reference"):                    # testing: the diagnostic r8 reference itself
                atomic_save(output / f"diag_r8_round_{t:04d}.pt", r8_state)
            del r8_disp, r8_state
        del deltas, full_deltas
        g_state = new_state
        rec = {"round": t, "strong": len(strong), "weak": len(weak), "loss": float(np.mean(losses)),
               "update_norm": float(np.sqrt(sum(float(disp[w].pow(2).sum()) for w in disp))),
               "diag_weak_full_loss": float(np.mean(loss_diag["full"])) if loss_diag["full"] else None,
               "diag_weak_restricted_loss": float(np.mean(loss_diag["restricted"])) if loss_diag["restricted"] else None,
               "mechanism": diag_out, "seconds": time.perf_counter() - started}
        st["rounds"].append(rec)
        if t % cf["eval_every"] == 0 or t == cf["rounds"]:
            ev = evaluate_llm(model, data, g_state, subset, device, cfg["eval_batch_size"])
            st["eval"].append({"round": t, "core_accuracy": _core_acc(ev, core), **ev})
            print(f"[C2-1 {method} α={alpha:g} lr={cfg['lr']:g}] round {t}: acc={ev['client_macro_accuracy']:.4f} "
                  f"core={st['eval'][-1]['core_accuracy']:.4f} loss={rec['loss']:.4f}", flush=True)
        st.update({"round": t, "g_state": g_state, "positions": {c: s.position for c, s in streams.items()}})
        st["elapsed"] += time.perf_counter() - started
        if t % cf.get("checkpoint_every", 5) == 0 or t == cf["rounds"]:
            atomic_save(ckpt, st)
        if stop_after_round is not None and t >= stop_after_round and t < cf["rounds"]:
            atomic_save(ckpt, st)
            return ckpt
    atomic_json(out_json, {"method": method, "seed": seed, "lr": cfg["lr"], "R": R, "r": r,
                           "amp_alpha": alpha if SPEC[method][0] == "amp" else None, "server_step": server_step,
                           "sketch_alpha": sketch if SPEC[method][0] in ("sketch", "sketch_st") else None,
                           "spec": dict(zip(("weak_training", "allocation", "aggregation"), SPEC[method])),
                           "strong_clients": strong_all, "core_weak_clients": core, "rounds": st["rounds"],
                           "eval": st["eval"], "elapsed_seconds": st["elapsed"],
                           "peak_memory_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                                               if device.type == "cuda" else None),
                           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    atomic_save(output / "final_state.pt", g_state)
    ckpt.unlink(missing_ok=True)
    return out_json


# ---------------------------------------------------------------- reports (pre-registered, design §3-§4)
def _window(ev: list[dict], lo: int, hi: int, key="core_accuracy") -> float | None:
    vals = [e[key] for e in ev if lo <= e["round"] <= hi]
    return float(np.mean(vals)) if vals else None


def _osc(ev: list[dict], start: int) -> float | None:
    acc = [100 * e["client_macro_accuracy"] for e in ev if e["round"] >= start]
    return float(np.mean(np.abs(np.diff(acc)))) if len(acc) > 1 else None


def _med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def select_lr(runs: dict[float, dict], at_round: int, tie_pp: float = 0.5) -> dict:
    """Stage A rule: highest core-weak accuracy at `at_round`; within `tie_pp` of the best, the larger lr."""
    scores = {lr: next(e["core_accuracy"] for e in b["eval"] if e["round"] == at_round) for lr, b in runs.items()}
    best = max(scores.values())
    chosen = max(lr for lr, s in scores.items() if 100 * (best - s) <= tie_pp)
    return {"scores": scores, "chosen_lr": chosen}


def mechanism_summary(blob: dict) -> list[dict]:
    """Per diagnostic round: medians over weak clients and the aggregate against this round's r8."""
    out = []
    for rec in blob["rounds"]:
        m = rec.get("mechanism")
        if not m:
            continue
        cl = m["clients"]
        out.append({"round": rec["round"],
                    "client_product_step": _med([c["product"]["step"] for c in cl]),
                    "client_product_cos": _med([c["product"]["cos"] for c in cl]),
                    "client_function_step": _med([c["function"]["step"] for c in cl]),
                    "client_local_loss": _med([c["local_loss"] for c in cl]),
                    "client_full_local_loss": _med([c["full_local_loss"] for c in cl]),
                    "aggregate_product_step": m["aggregate"]["product"]["step"],
                    "aggregate_product_cos": m["aggregate"]["product"]["cos"],
                    "aggregate_function_step": m["aggregate"]["function"]["step"],
                    "aggregate_function_cos": m["aggregate"]["function"]["cos"]})
    return out


MULTISEED_METHODS = ("r8", "vref", "amp", "fslora", "fslora_paper", "amp_zf", "fslora_zf", "vref_zf")
# key -> (a, b, kind, criterion text); pre-registered in design §9 (MS1-MS4) and §13 (MS5-MS10), before the runs.
#   ge_minus: mean >= -0.5 pp;  equiv3: |mean| < 0.5 pp -> equivalent, >= 0.5 pp with all seeds same sign -> that
#   side better, otherwise unclear;  dir: mean >= 0.5 pp and every seed > 0;  desc: descriptive only
MULTISEED_COMPARISONS = {
    "MS1": ("amp", "vref", "ge_minus", "mean ≥ −0.5 pp → amp matches vref"),
    "MS2": ("amp", "fslora", "equiv3", "|mean| < 0.5 pp → equivalent; ≥ 0.5 pp with all seeds same sign → that side better"),
    "MS3": ("r8", "amp", "desc", "descriptive (gap to the upper bound)"),
    "MS4": ("r8", "vref", "desc", "descriptive (cost of uploading only assigned components, with Hájek)"),
    "MS5": ("amp", "fslora_paper", "equiv3", "|mean| < 0.5 pp → equivalent; ≥ 0.5 pp with all seeds same sign → that side better"),
    "MS6": ("fslora_paper", "vref", "desc", "descriptive (zero-fill vs the Hájek reference)"),
    "MS7": ("amp_zf", "amp", "dir", "mean ≥ 0.5 pp and every seed > 0 → zero-fill helps amp"),
    "MS8": ("amp_zf", "fslora_paper", "equiv3", "external positioning, not single-factor: |mean| < 0.5 pp → equivalent; otherwise the better side (all seeds same sign)"),
    "MS9": ("vref_zf", "vref", "dir", "mean ≥ 0.5 pp and every seed > 0 → zero-fill helps without restricted training"),
    "MS10": ("r8", "vref_zf", "desc", "descriptive (cost of uploading only assigned components, with zero-fill)"),
    "MS11": ("fslora_zf", "fslora", "dir", "mean ≥ 0.5 pp and every seed > 0 → zero-fill helps FSLoRA's forward (balanced)"),
    "MS12": ("fslora_zf", "fslora_paper", "equiv3", "|mean| < 0.5 pp → allocation (balanced vs random) does not matter"),
}


def _judge(kind: str, p: dict, a: str, b: str):
    n = len(p["per_seed"])
    if kind == "ge_minus":
        return bool(p["mean"] >= -0.5)
    if kind == "dir":
        return bool(p["mean"] >= 0.5 and p["n_pos"] == n)
    if kind == "equiv3":
        if abs(p["mean"]) < 0.5:
            return "equivalent"
        if p["mean"] >= 0.5 and p["n_pos"] == n:
            return f"{a} better"
        if p["mean"] <= -0.5 and p["n_neg"] == n:
            return f"{b} better"
        return "unclear (seeds disagree)"
    return "descriptive"


def current_c2fed_hash() -> str:
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _late(blob: dict, window) -> float | None:
    v = _window(blob["eval"], *window)
    return None if v is None else 100 * v


def replay_checks(fed_root: Path, late_window=(40, 60)) -> dict:
    """Full-length re-runs with the current code of runs produced by older versions of c2fed.py.
    Pass rule (pre-registered, design §13 second revision): round-1 mean local loss within 0.1 % relative AND
    |late core accuracy (rounds 40-60) of replay - original| <= 0.5 pp. A replay that does not reach the late
    window is 'incomplete' and does not verify anything. Per-evaluation differences are reported descriptively."""
    out = {}
    for p in sorted(fed_root.glob("replay_*/fed.json")):
        orig_name = p.parent.name.removeprefix("replay_")
        orig = fed_root / orig_name / "fed.json"
        if not orig.exists():
            continue
        a, b = json.loads(p.read_text()), json.loads(orig.read_text())
        ia, ib = (json.loads((d / "identity.json").read_text()) if (d / "identity.json").exists() else {}
                  for d in (p.parent, orig.parent))
        la, lb = a["rounds"][0]["loss"], b["rounds"][0]["loss"]
        rel = abs(la - lb) / max(abs(lb), 1e-12)
        ea = {e["round"]: e["core_accuracy"] for e in a["eval"]}
        eb = {e["round"]: e["core_accuracy"] for e in b["eval"]}
        diffs = {t: 100 * (ea[t] - eb[t]) for t in sorted(ea) if t in eb and t > 0}
        late_a, late_b = _late(a, late_window), _late(b, late_window)
        complete = late_a is not None and late_b is not None and a["eval"][-1]["round"] >= late_window[1]
        late_diff = (late_a - late_b) if complete else None
        out[orig_name] = {"round1_loss_rel_diff": rel, "late_diff_pp": late_diff, "core_diff_pp": diffs,
                          "max_abs_eval_diff_pp": max((abs(d) for d in diffs.values()), default=None),
                          "original_version": ib.get("code_hashes", {}).get("scra/c2fed.py"),
                          "replay_version": ia.get("code_hashes", {}).get("scra/c2fed.py"),
                          "status": ("incomplete" if not complete else
                                     "pass" if rel <= 1e-3 and abs(late_diff) <= 0.5 else "fail")}
    return out


def verified_versions(replays: dict, current: str) -> dict:
    """c2fed.py version -> how it is verified: the current code, or an older version whose replay (run with the
    current code) passed. Versions not in the result are unverified."""
    ok = {current: "current code"}
    for orig, r in replays.items():
        if r["status"] == "pass" and r["replay_version"] == current and r["original_version"]:
            ok.setdefault(r["original_version"], f"replay of {orig} passed")
    return ok


def multiseed(fed_root: Path, cf_report: dict, current_hash: str | None = None) -> dict:
    """Late core-weak accuracy (pp) for the multi-seed methods: seed 2040 from main_<m>, others from seed<s>_<m>.

    Version gate (design §13 second revision): a run is used only if its scra/c2fed.py version is the current one
    or was verified by a passing full-length replay. Otherwise, if replay_<run> exists and was produced by the
    current code, the replay replaces it; if not, the run is blocked and every comparison that needs it is
    reported as blocked instead of being computed. Every used run must also share the protocol fields and the
    model / data / cache content (comparable_fields)."""
    from .batch2 import paired
    current = current_hash or current_c2fed_hash()
    window = tuple(cf_report["late_window"])
    replays = replay_checks(fed_root, window)
    ok_versions = verified_versions(replays, current)
    runs, versions, used, blocked, reference = {}, {}, {}, {}, None

    def load(d: Path):
        b = json.loads((d / "fed.json").read_text())
        ident = json.loads((d / "identity.json").read_text()) if (d / "identity.json").exists() else None
        return b, ident

    for m in MULTISEED_METHODS:
        for seed, d in ((2040, fed_root / f"main_{m}"), (2041, fed_root / f"seed2041_{m}"),
                        (2042, fed_root / f"seed2042_{m}")):
            if not (d / "fed.json").exists():
                continue
            b, ident = load(d)
            if b["seed"] != seed:
                if seed == 2040:              # stage-B runs on another seed are not part of this comparison
                    continue
                raise ValueError(f"{d / 'fed.json'} has seed {b['seed']}, expected {seed}")
            key = f"{m}_s{seed}"
            version = (ident or {}).get("code_hashes", {}).get("scra/c2fed.py")
            source = d.name
            if version not in ok_versions:
                rd = fed_root / f"replay_{d.name}"
                if (rd / "fed.json").exists():
                    rb, rident = load(rd)
                    if (rident or {}).get("code_hashes", {}).get("scra/c2fed.py") == current and \
                            rb["eval"][-1]["round"] >= window[1]:
                        b, ident, version, source = rb, rident, current, rd.name
                if version not in ok_versions:
                    blocked[key] = f"{d.name}: c2fed.py {str(version)[:8]} not verified"
                    continue
            fields = comparable_fields(ident)
            if reference is None:
                reference = (source, fields)
            elif fields != reference[1]:
                bad = sorted(k for k in fields if fields[k] != reference[1][k])
                raise ValueError(f"{source} differs from {reference[0]} in {bad}")
            versions[source] = version[:8]
            used[key] = source
            runs[key] = {"core": _late(b, window)}
    pairs, v = {}, {}
    for k, (a, b2, kind, _) in MULTISEED_COMPARISONS.items():
        need = [f"{x}_s{s}" for x in (a, b2) for s in (2040, 2041, 2042)]
        why = [blocked[n] for n in need if n in blocked]
        if why:
            pairs[k], v[k] = None, "blocked: " + "; ".join(why)
            continue
        pairs[k] = paired(runs, a, b2)
        if pairs[k] is not None:
            v[k] = _judge(kind, pairs[k], a, b2)
    return {"runs": runs, "paired": pairs, "verdicts": v, "c2fed_versions": versions, "used": used,
            "blocked": blocked, "verified_versions": {h[:8]: why for h, why in ok_versions.items()},
            "current_version": current[:8]}


def write_c2fed_report(fed_root: Path, cf_report: dict) -> dict:
    """fed_root = runs/c2_fed_20261018; reads lr_* and main_* runs that exist."""
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = ["# C2-1 decoder-LLM federated runs", "",
             "Core-weak accuracy (%) on the evaluation subset. Scope as in C2-0: no resource measurement.", ""]
    out = {}
    lr_runs = {}
    for p in sorted(fed_root.glob("lr_*/fed.json")):
        b = json.loads(p.read_text())
        lr_runs[float(b["lr"])] = b
    if lr_runs:
        at = cf_report["lr_select_round"]
        sel = select_lr(lr_runs, at)
        out["lr_check"] = sel

        def dmean(b, key):
            vals = [x[key] for x in b["rounds"] if x.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        diag = {lr: {"full": dmean(b, "diag_weak_full_loss"), "restricted": dmean(b, "diag_weak_restricted_loss")}
                for lr, b in lr_runs.items()}
        sel["diagnostic_weak_local_loss"] = diag
        first = next(iter(lr_runs.values()))
        lines += ["## Stage A: learning-rate check (homogeneous R=64, development seed)", "",
                  "| lr | " + " | ".join(f"core r{e['round']}" for e in first["eval"])
                  + " | weak local loss: full | restricted (replay) | full ≤ restricted |",
                  "|---|" + "---:|" * (len(first["eval"]) + 3)]
        for lr in sorted(lr_runs):
            b, dg = lr_runs[lr], diag[lr]
            ok = None if None in (dg["full"], dg["restricted"]) else dg["full"] <= dg["restricted"]
            lines.append(f"| {lr:g} | " + " | ".join(f"{100 * e['core_accuracy']:.2f}" for e in b["eval"]) + " | "
                         + f"{f(dg['full'], 4)} | {f(dg['restricted'], 4)} | {'—' if ok is None else ok} |")
        lines += ["", f"Selection rule (pre-registered): highest core accuracy at round {at}; within 0.5 pp of the "
                  f"best, the larger lr. **Chosen lr = {sel['chosen_lr']:g}.** The loss columns are a diagnostic, not "
                  "part of the rule: in C2-0 (lr 2e-4) full training had a higher local loss than restricted training "
                  "on the same batches, a sign that the full-rank reference steps too far; at a well-chosen lr full "
                  "training should not lose to restricted training locally.", ""]
    main = {p.parent.name.removeprefix("main_"): json.loads(p.read_text())
            for p in sorted(fed_root.glob("main_*/fed.json"))}
    ms = multiseed(fed_root, cf_report)
    order = [k for k in ("r8", "vref", "L2", "amp", "fslora_st", "fslora", "amp_a4", "amp_a16", "amp_a32",
                         "fslora_paper", "fedplora", "amp_zf", "fslora_zf", "vref_zf", "amp_step025") if k in main]
    order += [k for k in sorted(main) if k not in order]
    if main:
        lo, hi = cf_report["late_window"]
        elo, ehi = cf_report["early_window"]
        osc_from = cf_report["osc_from"]
        summ = {k: {"late": _window(b["eval"], lo, hi), "early": _window(b["eval"], elo, ehi),
                    "final": b["eval"][-1]["core_accuracy"], "osc": _osc(b["eval"], osc_from), "lr": b["lr"],
                    "method": b["method"], "amp_alpha": b.get("amp_alpha"), "mechanism": mechanism_summary(b)}
                for k, b in main.items()}
        out["main"] = summ
        lines += ["## Stage B: main comparison", "",
                  f"`late` = mean over evaluations in rounds {lo}–{hi}; `early` = rounds {elo}–{ehi}; `osc` = mean "
                  f"|Δ all-client accuracy| between evaluations from round {osc_from}.", "",
                  "| run | method | α | late | early | final | osc | lr |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for k in order:
            s = summ[k]
            lines.append(f"| {k} | {s['method']} | {f(s['amp_alpha'], 0) if s['amp_alpha'] else '—'} | {f(100 * s['late'])} | "
                         f"{f(100 * s['early'])} | {f(100 * s['final'])} | {f(s['osc'])} | {s['lr']:g} |")
        curves = sorted({e["round"] for b in main.values() for e in b["eval"]})
        lines += ["", "Curves (core %):", "", "| round | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
        for t in curves:
            lines.append(f"| {t} | " + " | ".join(
                f(100 * next((e["core_accuracy"] for e in main[k]["eval"] if e["round"] == t), float("nan")))
                for k in order) + " |")
        mech_rows = [(k, m) for k in order for m in summ[k]["mechanism"]]
        if mech_rows:
            lines += ["", "## Mechanism diagnostics (replayed full training on the same batches; not aggregated)", "",
                      "Per weak client (medians): its update on its assigned components against its own full run "
                      "(product step / cos), and the function-space step against its own full run (≈ r/R when matched "
                      "component by component, more when overshooting). Aggregate: the method's global update against "
                      "this round's r8 aggregate (all participants full).", "",
                      "| run | round | client step | client cos | client function step | local loss (method / full) | "
                      "aggregate step | aggregate cos | aggregate function step | aggregate function cos |",
                      "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
            for k, m in mech_rows:
                lines.append(f"| {k} | {m['round']} | {f(m['client_product_step'])} | {f(m['client_product_cos'])} | "
                             f"{f(m['client_function_step'], 3)} | {f(m['client_local_loss'], 3)} / "
                             f"{f(m['client_full_local_loss'], 3)} | {f(m['aggregate_product_step'])} | "
                             f"{f(m['aggregate_product_cos'])} | {f(m['aggregate_function_step'])} | "
                             f"{f(m['aggregate_function_cos'])} |")
        g = lambda k: None if k not in summ else 100 * summ[k]["late"]  # noqa: E731
        v = {}
        if all(k in summ for k in ("r8", "L2", "vref", "amp")):
            v["H0_restricted_costs_accuracy"] = bool(g("L2") <= g("r8") - 1.0)
            v["H1_amp_matches_vref"] = bool(g("amp") >= g("vref") - 0.5)
            v["H2_amp_beats_L2"] = bool(g("amp") >= g("L2") + 1.0)
            v["H3_cold_start_acceptable"] = bool(100 * summ["amp"]["early"] >= 100 * summ["vref"]["early"] - 2.0)
            v["H4_amp_stable"] = bool(summ["amp"]["osc"] <= 1.5 * summ["vref"]["osc"]) if summ["vref"]["osc"] else None
        if "amp" in summ and "fslora_st" in summ:
            v["H5a_increment_only_beats_state_sketch"] = bool(g("amp") >= g("fslora_st") + 0.3)
        if "fslora_st" in summ and "fslora" in summ:
            v["H5b_gradient_path_diff_pp"] = g("fslora_st") - g("fslora")
            v["H5b_gradient_path_matters"] = bool(abs(g("fslora_st") - g("fslora")) >= 0.3)
        alts = [k for k in ("amp_a4", "amp_a16", "amp_a32") if k in summ]
        if "amp" in summ and alts:
            v["H6_alpha_R_over_r_ok"] = bool(g("amp") >= max(g(k) for k in alts) - 0.3)
            v["H6_best_alpha_run"] = max(["amp"] + alts, key=g)
        if "amp" in summ and "fslora_paper" in summ:
            v["E1_amp_minus_fslora_paper_pp"] = g("amp") - g("fslora_paper")
            v["E1_clear_advantage_over_published_fslora"] = bool(g("amp") - g("fslora_paper") >= 1.5)
            if "fslora" in summ:
                v["E1b_fslora_hajek_minus_paper_pp"] = g("fslora") - g("fslora_paper")
        if "fedplora" in summ and "r8" in summ:
            v["E2_fedplora_collapses"] = bool(g("fedplora") <= g("r8") - 10.0)
            if "amp" in summ:
                v["E2_amp_minus_fedplora_pp"] = g("amp") - g("fedplora")
        if all(k in summ for k in ("amp", "amp_zf", "fslora", "fslora_zf")):
            v["F1_zero_fill_drives_gain"] = bool(g("amp_zf") >= g("amp") + 1.0 and g("fslora_zf") >= g("fslora") + 1.0)
            v["F1_amp_zf_minus_amp_pp"] = g("amp_zf") - g("amp")
            v["F1_fslora_zf_minus_fslora_pp"] = g("fslora_zf") - g("fslora")
        if "fslora_zf" in summ and "fslora_paper" in summ:
            v["F2_allocation_irrelevant"] = bool(abs(g("fslora_zf") - g("fslora_paper")) < 0.5)
            v["F2_fslora_zf_minus_paper_pp"] = g("fslora_zf") - g("fslora_paper")
        if "amp_zf" in summ and "fslora_paper" in summ:
            d = g("amp_zf") - g("fslora_paper")
            v["F3_amp_zf_minus_fslora_paper_pp"] = d
            v["F3_reading"] = "equivalent" if abs(d) < 0.5 else ("amp_zf better" if d > 0 else "fslora_paper better")
        if all(k in summ for k in ("amp_step025", "amp_zf", "amp")):
            d_zf, d_amp = g("amp_step025") - g("amp_zf"), g("amp_step025") - g("amp")
            v["G1_step025_minus_amp_zf_pp"], v["G1_step025_minus_amp_pp"] = d_zf, d_amp
            v["G1_reading"] = ("consistent with the step-size explanation (single seed; not a proof)" if abs(d_zf) < 0.5
                               else "not consistent with step size alone (single seed)" if abs(d_amp) < 0.5
                               else "in between / unclear (single seed)")
        out["verdicts"] = v
        lines += ["", "## Pre-registered reading (single seed)", "", "| question | criterion | result |", "|---|---|---|",
                  f"| H0 restricted training costs accuracy over many rounds | L2 late ≤ r8 late − 1.0 pp | {v.get('H0_restricted_costs_accuracy', '—')} |",
                  f"| H1 amp matches vref | amp late ≥ vref late − 0.5 pp | {v.get('H1_amp_matches_vref', '—')} |",
                  f"| H2 amp beats L2 | amp late ≥ L2 late + 1.0 pp | {v.get('H2_amp_beats_L2', '—')} |",
                  f"| H3 cold start acceptable | amp early ≥ vref early − 2.0 pp | {v.get('H3_cold_start_acceptable', '—')} |",
                  f"| H4 amp stable | osc(amp) ≤ 1.5 × osc(vref) | {v.get('H4_amp_stable', '—')} |",
                  f"| H5a amplifying only the increment beats sketching the state | amp late ≥ fslora_st late + 0.3 pp | "
                  f"{v.get('H5a_increment_only_beats_state_sketch', '—')} |",
                  f"| H5b gradient path (fslora_st − fslora), reported as an implementation effect | \\|diff\\| ≥ 0.3 pp | "
                  f"{v.get('H5b_gradient_path_matters', '—')} ({f(v.get('H5b_gradient_path_diff_pp'))} pp) |",
                  f"| H6 α = R/r is right (not re-selected) | amp(α=R/r) late ≥ best of α ∈ {{4, 16, 32}} − 0.3 pp | "
                  f"{v.get('H6_alpha_R_over_r_ok', '—')} (best run: {v.get('H6_best_alpha_run', '—')}) |",
                  f"| E1 external: amp vs FSLoRA as published (random + zero-fill, our AdamW protocol) | amp − fslora_paper "
                  f"≥ 1.5 pp; < 1.5 pp → add seeds 2041 / 2042 for fslora_paper | {v.get('E1_clear_advantage_over_published_fslora', '—')} "
                  f"({f(v.get('E1_amp_minus_fslora_paper_pp'))} pp; fslora − fslora_paper = {f(v.get('E1b_fslora_hajek_minus_paper_pp'))} pp) |",
                  f"| E2 external: Fed-PLoRA style collapses like L2 | fedplora late ≤ r8 late − 10 pp | "
                  f"{v.get('E2_fedplora_collapses', '—')} (amp − fedplora = {f(v.get('E2_amp_minus_fedplora_pp'))} pp) |",
                  f"| F1 zero-fill aggregation drives the gain (balanced allocation) | amp_zf ≥ amp + 1.0 and fslora_zf ≥ "
                  f"fslora + 1.0 pp | {v.get('F1_zero_fill_drives_gain', '—')} (amp: {f(v.get('F1_amp_zf_minus_amp_pp'))}, "
                  f"fslora: {f(v.get('F1_fslora_zf_minus_fslora_pp'))} pp) |",
                  f"| F2 allocation (random vs balanced) does not matter | \\|fslora_zf − fslora_paper\\| < 0.5 pp | "
                  f"{v.get('F2_allocation_irrelevant', '—')} ({f(v.get('F2_fslora_zf_minus_paper_pp'))} pp) |",
                  f"| F3 (external positioning, NOT a single-factor test: local rule and allocation both differ) amp with "
                  f"zero-fill vs FSLoRA as published | \\|amp_zf − fslora_paper\\| < 0.5 pp → equivalent | "
                  f"{v.get('F3_reading', '—')} ({f(v.get('F3_amp_zf_minus_fslora_paper_pp'))} pp) |",
                  f"| G1 step size vs zero-fill: amp with Hájek and server step 0.25 | within 0.5 pp of amp_zf → consistent "
                  f"with the step-size explanation; within 0.5 pp of amp → not consistent with step size alone | "
                  f"{v.get('G1_reading', '—')} (− amp_zf "
                  f"{f(v.get('G1_step025_minus_amp_zf_pp'))}, − amp {f(v.get('G1_step025_minus_amp_pp'))} pp) |", "",
                  "F1–F3 and G1 are single-seed early reads; the claims use the three-seed comparisons. F3 (and MS8) is "
                  "an external positioning comparison, not a single-factor mechanism test. Causal reading of stage E: "
                  "MS7 / MS11 (does zero-fill help amp / FSLoRA's forward), MS12 (does allocation matter), MS9 (does "
                  "zero-fill help without restricted training); G1 is auxiliary evidence for the step-size explanation "
                  "only.", "",
                  "Reading: H0–H2 true → the method holds on the decoder LLM; add seeds 2041 / 2042. H0 false → "
                  "restricted training costs little here over many rounds (scope statement). H0 true, H1 false → use "
                  "H3, H6 and the mechanism table to separate cold start, α, direction and the method itself. H5a is "
                  "the state-sketch claim; H5b is an implementation effect, not part of the method claim. H6 false → "
                  "report that R/r is not the best α here; the method's α is not re-chosen on these results. One seed: "
                  "differences below ~0.5 pp are not interpreted."]
    rp = replay_checks(fed_root, tuple(cf_report["late_window"]))
    if rp:
        out["replay_checks"] = rp
        lines += ["", "## Replay checks: older code versions re-run with the current code (full 60 rounds)", "",
                  "Pass rule (pre-registered, design §13 second revision): round-1 mean local loss within 0.1% relative "
                  "and |late core accuracy (rounds 40–60) of replay − original| ≤ 0.5 pp. Only a passing replay verifies "
                  "its older version; multi-seed comparisons enforce this (see below). Per-evaluation differences are "
                  "descriptive.", "",
                  "| original run | version (original → replay) | round-1 loss rel. diff | late diff (pp) | max abs eval diff (pp) | status |",
                  "|---|---|---:|---:|---:|---|"]
        for k, x in rp.items():
            lines.append(f"| {k} | {str(x['original_version'])[:8]} → {str(x['replay_version'])[:8]} | "
                         f"{x['round1_loss_rel_diff']:.2e} | {f(x['late_diff_pp'])} | {f(x['max_abs_eval_diff_pp'])} | "
                         f"{x['status']} |")
    if ms["runs"]:
        out["multiseed"] = ms
        lines += ["", "## Stage C: three seeds (2040 from stage B; 2041, 2042), late core-weak accuracy", "",
                  "Same plan and lr (stage A choice) for every seed; seeds share schedule and initialisation across "
                  "methods, so differences are paired per seed. 95% t interval with 2 degrees of freedom (t = 4.30).", "",
                  "| method | s2040 | s2041 | s2042 | mean ± sd |", "|---|---:|---:|---:|---:|"]
        for m in MULTISEED_METHODS:
            vals = [ms["runs"].get(f"{m}_s{s}", {}).get("core") for s in (2040, 2041, 2042)]
            ok = [x for x in vals if x is not None]
            lines.append(f"| {m} | " + " | ".join(f(x) for x in vals) + " | "
                         + (f"{np.mean(ok):.2f} ± {np.std(ok, ddof=1) if len(ok) > 1 else 0:.2f}" if ok else "—") + " |")
        if ms["c2fed_versions"] or ms["blocked"]:
            groups_v = {}
            for run, h in sorted(ms["c2fed_versions"].items()):
                groups_v.setdefault(h, []).append(run)
            lines += ["", f"Version gate: current `scra/c2fed.py` = `{ms['current_version']}`; verified versions: "
                      + "; ".join(f"`{h}` ({why})" for h, why in ms["verified_versions"].items()) + ". Runs used, by "
                      "version (protocol fields and model / data / cache content checked identical): "
                      + "; ".join(f"`{h}`: {', '.join(r)}" for h, r in groups_v.items()) + "."]
            subs = {k: v for k, v in ms["used"].items() if v.startswith("replay_")}
            if subs:
                lines += ["Replaced by current-code replays (original version unverified): "
                          + ", ".join(f"{k} ← {v}" for k, v in subs.items()) + "."]
            if ms["blocked"]:
                lines += ["**Blocked runs (unverified version, no current-code replay):** "
                          + "; ".join(ms["blocked"].values()) + ". Comparisons that need them are not computed."]
        lines += ["", "| comparison | per seed | mean ± sd | 95% CI | criterion | result |", "|---|---|---:|---|---|---|"]
        for key, (a, b, _, text) in MULTISEED_COMPARISONS.items():
            pr = ms["paired"].get(key)
            lines.append(f"| {key}: {a} − {b} | "
                         + ("—" if pr is None else ", ".join(f"{x:+.2f}" for x in pr["per_seed"])) + " | "
                         + ("—" if pr is None else f"{pr['mean']:+.2f} ± {pr['sd']:.2f}") + " | "
                         + ("—" if pr is None else f"[{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}]") + " | "
                         + f"{text} | {ms['verdicts'].get(key, '—')} |")
        lines += ["", "MS5–MS12 (design §13): MS5 amp vs FSLoRA as published; MS7 / MS11 whether zero-fill helps amp / "
                  "FSLoRA's forward under balanced allocation; MS8 amp with zero-fill vs FSLoRA as published; MS9 whether "
                  "zero-fill helps even with full local training (vref); MS12 balanced vs random allocation under zero-fill; "
                  "MS6 / MS10 descriptive. Update norms are reported only descriptively and are not a success criterion. "
                  "G1 (amp_step025 vs amp_zf) differs in aggregation as well as step size, so agreement is only "
                  "'consistent with' the step-size explanation.",
                  "", "Reading: MS1 true → amp matches vref across seeds (the single-seed H1 gap was noise); false → "
                  "report the gap with its interval. MS2 decides amp vs FSLoRA's forward with true gradient (both with "
                  "Hájek): 'equivalent', 'amp better' or 'fslora better'. MS3 / MS4 are descriptive (distance to the "
                  "homogeneous upper bound; cost of uploading only assigned components)."]
    fed_root.mkdir(parents=True, exist_ok=True)
    (fed_root / "REPORT_C2_1.md").write_text("\n".join(lines) + "\n")
    atomic_json(fed_root / "report_c2_1.json", out)
    return out
