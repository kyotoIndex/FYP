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
import time

import numpy as np
import torch

from .c2probe import _core_acc, _fstep, _probe_rows, _pstep, _setup, identity, restrict_to_sets
from .core import atomic_json, atomic_save, balanced_allocation, local_train, participants, random_allocation
from .d2 import core_weak_clients, lora_update
from .data import ClientStream
from .llm import Groups, evaluate_llm, label_logprobs, product_gram

METHODS = ("r8", "L2", "vref", "amp", "fslora", "fslora_st", "fslora_paper", "fedplora")
RESTRICTED = ("L2", "amp", "fslora", "fslora_st", "fslora_paper", "fedplora")
# method -> (weak local training, allocation, aggregation)
SPEC = {"r8": ("full", None, "all"), "vref": ("full", "balanced", "hajek"),
        "L2": ("plain", "balanced", "owner"), "amp": ("amp", "balanced", "hajek"),
        "fslora": ("sketch", "balanced", "hajek"), "fslora_st": ("sketch_st", "balanced", "hajek"),
        "fslora_paper": ("sketch", "random", "zerofill"), "fedplora": ("plain", "random", "owner")}


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
                amp_alpha=alpha if (restricted and method == "amp") else None,
                sketch_alpha=sketch if (restricted and method in ("fslora", "fslora_st", "fslora_paper")) else None,
                sketch_straight_through=restricted and method == "fslora_st")
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
                           "amp_alpha": alpha if method == "amp" else None,
                           "sketch_alpha": sketch if method in ("fslora", "fslora_st", "fslora_paper") else None,
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


MULTISEED_METHODS = ("r8", "vref", "amp", "fslora")
MULTISEED_COMPARISONS = {   # pre-registered before the seed runs (design §9)
    "MS1": ("amp", "vref", "mean ≥ −0.5 pp → amp matches vref"),
    "MS2": ("amp", "fslora", "|mean| < 0.5 pp → equivalent; ≥ 0.5 pp with all seeds same sign → that side better"),
    "MS3": ("r8", "amp", "descriptive (gap to the upper bound)"),
    "MS4": ("r8", "vref", "descriptive (cost of uploading only assigned components)"),
}


def multiseed(fed_root: Path, cf_report: dict) -> dict:
    """Late core-weak accuracy (pp) for the multi-seed methods: seed 2040 from main_<m>, others from seed<s>_<m>."""
    from .batch2 import paired
    lo, hi = cf_report["late_window"]
    runs = {}
    for m in MULTISEED_METHODS:
        for seed, path in ((2040, fed_root / f"main_{m}" / "fed.json"),
                           (2041, fed_root / f"seed2041_{m}" / "fed.json"),
                           (2042, fed_root / f"seed2042_{m}" / "fed.json")):
            if path.exists():
                b = json.loads(path.read_text())
                if b["seed"] != seed:
                    if seed == 2040:          # stage-B runs on another seed are not part of this comparison
                        continue
                    raise ValueError(f"{path} has seed {b['seed']}, expected {seed}")
                runs[f"{m}_s{seed}"] = {"core": 100 * _window(b["eval"], lo, hi)}
    pairs = {k: paired(runs, a, b) for k, (a, b, _) in MULTISEED_COMPARISONS.items()}
    v = {}
    if pairs["MS1"] is not None:
        v["MS1"] = bool(pairs["MS1"]["mean"] >= -0.5)
    if pairs["MS2"] is not None:
        p2 = pairs["MS2"]
        n = len(p2["per_seed"])
        if abs(p2["mean"]) < 0.5:
            v["MS2"] = "equivalent"
        elif p2["mean"] >= 0.5 and p2["n_pos"] == n:
            v["MS2"] = "amp better"
        elif p2["mean"] <= -0.5 and p2["n_neg"] == n:
            v["MS2"] = "fslora better"
        else:
            v["MS2"] = "unclear (seeds disagree)"
    for k in ("MS3", "MS4"):
        if pairs[k] is not None:
            v[k] = "descriptive"
    return {"runs": runs, "paired": pairs, "verdicts": v}


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
                         "fslora_paper", "fedplora") if k in main]
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
                  f"{v.get('E2_fedplora_collapses', '—')} (amp − fedplora = {f(v.get('E2_amp_minus_fedplora_pp'))} pp) |", "",
                  "Reading: H0–H2 true → the method holds on the decoder LLM; add seeds 2041 / 2042. H0 false → "
                  "restricted training costs little here over many rounds (scope statement). H0 true, H1 false → use "
                  "H3, H6 and the mechanism table to separate cold start, α, direction and the method itself. H5a is "
                  "the state-sketch claim; H5b is an implementation effect, not part of the method claim. H6 false → "
                  "report that R/r is not the best α here; the method's α is not re-chosen on these results. One seed: "
                  "differences below ~0.5 pp are not interpreted."]
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
        lines += ["", "| comparison | per seed | mean ± sd | 95% CI | criterion | result |", "|---|---|---:|---|---|---|"]
        for key, (a, b, text) in MULTISEED_COMPARISONS.items():
            pr = ms["paired"].get(key)
            lines.append(f"| {key}: {a} − {b} | "
                         + ("—" if pr is None else ", ".join(f"{x:+.2f}" for x in pr["per_seed"])) + " | "
                         + ("—" if pr is None else f"{pr['mean']:+.2f} ± {pr['sd']:.2f}") + " | "
                         + ("—" if pr is None else f"[{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}]") + " | "
                         + f"{text} | {ms['verdicts'].get(key, '—')} |")
        lines += ["", "Reading: MS1 true → amp matches vref across seeds (the single-seed H1 gap was noise); false → "
                  "report the gap with its interval. MS2 decides amp vs FSLoRA's forward with true gradient (both with "
                  "Hájek): 'equivalent', 'amp better' or 'fslora better'. MS3 / MS4 are descriptive (distance to the "
                  "homogeneous upper bound; cost of uploading only assigned components)."]
    fed_root.mkdir(parents=True, exist_ok=True)
    (fed_root / "REPORT_C2_1.md").write_text("\n".join(lines) + "\n")
    atomic_json(fed_root / "report_c2_1.json", out)
    return out
