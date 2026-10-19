"""D0: one logged federated run (design §7.1).

Training rule: coverage-balanced allocation + owner averaging (ladder L2; `aggregation` can be set to
"hajek"/"mix" for later use, "prefix" allocation exists for an equivalence test). The run also computes,
without using them, the Hajek aggregate and the A-SCRA shadow quantities (strong clients' restricted
replay -> online kappa, a_t). Logged per block (adapter j, component k) and per round:

  o_t = strong mean - weak mean displacement            (= D - m in design §5.6)
  owner - hajek, |hajek|, weak within-group dispersion, design variances of both estimators
  shadow: m_hat = strong restricted - strong full on the strong client's random pair

Method variants (2026-10-17, discriminating experiments):
  weak_lr_scale        weak clients train with lr * scale (fixed, decided before the run)
  weak_shrink.mode     "none" | "fixed" (multiply weak displacements by `value`) |
                       "rcs" (Replay-Calibrated Shrinkage: per-adapter c_j estimated online from strong
                       clients' restricted replays; c_j = <EMA restr, EMA full> / |EMA restr|^2, vectors
                       averaged per block before the ratio)

Batch 2026-10-18 (research_two_directions §6):
  weak_amp_alpha       None | number | "auto" (= R / r_i): amplified restricted training of weak clients
  lora.weak_rank_choices  optional list, e.g. [1, 2, 4]: each weak client gets a fixed rank r_i drawn once
                       from the list (seeded by client id); coverage-balanced hetero dealing and Hajek with
                       per-client inclusion r_i / R
  save_stats           false skips writing d0_stats.pt (large; only needed for the diagnostics report)

Batch 2 controls (roadmap_2026-10-18 stage A):
  weak_forward         "plain" | "fslora": FSLoRA forward (selected components x R/r_i as a whole, unselected
                       components dropped), true gradient | "fslora_st": same forward value, gradient through
                       the unscaled product as in amp (isolates sketching the state; FIX_BATCH2 §2)
  weak_lr_compensation None | "auto" | number: weak LoRA lr x alpha (head lr unchanged), server divides the
                       weak LoRA displacement by alpha (first-order twin of amplification)
  allocation "random"  Fed-PLoRA-style independent random component selection
  aggregation "zerofill"  every participant counts in every block, zeros where untrained / unsent (FSLoRA, FLASC)
  weak_upload "topk"   FLASC-style: weak clients train fully and upload the global top r/R fraction of LoRA
                       entries by magnitude (needs weak_train_full and zerofill). Strong clients upload fully, as
                       in every method here, so per-client budgets equal amp's (heterogeneous budgets). The
                       owner / Hajek diagnostics treat all blocks as sent in these runs and are not meaningful;
                       training uses zerofill only.

Outputs in <run_root>/d0/: d0_stats.pt (accumulated tensors), d0_rounds.json (per-round scalars and
evaluation curve), snapshots/round_XXXX.pt (global task state for D1), checkpoint.pt (resume).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .core import (adapter_names, atomic_json, atomic_save, balanced_allocation, block_aggregates,
                   evaluate_global, from_blocks, hetero_allocation, local_train, mix, participants,
                   prefix_allocation, random_allocation, random_pairs, rcs_coefficients, to_blocks, topk_sparsify,
                   weighted_head_mean)
from .data import ClientStream, Dataset, derived_seed, override_roles
from .model import TaskModel


def _zeros(J, R, D=None):
    return torch.zeros(J, R) if D is None else torch.zeros(J, R, D)


def new_stats(J, R, D, windows):
    s = {
        # systematic part of o (sum of vectors) overall and per window; energy for persistence ratio
        "o_sum": _zeros(J, R, D), "o_sq": _zeros(J, R), "o_n": _zeros(J, R),
        "o_win_sum": torch.zeros(len(windows), J, R, D), "o_win_n": torch.zeros(len(windows), J, R),
        "o_prev": _zeros(J, R, D), "o_prev_ok": torch.zeros(J, R, dtype=torch.bool),
        "o_cos_sum": _zeros(J, R), "o_cos_n": _zeros(J, R),
        # estimator magnitudes and gap
        "gap_sq": _zeros(J, R), "hajek_sq": _zeros(J, R), "owner_sq": _zeros(J, R),
        "rho_gap_sum": _zeros(J, R),            # sum of (rho_own - rho) over rounds with strong
        # weak dispersion and design variances (per round, rounds with >=2 weak contributors)
        "sigma2_sum": _zeros(J, R), "var_own_sum": _zeros(J, R), "var_haj_sum": _zeros(J, R), "var_n": _zeros(J, R),
        # shadow A-SCRA: strong mode effect
        "m_sq": _zeros(J, R), "m_full_sq": _zeros(J, R), "m_n": _zeros(J, R),
        "m_sum": _zeros(J, R, D), "m_dot_o": _zeros(J, R),
        # windowed vector sums of m_hat, for the self-correlation-free kappa (average vectors, then dot)
        "m_win_sum": torch.zeros(len(windows), J, R, D), "m_win_n": torch.zeros(len(windows), J, R),
    }
    return s


def weak_ranks(plan: dict, clients: list[int], seed: int) -> dict[int, int] | None:
    """Fixed per-client weak rank when lora.weak_rank_choices is set (resource property, not per round)."""
    choices = plan["lora"].get("weak_rank_choices")
    if not choices:
        return None
    return {c: random.Random(derived_seed(seed, "weak-rank", c)).choice(choices) for c in clients}


def code_hashes() -> dict[str, str]:
    """sha256 of every source file of this package, written to identity.json (not part of the resume check)."""
    here = Path(__file__).resolve().parent
    return {f"scra/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}


def amp_alpha_for(setting, R: int, r_i: int) -> float | None:
    if setting is None:
        return None
    return R / r_i if setting == "auto" else float(setting)


def run_d0(plan: dict, output: Path, device: torch.device, resume: bool = False,
           stop_after_round: int | None = None) -> dict:
    cfg, dz = plan["train"], plan["d0"]
    seed = dz["seed"]
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / "checkpoint.pt"
    if (output / "d0_rounds.json").exists() and not resume:
        raise FileExistsError(f"{output} already finished; refusing to overwrite")
    if ckpt.exists() and not resume:
        raise FileExistsError(f"{ckpt} exists; pass --resume")
    torch.set_num_threads(plan.get("cpu_threads", 4))

    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    strong_all = override_roles(data.clients, dz["strong_fraction"])
    R, r = plan["lora"]["full_rank"], plan["lora"]["weak_rank"]
    model = TaskModel.build(plan["model"], plan["num_labels"], R, plan["lora"]["scale"], seed).to(device)
    names = adapter_names(model)
    client_ids = sorted(data.clients)
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    rho_pop = sum(weights[c] for c in strong_all) / sum(weights.values())
    rounds, windows = plan["federation"]["rounds"], dz["windows"]
    identity = {"plan_id": plan["id"], "seed": seed, "strong_clients": strong_all, "allocation": dz["allocation"],
                "aggregation": dz["aggregation"], "R": R, "r": r,
                "weak_train_full": bool(dz.get("weak_train_full", False)),
                "weak_lr_scale": float(dz.get("weak_lr_scale", 1.0)),
                "weak_shrink": dict(dz.get("weak_shrink", {"mode": "none"}))}
    if dz.get("weak_amp_alpha") is not None or plan["lora"].get("weak_rank_choices"):
        identity.update({"weak_amp_alpha": dz.get("weak_amp_alpha"),
                         "weak_rank_choices": plan["lora"].get("weak_rank_choices")})
    extra = {k: dz[k] for k in ("weak_forward", "weak_lr_compensation", "weak_upload") if dz.get(k) is not None}
    identity.update(extra)
    if dz.get("weak_upload") == "topk" and not (identity["weak_train_full"] and dz["aggregation"] == "zerofill"):
        raise ValueError("weak_upload=topk needs weak_train_full and zerofill aggregation (FLASC)")
    ranks = weak_ranks(plan, [c for c in client_ids if c not in strong_all], seed)
    if ranks is not None and dz["allocation"] != "balanced":
        raise ValueError("weak_rank_choices needs balanced allocation")
    shrink = identity["weak_shrink"]
    if shrink["mode"] == "rcs" and not dz["shadow_strong_replay"]:
        raise ValueError("RCS needs shadow_strong_replay=true")
    cfg_weak = dict(cfg, lr=cfg["lr"] * identity["weak_lr_scale"])

    if resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st["identity"] != identity:
            raise ValueError("checkpoint identity differs")
        print(f"resuming D0 after round {st['round']}", flush=True)
    else:
        start = model.task_state()
        blocks0 = to_blocks(start, names)
        J, _, D = blocks0.shape
        st = {"identity": identity, "round": 0, "global": start, "positions": {c: 0 for c in client_ids},
              "stats": new_stats(J, R, D, windows), "rounds": [], "eval": [],
              "kappa_num": 0.0, "kappa_den": 0.0, "kappa_seen": 0, "elapsed": 0.0,
              "rcs_restr": torch.zeros(J, R, D), "rcs_full": torch.zeros(J, R, D), "rcs_n": torch.zeros(J, R)}
        st["eval"].append({"round": 0, **evaluate_global(model, data, start, device, cfg["eval_batch_size"])})
        atomic_json(output / "identity.json", {**identity, "dataset": data.manifest_summary, "rho_population": rho_pop,
                                               "client_domains": {str(c): data.clients[c]["domain"] for c in client_ids},
                                               "plan": plan, "torch": torch.__version__, "host": platform.node(),
                                               "code_hashes": code_hashes(),
                                               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    streams = {c: ClientStream(data.clients[c]["train_indices"], seed, c, st["positions"][c]) for c in client_ids}
    S_ = st["stats"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for t in range(st["round"] + 1, rounds + 1):
        started = time.perf_counter()
        chosen = participants(seed, t, client_ids, plan["federation"]["clients_per_round"])
        strong = [c for c in chosen if c in strong_all]
        weak = [c for c in chosen if c not in strong_all]
        inclusion = None
        if ranks is not None:
            sets, counts = hetero_allocation(weak, ranks, len(names), R, seed, t)
            inclusion = {c: ranks[c] / R for c in weak}
        elif dz["allocation"] == "prefix":
            sets, counts = prefix_allocation(weak, len(names), R, r)
        elif dz["allocation"] == "random":
            sets, counts = random_allocation(weak, len(names), R, r, seed, t)
        else:
            sets, counts = balanced_allocation(weak, len(names), R, r, seed, t)
        g_state = st["global"]
        g_blocks = to_blocks(g_state, names)
        states, deltas, losses, shadow = {}, {}, [], {}
        for c in chosen:
            position = streams[c].position
            # weak_train_full: variance-matched reference — weak clients train every component, but the
            # server still only uses their assigned blocks (same contributor counts, no training-mode effect)
            train_sets = None if (c in strong or identity["weak_train_full"]) else sets[c]
            r_c = ranks[c] if ranks and c in ranks else r
            alpha = None if train_sets is None else amp_alpha_for(dz.get("weak_amp_alpha"), R, r_c)
            sketch = (R / r_c if (train_sets is not None and dz.get("weak_forward") in ("fslora", "fslora_st"))
                      else None)
            lr_comp = (None if train_sets is None
                       else amp_alpha_for(dz.get("weak_lr_compensation"), R, r_c))
            new_state, rec = local_train(model, g_state, data, streams[c], train_sets,
                                         cfg if c in strong else cfg_weak, device, amp_alpha=alpha,
                                         sketch_alpha=sketch, lora_lr_scale=lr_comp,
                                         sketch_straight_through=dz.get("weak_forward") == "fslora_st")
            states[c] = new_state
            deltas[c] = to_blocks(new_state, names) - g_blocks
            if lr_comp is not None:
                deltas[c] = deltas[c] / lr_comp                # server-side / alpha of the LoRA displacement
            if c in weak and dz.get("weak_upload") == "topk":
                deltas[c] = topk_sparsify(deltas[c], r_c / R)
            losses.append(rec["mean_loss"])
            if c in strong and dz["shadow_strong_replay"]:
                pairs = random_pairs(len(names), R, r, seed, "shadow", t, c)
                replay = ClientStream(data.clients[c]["train_indices"], seed, c, position)  # same batches
                restricted, _ = local_train(model, g_state, data, replay, pairs, cfg, device)
                shadow[c] = (pairs, to_blocks(restricted, names) - g_blocks)
        weak_sets = {c: sets[c] for c in weak}
        if dz.get("weak_upload") == "topk":                     # sparse entries anywhere: all blocks count
            weak_sets = {c: [list(range(R)) for _ in names] for c in weak}
        # ---------------- weak shrinkage (fixed or Replay-Calibrated, RCS)
        J_ = len(names)
        c_vec = torch.ones(J_)
        if shrink["mode"] == "rcs":
            beta = shrink.get("beta", 0.9)
            for c, (pairs, restr) in shadow.items():            # update EMA vectors with this round's replays
                for j, pair in enumerate(pairs):
                    for k in pair:
                        st["rcs_restr"][j, k] = beta * st["rcs_restr"][j, k] + (1 - beta) * restr[j, k]
                        st["rcs_full"][j, k] = beta * st["rcs_full"][j, k] + (1 - beta) * deltas[c][j, k]
                        st["rcs_n"][j, k] += 1
            c_vec = rcs_coefficients(st["rcs_restr"], st["rcs_full"], st["rcs_n"], shrink.get("warmup_replays", 3),
                                     shrink.get("c_min", 0.2), shrink.get("c_max", 1.0))
        elif shrink["mode"] == "fixed":
            c_vec = torch.full((J_,), float(shrink["value"]))
        if shrink["mode"] != "none":
            for c in weak:
                deltas[c] = deltas[c] * c_vec[:, None, None]
        agg = block_aggregates(deltas, weights, strong, weak_sets, counts, inclusion)

        # ---------------- training update
        if dz["aggregation"] == "zerofill":
            disp = agg["zerofill"]
        else:
            a = {"owner": 0.0, "hajek": 1.0}.get(dz["aggregation"], dz.get("mix_a", 0.0))
            disp = mix(agg, a)
        new_blocks = g_blocks + torch.where(agg["covered"][..., None], disp, torch.zeros_like(disp))
        new_global = from_blocks(new_blocks, names, g_state)
        new_global.update(weighted_head_mean(states, weights, g_state))
        st["global"] = new_global

        # ---------------- diagnostics
        rec = {"round": t, "strong": len(strong), "weak": len(weak), "loss": float(np.mean(losses)),
               "update_norm": float(disp.norm()), "hajek_minus_owner_norm": float((agg["hajek"] - agg["owner"]).norm()),
               "shrink_mean": float(c_vec.mean()), "shrink_min": float(c_vec.min()), "shrink_max": float(c_vec.max())}
        n_weak = len(weak)
        s_w, w_w, w_tot = agg["strong_weight"], agg["weak_weight"], agg["weak_total"]
        if strong and n_weak:
            rho_t = s_w / (s_w + w_tot)
            rho_own = s_w / (s_w + w_w)                       # [J, R]
            o = agg["strong_mean"] - agg["weak_mean"]          # [J, R, D]
            ok = w_w > 0
            okf = ok.float()
            S_["o_sum"] += o * okf[..., None]
            S_["o_sq"] += (o.pow(2).sum(-1)) * okf
            S_["o_n"] += okf
            for wi, (lo, hi) in enumerate(windows):
                if lo <= t <= hi:
                    S_["o_win_sum"][wi] += o * okf[..., None]; S_["o_win_n"][wi] += okf
            both = ok & S_["o_prev_ok"]
            cos = torch.nn.functional.cosine_similarity(o, S_["o_prev"], dim=-1)
            S_["o_cos_sum"] += torch.where(both, cos, torch.zeros_like(cos))
            S_["o_cos_n"] += both.float()
            S_["o_prev"] = torch.where(ok[..., None], o, S_["o_prev"])
            S_["o_prev_ok"] |= ok
            gap = agg["hajek"] - agg["owner"]
            S_["gap_sq"] += gap.pow(2).sum(-1) * okf
            S_["hajek_sq"] += agg["hajek"].pow(2).sum(-1) * okf
            S_["owner_sq"] += agg["owner"].pow(2).sum(-1) * okf
            S_["rho_gap_sum"] += (rho_own - rho_t) * okf
            rec.update({"rho_t": float(rho_t), "rho_own_mean": float(rho_own[ok].mean())})
            # shadow A-SCRA kappa from strong restricted replay
            num = den = 0.0
            for c, (pairs, restr) in shadow.items():
                for j, pair in enumerate(pairs):
                    for k in pair:
                        if not ok[j, k]:
                            continue
                        m_hat = restr[j, k] - deltas[c][j, k]
                        num += float(m_hat @ o[j, k]); den += float(o[j, k] @ o[j, k])
                        S_["m_sq"][j, k] += float(m_hat @ m_hat)
                        S_["m_full_sq"][j, k] += float(deltas[c][j, k] @ deltas[c][j, k])
                        S_["m_n"][j, k] += 1
                        S_["m_sum"][j, k] += m_hat
                        S_["m_dot_o"][j, k] += float(m_hat @ o[j, k])
                        for wi, (lo, hi) in enumerate(windows):
                            if lo <= t <= hi:
                                S_["m_win_sum"][wi, j, k] += m_hat; S_["m_win_n"][wi, j, k] += 1
            if den > 0:
                beta = dz["kappa_beta"]
                st["kappa_num"] = beta * st["kappa_num"] + (1 - beta) * num
                st["kappa_den"] = beta * st["kappa_den"] + (1 - beta) * den
                st["kappa_seen"] += 1
            if st["kappa_seen"] >= dz["kappa_warmup"] and st["kappa_den"] > 0:
                kappa = -st["kappa_num"] / st["kappa_den"]
                rho_own_typ = s_w / (s_w + float(w_w[ok].mean()))
                a_shadow = float(np.clip(1 - (1 - rho_t) * kappa / max(rho_own_typ - rho_t, 1e-12), 0, 1))
                rec.update({"kappa": kappa, "a_shadow": a_shadow})
        # weak dispersion and design variances (any round with >=2 weak contributors)
        if n_weak:
            for j in range(len(names)):
                for k in range(R):
                    contrib = [c for c in weak if k in sets[c][j]]
                    if len(contrib) < 2:
                        continue
                    ws = torch.tensor([weights[c] for c in contrib])
                    vals = torch.stack([deltas[c][j, k] for c in contrib])
                    mean = (ws[:, None] * vals).sum(0) / ws.sum()
                    sigma2 = float((ws * (vals - mean).pow(2).sum(-1)).sum() / ws.sum())
                    cnt = len(contrib)
                    s_share_own = s_w / (s_w + float(w_w[j, k]))
                    s_share_haj = s_w / (s_w + w_tot) if strong else 0.0
                    var_own = (1 - s_share_own) ** 2 * sigma2 / cnt
                    var_haj = (1 - s_share_haj) ** 2 * (1 - cnt / n_weak) * sigma2 / cnt
                    S_["sigma2_sum"][j, k] += sigma2; S_["var_own_sum"][j, k] += var_own
                    S_["var_haj_sum"][j, k] += var_haj; S_["var_n"][j, k] += 1
        rec["seconds"] = time.perf_counter() - started
        st["rounds"].append(rec)
        st["positions"] = {c: s.position for c, s in streams.items()}
        if t % plan["evaluation"]["every_rounds"] == 0 or t == rounds:
            ev = evaluate_global(model, data, st["global"], device, cfg["eval_batch_size"])
            st["eval"].append({"round": t, **ev})
            print(f"[D0] round {t}: macro_acc={ev['client_macro_accuracy']:.4f} loss={rec['loss']:.4f} "
                  f"shrink={rec['shrink_mean']:.3f} [{rec['shrink_min']:.2f},{rec['shrink_max']:.2f}] "
                  f"kappa={rec.get('kappa', float('nan')):.3f} a_shadow={rec.get('a_shadow', float('nan')):.2f}", flush=True)
        if t in dz["snapshot_rounds"]:
            atomic_save(output / "snapshots" / f"round_{t:04d}.pt", st["global"])
        st["round"] = t
        st["elapsed"] += time.perf_counter() - started
        if t % plan["checkpoint_every"] == 0 or t == rounds or (stop_after_round and t >= stop_after_round):
            atomic_save(ckpt, st)
        if stop_after_round is not None and t >= stop_after_round:
            return {"stopped_after_round": t}

    if dz.get("save_stats", True):
        atomic_save(output / "d0_stats.pt", {"stats": S_, "adapter_names": names, "windows": windows,
                                             "rho_population": rho_pop, "R": R, "r": r})
    out = {**identity, "rho_population": rho_pop, "weak_ranks": ({str(c): v for c, v in ranks.items()} if ranks else None),
           "rounds": st["rounds"], "eval": st["eval"],
           "elapsed_seconds": st["elapsed"],
           "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
    atomic_save(output / "final_state.pt", st["global"])
    atomic_json(output / "d0_rounds.json", out)
    return out
