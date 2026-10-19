"""D3: screening realizable fixes for the restricted-training gap at fixed global states
(research_two_directions §5).

D2 located most of the one-step loss in the classifier head: heads trained next to restricted LoRA move
1.6-1.9x as far along the full-training head direction, and plain head averaging does not cancel it.
D3 keeps D2's protocol (same snapshots, real schedule of the following rounds, one update applied to the
snapshot, paired difference to r8 on the same draw) and compares rules a deployed system could use.

Local training per draw: strong clients train full and replay one restricted run (random pair per
adapter); weak clients train full (reference only), restricted, restricted with a frozen head, and
restricted with amplified forward (alpha = R / r, scra.core.local_train).

  reference      r8, vref, shrink (LoRA x0.63, restricted heads), shrink_fullhead (oracle head)
  head scalar    shrink_h40 / h55 / h70: weak head displacement x c
  head drop      shrink_hstrong: head from strong clients only; shrink_hnone: head not updated
  head only      hajek_h55: unshrunk Hajek LoRA, weak head x0.55
  frozen head    fh_hajek / fh_shrink: weak clients never train the head, head from strong clients only
  amplified      amp_hajek: amplified restricted training, Hajek, plain head mean (no constants);
                 amp_fullhead: same LoRA with full-training heads (isolates the LoRA part)
  calibrated     rcs2: per-adapter LoRA coefficient and one head coefficient, both from strong replays pooled
                 over the draws (a stand-in for the online estimate); oracle2: same from weak clients' own
                 full runs (not realizable, upper reference for scalar calibration)

Hajek LoRA and head means are linear in the weak displacements, so every scaled rule is base + c * slope;
rcs2 / oracle2 are evaluated in a second pass once the pooled coefficients are known.
Outputs <run_root>/d3/round_XXXX.json.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .core import (adapter_names, atomic_json, balanced_allocation, evaluate_global, from_blocks, head_names,
                   local_train, participants, random_pairs, rcs_coefficients, to_blocks, weighted_head_mean)
from .d2 import core_weak_clients, dispersion, lora_update, product_deltas
from .data import ClientStream, Dataset, override_roles
from .model import TaskModel

PASS1 = ["r8", "vref", "shrink", "shrink_fullhead", "shrink_h40", "shrink_h55", "shrink_h70", "shrink_hstrong",
         "shrink_hnone", "hajek_h55", "fh_hajek", "fh_shrink", "amp_hajek", "amp_fullhead"]
PASS2 = ["rcs2", "oracle2"]
RULES = PASS1 + PASS2
DISP_RULES = ["r8", "vref", "shrink", "shrink_h55", "fh_hajek", "amp_hajek", "rcs2", "oracle2"]


# ---------------------------------------------------------------- linear pieces
def scaled_parts(full, weak_disp, weights, strong, weak, sets, counts):
    """Hajek LoRA as base + c * slope in the weak displacements (c may be a per-adapter [J] vector)."""
    weak_sets = {c: sets[c] for c in weak}
    zero = {c: (torch.zeros_like(weak_disp[c]) if c in weak else full[c]) for c in strong + weak}
    one = {c: (weak_disp[c] if c in weak else full[c]) for c in strong + weak}
    base = lora_update(zero, weights, strong, weak_sets, counts, "hajek")
    return base, lora_update(one, weights, strong, weak_sets, counts, "hajek") - base


def head_parts(heads_full, weak_heads, weights, strong, weak, g_state):
    """Plain weighted head mean as base + c * slope in the weak head displacements."""
    names = list(next(iter(heads_full.values())))
    zero = {c: ({n: g_state[n] for n in names} if c in weak else heads_full[c]) for c in strong + weak}
    one = {c: (weak_heads[c] if c in weak else heads_full[c]) for c in strong + weak}
    base = weighted_head_mean(zero, weights, g_state)
    top = weighted_head_mean(one, weights, g_state)
    return base, {n: top[n] - base[n] for n in names}


def apply_c(base, slope, c):
    if isinstance(base, dict):
        return {n: base[n] + c * slope[n] for n in base}
    c = torch.as_tensor(c, dtype=base.dtype)
    return base + (c[:, None, None] if c.ndim == 1 else c) * slope


def strong_head(heads_full, weights, strong, g_state):
    names = list(next(iter(heads_full.values())))
    if not strong:
        return {n: g_state[n] for n in names}
    return weighted_head_mean({c: heads_full[c] for c in strong}, weights, g_state)


class Pooled:
    """Pooled restricted-vs-full statistics (vectors summed first, then the ratio)."""

    def __init__(self, J, R, D):
        self.restr, self.full, self.n = torch.zeros(J, R, D), torch.zeros(J, R, D), torch.zeros(J, R)
        self.h_restr, self.h_full = None, None

    def add(self, restr, full, mask, h_restr, h_full):
        m = mask.float()
        self.restr += m[..., None] * restr
        self.full += m[..., None] * full
        self.n += m
        self.h_restr = h_restr.clone() if self.h_restr is None else self.h_restr + h_restr
        self.h_full = h_full.clone() if self.h_full is None else self.h_full + h_full

    def coefficients(self, c_min, c_max):
        lora = rcs_coefficients(self.restr, self.full, self.n, 1, c_min, c_max)
        if self.h_restr is None:
            return lora, 1.0
        head = float((self.h_restr @ self.h_full) / (self.h_restr @ self.h_restr).clamp_min(1e-30))
        return lora, float(min(max(head, c_min), c_max))


def flat(head, g_state, names):
    return torch.cat([(head[n] - g_state[n]).reshape(-1).float() for n in names])


def sets_mask(sets_c, J, R):
    mask = torch.zeros(J, R, dtype=torch.bool)
    for j, chosen in enumerate(sets_c):
        mask[j, chosen] = True
    return mask


# ---------------------------------------------------------------- run
def run_d3(plan: dict, snapshot_round: int, output: Path, device: torch.device) -> Path:
    cfg, dz, d3 = plan["train"], plan["d0"], plan["d3"]
    out_path = output / f"round_{snapshot_round:04d}.json"
    if out_path.exists():
        print(f"{out_path} exists; skipping", flush=True)
        return out_path
    torch.set_num_threads(plan.get("cpu_threads", 4))
    started_all = time.perf_counter()
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    strong_all = override_roles(data.clients, dz["strong_fraction"])
    core = core_weak_clients(data, strong_all)
    R, r, seed = plan["lora"]["full_rank"], plan["lora"]["weak_rank"], dz["seed"]
    alpha = d3.get("amp_alpha", R / r)
    shrink = d3["shrink"]
    model = TaskModel.build(plan["model"], plan["num_labels"], R, plan["lora"]["scale"], seed).to(device)
    names = adapter_names(model)
    client_ids = sorted(data.clients)
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    snap = Path(plan["run_root"]) / "d0" / "snapshots" / f"round_{snapshot_round:04d}.pt"
    g_state = torch.load(snap, map_location="cpu", weights_only=True)
    g_blocks = to_blocks(g_state, names)
    hnames = head_names(g_state)
    J, _, D = g_blocks.shape
    pooled_strong, pooled_weak = Pooled(J, R, D), Pooled(J, R, D)

    def evaluate(state):
        ev = evaluate_global(model, data, state, device, cfg["eval_batch_size"])
        cl = ev["clients"]
        return {"acc": ev["client_macro_accuracy"], "nll": ev["client_macro_nll"],
                "core": float(np.mean([cl[str(c)][0] for c in core]))}

    def assemble(lora, head):
        state = from_blocks(g_blocks + lora, names, g_state)
        state.update(head)
        return state

    base_eval = evaluate(g_state)
    rows, stored = [], []
    lora_keep = {k: [] for k in DISP_RULES}
    head_keep = {k: [] for k in DISP_RULES}
    for d in range(1, d3["draws"] + 1):
        t = snapshot_round + d
        started = time.perf_counter()
        chosen = participants(seed, t, client_ids, plan["federation"]["clients_per_round"])
        strong = [c for c in chosen if c in strong_all]
        weak = [c for c in chosen if c not in strong_all]
        sets, counts = balanced_allocation(weak, J, R, r, seed, t)
        full, heads_full = {}, {}
        disp = {k: {} for k in ("restr", "fh", "amp")}
        heads = {k: {} for k in ("restr", "fh", "amp")}
        losses = {k: [] for k in ("full", "restr", "fh", "amp")}
        probe = {k: {"cos": [], "ratio": [], "head_step": []} for k in ("restr", "amp")}
        for c in chosen:
            def stream():
                return ClientStream(data.clients[c]["train_indices"], d3["seed"] + t, c, 0)
            state, rec = local_train(model, g_state, data, stream(), None, cfg, device)
            full[c] = to_blocks(state, names) - g_blocks
            heads_full[c] = {n: state[n] for n in hnames}
            hf = flat(heads_full[c], g_state, hnames)
            if c in strong:
                pair = random_pairs(J, R, r, d3["seed"], "d3replay", t, c)
                state, _ = local_train(model, g_state, data, stream(), pair, cfg, device)
                pooled_strong.add(to_blocks(state, names) - g_blocks, full[c], sets_mask(pair, J, R),
                                  flat(state, g_state, hnames), hf)
                continue
            losses["full"].append(rec["mean_loss"])
            mask = sets_mask(sets[c], J, R)
            variants = {"restr": {}, "fh": {"freeze_head": True}, "amp": {"amp_alpha": alpha}}
            for kind, extra in variants.items():
                state, rec = local_train(model, g_state, data, stream(), sets[c], cfg, device, **extra)
                disp[kind][c] = to_blocks(state, names) - g_blocks
                heads[kind][c] = {n: state[n] for n in hnames}
                losses[kind].append(rec["mean_loss"])
                if kind in probe:
                    p = probe[kind]
                    p["cos"].append(float(torch.nn.functional.cosine_similarity(
                        disp[kind][c], full[c], dim=-1)[mask].median()))
                    p["ratio"].append(float((disp[kind][c].norm(dim=-1)
                                             / full[c].norm(dim=-1).clamp_min(1e-12))[mask].median()))
                    p["head_step"].append(float((flat(heads[kind][c], g_state, hnames) @ hf)
                                                / (hf @ hf).clamp_min(1e-30)))
            pooled_weak.add(disp["restr"][c], full[c], mask, flat(heads["restr"][c], g_state, hnames), hf)

        weak_sets = {c: sets[c] for c in weak}
        zero = torch.zeros_like(counts)
        L_base, L_slope = scaled_parts(full, disp["restr"], weights, strong, weak, sets, counts)
        H_base, H_slope = head_parts(heads_full, heads["restr"], weights, strong, weak, g_state)
        fh_base, fh_slope = scaled_parts(full, disp["fh"], weights, strong, weak, sets, counts)
        head_full = weighted_head_mean(heads_full, weights, g_state)
        head_strong = strong_head(heads_full, weights, strong, g_state)
        head_none = {n: g_state[n] for n in hnames}
        lora_shrink = apply_c(L_base, L_slope, shrink)
        lora_amp = lora_update({c: (disp["amp"][c] if c in weak else full[c]) for c in chosen}, weights,
                               strong, weak_sets, counts, "hajek")
        rules = {
            "r8": (lora_update(full, weights, chosen, {}, zero, "owner"), head_full),
            "vref": (lora_update(full, weights, strong, weak_sets, counts, "hajek"), head_full),
            "shrink": (lora_shrink, apply_c(H_base, H_slope, 1.0)),
            "shrink_fullhead": (lora_shrink, head_full),
            "shrink_h40": (lora_shrink, apply_c(H_base, H_slope, 0.40)),
            "shrink_h55": (lora_shrink, apply_c(H_base, H_slope, 0.55)),
            "shrink_h70": (lora_shrink, apply_c(H_base, H_slope, 0.70)),
            "shrink_hstrong": (lora_shrink, head_strong),
            "shrink_hnone": (lora_shrink, head_none),
            "hajek_h55": (L_base + L_slope, apply_c(H_base, H_slope, 0.55)),
            "fh_hajek": (fh_base + fh_slope, head_strong),
            "fh_shrink": (apply_c(fh_base, fh_slope, shrink), head_strong),
            "amp_hajek": (lora_amp, weighted_head_mean({c: (heads["amp"][c] if c in weak else heads_full[c])
                                                        for c in chosen}, weights, g_state)),
            "amp_fullhead": (lora_amp, head_full),
        }
        row = {"round": t, "strong": len(strong), "weak": len(weak),
               "local_loss": {k: (float(np.mean(v)) if v else None) for k, v in losses.items()},
               "probe": {k: {m: (float(np.median(v)) if v else None) for m, v in p.items()} for k, p in probe.items()}}
        for rule in PASS1:
            lora, head = rules[rule]
            row[rule] = evaluate(assemble(lora, head))
            if rule in lora_keep:
                lora_keep[rule].append(lora)
                head_keep[rule].append(flat(head, g_state, hnames))
        stored.append((L_base, L_slope, H_base, H_slope))
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"[D3 r{snapshot_round}] draw {d}/{d3['draws']} (round {t}, strong {len(strong)}): "
              + " ".join(f"{k}={row[k]['core']:.4f}" for k in PASS1) + f" ({row['seconds']:.0f}s)", flush=True)

    # second pass: pooled coefficients (strong replays = realizable estimate; weak = oracle)
    lim = (d3["c_min"], d3["c_max"])
    coeffs = {"rcs2": pooled_strong.coefficients(*lim), "oracle2": pooled_weak.coefficients(*lim)}
    for row, (L_base, L_slope, H_base, H_slope) in zip(rows, stored):
        for rule, (c_lora, c_head) in coeffs.items():
            lora, head = apply_c(L_base, L_slope, c_lora), apply_c(H_base, H_slope, c_head)
            row[rule] = evaluate(assemble(lora, head))
            lora_keep[rule].append(lora)
            head_keep[rule].append(flat(head, g_state, hnames))
    del stored

    ref = product_deltas(lora_keep["r8"], g_blocks, names, g_state)
    head_ref = torch.stack(head_keep["r8"])
    disp_summary = {"r8": {"lora_product": dispersion(ref, ref), "head": dispersion(head_ref, head_ref)}}
    for rule in DISP_RULES[1:]:
        X = product_deltas(lora_keep[rule], g_blocks, names, g_state)
        disp_summary[rule] = {"lora_product": dispersion(X, ref), "head": dispersion(torch.stack(head_keep[rule]),
                                                                                     head_ref)}
        del X
    summary = {}
    for rule in RULES:
        acc = np.array([row[rule]["acc"] for row in rows])
        core_acc = np.array([row[rule]["core"] for row in rows])
        diff = acc - np.array([row["r8"]["acc"] for row in rows])
        core_diff = core_acc - np.array([row["r8"]["core"] for row in rows])
        summary[rule] = {
            "acc_mean": float(acc.mean()), "acc_std": float(acc.std()), "acc_min": float(acc.min()),
            "core_mean": float(core_acc.mean()), "core_std": float(core_acc.std()),
            "nll_mean": float(np.mean([row[rule]["nll"] for row in rows])),
            "diff_vs_r8_mean": float(diff.mean()), "diff_vs_r8_std": float(diff.std()),
            "core_diff_vs_r8_mean": float(core_diff.mean()), "core_diff_vs_r8_std": float(core_diff.std()),
            **disp_summary.get(rule, {}),
        }
    out = {"snapshot_round": snapshot_round, "draws": d3["draws"], "shrink": shrink, "amp_alpha": alpha,
           "rules": RULES, "core_weak_clients": core, "baseline": base_eval, "summary": summary, "rows": rows,
           "coefficients": {k: {"lora_per_adapter": [float(x) for x in v[0]], "lora_mean": float(v[0].mean()),
                                "head": v[1]} for k, v in coeffs.items()},
           "elapsed_seconds": time.perf_counter() - started_all,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    atomic_json(out_path, out)
    return out_path


# ---------------------------------------------------------------- report (pre-registered reading)
def closure(S: dict, rule: str) -> float | None:
    """Share of the shrink -> vref gap (core weak, one step) that `rule` closes; 0 = shrink, 1 = vref."""
    lo, hi = S["shrink"]["core_diff_vs_r8_mean"], S["vref"]["core_diff_vs_r8_mean"]
    if hi - lo <= 1e-9:
        return None
    return float((S[rule]["core_diff_vs_r8_mean"] - lo) / (hi - lo))


def d3_checks(blob: dict) -> dict:
    S = blob["summary"]
    cl = {k: closure(S, k) for k in S}
    core = {k: S[k]["core_diff_vs_r8_mean"] for k in S}
    best_h = max((cl[k] for k in ("shrink_h40", "shrink_h55", "shrink_h70") if cl[k] is not None), default=None)
    return {
        "closure": cl, "best_head_scalar_closure": best_h,
        "rcs2_minus_oracle2_pp": 100 * (core["rcs2"] - core["oracle2"]),
        "amp_minus_rcs2_pp": 100 * (core["amp_hajek"] - core["rcs2"]),
        "hstrong_minus_best_head_scalar": (None if best_h is None or cl["shrink_hstrong"] is None
                                           else cl["shrink_hstrong"] - best_h),
    }


def d3_verdicts(checks: list[dict]) -> dict:
    def majority(pred):
        def safe(c):
            try:
                return bool(pred(c))
            except TypeError:
                return False
        return sum(safe(c) for c in checks) >= (2 if len(checks) >= 3 else len(checks))
    return {
        "head_scalar_enough": majority(lambda c: c["best_head_scalar_closure"] >= 0.6),
        "rcs2_realizable": majority(lambda c: c["closure"]["rcs2"] >= 0.6 and abs(c["rcs2_minus_oracle2_pp"]) <= 0.3),
        "amplification_works": majority(lambda c: c["closure"]["amp_hajek"] >= 0.6),
        "amp_not_worse_than_rcs2": majority(lambda c: c["amp_minus_rcs2_pp"] >= -0.2),
        "weak_heads_carry_signal": majority(lambda c: c["hstrong_minus_best_head_scalar"] <= -0.2),
    }


def write_d3_report(run_root: Path) -> dict:
    import json
    files = sorted((run_root / "d3").glob("round_*.json"))
    if not files:
        raise FileNotFoundError(f"no D3 results under {run_root / 'd3'}")
    blobs = [json.loads(f.read_text()) for f in files]
    checks = [d3_checks(b) for b in blobs]
    verdicts = d3_verdicts(checks)
    lines = [f"# D3 realizable-fix screening ({run_root.name})", "",
             "Same protocol as D2: one update from each snapshot over the real schedule of the following rounds. "
             "`core diff` = core-weak accuracy minus r8 on the same draw. `closure` = share of the shrink → vref "
             "core gap closed (0 = shrink, 1 = vref). LoRA / head numbers are relative to |mean r8 update|².", ""]
    for b, c in zip(blobs, checks):
        S = b["summary"]
        co = b["coefficients"]
        lines += [f"## Snapshot round {b['snapshot_round']} ({b['draws']} draws, shrink {b['shrink']}, "
                  f"amp alpha {b['amp_alpha']}; before update: acc {100 * b['baseline']['acc']:.2f}, "
                  f"core {100 * b['baseline']['core']:.2f})", "",
                  "| rule | core diff mean | core diff std | diff mean | min acc | closure | LoRA paired | "
                  "LoRA step along r8 | head paired | head step along r8 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
        for rule in b["rules"]:
            s = S[rule]
            lp, hp = s.get("lora_product"), s.get("head")
            cl = c["closure"][rule]
            lines.append(
                f"| {rule} | {100 * s['core_diff_vs_r8_mean']:+.2f} | {100 * s['core_diff_vs_r8_std']:.2f} | "
                f"{100 * s['diff_vs_r8_mean']:+.2f} | {100 * s['acc_min']:.2f} | "
                + ("—" if cl is None else f"{cl:.2f}") + " | "
                + (f"{lp['paired_error']:.3f} | {lp['step_along_ref_mean']:.2f} ± {lp['step_along_ref_std']:.2f}"
                   if lp else "— | —") + " | "
                + (f"{hp['paired_error']:.3f} | {hp['step_along_ref_mean']:.2f} ± {hp['step_along_ref_std']:.2f}"
                   if hp else "— | —") + " |")
        med = lambda kind, m: np.median([row["probe"][kind][m] for row in b["rows"] if row["probe"][kind][m] is not None])
        loss = lambda k: np.mean([row["local_loss"][k] for row in b["rows"] if row["local_loss"][k] is not None])
        lines += ["",
                  f"Pooled coefficients: strong replays (rcs2) LoRA mean {co['rcs2']['lora_mean']:.3f}, head "
                  f"{co['rcs2']['head']:.3f}; weak oracle LoRA mean {co['oracle2']['lora_mean']:.3f}, head "
                  f"{co['oracle2']['head']:.3f}.",
                  f"Per weak client vs its full run (median): restricted block cos {med('restr', 'cos'):.2f}, norm "
                  f"ratio {med('restr', 'ratio'):.2f}, head step {med('restr', 'head_step'):.2f}; amplified block cos "
                  f"{med('amp', 'cos'):.2f}, norm ratio {med('amp', 'ratio'):.2f}, head step "
                  f"{med('amp', 'head_step'):.2f}.",
                  f"Mean local training loss (weak): full {loss('full'):.4f}, restricted {loss('restr'):.4f}, "
                  f"frozen head {loss('fh'):.4f}, amplified {loss('amp'):.4f}.", ""]
    lines += ["## Pre-registered reading (≥ 2 of 3 snapshots)", "",
              "| question | criterion | result |", "|---|---|---|",
              f"| a scalar on weak heads fixes most of it | best of shrink_h40/55/70 closure ≥ 0.6 | "
              f"{verdicts['head_scalar_enough']} |",
              f"| calibration from strong replays is realizable | rcs2 closure ≥ 0.6 and within 0.3 pp of oracle2 | "
              f"{verdicts['rcs2_realizable']} |",
              f"| amplified restricted training works without constants | amp_hajek closure ≥ 0.6 | "
              f"{verdicts['amplification_works']} |",
              f"| amplification is not worse than calibration | amp_hajek ≥ rcs2 − 0.2 pp | "
              f"{verdicts['amp_not_worse_than_rcs2']} |",
              f"| weak heads carry signal (dropping them hurts) | closure(hstrong) ≤ best head scalar − 0.2 | "
              f"{verdicts['weak_heads_carry_signal']} |", "",
              "Next step by outcome: amplification works and is not worse → main candidate is amplified restricted "
              "training, confirm with a full D0 run; else rcs2 realizable → RCS v2 (head + LoRA calibration) full "
              "D0 run; head scalar works but rcs2 not → estimator transfer problem, inspect coefficients; nothing "
              "closes ≥ 0.6 → the one-step view misses the mechanism, rethink before any method."]
    out_dir = run_root / "diagnostics_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT_D3.md").write_text("\n".join(lines) + "\n")
    report = {"checks": checks, "verdicts": verdicts, "snapshots": [b["snapshot_round"] for b in blobs]}
    atomic_json(out_dir / "report_d3.json", report)
    return report
