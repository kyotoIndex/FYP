"""D2: single-round dispersion at fixed global states (research_two_directions §4, item 6).

Question: is the restricted-training gap a per-round fluctuation (stitching variance) rather than an
expected bias, and does it live in the LoRA blocks or in the classifier head?

From a D0 snapshot s, simulate `draws` independent rounds that use the real participant schedule and
coverage-balanced allocation of rounds s+1..s+draws. Every participant trains in both modes on the same
batches (weak: full and restricted to its assigned components; strong: full). Each rule below forms a
one-step update that is applied to the snapshot (training does not continue) and evaluated:

  r8               every client trains full, all blocks, weighted mean (homogeneous rank-8 FedAvg)
  vref             full training, weak clients contribute only their assigned blocks, Hajek
  L2               restricted training, owner averaging                       (the D0 training rule)
  hajek            restricted training, Hajek
  shrink           restricted training x fixed c, Hajek                       (the fixedshrink rule)
  shrink_fullhead  `shrink` LoRA blocks + classifier heads from full training
  vref_restrhead   `vref` LoRA blocks + classifier heads from restricted training

Per rule and snapshot we report one-step accuracy (mean, spread across draws, paired difference to r8 on
the same draw) and, in exact product space (Delta W = B'A' - BA summed over adapters), the spread across
draws, the bias of the mean against r8's mean, and the step-size jitter along r8's mean direction.
Outputs <run_root>/d2/round_XXXX.json.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .core import (adapter_names, atomic_json, balanced_allocation, block_aggregates, evaluate_global,
                   from_blocks, head_names, local_train, participants, to_blocks, weighted_head_mean)
from .data import ClientStream, Dataset, override_roles
from .model import TaskModel

RULES = ["r8", "vref", "L2", "hajek", "shrink", "shrink_fullhead", "vref_restrhead"]


def core_weak_clients(data: Dataset, strong: list[int]) -> list[int]:
    strong_domains = {data.clients[c]["domain"] for c in strong}
    return sorted(c for c in data.clients if c not in strong and data.clients[c]["domain"] not in strong_domains)


def lora_update(deltas, weights, strong, weak_sets, counts, kind):
    agg = block_aggregates(deltas, weights, strong, weak_sets, counts)
    return torch.where(agg["covered"][..., None], agg[kind], torch.zeros_like(agg[kind]))


def round_updates(full, restr, heads_full, heads_restr, weights, strong, weak, sets, counts, shrink, g_state):
    """LoRA displacement [J,R,D] and new head state for every rule, for one simulated round."""
    chosen = strong + weak
    weak_sets = {c: sets[c] for c in weak}
    zero = torch.zeros_like(counts)
    mixed = {c: (restr[c] if c in weak else full[c]) for c in chosen}
    shrunk = {c: (restr[c] * shrink if c in weak else full[c]) for c in chosen}
    lora = {
        "r8": lora_update(full, weights, chosen, {}, zero, "owner"),
        "vref": lora_update(full, weights, strong, weak_sets, counts, "hajek"),
        "L2": lora_update(mixed, weights, strong, weak_sets, counts, "owner"),
        "hajek": lora_update(mixed, weights, strong, weak_sets, counts, "hajek"),
        "shrink": lora_update(shrunk, weights, strong, weak_sets, counts, "hajek"),
    }
    lora["shrink_fullhead"] = lora["shrink"]
    lora["vref_restrhead"] = lora["vref"]
    head_full = weighted_head_mean(heads_full, weights, g_state)
    head_restr = weighted_head_mean({c: (heads_restr[c] if c in weak else heads_full[c]) for c in chosen},
                                    weights, g_state)
    heads = {"r8": head_full, "vref": head_full, "L2": head_restr, "hajek": head_restr, "shrink": head_restr,
             "shrink_fullhead": head_full, "vref_restrhead": head_restr}
    return lora, heads


def product_deltas(disps: list[torch.Tensor], g_blocks: torch.Tensor, names, g_state) -> torch.Tensor:
    """[N, P] exact Delta W = B'A' - BA of every adapter, flattened and concatenated (scale omitted)."""
    before = [g_state[f"{n}.lora_b"].float() @ g_state[f"{n}.lora_a"].float() for n in names]
    size = sum(b.numel() for b in before)
    out = torch.empty(len(disps), size)
    for i, disp in enumerate(disps):
        new = from_blocks(g_blocks + disp, names, g_state)
        pos = 0
        for name, b0 in zip(names, before):
            after = new[f"{name}.lora_b"].float() @ new[f"{name}.lora_a"].float()
            out[i, pos:pos + b0.numel()] = (after - b0).reshape(-1)
            pos += b0.numel()
    return out


def dispersion(X: torch.Tensor, ref: torch.Tensor) -> dict:
    """Spread of the N draws in X [N, P] across draws, bias of their mean against ref's mean, paired error
    against ref on the same draw, and step along ref's mean direction; all relative to |mean(ref)|^2.
    Computed from Gram matrices, so only inner products of the draws are needed."""
    n = X.shape[0]
    gxx, gxr, grr = (X @ X.T).double(), (X @ ref.T).double(), (ref @ ref.T).double()
    sq = gxx.diagonal()
    mx2, mr2, mxr = gxx.sum() / n ** 2, grr.sum() / n ** 2, gxr.sum() / n ** 2
    proj = gxr.sum(1) / n / mr2                    # <x_d, mean(ref)> / |mean(ref)|^2
    return {
        "mean_sq": float(sq.mean() / mr2),
        "variance": float((sq.mean() - mx2) / mr2),
        "bias_sq": float((mx2 + mr2 - 2 * mxr) / mr2),
        "paired_error": float((sq + grr.diagonal() - 2 * gxr.diagonal()).mean() / mr2),
        "step_along_ref_mean": float(proj.mean()),
        "step_along_ref_std": float(proj.std(unbiased=False)),
        "orthogonal": float((sq - proj ** 2 * mr2).mean() / mr2),
    }


def run_d2(plan: dict, snapshot_round: int, output: Path, device: torch.device) -> Path:
    cfg, dz, d2 = plan["train"], plan["d0"], plan["d2"]
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
    model = TaskModel.build(plan["model"], plan["num_labels"], R, plan["lora"]["scale"], seed).to(device)
    names = adapter_names(model)
    client_ids = sorted(data.clients)
    weights = {c: float(data.clients[c]["samples"]) for c in client_ids}
    snap = Path(plan["run_root"]) / "d0" / "snapshots" / f"round_{snapshot_round:04d}.pt"
    g_state = torch.load(snap, map_location="cpu", weights_only=True)
    g_blocks = to_blocks(g_state, names)
    heads = head_names(g_state)

    def evaluate(state):
        ev = evaluate_global(model, data, state, device, cfg["eval_batch_size"])
        cl = ev["clients"]
        return {"acc": ev["client_macro_accuracy"], "nll": ev["client_macro_nll"],
                "core": float(np.mean([cl[str(c)][0] for c in core]))}

    base = evaluate(g_state)
    rows, lora_by_rule, head_by_rule = [], {k: [] for k in RULES}, {k: [] for k in RULES}
    for d in range(1, d2["draws"] + 1):
        t = snapshot_round + d
        started = time.perf_counter()
        chosen = participants(seed, t, client_ids, plan["federation"]["clients_per_round"])
        strong = [c for c in chosen if c in strong_all]
        weak = [c for c in chosen if c not in strong_all]
        sets, counts = balanced_allocation(weak, len(names), R, r, seed, t)
        full, restr, heads_full, heads_restr, pair_stats = {}, {}, {}, {}, []
        for c in chosen:
            def stream():
                return ClientStream(data.clients[c]["train_indices"], d2["seed"] + t, c, 0)
            state, _ = local_train(model, g_state, data, stream(), None, cfg, device)
            full[c] = to_blocks(state, names) - g_blocks
            heads_full[c] = {n: state[n] for n in heads}
            if c in weak:
                state, _ = local_train(model, g_state, data, stream(), sets[c], cfg, device)
                restr[c] = to_blocks(state, names) - g_blocks
                heads_restr[c] = {n: state[n] for n in heads}
                mask = torch.zeros(len(names), R, dtype=torch.bool)
                for j, chosen_k in enumerate(sets[c]):
                    mask[j, chosen_k] = True
                cos = torch.nn.functional.cosine_similarity(restr[c], full[c], dim=-1)[mask]
                ratio = (restr[c].norm(dim=-1) / full[c].norm(dim=-1).clamp_min(1e-12))[mask]
                pair_stats.append((float(cos.median()), float(ratio.median())))
        lora, new_heads = round_updates(full, restr, heads_full, heads_restr, weights, strong, weak, sets, counts,
                                        d2["shrink"], g_state)
        row = {"round": t, "strong": len(strong), "weak": len(weak),
               "weak_single_pair_cos_median": float(np.median([p[0] for p in pair_stats])) if pair_stats else None,
               "weak_single_pair_norm_ratio_median": float(np.median([p[1] for p in pair_stats])) if pair_stats else None}
        for rule in RULES:
            state = from_blocks(g_blocks + lora[rule], names, g_state)
            state.update(new_heads[rule])
            row[rule] = evaluate(state)
            lora_by_rule[rule].append(lora[rule])
            head_by_rule[rule].append(torch.cat([(new_heads[rule][n] - g_state[n]).reshape(-1).float()
                                                 for n in heads]))
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"[D2 r{snapshot_round}] draw {d}/{d2['draws']} (round {t}, strong {len(strong)}): "
              + " ".join(f"{k}={row[k]['acc']:.4f}" for k in RULES) + f" ({row['seconds']:.0f}s)", flush=True)

    # product-space and head dispersion, relative to r8's mean over draws (one rule in memory at a time)
    ref = product_deltas(lora_by_rule["r8"], g_blocks, names, g_state)
    head_ref = torch.stack(head_by_rule["r8"])
    lora_disp = {"r8": dispersion(ref, ref)}
    for rule in RULES[1:]:
        X = product_deltas(lora_by_rule[rule], g_blocks, names, g_state)
        lora_disp[rule] = dispersion(X, ref)
        del X
    summary = {}
    for rule in RULES:
        acc = np.array([row[rule]["acc"] for row in rows])
        core_acc = np.array([row[rule]["core"] for row in rows])
        diff = acc - np.array([row["r8"]["acc"] for row in rows])
        core_diff = core_acc - np.array([row["r8"]["core"] for row in rows])
        summary[rule] = {
            "acc_mean": float(acc.mean()), "acc_std": float(acc.std()), "acc_min": float(acc.min()),
            "core_mean": float(core_acc.mean()), "core_std": float(core_acc.std()), "core_min": float(core_acc.min()),
            "nll_mean": float(np.mean([row[rule]["nll"] for row in rows])),
            "diff_vs_r8_mean": float(diff.mean()), "diff_vs_r8_std": float(diff.std()),
            "core_diff_vs_r8_mean": float(core_diff.mean()), "core_diff_vs_r8_std": float(core_diff.std()),
            "lora_product": lora_disp[rule],
            "head": dispersion(torch.stack(head_by_rule[rule]), head_ref),
        }
    out = {"snapshot_round": snapshot_round, "draws": d2["draws"], "shrink": d2["shrink"], "rules": RULES,
           "core_weak_clients": core, "baseline": base, "summary": summary, "rows": rows,
           "elapsed_seconds": time.perf_counter() - started_all,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    atomic_json(out_path, out)
    return out_path


# ---------------------------------------------------------------- report (pre-registered reading)
def _ratio(a, b):
    return float(a / b) if b > 0 else None          # undefined when the reference does not vary


def d2_checks(blob: dict) -> dict:
    S = blob["summary"]
    std = {k: S[k]["diff_vs_r8_std"] for k in S}
    lp = {k: S[k]["lora_product"] for k in S}
    return {
        "acc_spread_ratio_shrink_vs_vref": _ratio(std["shrink"], std["vref"]),
        "lora_paired_error_ratio_shrink_vs_vref": _ratio(lp["shrink"]["paired_error"], lp["vref"]["paired_error"]),
        "lora_variance_ratio_shrink_vs_vref": _ratio(lp["shrink"]["variance"], lp["vref"]["variance"]),
        "lora_bias_over_variance_shrink": _ratio(lp["shrink"]["bias_sq"], lp["shrink"]["variance"]),
        "head_swap_shrink_fullhead_over_shrink": _ratio(std["shrink_fullhead"], std["shrink"]),
        "head_swap_vref_restrhead_over_vref": _ratio(std["vref_restrhead"], std["vref"]),
    }


def d2_verdicts(checks: list[dict]) -> dict:
    def majority(pred):
        def safe(c):
            try:
                return bool(pred(c))
            except TypeError:                         # an undefined ratio never counts as passing
                return False
        return sum(safe(c) for c in checks) >= (2 if len(checks) >= 3 else len(checks))
    single_round = majority(lambda c: c["acc_spread_ratio_shrink_vs_vref"] >= 2
                            and c["lora_paired_error_ratio_shrink_vs_vref"] >= 2)
    not_single_round = majority(lambda c: c["acc_spread_ratio_shrink_vs_vref"] < 1.5)
    head = majority(lambda c: c["head_swap_shrink_fullhead_over_shrink"] <= 0.6
                    or c["head_swap_vref_restrhead_over_vref"] >= 2)
    lora = majority(lambda c: c["head_swap_shrink_fullhead_over_shrink"] >= 0.8
                    and c["head_swap_vref_restrhead_over_vref"] < 1.5)
    return {"H1_single_round_dispersion": single_round, "H1_rejected_multi_round_effect": not_single_round,
            "bias_not_dominant": majority(lambda c: c["lora_bias_over_variance_shrink"] < 1),
            "source": "head" if head and not lora else ("lora" if lora and not head else "mixed/unclear")}


def write_d2_report(run_root: Path) -> dict:
    files = sorted((run_root / "d2").glob("round_*.json"))
    if not files:
        raise FileNotFoundError(f"no D2 results under {run_root / 'd2'}")
    import json
    blobs = [json.loads(f.read_text()) for f in files]
    checks = [d2_checks(b) for b in blobs]
    verdicts = d2_verdicts(checks)
    lines = [f"# D2 single-round dispersion ({run_root.name})", "",
             "One simulated round from each snapshot, repeated over the real schedule of the following rounds; "
             "every rule is applied once to the same snapshot. `diff` = accuracy minus r8 on the same draw "
             "(removes participant-sampling noise shared by all rules). Product-space numbers are relative to "
             "|mean r8 update|^2.", ""]
    for b, c in zip(blobs, checks):
        S = b["summary"]
        lines += [f"## Snapshot round {b['snapshot_round']} ({b['draws']} draws, shrink {b['shrink']}; "
                  f"before update: acc {100 * b['baseline']['acc']:.2f}, core {100 * b['baseline']['core']:.2f})", "",
                  "| rule | acc mean | acc std | diff mean | diff std | core diff mean | core diff std | min acc "
                  "| LoRA var | LoRA bias² | LoRA paired | step along r8 (mean ± std) | head paired |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
        for rule in b["rules"]:
            s, lp = S[rule], S[rule]["lora_product"]
            lines.append(f"| {rule} | {100 * s['acc_mean']:.2f} | {100 * s['acc_std']:.2f} | "
                         f"{100 * s['diff_vs_r8_mean']:+.2f} | {100 * s['diff_vs_r8_std']:.2f} | "
                         f"{100 * s['core_diff_vs_r8_mean']:+.2f} | {100 * s['core_diff_vs_r8_std']:.2f} | "
                         f"{100 * s['acc_min']:.2f} | {lp['variance']:.3f} | {lp['bias_sq']:.3f} | "
                         f"{lp['paired_error']:.3f} | {lp['step_along_ref_mean']:.2f} ± {lp['step_along_ref_std']:.2f} | "
                         f"{s['head']['paired_error']:.3f} |")
        pair_cos = [row["weak_single_pair_cos_median"] for row in b["rows"] if row["weak_single_pair_cos_median"]]
        pair_ratio = [row["weak_single_pair_norm_ratio_median"] for row in b["rows"]
                      if row["weak_single_pair_norm_ratio_median"]]
        lines += ["", "Checks: " + ", ".join(f"{k} = " + ("—" if v is None else f"{v:.2f}") for k, v in c.items()),
                  f"Weak single-pair restricted vs full (assigned blocks, median over draws): cos "
                  f"{np.median(pair_cos):.2f}, norm ratio {np.median(pair_ratio):.2f} "
                  f"(D1 pair-averaged: cos ≈ 0.9, ratio ≈ 1.1).", ""]
    lines += ["## Pre-registered reading (≥ 2 of 3 snapshots)", "",
              "| question | criterion | result |", "|---|---|---|",
              f"| H1: the gap is single-round dispersion from stitching | acc spread ratio shrink/vref ≥ 2 and LoRA "
              f"paired-error ratio ≥ 2 | {verdicts['H1_single_round_dispersion']} |",
              f"| H1 rejected: instability builds up over rounds instead | acc spread ratio < 1.5 | "
              f"{verdicts['H1_rejected_multi_round_effect']} |",
              f"| expected bias is not the main term after shrinking | LoRA bias² / variance < 1 | "
              f"{verdicts['bias_not_dominant']} |",
              f"| where it lives | head: shrink_fullhead/shrink ≤ 0.6 or vref_restrhead/vref ≥ 2; LoRA: ≥ 0.8 and < 1.5 | "
              f"{verdicts['source']} |", "",
              "Next step by outcome: H1 true + LoRA → fix the local training dynamics of the restricted components; "
              "H1 true + head → fix head aggregation / co-adaptation (cheaper); H1 rejected → multi-round rollout "
              "diagnostic before any method."]
    out_dir = run_root / "diagnostics_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT_D2.md").write_text("\n".join(lines) + "\n")
    report = {"checks": checks, "verdicts": verdicts, "snapshots": [b["snapshot_round"] for b in blobs]}
    atomic_json(out_dir / "report_d2.json", report)
    return report
