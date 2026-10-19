"""Turn D0/D1 outputs into the design-doc criteria (§7.1, §5.6, §5.7, failure modes F2/F3/F9/F11, E1).

All thresholds come from the design doc and plan.json["criteria"]; nothing here is tuned on task accuracy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def analyze_d0(run_root: Path, crit: dict) -> dict:
    d0 = Path(run_root) / "d0"
    blob = torch.load(d0 / "d0_stats.pt", map_location="cpu", weights_only=False)
    rounds = json.loads((d0 / "d0_rounds.json").read_text())
    s = blob["stats"]
    n = s["o_n"].clamp_min(1)
    o_mean = s["o_sum"] / n[..., None]                       # systematic part of o per block
    gap = s["rho_gap_sum"] / n                                # mean (rho_own - rho) when strong present
    b_sys = gap[..., None] * o_mean                           # systematic owner-minus-SCRA per round
    haj_rms_sq = s["hajek_sq"] / n
    rel_bias = math.sqrt(float(b_sys.pow(2).sum()) / max(float(haj_rms_sq.sum()), 1e-30))
    # persistence: adjacent window means
    wn = s["o_win_n"].clamp_min(1)
    wmeans = s["o_win_sum"] / wn[..., None]
    cosines = []
    for w in range(wmeans.shape[0] - 1):
        a, b = wmeans[w].flatten(), wmeans[w + 1].flatten()
        if a.norm() > 0 and b.norm() > 0:
            cosines.append(float(torch.dot(a, b) / (a.norm() * b.norm())))
    window_cos = sum(cosines) / len(cosines) if cosines else None
    round_cos = float((s["o_cos_sum"].sum() / s["o_cos_n"].sum().clamp_min(1)))
    energy_ratio = float(o_mean.pow(2).sum() / (s["o_sq"] / n).sum().clamp_min(1e-30))
    vn = s["var_n"].clamp_min(1)
    var_own, var_haj = float((s["var_own_sum"] / vn).sum()), float((s["var_haj_sum"] / vn).sum())
    bias_sq = float(b_sys.pow(2).sum())
    k_star = (var_haj - var_own) / bias_sq if bias_sq > 0 else float("inf")
    sigma = (s["sigma2_sum"] / vn).sqrt()                    # [J, R]
    cv2_within = float(((sigma.std(dim=1, unbiased=False) / sigma.mean(dim=1).clamp_min(1e-30)) ** 2).mean())
    cv2_all = float((sigma.std(unbiased=False) / sigma.mean().clamp_min(1e-30)) ** 2)
    rel_sigma = sigma / haj_rms_sq.sqrt().clamp_min(1e-30)
    cv2_all_rel = float((rel_sigma.std(unbiased=False) / rel_sigma.mean().clamp_min(1e-30)) ** 2)
    mn = s["m_n"].clamp_min(1)
    strong_mode_rel = math.sqrt(float((s["m_sq"] / mn).sum()) / max(float((s["m_full_sq"] / mn).sum()), 1e-30))
    rec = rounds["rounds"]
    kappas = [r["kappa"] for r in rec if "kappa" in r]
    a_sh = [r["a_shadow"] for r in rec if "a_shadow" in r]
    upd = [r["update_norm"] for r in rec]
    third = max(len(upd) // 3, 1)
    growth = (sum(upd[-third:]) / third) / max(sum(upd[:third]) / third, 1e-30)
    ev = rounds["eval"]
    # self-correlation-free kappa: average vectors within each window first, then take inner products
    r_ratio = blob["r"] / blob["R"]
    rho = blob["rho_population"]
    rho_own = rho / (rho + r_ratio * (1 - rho))
    kappa_windows = []
    if "m_win_sum" in s:
        for w in range(s["o_win_sum"].shape[0]):
            mn_w = s["m_win_n"][w]
            ok = (mn_w > 0) & (s["o_win_n"][w] > 0)
            if ok.sum() == 0:
                continue
            mbar = s["m_win_sum"][w] / mn_w.clamp_min(1)[..., None]
            obar = s["o_win_sum"][w] / s["o_win_n"][w].clamp_min(1)[..., None]
            num = float((mbar * obar).sum(-1)[ok].sum()); den = float(obar.pow(2).sum(-1)[ok].sum())
            if den > 0:
                kappa_windows.append(-num / den)
    a_of = lambda k: min(max(1 - (1 - rho) * k / max(rho_own - rho, 1e-12), 0.0), 1.0)
    acc = d0_accuracy(run_root, rounds, crit.get("late_start_round", 110))
    out = {
        "relative_systematic_bias": rel_bias,
        "adjacent_window_cosine": window_cos,
        "adjacent_round_cosine_descriptive": round_cos,
        "systematic_energy_ratio": energy_ratio,
        "var_owner_sum": var_own, "var_hajek_sum": var_haj, "bias_sq_sum": bias_sq, "K_star_rounds": k_star,
        "sigma_cv2_within_adapter": cv2_within, "sigma_cv2_all_blocks": cv2_all,
        "sigma_cv2_all_blocks_relative_to_update": cv2_all_rel,
        "shadow_strong_mode_relative": strong_mode_rel,
        "shadow_kappa_last": kappas[-1] if kappas else None,
        "shadow_kappa_mean_last50": sum(kappas[-50:]) / len(kappas[-50:]) if kappas else None,
        "shadow_a_mean_last50": sum(a_sh[-50:]) / len(a_sh[-50:]) if a_sh else None,
        "vector_kappa_windows": kappa_windows,
        "vector_kappa_last3_mean": sum(kappa_windows[-3:]) / len(kappa_windows[-3:]) if kappa_windows else None,
        "vector_a_last3": a_of(sum(kappa_windows[-3:]) / len(kappa_windows[-3:])) if kappa_windows else None,
        "accuracy": acc,
        "update_norm_late_over_early": growth,
        "loss_first10": sum(r["loss"] for r in rec[:10]) / min(10, len(rec)),
        "loss_last10": sum(r["loss"] for r in rec[-10:]) / min(10, len(rec)),
        "final_macro_accuracy": ev[-1]["client_macro_accuracy"],
        "curve": [(e["round"], e["client_macro_accuracy"]) for e in ev],
    }
    out["checks"] = {
        "F2_bias_large_enough": rel_bias >= crit["min_relative_bias"],
        "F3_direction_persistent": window_cos is not None and window_cos >= crit["min_window_cosine"],
        "F3_K_star_small_enough": k_star <= crit["max_K_star"],
        "E1_neyman_worthwhile": cv2_within >= crit["min_cv2_for_neyman"],
        "F9_no_update_blowup": growth <= crit["max_update_growth"],
    }
    return out


def analyze_d1_round(path: Path, rho_override: float | None = None) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    means, pop, R, r = blob["means"], blob["population"], blob["R"], blob["r"]
    rho = rho_override if rho_override is not None else pop["strong"] / (pop["strong"] + pop["weak"])
    rho_own = rho / (rho + (r / R) * (1 - rho))
    core_share = pop["core_total"] / pop["weak"] if pop["weak"] else 1.0
    if means["demoted"]["n"] == 0:
        core_share = 1.0

    def weak(mode):
        return core_share * means["core"][mode] + (1 - core_share) * means["demoted"][mode]

    s_full, s_restr = means["strong"]["full"], means["strong"]["restr"]
    w_full, w_restr = weak("full"), weak("restr")
    T = rho * s_full + (1 - rho) * w_full
    D = s_full - w_full
    m_w = w_restr - w_full
    m_s = s_restr - s_full
    e_scra = (rho * s_full + (1 - rho) * w_restr) - T
    e_owner = (rho_own * s_full + (1 - rho_own) * w_restr) - T
    identity_err = float((e_owner - (e_scra + (rho_own - rho) * (D - m_w))).abs().max())
    o = D - m_w

    def tot(x):
        return float(x.pow(2).sum().sqrt())

    def kappa(m):
        return -float((m * o).sum()) / max(float(o.pow(2).sum()), 1e-30)

    def a_of(k):
        return min(max(1 - (1 - rho) * k / max(rho_own - rho, 1e-12), 0.0), 1.0)

    blocks_scra_better = float(((e_scra.pow(2).sum(-1)) < (e_owner.pow(2).sum(-1))).float().mean())
    product = {}
    snap = Path(path).parent.parent / "d0" / "snapshots" / f"round_{blob['snapshot_round']:04d}.pt"
    if snap.exists():
        from .core import to_blocks
        state = torch.load(snap, map_location="cpu", weights_only=True)
        g = to_blocks(state, blob["adapter_names"])
        d_out = state[f"{blob['adapter_names'][0]}.lora_b"].shape[0]
        pn = lambda x: math.sqrt(max(product_ip(x, x, g, d_out), 0.0))
        oo = product_ip(o, o, g, d_out)
        k_pw = -product_ip(m_w, o, g, d_out) / max(oo, 1e-30)
        k_ps = -product_ip(m_s, o, g, d_out) / max(oo, 1e-30)
        product = {"norm_e_scra": pn(e_scra), "norm_e_owner": pn(e_owner), "norm_T": pn(T),
                   "kappa_weak_oracle": k_pw, "kappa_strong_estimate": k_ps,
                   "a_star_oracle": None, "a_from_strong": None}
        product["scra_beats_owner"] = product["norm_e_scra"] < product["norm_e_owner"]
    k_w, k_s = kappa(m_w), kappa(m_s)
    cos_ms_mw = float((m_w * m_s).sum() / (m_w.norm() * m_s.norm()).clamp_min(1e-30))
    per = blob["per_client"]
    m_rel_by_group = {}
    for g in ("strong", "core", "demoted"):
        vals = [per[c]["m_rel"].median().item() for c in per if per[c]["group"] == g]
        if vals:
            m_rel_by_group[g] = sum(vals) / len(vals)
    out = {
        "snapshot_round": blob["snapshot_round"], "rho": rho, "rho_owner": rho_own,
        "norm_T": tot(T), "norm_e_scra": tot(e_scra), "norm_e_owner": tot(e_owner),
        "norm_data_bias": tot((rho_own - rho) * D), "norm_mode_weak": tot(m_w),
        "relative_mode_weak": tot(m_w) / max(tot(w_full), 1e-30),
        "relative_mode_strong": tot(m_s) / max(tot(s_full), 1e-30),
        "blocks_where_scra_better": blocks_scra_better,
        "kappa_weak_oracle": k_w, "kappa_strong_estimate": k_s,
        "a_star_oracle": a_of(k_w), "a_from_strong": a_of(k_s),
        "cos_mode_weak_vs_strong": cos_ms_mw,
        "norm_D_core": tot(s_full - means["core"]["full"]),
        "norm_D_demoted": tot(s_full - means["demoted"]["full"]) if means["demoted"]["n"] else None,
        "median_client_mode_rel_by_group": m_rel_by_group,
        "identity_check_max_abs": identity_err,
        "product_space": product,
    }
    if product:
        product["a_star_oracle"] = a_of(product["kappa_weak_oracle"])
        product["a_from_strong"] = a_of(product["kappa_strong_estimate"])
    out["checks"] = {
        "F1_scra_beats_owner_D1": out["norm_e_scra"] < out["norm_e_owner"],
        "F11_A4_mode_cosine_ok": cos_ms_mw >= 0.5,
        # compare the decisions the two kappas imply (a sign test is meaningless when both are ~0)
        "F11_A4_implied_a_agrees": abs(a_of(k_w) - a_of(k_s)) <= 0.2,
    }
    return out


def _late(history, clients, start=110):
    pts = [h for h in history if h["round"] >= start]
    if not pts or not clients:
        return None
    return 100.0 * sum(sum(h["clients"][str(c)][0] for c in clients) / len(clients) for h in pts) / len(pts)


def d0_accuracy(run_root: Path, rounds: dict, start: int = 110) -> dict:
    """Core-weak / all-client late accuracy of D0 (ladder L2), next to the strong-fraction sweep cells
    with the same seed and schedule (fedavg_r8, fedavg_r2, prefix hetero = L0) when available."""
    ident = json.loads((Path(run_root) / "d0" / "identity.json").read_text())
    sweep_summary = Path(run_root).parent / "headroom_strongfrac_20261017_v1" / "summary_strongfrac.json"
    if "client_domains" in ident:
        domains = {int(c): d for c, d in ident["client_domains"].items()}
        strong_domains = {domains[c] for c in ident["strong_clients"]}
        core = sorted(c for c, d in domains.items() if d not in strong_domains)
        everyone = sorted(domains)
    elif sweep_summary.exists():
        # D0 run with the pre-review code (no client_domains in identity): reuse the sweep's fixed core-weak set
        core = json.loads(sweep_summary.read_text())["core_weak_clients"]
        everyone = sorted(int(c) for c in rounds["eval"][0]["clients"])
        domains = {c: None for c in everyone}
    else:
        return {"skipped": "no client_domains in identity and no sweep summary"}
    out = {"core_clients": len(core), "late_start_round": start,
           "L2_d0": {"core": _late(rounds["eval"], core, start), "all": _late(rounds["eval"], everyone, start)}}
    sweep = Path(run_root).parent / "headroom_strongfrac_20261017_v1"
    seed = ident["seed"]
    frac_tag = f"sf{round(100 * len(ident['strong_clients']) / len(domains)):02d}"
    for name, cell in (("r8", f"fedavg_r8_seed{seed}"), ("r2", f"fedavg_r2_seed{seed}"),
                       ("L0_prefix", f"hetero_r8r2_{frac_tag}_seed{seed}")):
        path = sweep / cell / "result.json"
        if path.exists():
            hist = json.loads(path.read_text())["history"]
            out[name] = {"core": _late(hist, core, start), "all": _late(hist, everyone, start)}
    main_rounds = Path(run_root).parent / "scra_diagnostics_20261017_v1" / "d0" / "d0_rounds.json"
    if main_rounds.exists() and Path(run_root).resolve() != main_rounds.parents[1].resolve():
        # e.g. the variance-matched reference: put the main D0 (L2) next to it, same seed and schedule
        main_eval = json.loads(main_rounds.read_text())["eval"]
        out["L2_main"] = {"core": _late(main_eval, core, start), "all": _late(main_eval, everyone, start)}
    vref_rounds = Path(run_root).parent / "scra_diagnostics_20261017_v1_vref" / "d0" / "d0_rounds.json"
    if vref_rounds.exists() and Path(run_root).resolve() != vref_rounds.parents[1].resolve():
        vref_eval = json.loads(vref_rounds.read_text())["eval"]
        out["vref_ref"] = {"core": _late(vref_eval, core, start), "all": _late(vref_eval, everyone, start)}
    out["this_run_label"] = _run_label(ident)
    if "L2_main" in out and "vref_ref" in out and out["L2_d0"]["core"] is not None:
        span = out["vref_ref"]["core"] - out["L2_main"]["core"]
        out["share_of_vref_minus_L2_recovered_core"] = (out["L2_d0"]["core"] - out["L2_main"]["core"]) / span if abs(span) > 1e-9 else None
    if all(k in out for k in ("r8", "r2")) and out["L2_d0"]["core"] is not None:
        span = out["r8"]["core"] - out["r2"]["core"]
        out["L2_recovered_fraction_core"] = (out["L2_d0"]["core"] - out["r2"]["core"]) / span if abs(span) > 1e-9 else None
        if "L0_prefix" in out:
            out["L0_recovered_fraction_core"] = (out["L0_prefix"]["core"] - out["r2"]["core"]) / span if abs(span) > 1e-9 else None
        out["remaining_gap_core_pp"] = out["r8"]["core"] - out["L2_d0"]["core"]
    return out


def _run_label(ident: dict) -> str:
    parts = []
    if ident.get("weak_train_full"):
        parts.append(f"weak train full, upload {ident.get('r')} of {ident.get('R')}")
    if ident.get("weak_lr_scale", 1.0) != 1.0:
        parts.append(f"weak lr x{ident['weak_lr_scale']}")
    shrink = ident.get("weak_shrink", {"mode": "none"})
    if shrink.get("mode") == "rcs":
        parts.append("RCS shrink")
    elif shrink.get("mode") == "fixed":
        parts.append(f"fixed shrink {shrink.get('value')}")
    parts.append(ident.get("aggregation", "?"))
    return "this run: " + ", ".join(parts)


def product_ip(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor, d_out: int) -> float:
    """Sum over blocks of <L x, L y>_F with L(v) = v_b a^T + b v_a^T linearised at global blocks g."""
    xb, xa, yb, ya = x[..., :d_out], x[..., d_out:], y[..., :d_out], y[..., d_out:]
    b, a = g[..., :d_out], g[..., d_out:]
    aa, bb = a.pow(2).sum(-1), b.pow(2).sum(-1)
    val = ((xb * yb).sum(-1) * aa + bb * (xa * ya).sum(-1)
           + (xb * b).sum(-1) * (a * ya).sum(-1) + (yb * b).sum(-1) * (a * xa).sum(-1))
    return float(val.sum())


def write_report(run_root: Path, plan: dict) -> dict:
    run_root = Path(run_root)
    crit = plan["criteria"]
    report = {"plan_id": plan["id"], "criteria": crit}
    if (run_root / "d0" / "d0_stats.pt").exists():
        report["d0"] = analyze_d0(run_root, crit)
    d1_files = sorted((run_root / "d1").glob("round_*.pt")) if (run_root / "d1").exists() else []
    report["d1"] = [analyze_d1_round(p) for p in d1_files]
    lines = [f"# {plan['id']} diagnostics", ""]
    if "d0" in report:
        d0 = report["d0"]
        lines += ["## D0 (logged training run)", "",
                  "| quantity | value | criterion | pass |", "|---|---:|---|---|",
                  f"| relative systematic owner–SCRA gap | {d0['relative_systematic_bias']:.4f} | ≥ {crit['min_relative_bias']} | {d0['checks']['F2_bias_large_enough']} |",
                  f"| adjacent 10-round window cosine of o | {_fmt(d0['adjacent_window_cosine'])} | ≥ {crit['min_window_cosine']} | {d0['checks']['F3_direction_persistent']} |",
                  f"| crossover rounds K* | {_fmt(d0['K_star_rounds'], 1)} | ≤ {crit['max_K_star']} | {d0['checks']['F3_K_star_small_enough']} |",
                  f"| CV² of weak dispersion within adapter | {d0['sigma_cv2_within_adapter']:.4f} | ≥ {crit['min_cv2_for_neyman']} (enable E1) | {d0['checks']['E1_neyman_worthwhile']} |",
                  f"| update norm late/early | {d0['update_norm_late_over_early']:.3f} | ≤ {crit['max_update_growth']} | {d0['checks']['F9_no_update_blowup']} |",
                  "", f"Descriptive: adjacent-round cosine {d0['adjacent_round_cosine_descriptive']:.3f} (noise-dominated), "
                  f"systematic energy ratio {d0['systematic_energy_ratio']:.4f}, CV² all blocks {d0['sigma_cv2_all_blocks']:.3f} "
                  f"(relative {d0['sigma_cv2_all_blocks_relative_to_update']:.3f}), shadow strong relative mode "
                  f"{d0['shadow_strong_mode_relative']:.3f}, shadow κ (last 50) {_fmt(d0['shadow_kappa_mean_last50'], 3)}, "
                  f"shadow a (last 50) {_fmt(d0['shadow_a_mean_last50'], 3)}, final macro accuracy {d0['final_macro_accuracy']:.4f}.", "",
                  f"**A-SCRA kappa.** The per-round EMA above is biased by self-correlation (critical review C1); use the "
                  f"window-averaged vector estimate: κ (last 3 windows) = {_fmt(d0['vector_kappa_last3_mean'], 3)}, implied a = "
                  f"{_fmt(d0['vector_a_last3'], 2)}; per window: " + ", ".join(f"{k:.3f}" for k in d0["vector_kappa_windows"]) + ".", ""]
        acc = d0["accuracy"]
        lines += [f"**Accuracy, rounds ≥ {acc['late_start_round']} (core weak / all clients, %).** Same seed and participant schedule as the sweep.", "",
                  "| run | core weak | all |", "|---|---:|---:|"]
        this_run = acc.get("this_run_label", "this run")
        labels = {"r8": "r8", "r2": "r2", "L0_prefix": "L0 prefix", "L2_main": "L2 (main D0)",
                  "vref_ref": "vref (weak train full, upload 2 of 8, Hajek)", "L2_d0": this_run}
        for name in ("r8", "r2", "L0_prefix", "L2_main", "vref_ref", "L2_d0"):
            if name in acc:
                lines.append(f"| {labels[name]} | {_fmt(acc[name]['core'], 2)} | {_fmt(acc[name]['all'], 2)} |")
        is_vref = "L2_main" in acc and "vref_ref" not in acc   # this run is the variance-matched reference itself
        if is_vref and "r8" in acc and acc["L2_main"]["core"] is not None:
            v, l2, r8 = acc["L2_d0"]["core"], acc["L2_main"]["core"], acc["r8"]["core"]
            lines += ["", f"Decomposition of the core-weak gap r8 − L2 = {r8 - l2:.2f} pp: fewer contributors (r8 − vref) "
                      f"= {r8 - v:.2f} pp; restricted training mode (+ small owner bias) (vref − L2) = {v - l2:.2f} pp "
                      "(1 seed; per-seed noise ≈ 0.2–0.4 pp)."]
        if acc.get("share_of_vref_minus_L2_recovered_core") is not None:
            lines += ["", f"This run recovers {100 * acc['share_of_vref_minus_L2_recovered_core']:.0f}% of the vref − L2 "
                      f"core-weak gap ({acc['vref_ref']['core'] - acc['L2_main']['core']:.2f} pp); 0% = L2, 100% = vref "
                      "(1 seed; per-seed noise ≈ 0.2–0.4 pp)."]
        if "L2_recovered_fraction_core" in acc:
            lines += ["", f"Share of the r8−r2 core-weak gap recovered: L0 prefix {_fmt(acc.get('L0_recovered_fraction_core'), 2)}, "
                      f"this run {_fmt(acc['L2_recovered_fraction_core'], 2)}; remaining gap to r8 "
                      f"{_fmt(acc['remaining_gap_core_pp'], 2)} pp (1 seed, descriptive)."]
        lines += [""]
    if report["d1"]:
        lines += ["## D1 (fixed states)", "",
                  "| round | ‖e_SCRA‖ | ‖e_owner‖ | blocks SCRA better | rel. mode (weak / strong) | κ weak (oracle) | κ from strong | a* oracle / from strong | cos(m_w, m_s) | ‖D_core‖ / ‖D_demoted‖ |",
                  "|---:|---:|---:|---:|---|---:|---:|---|---:|---|"]
        for d in report["d1"]:
            lines.append(f"| {d['snapshot_round']} | {d['norm_e_scra']:.4g} | {d['norm_e_owner']:.4g} | {d['blocks_where_scra_better']:.2f} | "
                         f"{d['relative_mode_weak']:.3f} / {d['relative_mode_strong']:.3f} | {d['kappa_weak_oracle']:.3f} | "
                         f"{d['kappa_strong_estimate']:.3f} | {d['a_star_oracle']:.2f} / {d['a_from_strong']:.2f} | "
                         f"{d['cos_mode_weak_vs_strong']:.3f} | {d['norm_D_core']:.4g} / {_fmt(d['norm_D_demoted'], 4)} |")
        if any(d["product_space"] for d in report["d1"]):
            lines += ["", "Product space (linearised ΔW = Δb aᵀ + b Δaᵀ at the snapshot):", "",
                      "| round | ‖e_SCRA‖ | ‖e_owner‖ | SCRA better | κ weak (oracle) | κ from strong | a* oracle / from strong |",
                      "|---:|---:|---:|---|---:|---:|---|"]
            for d in report["d1"]:
                p = d["product_space"]
                if p:
                    lines.append(f"| {d['snapshot_round']} | {p['norm_e_scra']:.4g} | {p['norm_e_owner']:.4g} | {p['scra_beats_owner']} | "
                                 f"{p['kappa_weak_oracle']:.3f} | {p['kappa_strong_estimate']:.3f} | {p['a_star_oracle']:.2f} / {p['a_from_strong']:.2f} |")
        lines += ["", "Checks per round: " + "; ".join(f"r{d['snapshot_round']}: " + ", ".join(f"{k}={v}" for k, v in d["checks"].items())
                                                    for d in report["d1"]), ""]
    lines += ["Thresholds and their meaning: docs/designs/algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md §7.1, §6."]
    out_dir = run_root / "diagnostics_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_f) + "\n")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return report


def _fmt(x, digits=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"
