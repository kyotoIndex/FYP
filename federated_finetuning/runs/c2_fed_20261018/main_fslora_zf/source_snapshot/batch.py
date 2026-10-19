"""Summary and pre-registered reading of the 2026-10-18 validation batch (README, "批量验证").

Reads every finished run under runs/scra_batch_20261018/<name>/d0/d0_rounds.json plus the existing
references: L2 / vref / vref-r1 at seed 2040 from the diagnostics runs, r8 / r2 / prefix (sf10, sf20) for
seeds 2040-2042 from the strong-fraction sweep. Missing runs are reported as missing, never guessed.

Metric: accuracy on core weak clients (not strong, domain not held by any strong client), averaged over the
evaluations at rounds >= 110 (5 evaluations). `core10` always uses the 10%-strong core set (30 clients), so
runs with a different strong fraction stay comparable with the sweep. `osc` = mean |change| of all-client
accuracy between consecutive evaluations from round 50 on (stability).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEEDS = [2040, 2041, 2042]
LATE, OSC_FROM = 110, 50


# ---------------------------------------------------------------- loading
def _curve(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return blob.get("eval") or blob.get("history")


def _identity(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def core_set(strong: list[int], domains: dict[str, str]) -> list[str]:
    held = {domains[str(c)] for c in strong}
    return sorted((c for c, d in domains.items() if int(c) not in strong and d not in held), key=int)


def metrics(curve: list[dict], core: list[str], core10: list[str]) -> dict:
    late = [e for e in curve if e["round"] >= LATE]
    mean_over = lambda e, cs: float(np.mean([e["clients"][c][0] for c in cs]))  # noqa: E731
    acc = [100 * e["client_macro_accuracy"] for e in curve if e["round"] >= OSC_FROM]
    return {"core": 100 * float(np.mean([mean_over(e, core) for e in late])),
            "core10": 100 * float(np.mean([mean_over(e, core10) for e in late])),
            "all": 100 * float(np.mean([e["client_macro_accuracy"] for e in late])),
            "osc": float(np.mean(np.abs(np.diff(acc)))) if len(acc) > 1 else None,
            "min_after_50": float(min(acc)) if acc else None,
            "final": 100 * curve[-1]["client_macro_accuracy"], "last_round": curve[-1]["round"]}


def collect(runs_dir: Path) -> dict:
    """name -> metrics for batch runs and references that exist."""
    batch = runs_dir / "scra_batch_20261018"
    diag = runs_dir / "scra_diagnostics_20261017_v1"
    sweep = runs_dir / "headroom_strongfrac_20261017_v1"
    idents = {p.parent.parent.name: _identity(p) for p in sorted(batch.glob("*/d0/identity.json"))}
    # the 10%-strong reference fixes core10 (vref first: the main L2 run predates client_domains)
    candidates = [_identity(runs_dir / "scra_diagnostics_20261017_v1_vref" / "d0" / "identity.json"),
                  _identity(diag / "d0" / "identity.json"), *idents.values()]
    ref_ident = next((i for i in candidates if i and "client_domains" in i and i["strong_clients"]), None)
    if ref_ident is None:
        raise FileNotFoundError("need one identity.json with client_domains (vref reference or any batch run)")
    domains = ref_ident["client_domains"]
    core10 = core_set(ref_ident["strong_clients"], domains)
    out = {}

    def add(name, curve_path, strong):
        curve = _curve(curve_path)
        if curve is None or not any(e["round"] >= LATE for e in curve):
            return
        rec = metrics(curve, core_set(strong, domains), core10)
        blob = json.loads(curve_path.read_text())
        rec["peak_memory_mib"] = (blob.get("peak_memory_mib") or (blob.get("cost") or {}).get("peak_memory_mib"))
        rec["elapsed_seconds"] = blob.get("elapsed_seconds")
        out[name] = rec

    for name, ident in idents.items():
        add(name, batch / name / "d0" / "d0_rounds.json", ident["strong_clients"])
    s10 = ref_ident["strong_clients"]
    add("L2_s2040", diag / "d0" / "d0_rounds.json", s10)
    add("vref_s2040", runs_dir / "scra_diagnostics_20261017_v1_vref" / "d0" / "d0_rounds.json", s10)
    add("vref_r1_s2040", runs_dir / "scra_diagnostics_20261017_v1_budget8" / "d0" / "d0_rounds.json", s10)
    for s in SEEDS:
        add(f"r8_s{s}", sweep / f"fedavg_r8_seed{s}" / "result.json", s10)
        add(f"r2_s{s}", sweep / f"fedavg_r2_seed{s}" / "result.json", s10)
        add(f"prefix_s{s}", sweep / f"hetero_r8r2_sf10_seed{s}" / "result.json", s10)
        p20 = sweep / f"hetero_r8r2_sf20_seed{s}" / "result.json"
        if p20.exists():
            add(f"prefix_sf20_s{s}", p20, json.loads(p20.read_text())["strong_clients"])
    return {"runs": out, "core10_size": len(core10)}


# ---------------------------------------------------------------- reading
def _get(runs, name, key="core"):
    return runs[name][key] if name in runs and runs[name].get(key) is not None else None


def closure(runs, rule, lo, hi, key="core", min_gap=0.5):
    """Share of the lo -> hi gap closed by rule; None if a run is missing or the gap is below min_gap pp."""
    a, b, x = _get(runs, lo, key), _get(runs, hi, key), _get(runs, rule, key)
    if None in (a, b, x) or b - a < min_gap:
        return None
    return (x - a) / (b - a)


def verdicts(runs: dict) -> dict:
    def seeds(prefix):
        vals = [_get(runs, f"{prefix}_s{s}") for s in SEEDS]
        return vals if None not in vals else None

    amp, l2, vref = seeds("amp"), seeds("L2"), seeds("vref")
    v = {}
    if amp and l2 and vref:
        v["M1_amp_matches_vref_and_beats_L2"] = bool(np.mean(amp) >= np.mean(vref) - 0.3
                                                     and np.mean(amp) >= np.mean(l2) + 1.0
                                                     and all(a > b for a, b in zip(amp, l2)))
        osc_a = [_get(runs, f"amp_s{s}", "osc") for s in SEEDS]
        osc_v = [_get(runs, f"vref_s{s}", "osc") for s in SEEDS]
        v["M2_amp_stable"] = bool(np.mean(osc_a) <= 1.5 * np.mean(osc_v))
    a0 = _get(runs, "amp_s2040")

    def worse(name):
        x = _get(runs, name)
        return None if None in (a0, x) else bool(x <= a0 - 0.3)

    v["A1_hajek_needed"] = worse("amp_owner_s2040")
    v["A2_balanced_allocation_needed"] = worse("amp_prefix_s2040")
    alts = [_get(runs, n) for n in ("amp_a2_s2040", "amp_a8_s2040")]
    v["A3_alpha_R_over_r_ok"] = None if a0 is None or None in alts else bool(a0 >= max(alts) - 0.3)
    for key, (rule, lo, hi) in {"R1_rank1": ("amp_r1_s2040", "L2_r1_s2040", "vref_r1_s2040"),
                                "R2_hetero_ranks": ("amp_het_s2040", "L2_het_s2040", "vref_het_s2040"),
                                "R3_strong20": ("amp_sf20_s2040", "L2_sf20_s2040", "vref_sf20_s2040")}.items():
        c = closure(runs, rule, lo, hi)
        v[key] = None if c is None else bool(c >= 0.7)
        v[key + "_closure"] = None if c is None else float(c)
    return v


def write_batch_report(runs_dir: Path) -> dict:
    data = collect(runs_dir)
    runs = data["runs"]
    v = verdicts(runs)
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = ["# Validation batch 2026-10-18", "",
             f"Core weak accuracy (%) averaged over rounds ≥ {LATE}; `core10` = the fixed {data['core10_size']}-client "
             f"core set of the 10%-strong setting; `osc` = mean |Δ all-client accuracy| between evaluations from "
             f"round {OSC_FROM}. 1 seed noise ≈ 0.2–0.4 pp.", "",
             "## Main comparison (10% strong, R=8, r=2)", "",
             "| method | " + " | ".join(f"core s{s}" for s in SEEDS) + " | core mean ± sd | all mean | osc mean |",
             "|---|" + "---:|" * (len(SEEDS) + 3)]
    for label, prefix in [("r8 (upper)", "r8"), ("r2", "r2"), ("prefix hetero (L0)", "prefix"),
                          ("L2 restricted, owner", "L2"), ("vref (full train, upload r)", "vref"),
                          ("**amp (amplified restricted, Hájek)**", "amp")]:
        vals = [_get(runs, f"{prefix}_s{s}") for s in SEEDS]
        ok = [x for x in vals if x is not None]
        alls = [_get(runs, f"{prefix}_s{s}", "all") for s in SEEDS]
        oscs = [_get(runs, f"{prefix}_s{s}", "osc") for s in SEEDS]
        mean_sd = f"{np.mean(ok):.2f} ± {np.std(ok):.2f}" if ok else "—"
        lines.append(f"| {label} | " + " | ".join(f(x) for x in vals) + f" | {mean_sd} | "
                     + f(np.mean([x for x in alls if x is not None]) if any(x is not None for x in alls) else None)
                     + " | " + f(np.mean([x for x in oscs if x is not None]) if any(x is not None for x in oscs) else None)
                     + " |")
    lines += ["", "## Ablations (seed 2040)", "", "| run | core | Δ vs amp | closure L2→vref | osc |", "|---|---:|---:|---:|---:|"]
    a0 = _get(runs, "amp_s2040")
    for name in ["amp_s2040", "amp_owner_s2040", "amp_prefix_s2040", "amp_a2_s2040", "amp_a8_s2040"]:
        x = _get(runs, name)
        lines.append(f"| {name} | {f(x)} | {f(None if None in (x, a0) else x - a0)} | "
                     f"{f(closure(runs, name, 'L2_s2040', 'vref_s2040'))} | {f(_get(runs, name, 'osc'))} |")
    lines += ["", "## Robustness (seed 2040)", "",
              "| setting | L2 | vref | amp | closure | core10 L2 / vref / amp | osc amp |", "|---|---:|---:|---:|---:|---|---:|"]
    for label, (amp_n, lo, hi) in {"r = 1 (upload 1/8)": ("amp_r1_s2040", "L2_r1_s2040", "vref_r1_s2040"),
                                   "ranks {1,2,4}": ("amp_het_s2040", "L2_het_s2040", "vref_het_s2040"),
                                   "20% strong": ("amp_sf20_s2040", "L2_sf20_s2040", "vref_sf20_s2040")}.items():
        lines.append(f"| {label} | {f(_get(runs, lo))} | {f(_get(runs, hi))} | {f(_get(runs, amp_n))} | "
                     f"{f(closure(runs, amp_n, lo, hi))} | {f(_get(runs, lo, 'core10'))} / "
                     f"{f(_get(runs, hi, 'core10'))} / {f(_get(runs, amp_n, 'core10'))} | {f(_get(runs, amp_n, 'osc'))} |")
    lines += ["", "## Cost per run", "", "| run | peak memory MiB | seconds |", "|---|---:|---:|"]
    for name in sorted(runs):
        if runs[name].get("elapsed_seconds") is not None:
            lines.append(f"| {name} | {f(runs[name]['peak_memory_mib'], 0)} | {f(runs[name]['elapsed_seconds'], 0)} |")
    lines += ["", "## Pre-registered reading", "", "| question | criterion | result |", "|---|---|---|",
              f"| M1 amp matches vref and beats L2 | 3-seed mean ≥ vref − 0.3, ≥ L2 + 1.0, amp > L2 in every seed | "
              f"{v.get('M1_amp_matches_vref_and_beats_L2', '—')} |",
              f"| M2 amp is stable | mean osc ≤ 1.5 × vref | {v.get('M2_amp_stable', '—')} |",
              f"| A1 Hájek needed | amp_owner ≤ amp − 0.3 | {v['A1_hajek_needed']} |",
              f"| A2 balanced allocation needed | amp_prefix ≤ amp − 0.3 | {v['A2_balanced_allocation_needed']} |",
              f"| A3 α = R/r is right | amp ≥ max(α=2, α=8) − 0.3 | {v['A3_alpha_R_over_r_ok']} |",
              f"| R1 rank 1 | closure ≥ 0.7 (gap ≥ 0.5 pp) | {v['R1_rank1']} |",
              f"| R2 heterogeneous ranks | closure ≥ 0.7 | {v['R2_hetero_ranks']} |",
              f"| R3 20% strong | closure ≥ 0.7 | {v['R3_strong20']} |", "",
              "Reading: M1 and M2 true → amplified restricted training is the main method; next are baselines "
              "(FLASC, FSLoRA, Fed-PLoRA), literature check and scale. A1/A2 false → that component is not needed "
              "and is dropped from the method (single seed, so only a gap ≥ 0.3 pp counts). M1 false while D3 "
              "passed → the one-step view missed a multi-round effect; probe amp states with D3 before changing "
              "the method. A closure of None means a run is missing or the setting has < 0.5 pp headroom."]
    out_dir = runs_dir / "scra_batch_20261018"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"runs": runs, "verdicts": v}, indent=2, default=float) + "\n")
    return {"verdicts": v, "n_runs": len(runs)}
