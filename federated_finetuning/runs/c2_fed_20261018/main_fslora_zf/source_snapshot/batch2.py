"""Summary and pre-registered reading of batch 2 (positioning controls, roadmap_2026-10-18 stage A).

Batch-2 runs live under runs/scra_batch2_20261018/<method>_s<seed>/d0; the references (amp, vref, L2, r8)
come from batch 1, the diagnostics runs and the strong-fraction sweep (see scra.batch). Same metric as
batch 1 (core weak accuracy, rounds >= 110), plus window means, because FSLoRA's forward equals amp's at
cold start (B = 0) in the Adam eps -> 0 limit, so the two can only separate later in training. Comparisons
are paired per seed and split into mechanism controls (one factor differs) and external positioning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .batch import SEEDS, _curve, _identity, core_set, metrics

WINDOWS = {"early": (10, 50), "mid": (60, 100), "late": (110, 150)}
METHODS = ["r8", "vref", "amp", "fslora_st", "fslora_hajek", "fslora_zerofill", "lrcomp", "fslora_paper",
           "fedplora", "flasc", "L2"]
LABELS = {"r8": "r8 (upper)", "vref": "vref (full train, upload r/R, Hájek)",
          "amp": "**amp (ours)**",
          "fslora_st": "A1a FSLoRA forward, amp gradient path, balanced + Hájek",
          "fslora_hajek": "A1b FSLoRA forward, true gradient, balanced + Hájek",
          "fslora_zerofill": "A1c FSLoRA forward, true gradient, balanced + zero-fill",
          "lrcomp": "A2 LoRA lr × R/r + server ÷ R/r",
          "fslora_paper": "ext. FSLoRA-style (our AdamW protocol; random, zero-fill)",
          "fedplora": "ext. Fed-PLoRA style (random, owner)",
          "flasc": "ext. FLASC-style (full train, global top-r/R upload by weak, zero-fill)",
          "L2": "L2 (balanced, owner)"}
T975_DF2 = 4.303   # two-sided 95% t quantile, 2 degrees of freedom (3 paired seeds)
# (key, a, b, kind, criterion): diff = a - b per seed. "dir" needs mean >= thr and all seeds > 0;
# "two_sided" needs |mean| >= thr and all seeds of the same sign; "equiv" needs |mean| < thr;
# "noninf" needs mean >= thr.
COMPARISONS = [
    ("A1a_state_sketch", "amp", "fslora_st", "mechanism", ("dir", 0.3)),
    ("A1b_gradient_path", "fslora_st", "fslora_hajek", "mechanism", ("two_sided", 0.3)),
    ("A1c_aggregation", "fslora_hajek", "fslora_zerofill", "mechanism", ("dir", 0.3)),
    ("A1_total_vs_fslora_forward", "amp", "fslora_hajek", "mechanism", ("dir", 0.3)),
    ("A2_lr_compensation", "amp", "lrcomp", "mechanism", ("equiv", 0.3)),
    ("E1_vs_fslora_style", "amp", "fslora_paper", "external", ("dir", 0.3)),
    ("E2_vs_fedplora_style", "amp", "fedplora", "external", ("dir", 1.0)),
    ("E3_vs_flasc_style", "amp", "flasc", "external", ("noninf", -0.3)),
]


def _paths(runs_dir: Path, method: str, seed: int) -> Path:
    b1, b2 = runs_dir / "scra_batch_20261018", runs_dir / "scra_batch2_20261018"
    diag = runs_dir / "scra_diagnostics_20261017_v1"
    if method == "r8":
        return runs_dir / "headroom_strongfrac_20261017_v1" / f"fedavg_r8_seed{seed}" / "result.json"
    if method in ("vref", "L2") and seed == 2040:
        return (runs_dir / "scra_diagnostics_20261017_v1_vref" if method == "vref" else diag) / "d0" / "d0_rounds.json"
    root = b1 if method in ("vref", "L2", "amp") else b2
    return root / f"{method}_s{seed}" / "d0" / "d0_rounds.json"


def window_means(curve: list[dict], core: list[str]) -> dict:
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        vals = [100 * np.mean([e["clients"][c][0] for c in core]) for e in curve if lo <= e["round"] <= hi]
        out[name] = float(np.mean(vals)) if vals else None
    return out


def collect2(runs_dir: Path) -> dict:
    ident = _identity(runs_dir / "scra_diagnostics_20261017_v1_vref" / "d0" / "identity.json")
    if not ident or "client_domains" not in ident:
        raise FileNotFoundError("need the vref reference identity (client_domains)")
    core = core_set(ident["strong_clients"], ident["client_domains"])
    runs = {}
    for m in METHODS:
        for s in SEEDS:
            path = _paths(runs_dir, m, s)
            curve = _curve(path)
            if curve is None or curve[-1]["round"] < WINDOWS["late"][1]:
                continue
            runs[f"{m}_s{s}"] = {**metrics(curve, core, core), "windows": window_means(curve, core)}
    return runs


def _seed_vals(runs, m, key="core", window=None):
    vals = []
    for s in SEEDS:
        rec = runs.get(f"{m}_s{s}")
        if rec is None:
            return None
        vals.append(rec["windows"][window] if window else rec[key])
    return vals


def paired(runs: dict, a: str, b: str, window: str | None = None) -> dict | None:
    """Per-seed a - b (same seed = same participant schedule and initialisation), mean, sd, 95% t interval."""
    xa, xb = _seed_vals(runs, a, window=window), _seed_vals(runs, b, window=window)
    if xa is None or xb is None:
        return None
    d = np.array(xa) - np.array(xb)
    sd = float(d.std(ddof=1))
    half = T975_DF2 * sd / np.sqrt(len(d))
    return {"per_seed": [float(x) for x in d], "mean": float(d.mean()), "sd": sd,
            "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}


def judge(p: dict | None, criterion) -> bool | None:
    if p is None:
        return None
    kind, thr = criterion
    n = len(p["per_seed"])
    if kind == "dir":
        return bool(p["mean"] >= thr and p["n_pos"] == n)
    if kind == "two_sided":
        return bool(abs(p["mean"]) >= thr and max(p["n_pos"], p["n_neg"]) == n)
    if kind == "equiv":
        return bool(abs(p["mean"]) < thr)
    return bool(p["mean"] >= thr)                       # noninf


def verdicts2(runs: dict) -> dict:
    out = {}
    for key, a, b, kind, crit in COMPARISONS:
        p = paired(runs, a, b)
        out[key] = {"a": a, "b": b, "kind": kind, "criterion": list(crit), "paired": p, "result": judge(p, crit)}
    return out


def write_batch2_report(runs_dir: Path) -> dict:
    runs = collect2(runs_dir)
    v = verdicts2(runs)
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    crit_text = {"dir": "mean ≥ {t} pp and every seed > 0", "two_sided": "|mean| ≥ {t} pp, all seeds same sign",
                 "equiv": "|mean| < {t} pp", "noninf": "mean ≥ {t} pp"}
    lines = ["# Batch 2: positioning controls (2026-10-18)", "",
             "Core weak accuracy (%); `late` = rounds 110–150 (the batch-1 metric), `early` = 10–50, `mid` = 60–100. "
             "Seeds share participant schedule and initialisation, so differences are paired per seed. With 3 seeds "
             "the 95% t interval is wide (t = 4.30); verdicts use the mean threshold plus sign agreement. "
             "FSLoRA's forward equals amp's while the adapter is zero only in the Adam eps → 0 limit "
             "(FIX_BATCH2_CONTROLS_2026-10-18 §2); A1a removes that confound.", "",
             "| method | " + " | ".join(f"late s{s}" for s in SEEDS) + " | late mean ± sd | early | mid | osc |",
             "|---|" + "---:|" * (len(SEEDS) + 4)]
    for m in METHODS:
        vals = [runs.get(f"{m}_s{s}", {}).get("core") for s in SEEDS]
        ok = [x for x in vals if x is not None]
        win = {w: [runs[f"{m}_s{s}"]["windows"][w] for s in SEEDS if f"{m}_s{s}" in runs] for w in WINDOWS}
        osc = [runs[f"{m}_s{s}"]["osc"] for s in SEEDS if f"{m}_s{s}" in runs]
        lines.append(f"| {LABELS[m]} | " + " | ".join(f(x) for x in vals) + " | "
                     + (f"{np.mean(ok):.2f} ± {np.std(ok, ddof=1) if len(ok) > 1 else 0:.2f}" if ok else "—") + " | "
                     + " | ".join(f(np.mean(win[w])) if win[w] else "—" for w in ("early", "mid")) + " | "
                     + (f(np.mean(osc)) if osc else "—") + " |")
    for kind, title in (("mechanism", "Mechanism controls (one factor differs per pair)"),
                        ("external", "External positioning (several factors differ; not mechanism evidence)")):
        lines += ["", f"## {title}", "",
                  "| comparison | a − b | per seed (late) | mean ± sd | 95% CI | early | mid | criterion | result |",
                  "|---|---|---|---:|---|---:|---:|---|---|"]
        for key, a, b, k, crit in COMPARISONS:
            if k != kind:
                continue
            p = v[key]["paired"]
            wins = [paired(runs, a, b, w) for w in ("early", "mid")]
            lines.append(
                f"| {key} | {a} − {b} | "
                + ("—" if p is None else ", ".join(f"{x:+.2f}" for x in p["per_seed"])) + " | "
                + ("—" if p is None else f"{p['mean']:+.2f} ± {p['sd']:.2f}") + " | "
                + ("—" if p is None else f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]") + " | "
                + " | ".join("—" if w is None else f"{w['mean']:+.2f}" for w in wins) + " | "
                + crit_text[crit[0]].format(t=crit[1]) + f" | {v[key]['result']} |")
    lines += ["", "## Reading (pre-registered)", "",
              "- A1a true → amplifying only the increment (not sketching the state) has value beyond FSLoRA's forward; "
              "A1a false → narrow the claim to a new explanation of FSLoRA-style amplification (Adam progress matching, "
              "head / component overshoot) plus the diagnostics.",
              "- A1b true → FSLoRA's true-gradient implementation itself changes the outcome under AdamW (gradient scale "
              "/ eps); report it as an implementation effect, not as the state-sketch effect.",
              "- A1c true → Hájek helps FSLoRA-style training too.",
              "- A2 true → present the method as matching local progress to full-rank training, forward amplification "
              "being one implementation; A2 false → the forward form matters, report which side wins.",
              "- E1–E3 position the method against published-method styles under our unified protocol only.",
              "- A result of None means a run is missing."]
    out_dir = runs_dir / "scra_batch2_20261018"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"runs": runs, "verdicts": v}, indent=2) + "\n")
    return {"verdicts": {k: x["result"] for k, x in v.items()}, "n_runs": len(runs)}
