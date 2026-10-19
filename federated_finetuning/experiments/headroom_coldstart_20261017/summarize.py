"""Collect completed cells, compute paired per-seed gaps and apply the pre-registered rule.

    python summarize.py --plan plan.json [--project-root ...]

Writes SUMMARY.md and summary.json into the run root. Missing cells are listed, never imputed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_plan  # noqa: E402

THRESHOLD_PP = 1.0  # pre-registered in README before any result was read


def get(result: dict | None, *path):
    value = result
    for key in path:
        if value is None:
            return None
        value = value.get(key)
    return value


def pct(x):
    return None if x is None else 100.0 * x


def fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def paired(results, seeds, left, right, *path):
    """Per-seed difference left-right in percentage points (accuracy paths only)."""
    out = {}
    for s in seeds:
        a, b = get(results.get((left, s)), *path), get(results.get((right, s)), *path)
        out[s] = None if a is None or b is None else pct(a) - pct(b)
    values = [v for v in out.values() if v is not None]
    return {"per_seed": out, "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None, "complete": len(values) == len(seeds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--project-root")
    args = parser.parse_args()
    plan = load_plan(args.plan, args.project_root)
    root = Path(plan["run_root"])
    results, missing = {}, []
    for seed in plan["seeds"]:
        for method in plan["methods"]:
            path = root / f"{method}_seed{seed}" / "result.json"
            if path.exists():
                results[(method, seed)] = json.loads(path.read_text())
            else:
                missing.append(f"{method}_seed{seed}")
    seeds = plan["seeds"]
    acc = ("final", "client_macro_accuracy")
    weak_acc = ("final", "by_role", "weak", "client_macro_accuracy")
    contrasts = {
        "capacity_r8_minus_r2": paired(results, seeds, "fedavg_r8", "fedavg_r2", *acc),
        "hetero_cost_r8_minus_hetero": paired(results, seeds, "fedavg_r8", "hetero_r8r2", *acc),
        "hetero_weak_cost_r8_minus_hetero": paired(results, seeds, "fedavg_r8", "hetero_r8r2", *weak_acc),
        "hetero_gain_over_all_r2": paired(results, seeds, "hetero_r8r2", "fedavg_r2", *acc),
        "federation_hetero_minus_local": paired(results, seeds, "hetero_r8r2", "local_only", *acc),
        "federation_weak_hetero_minus_local": paired(results, seeds, "hetero_r8r2", "local_only", *weak_acc),
    }
    # FedAvg+FT reference against local-only (personalised vs personalised).
    ft = {}
    for s in seeds:
        a = get(results.get(("hetero_r8r2", s)), "personalized", "client_macro_accuracy")
        b = get(results.get(("local_only", s)), "final", "client_macro_accuracy")
        ft[s] = None if a is None or b is None else pct(a) - pct(b)
    vals = [v for v in ft.values() if v is not None]
    contrasts["federation_hetero_ft_minus_local"] = {"per_seed": ft, "mean": statistics.mean(vals) if vals else None,
                                                     "min": min(vals) if vals else None,
                                                     "complete": len(vals) == len(seeds)}

    def rule(name):
        c = contrasts[name]
        if not c["complete"]:
            return "incomplete"
        return "yes" if c["min"] >= THRESHOLD_PP else "no"

    decision = {
        "threshold_pp_every_seed": THRESHOLD_PP,
        "rank_capacity_headroom": rule("capacity_r8_minus_r2"),
        "weak_clients_lose_under_heterogeneity": rule("hetero_weak_cost_r8_minus_hetero"),
        "federation_beats_local_only": rule("federation_hetero_minus_local"),
        "federation_beats_local_after_personalization": rule("federation_hetero_ft_minus_local"),
    }
    summary = {"plan_id": plan["id"], "missing": missing, "contrasts": contrasts, "decision": decision,
               "cells": {f"{m}_seed{s}": {
                   "final_macro_accuracy_pct": pct(get(r, *acc)),
                   "final_macro_nll": get(r, "final", "client_macro_nll"),
                   "weak_macro_accuracy_pct": pct(get(r, *weak_acc)),
                   "strong_macro_accuracy_pct": pct(get(r, "final", "by_role", "strong", "client_macro_accuracy")),
                   "personalized_macro_accuracy_pct": pct(get(r, "personalized", "client_macro_accuracy")),
                   "elapsed_minutes": (get(r, "cost", "elapsed_seconds") or 0) / 60,
                   "upload_params_total": get(r, "cost", "upload_params_total"),
                   "curve": [(h["round"], pct(h["client_macro_accuracy"])) for h in r["history"]],
               } for (m, s), r in sorted(results.items())}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [f"# {plan['id']} summary", "",
             f"Completed cells: {len(results)}/{len(seeds) * len(plan['methods'])}."
             + (f" Missing: {', '.join(missing)}." if missing else ""), "",
             "## Final round, client-macro accuracy (%)", "",
             "| method | " + " | ".join(f"seed {s}" for s in seeds) + " | weak (mean) | strong (mean) | +FT (mean) | minutes (mean) |",
             "|---|" + "---:|" * (len(seeds) + 4)]
    for m in plan["methods"]:
        cells = [summary["cells"].get(f"{m}_seed{s}") for s in seeds]
        done = [c for c in cells if c]

        def mean(key):
            v = [c[key] for c in done if c[key] is not None]
            return statistics.mean(v) if v else None
        lines.append(f"| {m} | " + " | ".join(fmt(c and c["final_macro_accuracy_pct"]) for c in cells)
                     + f" | {fmt(mean('weak_macro_accuracy_pct'))} | {fmt(mean('strong_macro_accuracy_pct'))}"
                     + f" | {fmt(mean('personalized_macro_accuracy_pct'))} | {fmt(mean('elapsed_minutes'), 1)} |")
    lines += ["", "## Paired gaps (percentage points; per seed, mean)", "",
              "| contrast | " + " | ".join(f"seed {s}" for s in seeds) + " | mean |", "|---|" + "---:|" * (len(seeds) + 1)]
    for name, c in contrasts.items():
        lines.append(f"| {name} | " + " | ".join(fmt(c["per_seed"][s]) for s in seeds) + f" | {fmt(c['mean'])} |")
    lines += ["", f"## Pre-registered decision (gap ≥ {THRESHOLD_PP} pp in every seed)", ""]
    lines += [f"- {k}: **{v}**" for k, v in decision.items() if k != "threshold_pp_every_seed"]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
