"""Strong-fraction sweep summary: how much of the rank-8 vs rank-2 gap naive heterogeneity loses.

    python summarize_strongfrac.py --plan plan_strongfrac.json [--project-root ...]

fedavg_r8 / fedavg_r2 are role-independent, so their per-client curves are re-sliced onto
any client set. Because weak sets change composition with f (smaller f demotes some
government/travel clients that share domains with the remaining strong ones), the primary
comparison uses a FIXED "core weak" set: clients weak at every fraction (never strong in any
cell). The demoted clients and all weak clients at f are reported separately.
Primary metric (pre-registered before the sweep ran): core-weak client-macro accuracy
averaged over evaluations at rounds >= late_start_round. Final round is secondary.
Also reports how many strong clients the (method-independent) participant schedule draws
per round, since the tail components 2..R-1 average only over sampled strong clients.
Writes SUMMARY_STRONGFRAC.md and summary_strongfrac.json into the run root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom.fed import participants  # noqa: E402
from train import cell_name, load_plan  # noqa: E402


def macro(history_point: dict, clients: list[int]) -> float:
    values = history_point["clients"]
    return 100.0 * statistics.mean(values[str(c)][0] for c in clients)


def late(result: dict, clients: list[int], start: int) -> float:
    points = [h for h in result["history"] if h["round"] >= start]
    return statistics.mean(macro(h, clients) for h in points)


def final(result: dict, clients: list[int]) -> float:
    return macro(result["history"][-1], clients)


def rounds_to(result: dict, clients: list[int], target: float):
    return next((h["round"] for h in result["history"] if macro(h, clients) >= target), None)


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values), "n": len(values)}


def fmt(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan_strongfrac.json"))
    parser.add_argument("--project-root")
    args = parser.parse_args()
    plan = load_plan(args.plan, args.project_root)
    sweep = plan["sweep"]
    root = Path(plan["run_root"])
    seeds = sorted({c["seed"] for c in plan["cells"]})
    results, missing = {}, []
    for cell in plan["cells"]:
        path = root / cell_name(cell) / "result.json"
        key = (cell["method"], cell.get("strong_fraction"), cell["seed"])
        if path.exists():
            results[key] = json.loads(path.read_text())
        else:
            missing.append(cell_name(cell))

    start, threshold = sweep["late_start_round"], sweep["threshold_pp_every_seed"]
    strong_sets = {}
    for (method, f, _), res in results.items():
        if method == "hetero_r8r2":
            strong_sets.setdefault(f, set(res["strong_clients"]))
            if set(res["strong_clients"]) != strong_sets[f]:
                raise ValueError(f"Strong set differs across seeds at fraction {f}")
    ever_strong = set().union(*strong_sets.values()) if strong_sets else set()
    rows, decision = {}, {}
    for f in sweep["fractions"]:
        per_seed = {}
        for s in seeds:
            r8, r2 = results.get(("fedavg_r8", None, s)), results.get(("fedavg_r2", None, s))
            het = results.get(("hetero_r8r2", f, s))
            if not (r8 and r2 and het):
                per_seed[s] = None
                continue
            everyone = sorted(int(c) for c in het["client_ranks"])
            strong = het["strong_clients"]
            weak = [c for c in everyone if c not in set(strong)]
            core = [c for c in everyone if c not in ever_strong]
            demoted = [c for c in weak if c in ever_strong]
            m = {}
            for name, res in (("r8", r8), ("r2", r2), ("het", het)):
                m[name] = {"late_all": late(res, everyone, start), "late_weak": late(res, weak, start),
                           "late_core": late(res, core, start), "final_core": final(res, core),
                           "late_demoted": late(res, demoted, start) if demoted else None,
                           "late_strong": late(res, strong, start),
                           **{f"rounds_to_{t}": rounds_to(res, everyone, t) for t in sweep["targets"]}}
            span = m["r8"]["late_core"] - m["r2"]["late_core"]
            per_seed[s] = {
                "strong_clients": len(strong), "weak_clients": len(weak),
                "core_weak_clients": len(core), "demoted_clients": len(demoted), "models": m,
                "core_gap_r8_minus_het_late": m["r8"]["late_core"] - m["het"]["late_core"],
                "core_gap_r8_minus_het_final": m["r8"]["final_core"] - m["het"]["final_core"],
                "weak_gap_r8_minus_het_late": m["r8"]["late_weak"] - m["het"]["late_weak"],
                "demoted_gap_r8_minus_het_late": None if not demoted
                else m["r8"]["late_demoted"] - m["het"]["late_demoted"],
                "all_gap_r8_minus_het_late": m["r8"]["late_all"] - m["het"]["late_all"],
                "core_capacity_r8_minus_r2_late": span,
                "core_recovered_fraction": None if abs(span) < 1e-9
                else (m["het"]["late_core"] - m["r2"]["late_core"]) / span,
            }
        complete = [v for v in per_seed.values() if v is not None]
        rows[str(f)] = {
            "per_seed": {str(s): v for s, v in per_seed.items()},
            "core_gap_late": describe([v["core_gap_r8_minus_het_late"] for v in complete]),
            "core_gap_final": describe([v["core_gap_r8_minus_het_final"] for v in complete]),
            "weak_gap_late": describe([v["weak_gap_r8_minus_het_late"] for v in complete]),
            "demoted_gap_late": describe([v["demoted_gap_r8_minus_het_late"] for v in complete]),
            "all_gap_late": describe([v["all_gap_r8_minus_het_late"] for v in complete]),
            "core_recovered_fraction": describe([v["core_recovered_fraction"] for v in complete]),
            "het_rounds_to": {t: describe([v["models"]["het"][f"rounds_to_{t}"] for v in complete])
                              for t in sweep["targets"]},
        }
        if len(complete) < len(seeds):
            decision[str(f)] = "incomplete"
        else:
            decision[str(f)] = "yes" if rows[str(f)]["core_gap_late"]["min"] >= threshold else "no"

    # Strong participants per round under the shared participant schedule.
    participation = {}
    clients_all = list(range(len(next(iter(results.values()))["client_ranks"]))) if results else []
    k, rounds = plan["federation"]["clients_per_round"], plan["federation"]["rounds"]
    for f, strong in sorted(strong_sets.items()):
        counts = [sum(c in strong for c in participants(s, r, clients_all, k))
                  for s in seeds for r in range(1, rounds + 1)]
        participation[str(f)] = {"share_rounds_with_0_strong": counts.count(0) / len(counts),
                                 "share_rounds_with_1_strong": counts.count(1) / len(counts),
                                 "mean_strong_per_round": statistics.mean(counts)}

    capacity = {}
    for s in seeds:
        r8, r2 = results.get(("fedavg_r8", None, s)), results.get(("fedavg_r2", None, s))
        if r8 and r2:
            everyone = sorted(int(c) for c in r8["client_ranks"])
            capacity[str(s)] = {
                "late_r8": late(r8, everyone, start), "late_r2": late(r2, everyone, start),
                "gap_late": late(r8, everyone, start) - late(r2, everyone, start),
                **{f"rounds_to_{t}": {"r8": rounds_to(r8, everyone, t), "r2": rounds_to(r2, everyone, t)}
                   for t in sweep["targets"]}}

    summary = {"plan_id": plan["id"], "missing": missing, "late_start_round": start,
               "threshold_pp_every_seed": threshold, "capacity": capacity,
               "core_weak_clients": sorted(set(range(len(clients_all))) - ever_strong),
               "by_fraction": rows, "strong_participation": participation,
               "decision_core_weak_headroom": decision}
    (root / "summary_strongfrac.json").write_text(json.dumps(summary, indent=2) + "\n")

    n_cells = len(plan["cells"])
    lines = [f"# {plan['id']} summary", "",
             f"Completed cells: {n_cells - len(missing)}/{n_cells}." + (f" Missing: {', '.join(missing)}." if missing else ""),
             f"Primary metric: client-macro accuracy (%) of the fixed core-weak set "
             f"({len(summary['core_weak_clients'])} clients never strong in any cell), averaged over evaluations at rounds ≥ {start}.", "",
             "## Capacity reference (all clients)", "",
             "| seed | r8 late | r2 late | r8 − r2 | rounds to 80% (r8 / r2) |", "|---|---:|---:|---:|---:|"]
    for s, c in capacity.items():
        r80 = c.get("rounds_to_80", {})
        lines.append(f"| {s} | {fmt(c['late_r8'])} | {fmt(c['late_r2'])} | {fmt(c['gap_late'])} | "
                     f"{r80.get('r8') or '—'} / {r80.get('r2') or '—'} |")
    lines += ["", "## Core-weak gap by strong fraction (primary)", "",
              "| strong fraction | r8 − hetero, core weak, late (per seed) | mean | min | final round (mean) | "
              "share of r8−r2 gap recovered (mean) | decision |",
              "|---|---|---:|---:|---:|---:|---|"]
    for f in sweep["fractions"]:
        row = rows[str(f)]
        seeds_txt = ", ".join(fmt(v["core_gap_r8_minus_het_late"]) if v else "—" for v in row["per_seed"].values())
        lines.append(f"| {f:.0%} | {seeds_txt} | {fmt(row['core_gap_late']['mean'])} | "
                     f"{fmt(row['core_gap_late']['min'])} | {fmt(row['core_gap_final']['mean'])} | "
                     f"{fmt(row['core_recovered_fraction']['mean'])} | **{decision[str(f)]}** |")
    lines += ["", "## Secondary groups (r8 − hetero, late, mean over seeds)", "",
              "| strong fraction | strong / all-weak / demoted clients | all clients | all weak at f | "
              "demoted (gov/travel now weak) | rounds with 0 / 1 strong participant |",
              "|---|---|---:|---:|---:|---|"]
    for f in sweep["fractions"]:
        row = rows[str(f)]
        v = next((x for x in row["per_seed"].values() if x), None)
        counts = "—" if v is None else f"{v['strong_clients']} / {v['weak_clients']} / {v['demoted_clients']}"
        part = participation.get(str(f))
        part_txt = "—" if part is None else (f"{100 * part['share_rounds_with_0_strong']:.0f}% / "
                                              f"{100 * part['share_rounds_with_1_strong']:.0f}%")
        lines.append(f"| {f:.0%} | {counts} | {fmt(row['all_gap_late']['mean'])} | {fmt(row['weak_gap_late']['mean'])} | "
                     f"{fmt(row['demoted_gap_late']['mean'])} | {part_txt} |")
    lines += ["", f"Decision: **yes** when the late core-weak gap is ≥ {threshold} pp in every seed, "
              "i.e. naive heterogeneity leaves at least that much for an adaptive method to recover "
              "on the same fixed set of clients."]
    (root / "SUMMARY_STRONGFRAC.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
