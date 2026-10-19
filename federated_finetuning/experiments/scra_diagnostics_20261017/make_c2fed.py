"""Generate C2-1 plans (docs/designs/c2_1_design_2026-10-18.md) into plans_c2fed/.

    python make_c2fed.py lrcheck           # stage A: r8, development seed 2041, 10 rounds, 4 learning rates
    python make_c2fed.py main --lr 1e-4    # stage B: 9 runs (r8 / vref / L2 / amp / fslora_st / fslora / amp α=4,16,32)
    python make_c2fed.py seeds --lr 2e-4   # stage C: r8 / vref / amp / fslora on seeds 2041, 2042 (design §9)
    python make_c2fed.py external --lr 2e-4  # stage D: FSLoRA as published + Fed-PLoRA style, seed 2040 (design §11)
    python make_c2fed.py aggregation --lr 2e-4  # stage E (§13): amp_zf / fslora_zf / vref_zf on 3 seeds, fslora_paper
                                                #   seeds 2041/2042, amp with server step 0.25 (12 runs), plus 3
                                                #   full-length replay checks of older code versions
Run roots: runs/c2_fed_20261018/<name>; report: python run.py c2fedreport --plan plans_c2fed/<any>.json
"""
import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "runs/c2_fed_20261018"
LRS = [2e-4, 1e-4, 5e-5, 2.5e-5]
REPORT = {"lr_select_round": 10, "late_window": [40, 60], "early_window": [5, 20], "osc_from": 20}


def base_plan() -> dict:
    plan = json.loads((HERE / "plan_c2.json").read_text())
    plan["c2"]["eval_seed"] = 2040                      # same evaluation subset for every C2-1 run
    plan["c2fed_report"] = REPORT
    return plan


def write(name: str, plan: dict) -> None:
    out = HERE / "plans_c2fed"
    out.mkdir(exist_ok=True)
    plan["id"] = f"c2_fed_20261018_{name}"
    plan["run_root"] = f"{ROOT}/{name}"
    (out / f"{name}.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote plans_c2fed/{name}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lrcheck", "main", "seeds", "external", "aggregation"])
    ap.add_argument("--lr", type=float)
    args = ap.parse_args()
    if args.stage == "lrcheck":
        for lr in LRS:
            plan = base_plan()
            plan["d0"]["seed"] = 2041
            plan["train"]["lr"] = lr
            plan["c2fed"] = {"method": "r8", "rounds": 10, "eval_every": 5, "checkpoint_every": 5,
                             "diag_restricted_rounds": 5}
            write(f"lr_{lr:.1e}", plan)
    elif args.stage == "aggregation":
        if args.lr is None:
            raise SystemExit("aggregation needs --lr (the stage-A choice, 2e-4)")
        runs = [("main_amp_zf", "amp_zf", 2040, {}), ("main_fslora_zf", "fslora_zf", 2040, {}),
                ("main_vref_zf", "vref_zf", 2040, {}), ("main_amp_step025", "amp", 2040, {"server_step": 0.25})]
        for seed in (2041, 2042):
            runs += [(f"seed{seed}_{m}", m, seed, {}) for m in ("amp_zf", "fslora_zf", "vref_zf", "fslora_paper")]
        for name, method, seed, extra in runs:
            plan = base_plan()
            plan["d0"]["seed"] = seed
            plan["train"]["lr"] = args.lr
            plan["c2fed"] = {"method": method, "rounds": 60, "eval_every": 5, "checkpoint_every": 5,
                             "diag_every": 10, **extra}
            write(name, copy.deepcopy(plan))
        # replay checks (design §13, second revision): one run per older version of c2fed.py, re-run in full
        # (60 rounds) with the current code; c2fedreport compares them with the originals and the multi-seed
        # summary only pairs runs whose version is verified (or substitutes the current-code replay)
        for orig in ("main_amp", "seed2041_vref", "main_fslora_paper"):
            plan = json.loads((HERE / "plans_c2fed" / f"{orig}.json").read_text())
            write(f"replay_{orig}", plan)
    elif args.stage == "external":
        if args.lr is None:
            raise SystemExit("external needs --lr (the stage-A choice, 2e-4)")
        for method in ("fslora_paper", "fedplora"):
            plan = base_plan()
            plan["d0"]["seed"] = 2040
            plan["train"]["lr"] = args.lr
            plan["c2fed"] = {"method": method, "rounds": 60, "eval_every": 5, "checkpoint_every": 5,
                             "diag_every": 10}
            write(f"main_{method}", copy.deepcopy(plan))
    elif args.stage == "seeds":
        if args.lr is None:
            raise SystemExit("seeds needs --lr (the stage-A choice, 2e-4)")
        for seed in (2041, 2042):
            for method in ("r8", "vref", "amp", "fslora"):
                plan = base_plan()
                plan["d0"]["seed"] = seed
                plan["train"]["lr"] = args.lr
                plan["c2fed"] = {"method": method, "rounds": 60, "eval_every": 5, "checkpoint_every": 5,
                                 "diag_every": 10}
                write(f"seed{seed}_{method}", copy.deepcopy(plan))
    else:
        if args.lr is None:
            raise SystemExit("main needs --lr (the learning rate chosen by stage A, see REPORT_C2_1.md)")
        runs = [("r8", "r8", {}), ("vref", "vref", {}), ("L2", "L2", {}), ("amp", "amp", {}),
                ("fslora_st", "fslora_st", {}), ("fslora", "fslora", {}),
                ("amp_a4", "amp", {"amp_alpha": 4.0}), ("amp_a16", "amp", {"amp_alpha": 16.0}),
                ("amp_a32", "amp", {"amp_alpha": 32.0})]
        for name, method, extra in runs:
            plan = base_plan()
            plan["d0"]["seed"] = 2040
            plan["train"]["lr"] = args.lr
            plan["c2fed"] = {"method": method, "rounds": 60, "eval_every": 5, "checkpoint_every": 5,
                             "diag_every": 10, **extra}
            write(f"main_{name}", copy.deepcopy(plan))

if __name__ == "__main__":
    main()
