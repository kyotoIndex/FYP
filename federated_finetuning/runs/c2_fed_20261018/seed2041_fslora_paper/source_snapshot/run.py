"""Entry point for the A-SCRA diagnostics.

    python run.py d0 [--resume]                 # logged federated run (GPU, ~15 min)
    python run.py d1 --task-index 0|1|2         # fixed-state replay for snapshot_rounds[i] (GPU)
    python run.py d1 --round 60                 # same, by round
    python run.py d2 --task-index 0|1|2         # single-round dispersion on snapshot_rounds[i] (GPU)
    python run.py analyze                       # criteria report (CPU, seconds)
    python run.py d2report                      # D2 report (CPU, seconds)
    python run.py d3 --task-index 0|1|2         # realizable-fix screening on snapshot_rounds[i] (GPU)
    python run.py d3report                      # D3 report (CPU, seconds)
    python run.py batchreport --plan plans_batch/00_amp_s2040.json   # validation batch summary (CPU)
    python run.py batch2report --plan plans_batch2/00_fslora_st_s2040.json   # batch 2 summary (CPU)
    python run.py c2check --plan plan_c2.json                    # C2-0: asset check (cache, model, one step)
    python run.py c2snap --plan plan_c2.json                     # C2-0: neutral rank-R snapshots (GPU)
    python run.py c2probe --plan plan_c2.json --task-index 0|1|2 # C2-0: one-step probe per snapshot (GPU)
    python run.py c2report --plan plan_c2.json                   # C2-0 report (CPU)
    python run.py c2fed --plan plans_c2fed/<name>.json           # C2-1: one federated run (GPU)
    python run.py c2fedreport --plan plans_c2fed/<any>.json      # C2-1 report incl. lr choice (CPU)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scra.analyze import write_report  # noqa: E402
from scra.batch import write_batch_report  # noqa: E402
from scra.batch2 import write_batch2_report  # noqa: E402
from scra.c2fed import run_c2_fed, write_c2fed_report  # noqa: E402
from scra.c2probe import run_c2_check, run_c2_probe, run_c2_snapshots, write_c2_report  # noqa: E402
from scra.core import load_plan  # noqa: E402
from scra.d0 import run_d0  # noqa: E402
from scra.d1 import run_d1  # noqa: E402
from scra.d2 import run_d2, write_d2_report  # noqa: E402
from scra.d3 import run_d3, write_d3_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["d0", "d1", "d2", "d3", "analyze", "d2report", "d3report", "batchreport", "batch2report", "c2check", "c2snap", "c2probe", "c2report", "c2fed", "c2fedreport"])
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--project-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--stop-after-round", type=int, help="testing only")
    args = parser.parse_args()
    plan = load_plan(args.plan, args.project_root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    root = Path(plan["run_root"])
    print(f"plan={plan['id']} stage={args.stage} device={device} run_root={root}", flush=True)
    if args.stage == "d0":
        run_d0(plan, root / "d0", device, resume=args.resume, stop_after_round=args.stop_after_round)
    elif args.stage in ("d1", "d2", "d3"):
        rounds = plan["d0"]["snapshot_rounds"]
        snapshot = args.round if args.round is not None else rounds[args.task_index]
        {"d1": run_d1, "d2": run_d2, "d3": run_d3}[args.stage](plan, snapshot, root / args.stage, device)
    elif args.stage == "d2report":
        print(write_d2_report(root)["verdicts"], flush=True)
    elif args.stage == "d3report":
        print(write_d3_report(root)["verdicts"], flush=True)
    elif args.stage == "batchreport":
        print(write_batch_report(Path(plan["project_root"]) / "runs"), flush=True)
    elif args.stage == "batch2report":
        print(write_batch2_report(Path(plan["project_root"]) / "runs"), flush=True)
    elif args.stage == "c2check":
        run_c2_check(plan, root / "c2check", device)
    elif args.stage == "c2snap":
        run_c2_snapshots(plan, root / "c2snap", device)
    elif args.stage == "c2probe":
        rounds = plan["c2"]["snapshot_rounds"]
        snapshot = args.round if args.round is not None else rounds[args.task_index]
        run_c2_probe(plan, snapshot, root / "c2probe", device)
    elif args.stage == "c2fed":
        run_c2_fed(plan, root, device)
    elif args.stage == "c2fedreport":
        print(write_c2fed_report(root.parent, plan["c2fed_report"]), flush=True)
    elif args.stage == "c2report":
        print(write_c2_report(root, plan["c2"].get("learnable_min_core_acc", 0.45))["verdicts"], flush=True)
    else:
        write_report(root, plan)


if __name__ == "__main__":
    main()
