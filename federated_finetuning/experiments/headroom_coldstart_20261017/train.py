"""Run one (method, seed) cell of the cold-start headroom plan.

    python train.py --plan plan.json --task-index $SLURM_ARRAY_TASK_ID [--resume]
    python train.py --plan plan.json --method fedavg_r8 --seed 2040

A plan either lists explicit "cells" ({method, seed, strong_fraction?}; task index = list
position) or gives a methods x seeds grid (index = seed_idx * len(methods) + method_idx).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom import METHODS  # noqa: E402
from headroom.fed import run  # noqa: E402


def load_plan(path: Path, project_root: str | None) -> dict:
    plan = json.loads(path.read_text())
    root = Path(project_root or plan["project_root"]).expanduser()
    plan["project_root"] = str(root)
    for key in ("cache_directory", "manifest_path", "run_root"):
        value = Path(plan[key]).expanduser()
        plan[key] = str(value if value.is_absolute() else root / value)
    name = Path(plan["model"]["name"]).expanduser()
    plan["model"]["name"] = str(name if name.is_absolute() else root / name)
    if "cells" not in plan:
        plan["cells"] = [{"method": m, "seed": s} for s in plan["seeds"] for m in plan["methods"]]
    unknown = {c["method"] for c in plan["cells"]} - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods in plan: {sorted(unknown)}")
    return plan


def cell_name(cell: dict) -> str:
    fraction = cell.get("strong_fraction")
    tag = "" if fraction is None else f"_sf{round(100 * fraction):02d}"
    return f"{cell['method']}{tag}_seed{cell['seed']}"


def cell(plan: dict, args) -> dict:
    if args.task_index is not None:
        if not 0 <= args.task_index < len(plan["cells"]):
            raise IndexError(f"task index {args.task_index} outside plan ({len(plan['cells'])} cells)")
        return plan["cells"][args.task_index]
    if not (args.method and args.seed is not None):
        raise SystemExit("Give --task-index, or both --method and --seed")
    return {"method": args.method, "seed": args.seed, "strong_fraction": args.strong_fraction}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--project-root")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--strong-fraction", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-round", type=int, help="testing: checkpoint and stop early")
    args = parser.parse_args()

    plan = load_plan(args.plan, args.project_root)
    spec = cell(plan, args)
    method, seed, fraction = spec["method"], spec["seed"], spec.get("strong_fraction")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    output = Path(plan["run_root"]) / cell_name(spec)
    print(f"plan={plan['id']} method={method} seed={seed} strong_fraction={fraction} "
          f"device={device} output={output}", flush=True)
    result = run(plan, method, seed, output, device, resume=args.resume,
                 stop_after_round=args.stop_after_round, strong_fraction=fraction)
    if "final" in result:
        print(json.dumps({"method": method, "seed": seed, "strong_fraction": fraction,
                          "client_macro_accuracy": result["final"]["client_macro_accuracy"],
                          "client_macro_nll": result["final"]["client_macro_nll"]}), flush=True)


if __name__ == "__main__":
    main()
