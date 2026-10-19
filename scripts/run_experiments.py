#!/usr/bin/env python3
"""List or run one recorded experiment group, sequentially and with portable paths."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "federated_finetuning"
SCRA = CODE / "experiments/scra_diagnostics_20261017"
HEADROOM = CODE / "experiments/headroom_coldstart_20261017"
GROUPS = ("roberta-main", "roberta-ablation", "roberta-controls", "roberta-baselines",
          "roberta-diagnostics", "qwen-lr", "qwen-main", "qwen-seeds", "qwen-external",
          "qwen-aggregation", "qwen-replay", "reports")


@dataclass
class Job:
    name: str
    command: list[str]
    seed: int | None = None
    completed: Path | None = None


def make_jobs(group: str, root: Path, device: str) -> list[Job]:
    jobs: list[Job] = []

    def scra(plan_file: Path, stage: str = "d0", extra: tuple[str, ...] = ()) -> None:
        plan = json.loads(plan_file.read_text())
        destination = root / plan["run_root"]
        complete = destination / ("fed.json" if stage == "c2fed" else "d0/d0_rounds.json")
        command = [sys.executable, str(SCRA / "run.py"), stage, "--plan", str(plan_file),
                   "--project-root", str(root), "--device", device, *extra]
        if stage == "d0" and (destination / "d0/checkpoint.pt").exists():
            command.append("--resume")
        jobs.append(Job(plan_file.stem + (f"-{stage}" if stage != "d0" else ""), command,
                        plan["d0"]["seed"], complete if stage in ("d0", "c2fed") else None))

    def headroom(only_main: bool) -> None:
        plan_file = HEADROOM / "plan_strongfrac.json"
        plan = json.loads(plan_file.read_text())
        for index, cell in enumerate(plan["cells"]):
            if only_main and cell["method"] != "fedavg_r8":
                continue
            tag = "" if cell.get("strong_fraction") is None else f"_sf{round(100 * cell['strong_fraction']):02d}"
            name = f"{cell['method']}{tag}_seed{cell['seed']}"
            command = [sys.executable, str(HEADROOM / "train.py"), "--plan", str(plan_file),
                       "--task-index", str(index), "--project-root", str(root), "--device", device, "--resume"]
            jobs.append(Job(name, command, cell["seed"], root / plan["run_root"] / name / "result.json"))

    if group == "roberta-main":
        headroom(only_main=True)
        scra(SCRA / "plan.json")
        scra(SCRA / "plan_vref.json")
        for plan in sorted((SCRA / "plans_batch").glob("*.json")):
            if int(plan.name.split("_")[0]) <= 6:
                scra(plan)
    elif group == "roberta-ablation":
        scra(SCRA / "plan_budget8.json")
        for plan in sorted((SCRA / "plans_batch").glob("*.json")):
            if int(plan.name.split("_")[0]) >= 7:
                scra(plan)
    elif group == "roberta-controls":
        for plan in sorted((SCRA / "plans_batch2").glob("*.json")):
            scra(plan)
    elif group == "roberta-baselines":
        headroom(only_main=False)
    elif group == "roberta-diagnostics":
        # Requires the snapshots produced by roberta-main's plan.json run.
        for filename in ("plan_rcs.json", "plan_weaklr.json", "plan_fixedshrink.json"):
            scra(SCRA / filename)
        for filename, stage in (("plan_d2.json", "d2"), ("plan_d3.json", "d3")):
            for index in range(3):
                scra(SCRA / filename, stage, ("--task-index", str(index)))
                jobs[-1].name += f"-snapshot{index}"
    elif group.startswith("qwen-"):
        for plan_file in sorted((SCRA / "plans_c2fed").glob("*.json")):
            name = plan_file.stem
            plan = json.loads(plan_file.read_text())
            method = plan["c2fed"]["method"]
            choose = {
                "qwen-lr": name.startswith("lr_"),
                "qwen-main": name.startswith("main_") and method in
                    {"r8", "vref", "L2", "amp", "fslora_st", "fslora"} and name != "main_amp_step025",
                "qwen-seeds": name.startswith("seed") and method in {"r8", "vref", "amp", "fslora"},
                "qwen-external": name in {"main_fslora_paper", "main_fedplora"},
                "qwen-aggregation": (method in {"amp_zf", "fslora_zf", "vref_zf"}
                    or name in {"main_amp_step025", "seed2041_fslora_paper", "seed2042_fslora_paper"})
                    and not name.startswith("replay_"),
                "qwen-replay": name.startswith("replay_"),
            }[group]
            if choose:
                scra(plan_file, "c2fed")
    elif group == "reports":
        for stage, filename in (("batchreport", "plans_batch/00_amp_s2040.json"),
                                ("batch2report", "plans_batch2/00_fslora_st_s2040.json"),
                                ("c2fedreport", "plans_c2fed/main_amp_zf.json")):
            scra(SCRA / filename, stage)
            jobs[-1].seed = None
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--project-root", type=Path, default=REPO / ".reproduction",
                        help="Directory containing models/, data/ and runs/; defaults to .reproduction")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", help="Select training seeds within the group")
    parser.add_argument("--name", action="append", help="Select an exact listed job name; repeatable")
    parser.add_argument("--execute", action="store_true", help="Run the listed commands; otherwise only list")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    jobs = make_jobs(args.group, root, args.device)
    if args.seeds:
        jobs = [job for job in jobs if job.seed in args.seeds]
    if args.name:
        unknown = set(args.name) - {job.name for job in jobs}
        if unknown:
            parser.error(f"Unknown job names after seed filtering: {sorted(unknown)}")
        jobs = [job for job in jobs if job.name in args.name]
    if not jobs:
        parser.error("No experiments match this selection")
    for job in jobs:
        status = " [recorded result exists; skip]" if job.completed and job.completed.exists() else ""
        print(f"{job.name}{status}\n  {shlex.join(job.command)}", flush=True)
    print(f"{len(jobs)} selected jobs; execution is sequential.", flush=True)
    if not args.execute:
        print("Add --execute to run this selection.")
        return
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    for job in jobs:
        if job.completed and job.completed.exists():
            print(f"Skipping completed {job.name}", flush=True)
            continue
        print(f"Running {job.name}", flush=True)
        subprocess.run(job.command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
