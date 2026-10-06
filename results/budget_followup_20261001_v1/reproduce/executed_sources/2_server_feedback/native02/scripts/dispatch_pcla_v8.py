#!/usr/bin/env python3
"""Dispatch one v8 job at a time within an already allocated one-GPU Slurm step."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from pcla_research_v8 import queue


def assigned_cpus(limit=4):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < limit:
        raise RuntimeError("worker needs four allocated CPU cores")
    return tuple(available[:limit])


def worker(root: Path, run_root: Path, *, python: str, idle_seconds: int):
    slurm_job = os.getenv("SLURM_JOB_ID")
    slurm_step = os.getenv("SLURM_STEP_ID")
    if not slurm_job or slurm_step is None:
        raise RuntimeError("worker requires an existing Slurm GPU step")
    gpu = queue.gpu_identity()
    cpus = assigned_cpus()
    identity = f"{socket.gethostname()}:{os.getpid()}:{slurm_job}.{slurm_step}:{gpu}"
    idle_since = time.monotonic()
    while True:
        item = queue.claim(root, worker=identity, gpu_uuid=gpu, slurm_job=slurm_job,
                           slurm_step=slurm_step, cpu_cores=cpus)
        if item is None:
            if time.monotonic() - idle_since >= idle_seconds:
                return
            time.sleep(15)
            continue
        idle_since = time.monotonic()
        jid = item["job_id"]
        out = Path(item["output"])
        out.mkdir(parents=True, exist_ok=True)
        command = [python, str(SOURCE / "scripts/run_pcla_v8.py"), "--config", item["config"],
                   "--root", str(run_root), "--source-root", str(SOURCE), "--out", str(out)]
        if (out / "latest.pt").exists():
            command.append("--resume")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        with (out / "dispatch.stdout.log").open("a") as stdout, (out / "dispatch.stderr.log").open("a") as stderr:
            queue.write(out / "dispatch_identity.json", {
                "job_id": jid, "claim": item["claim"], "command": command,
                "config_sha256": item["config_sha256"], "source_root": str(SOURCE),
            })
            process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=False,
                                     preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        state = queue.finish(root, jid, item["claim"]["token"], return_code=process.returncode,
                             result_path=out / "result.json")
        print(json.dumps({"job_id": jid, "state": state, "return_code": process.returncode}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="v8 run root; queue is root/queues/development01")
    sub = p.add_subparsers(dest="action", required=True)
    init = sub.add_parser("initialize"); init.add_argument("--configs", type=Path)
    release = sub.add_parser("release-family"); release.add_argument("--family", required=True); release.add_argument("--gate-receipt", type=Path, required=True)
    run = sub.add_parser("worker"); run.add_argument("--python", default=sys.executable); run.add_argument("--idle-seconds", type=int, default=600)
    audit = sub.add_parser("complete-audit"); audit.add_argument("--job-id", required=True); audit.add_argument("--receipt", type=Path, required=True)
    sub.add_parser("status")
    args = p.parse_args()
    root = args.root / "queues" / "development01"
    if args.action == "initialize":
        queue.initialize(root, args.configs or args.root / "configs" / "development01")
    elif args.action == "release-family":
        print(queue.release_family(root, args.family, args.gate_receipt))
    elif args.action == "worker":
        worker(root, args.root, python=args.python, idle_seconds=args.idle_seconds)
    elif args.action == "complete-audit":
        queue.complete_audit(root, args.job_id, args.receipt)
    else:
        from collections import Counter
        print(json.dumps(Counter(value["state"] for _, value in queue._jobs(root)), sort_keys=True))


if __name__ == "__main__":
    main()
