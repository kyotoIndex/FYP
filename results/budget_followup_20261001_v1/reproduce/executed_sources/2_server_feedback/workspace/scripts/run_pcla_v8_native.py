#!/usr/bin/env python3
"""Run one v8 U005/U006 native Fed-SB trajectory on a single device."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from pcla_repair.runtime import atomic_json
from pcla_research_v7.execution_timing import ExecutionTimingLedger, required_breakdown
from pcla_research_v7.native_runtime import NATIVE_TIMING_LABELS
from pcla_research_v8.native_fedsb_runtime import execute_native_fedsb, native_contract


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    args = p.parse_args(argv)
    config_path = args.config.resolve()
    expanded = json.loads(config_path.read_text())
    native_contract(expanded)
    if args.source_root is not None and args.source_root.resolve() != ROOT:
        raise ValueError("native source-root differs from executing workspace")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = args.out or args.root / "results" / expanded["job"]["job_id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() != args.resume:
        raise ValueError("native --resume must match checkpoint presence")
    lock = out / ".run.lock"  # allowed before first native checkpoint
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("native run lock already exists") from exc
    os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node(),
                             "job": os.environ.get("SLURM_JOB_ID")}).encode())
    os.close(fd)
    timing = None
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
        timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS, synchronize=sync)
        rows, provenance = execute_native_fedsb(expanded, config_path, out, device,
                                                timing=timing, stop_after=args.stop_after,
                                                resume=args.resume)
        timing.end_envelope()
        receipt = timing.snapshot()
        attempts_file = out / "execution_attempts.json"
        attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
        atomic_json(out / f"timing_attempt_{len(attempts):03d}.json", receipt)
        breakdowns = [required_breakdown(json.loads(p.read_text()))
                      for p in sorted(out.glob("timing_attempt_*.json"))]
        cumulative = {key: sum(part[key] for part in breakdowns)
                      for key in breakdowns[0]}
        atomic_json(out / "result.json", {
            "job_id": expanded["job"]["job_id"], "recipe": expanded["recipe"],
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "method": "native_fedsb_r32", "rounds": len(rows),
            "budget_return_round": rows[-1]["budget_stop_round"] if rows else None,
            "required_cost_breakdown": cumulative,
            "required_cost_breakdown_current_attempt": required_breakdown(receipt),
            "timed_completed_attempts": len(breakdowns),
            "source_manifest_sha256": provenance["v8_source_manifest_sha256"],
            "state": "technical_paused" if args.stop_after is not None else "executed_pending_audit",
            "hardware": provenance["hardware"],
        })
        print(json.dumps({"job": expanded["job"]["job_id"], "rounds": len(rows),
                          "out": str(out)}), flush=True)
    except BaseException:
        trace = traceback.format_exc()
        (out / "exception.txt").write_text(trace)
        if timing is not None:
            try:
                atomic_json(out / "timing_events_partial.json", timing.partial_snapshot())
            except BaseException:
                (out / "timing_capture_exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
