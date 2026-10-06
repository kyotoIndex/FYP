#!/usr/bin/env python3
"""Release and run the independent V queue inside an existing GPU allocation.

Release requires a frozen receipt binding each config and the complete source
snapshot to reviewed V technical evidence. Execution success awaits V audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dispatch_pcla_v7 import lock, read, sha, verify_source, write
from scripts.initialize_pcla_v7_validation_queue import HARDWARE_FAMILIES, PROTOCOL, QUEUE_NAME

MODES = ("exclusive", "shared")
PARENTS = {"D072": 0, "D069": 1}


def checked_queue(queue: Path) -> tuple[Path, dict]:
    queue = queue.resolve()
    if queue.name != QUEUE_NAME:
        raise ValueError("expected run_root/validation_queue")
    identity = read(queue / "matrix_identity.json")
    if (identity.get("schema") != "pcla-v7-validation-queue-1"
            or identity.get("protocol") != PROTOCOL or identity.get("phase") != "V"
            or identity.get("count") != 156):
        raise ValueError("V queue identity is invalid")
    return queue, identity


def checked_job(queue: Path, identity: dict, jid: str) -> tuple[Path, dict]:
    if not jid.startswith("V_S") or "/" in jid or ".." in jid:
        raise ValueError(f"invalid V job ID: {jid}")
    path = queue / "jobs" / f"{jid}.json"
    item = read(path)
    job = item["job"]
    if (job.get("job_id") != jid or job.get("phase") != "V"
            or job.get("protocol") != PROTOCOL
            or item.get("pool") != identity["pool_map"].get(job.get("hardware_block"))
            or item.get("required_release_gates") != ["review", "configuration", "source"]):
        raise ValueError(f"V queue job identity differs: {jid}")
    expected_out = queue.parent / "V" / jid
    if job.get("output_dir") != f"runs/{PROTOCOL}/V/{jid}":
        raise ValueError(f"V output identity differs: {jid}")
    if expected_out.is_symlink():
        raise ValueError(f"V output is a symlink: {jid}")
    return path, item


def checked_receipt(path: Path, expected_sha: str, technical_path: Path,
                    expected_technical_sha: str, source_sha: str) -> tuple[dict, dict]:
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise ValueError("release receipt SHA-256 differs")
    if len(expected_technical_sha) != 64 or sha(technical_path) != expected_technical_sha:
        raise ValueError("technical receipt SHA-256 differs")
    receipt, technical = read(path), read(technical_path)
    if (receipt.get("schema") != "pcla-v7-release-gates-1"
            or receipt.get("phase") != "V" or receipt.get("protocol") != PROTOCOL
            or receipt.get("review_gate") != "approved"
            or receipt.get("configuration_gate") != "passed"
            or receipt.get("source_gate") != "passed"
            or receipt.get("technical_receipt_sha256") != expected_technical_sha
            or receipt.get("source_manifest_sha256") != source_sha):
        raise ValueError("V release gate or source identity is not satisfied")
    if (technical.get("schema") != "pcla-v7-technical-gate-1"
            or technical.get("phase") != "V" or technical.get("protocol") != PROTOCOL
            or technical.get("status") != "passed"
            or technical.get("source_manifest_sha256") != source_sha):
        raise ValueError("V technical gate is not satisfied")
    return receipt, technical


def release(queue: Path, ids: list[str], config_dir: Path, source: Path,
            receipt_path: Path, receipt_sha256: str, technical_path: Path,
            technical_sha256: str, hardware_family: str, mode: str = "exclusive",
            shared_profile_path: Path | None = None,
            shared_profile_sha256: str | None = None) -> list[str]:
    queue, identity = checked_queue(queue)
    if hardware_family not in HARDWARE_FAMILIES or mode not in MODES:
        raise ValueError("explicit hardware family and execution mode are required")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("release IDs must be unique and nonempty")
    source = source.resolve(strict=True)
    source_hash = verify_source(source)
    receipt, technical = checked_receipt(receipt_path, receipt_sha256, technical_path,
                                         technical_sha256, source_hash)
    if mode == "shared":
        if (shared_profile_path is None or shared_profile_sha256 is None
                or len(shared_profile_sha256) != 64
                or sha(shared_profile_path) != shared_profile_sha256):
            raise ValueError("shared mode requires a SHA-checked profile receipt")
        profile = read(shared_profile_path)
        if (profile.get("schema") != "pcla-v7-shared-profile-1"
                or profile.get("phase") != "V" or profile.get("status") != "passed"
                or profile.get("hardware_family") != hardware_family
                or profile.get("source_manifest_sha256") != source_hash
                or receipt.get("shared_profile_sha256") != shared_profile_sha256):
            raise ValueError("shared mode profile gate is not satisfied")
    changes = []
    with lock(queue):
        for jid in ids:
            path, item = checked_job(queue, identity, jid)
            if item["state"] != "blocked":
                raise ValueError(f"{jid}: release requires blocked state")
            if item["pool"] != hardware_family:
                raise ValueError(f"{jid}: hardware family differs from frozen seed map")
            cfg_path = config_dir / f"{jid}.json"
            cfg_hash = sha(cfg_path)
            cfg = read(cfg_path)
            if cfg.get("job") != item["job"] or cfg.get("recipe", {}).get("recipe_id") != item["job"]["recipe_id"]:
                raise ValueError(f"{jid}: V config identity differs")
            if (receipt.get("config_sha256_by_job", {}).get(jid) != cfg_hash
                    or technical.get("config_sha256_by_job", {}).get(jid) != cfg_hash):
                raise ValueError(f"{jid}: configuration or technical SHA-256 differs")
            if mode == "shared" and profile.get("config_sha256_by_job", {}).get(jid) != cfg_hash:
                raise ValueError(f"{jid}: shared profile configuration SHA-256 differs")
            out = queue.parent / "V" / jid
            if out.exists() and any(out.iterdir()):
                raise ValueError(f"{jid}: output already contains files")
            changes.append((path, dict(item, state="pending", reason=None,
                hardware_family=hardware_family, execution_mode=mode,
                config=str(cfg_path.resolve()), config_sha256=cfg_hash,
                source=str(source), source_manifest_sha256=source_hash,
                release_receipt=str(receipt_path.resolve()), release_receipt_sha256=receipt_sha256,
                technical_receipt=str(technical_path.resolve()), technical_receipt_sha256=technical_sha256,
                shared_profile_receipt=str(shared_profile_path.resolve()) if mode == "shared" else None,
                shared_profile_sha256=shared_profile_sha256 if mode == "shared" else None,
                out=str(out), released_at=time.time())))
        for path, item in changes:
            write(path, item)
    return ids


def dependencies_met(queue: Path, job: dict) -> bool:
    dependencies = job.get("depends_on") or []
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    for jid in dependencies:
        if not jid.startswith("V_S") or "/" in jid or ".." in jid:
            return False
        dep_path = queue / "jobs" / f"{jid}.json"
        if not dep_path.is_file() or read(dep_path).get("state") != "completed":
            return False
    return True


def claim(queue: Path, hardware_family: str, worker_id: str,
          mode: str = "exclusive") -> tuple[Path, dict] | None:
    queue, identity = checked_queue(queue)
    if hardware_family not in HARDWARE_FAMILIES or mode not in MODES:
        raise ValueError("explicit hardware family and execution mode are required")
    with lock(queue):
        candidates = []
        for path in sorted((queue / "jobs").glob("*.json")):
            jid = path.stem
            _, item = checked_job(queue, identity, jid)
            if (item["state"] == "pending" and item.get("hardware_family") == hardware_family
                    and item.get("execution_mode") == mode
                    and dependencies_met(queue, item["job"])):
                candidates.append((path, item))
        candidates.sort(key=lambda pv: (PARENTS.get(pv[1]["job"]["recipe_id"], 2),
                                          pv[1]["job"].get("priority", 100),
                                          pv[1]["job"]["job_id"]))
        if not candidates:
            return None
        path, item = candidates[0]
        if sha(item["config"]) != item["config_sha256"]:
            raise RuntimeError("released V config mutated")
        if verify_source(Path(item["source"])) != item["source_manifest_sha256"]:
            raise RuntimeError("released V source mutated")
        if (sha(item["release_receipt"]) != item["release_receipt_sha256"]
                or sha(item["technical_receipt"]) != item["technical_receipt_sha256"]):
            raise RuntimeError("released V gate receipt mutated")
        if (mode == "shared" and
                sha(item["shared_profile_receipt"]) != item["shared_profile_sha256"]):
            raise RuntimeError("released V shared profile mutated")
        now = time.time()
        item.update(state="running", worker=worker_id, started_at=now)
        item["attempts"].append({"worker": worker_id, "started_at": now,
            "slurm_job": os.getenv("SLURM_JOB_ID"), "slurm_step": os.getenv("SLURM_STEP_ID")})
        write(path, item)
        return path, item


def finish(queue: Path, path: Path, worker_id: str, exit_code: int) -> str:
    queue, identity = checked_queue(queue)
    with lock(queue):
        _, item = checked_job(queue, identity, path.stem)
        if item["state"] != "running" or item.get("worker") != worker_id:
            raise RuntimeError("lost V job ownership")
        state = "executed_pending_audit" if exit_code == 0 else "failed_needs_review"
        item.update(state=state, finished_at=time.time(), exit_code=exit_code)
        item["attempts"][-1].update(finished_at=item["finished_at"], exit_code=exit_code)
        write(path, item)
    return state


def worker(queue: Path, hardware_family: str, python: str = sys.executable,
           idle_seconds: int = 900, mode: str = "exclusive") -> None:
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("V worker requires an existing Slurm GPU step")
    wid = f"{socket.gethostname()}:{os.getpid()}:{os.getenv('SLURM_JOB_ID')}.{os.getenv('SLURM_STEP_ID')}"
    idle_since = time.monotonic()
    while True:
        claimed = claim(queue, hardware_family, wid, mode)
        if claimed is None:
            if time.monotonic() - idle_since >= idle_seconds:
                return
            time.sleep(3)
            continue
        idle_since = time.monotonic()
        path, item = claimed
        out = Path(item["out"])
        try:
            if out.exists() and any(out.iterdir()):
                raise RuntimeError("V output contains files before first execution")
            out.mkdir(parents=True, exist_ok=True)
            argv = [python, str(Path(item["source"]) / "scripts/run_pcla_v7.py"),
                    "--config", item["config"], "--out", str(out), "--device", "cuda"]
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       PCLA_GPU_SHARING=mode)
            write(out / "dispatch_identity.json", {"worker": wid, "argv": argv,
                  "config_sha256": item["config_sha256"],
                  "source_manifest_sha256": item["source_manifest_sha256"],
                  "release_receipt_sha256": item["release_receipt_sha256"],
                  "technical_receipt_sha256": item["technical_receipt_sha256"],
                  "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                  "hardware_family": hardware_family, "gpu_execution_mode": mode})
            with (out / "worker.log").open("ab") as log:
                code = subprocess.run(argv, cwd=item["source"], env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
        except Exception as exc:
            try:
                out.mkdir(parents=True, exist_ok=True)
                with (out / "worker.log").open("ab") as log:
                    log.write(f"\nV worker launch error: {exc!r}\n".encode())
            except OSError:
                pass
            code = 127
        state = finish(queue, path, wid, code)
        print(json.dumps({"job": item["job"]["job_id"], "state": state, "exit": code}), flush=True)


def status(queue: Path) -> dict:
    queue, identity = checked_queue(queue)
    with lock(queue):
        jobs = [checked_job(queue, identity, p.stem)[1]
                for p in sorted((queue / "jobs").glob("*.json"))]
    return {"phase": "V", "count": len(jobs),
            "counts": dict(Counter(j["state"] for j in jobs)),
            "by_hardware_family": {family: dict(Counter(j["state"] for j in jobs if j["pool"] == family))
                                   for family in sorted(HARDWARE_FAMILIES)},
            "running": [{"job_id": j["job"]["job_id"], "worker": j["worker"]}
                        for j in jobs if j["state"] == "running"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("release")
    r.add_argument("--ids", nargs="+", required=True)
    r.add_argument("--config-dir", type=Path, required=True)
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--release-receipt", type=Path, required=True)
    r.add_argument("--release-receipt-sha256", required=True)
    r.add_argument("--technical-receipt", type=Path, required=True)
    r.add_argument("--technical-receipt-sha256", required=True)
    r.add_argument("--hardware-family", choices=sorted(HARDWARE_FAMILIES), required=True)
    r.add_argument("--mode", choices=MODES, default="exclusive")
    r.add_argument("--shared-profile-receipt", type=Path)
    r.add_argument("--shared-profile-receipt-sha256")
    w = sub.add_parser("worker")
    w.add_argument("--hardware-family", choices=sorted(HARDWARE_FAMILIES), required=True)
    w.add_argument("--mode", choices=MODES, default="exclusive")
    w.add_argument("--python", default=sys.executable)
    w.add_argument("--idle-seconds", type=int, default=900)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "release":
        print(json.dumps({"released": release(args.queue, args.ids, args.config_dir, args.source,
              args.release_receipt, args.release_receipt_sha256, args.technical_receipt,
              args.technical_receipt_sha256, args.hardware_family, args.mode,
              args.shared_profile_receipt, args.shared_profile_receipt_sha256)}))
    elif args.command == "worker":
        worker(args.queue, args.hardware_family, args.python, args.idle_seconds, args.mode)
    else:
        print(json.dumps(status(args.queue), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
