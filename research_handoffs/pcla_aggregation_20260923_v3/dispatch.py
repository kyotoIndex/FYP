from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from common import (
    BATCH, SEEDS, SOURCES, archive_tree, atomic_json, extract_verified,
    fingerprint, handoff_hashes, read_json, require, resolve_site, sha256_file,
)

ROOT = Path(__file__).resolve().parent


def make_jobs(site):
    jobs = [
        {"id": "parity_pcla_7", "phase": 1, "kind": "trace",
         "source": "pcla", "rule": "fixed", "seed": 7,
         "check_source": True, "dependencies": [], "priority": 0,
         "timeout": site["full_run_timeout_seconds"]},
        {"id": "parity_replay_random_rank_7", "phase": 1, "kind": "trace",
         "source": "replay_random_rank", "rule": "fixed", "seed": 7,
         "check_source": True, "dependencies": [], "priority": 0,
         "timeout": site["full_run_timeout_seconds"]},
        {"id": "diagnose_7_5", "phase": 1, "kind": "diagnose",
         "seed": 7, "anchor": 5, "dependencies": [], "priority": 1,
         "timeout": site["diagnostic_timeout_seconds"]},
        {"id": "diagnose_7_15", "phase": 1, "kind": "diagnose",
         "seed": 7, "anchor": 15, "dependencies": [], "priority": 1,
         "timeout": site["diagnostic_timeout_seconds"]},
    ]
    for seed in (13, 21, 42, 87):
        for anchor in (5, 15):
            jobs.append({
                "id": f"diagnose_{seed}_{anchor}", "phase": 2,
                "kind": "diagnose", "seed": seed, "anchor": anchor,
                "dependencies": [], "priority": 1,
                "timeout": site["diagnostic_timeout_seconds"],
            })
    parity = ["parity_pcla_7", "parity_replay_random_rank_7"]
    for seed in SEEDS:
        phase = 3 if seed in (7, 13) else 4
        for source in SOURCES:
            for rule in ("contributors", "function_norm_matched"):
                jobs.append({
                    "id": f"trace_{source}_{rule}_{seed}",
                    "phase": phase, "kind": "trace", "source": source,
                    "rule": rule, "seed": seed, "dependencies": parity,
                    "priority": 3, "timeout": site["full_run_timeout_seconds"],
                })
        donor = f"online_pcla_contributors_{seed}"
        jobs.append({
            "id": donor, "phase": phase, "kind": "online",
            "source": "pcla", "rule": "contributors", "seed": seed,
            "dependencies": parity, "priority": 2,
            "timeout": site["full_run_timeout_seconds"],
        })
        jobs.append({
            "id": f"online_replay_contributors_{seed}",
            "phase": phase, "kind": "online_replay", "source": "pcla",
            "rule": "contributors", "seed": seed, "donor_job": donor,
            "dependencies": [donor], "priority": 2,
            "timeout": site["full_run_timeout_seconds"],
        })
    require(len(jobs) == 52 and len({x["id"] for x in jobs}) == 52,
            "unexpected job matrix")
    counts = {phase: sum(x["phase"] == phase for x in jobs)
              for phase in (1, 2, 3, 4)}
    require(counts == {1: 4, 2: 8, 3: 16, 4: 24},
            f"unexpected phase counts: {counts}")
    return jobs


def queue_paths(queue):
    queue = Path(queue)
    return {name: queue / name for name in (
        "jobs", "claims", "attempts", "done", "failed", "blocked",
        "archives", "leases", "gates", "recovery", "history",
    )}


def initialize(site_path, test_report):
    site = resolve_site(read_json(site_path))
    test = read_json(test_report)
    require(test.get("passed") is True, "fresh CPU tests are not passed")
    require(test.get("handoff_sources") == handoff_hashes(),
            "CPU tests were run on another handoff source")
    queue = Path(site["queue_root"])
    require(not queue.exists(), "queue root already exists")
    queue.mkdir(parents=True)
    paths = queue_paths(queue)
    for path in paths.values():
        path.mkdir()
    frozen_site = queue / "frozen_site.json"
    atomic_json(frozen_site, site)
    jobs = make_jobs(site)
    plan = {
        "batch": BATCH,
        "handoff_sources": handoff_hashes(),
        "frozen_site_sha256": sha256_file(frozen_site),
        "jobs": jobs,
    }
    atomic_json(queue / "plan.json", plan)
    atomic_json(queue / "control.json", {
        "open_phase": 1,
        "batch": BATCH,
        "history": [{"phase": 1, "action": "initialized",
                     "utc": time.time()}],
    })
    atomic_json(queue / "cpu_tests.json", test)
    for job in jobs:
        atomic_json(paths["jobs"] / f"{job['id']}.json", job)
    print(json.dumps({"queue": str(queue), "jobs": len(jobs),
                      "phase_counts": {str(p): sum(x["phase"] == p for x in jobs)
                                       for p in (1, 2, 3, 4)}}, indent=2))


def load_queue(queue):
    queue = Path(queue).resolve()
    plan = read_json(queue / "plan.json")
    site_path = queue / "frozen_site.json"
    require(sha256_file(site_path) == plan["frozen_site_sha256"],
            "frozen site changed")
    require(plan["handoff_sources"] == handoff_hashes(),
            "worker handoff source differs from queue")
    site = read_json(site_path)
    return queue, plan, site, queue_paths(queue)


def current_attempt(paths, job_id):
    records = sorted(paths["attempts"].glob(job_id + "__attempt_*.json"))
    return len(records) + 1


def process_exists(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def claim_ready(queue, plan, site, paths, gpu_uuid):
    control = read_json(queue / "control.json")
    open_phase = int(control["open_phase"])
    host = os.uname().nodename
    failed = {p.stem for p in paths["failed"].glob("*.json")}
    done = {p.stem for p in paths["done"].glob("*.json")}
    blocked = {p.stem for p in paths["blocked"].glob("*.json")}
    for job in sorted(plan["jobs"], key=lambda x: (x["phase"], x["priority"], x["id"])):
        job_id = job["id"]
        if job["phase"] > open_phase or job_id in done | failed | blocked:
            continue
        if any(dep in failed | blocked for dep in job["dependencies"]):
            continue
        if not all(dep in done for dep in job["dependencies"]):
            continue
        recovery = paths["recovery"] / f"{job_id}.json"
        recovery_record = read_json(recovery) if recovery.exists() else None
        if recovery_record and recovery_record.get("same_host_only"):
            if recovery_record["required_host"] != host:
                continue
        claim = paths["claims"] / f"{job_id}.json"
        try:
            descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        attempt = current_attempt(paths, job_id)
        record = {
            "job": job_id, "attempt": attempt, "gpu_uuid": gpu_uuid,
            "host": host, "pid": os.getpid(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "claimed_unix": time.time(),
            "recovery": recovery_record,
        }
        os.write(descriptor, json.dumps(record).encode())
        os.close(descriptor)
        return job, record
    return None, None


def dependency_roots(work, queue, paths, job):
    result = {}
    for dependency in job["dependencies"]:
        record = read_json(paths["done"] / f"{dependency}.json")
        destination = work / "dependencies" / dependency
        extract_verified(record["archive"]["path"],
                         record["archive"]["sha256"], destination)
        result[dependency] = str(destination)
    return result


def worker(queue_path, gpu_uuid, max_jobs):
    require(gpu_uuid.startswith("GPU-"), "pass a physical GPU UUID")
    queue, plan, site, paths = load_queue(queue_path)
    if site.get("require_slurm", True):
        require(os.environ.get("SLURM_JOB_ID"), "worker is outside Slurm")
        require(os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA visibility is unset")
    lease = paths["leases"] / f"{gpu_uuid}.json"
    descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, json.dumps({
        "pid": os.getpid(), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "started_unix": time.time(),
    }).encode())
    os.close(descriptor)
    completed = 0
    try:
        while completed < max_jobs:
            job, claim = claim_ready(queue, plan, site, paths, gpu_uuid)
            if job is None:
                return
            job_id = job["id"]
            attempt = claim["attempt"]
            stable_work = Path(site["scratch_root"]) / BATCH / job_id
            recovery = claim.get("recovery")
            if stable_work.exists() and not recovery:
                raise RuntimeError(f"unexpected existing scratch for {job_id}")
            stable_work.mkdir(parents=True, exist_ok=True)
            attempt_dir = stable_work / "attempts" / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            dependencies = dependency_roots(stable_work, queue, paths, job)
            job_file = attempt_dir / "job.json"
            deps_file = attempt_dir / "dependencies.json"
            atomic_json(job_file, job)
            atomic_json(deps_file, dependencies)
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in site.get("environment", {}).items()})
            env["PCLA_EXPECTED_GPU_UUID"] = gpu_uuid
            command = [
                site["python_pcla"], str(ROOT / "pcla_runner.py"),
                "--site", str(queue / "frozen_site.json"),
                "--job", str(job_file),
                "--out", str(stable_work / "result"),
                "--dependencies", str(deps_file),
            ]
            atomic_json(attempt_dir / "command.json", {
                "argv": command, "environment_overrides": site.get("environment", {}),
                "gpu_uuid": gpu_uuid, "host": os.uname().nodename,
                "handoff_sources": handoff_hashes(),
            })
            started = time.time()
            error = None
            returncode = None
            try:
                with (attempt_dir / "console.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True,
                    )
                    try:
                        returncode = process.wait(timeout=job["timeout"])
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError("job exceeded fixed technical timeout")
                require(returncode == 0, "job process returned nonzero")
                result = read_json(stable_work / "result" / "result.json")
                require(result["state"] == "completed", "missing completion record")
            except BaseException:
                error = traceback.format_exc()
                (attempt_dir / "worker_exception.txt").write_text(error, encoding="utf-8")
            attempt_record = {
                "job_id": job_id, "job_sha256": fingerprint(job),
                "attempt": attempt, "gpu_uuid": gpu_uuid,
                "host": os.uname().nodename, "pid": os.getpid(),
                "returncode": returncode, "started_unix": started,
                "ended_unix": time.time(),
                "state": "completed" if error is None else "failed",
                "recovery": recovery,
            }
            atomic_json(attempt_dir / "attempt.json", attempt_record)
            atomic_json(paths["attempts"] / f"{job_id}__attempt_{attempt:03d}.json",
                        attempt_record)
            payload = stable_work / f"payload_attempt_{attempt:03d}"
            require(not payload.exists(), "attempt payload already exists")
            payload.mkdir()
            if (stable_work / "result").exists():
                shutil.copytree(stable_work / "result", payload / "result")
            shutil.copytree(attempt_dir, payload / "attempt")
            archive = archive_tree(
                payload,
                paths["archives"] / f"{job_id}__attempt_{attempt:03d}.zip",
            )
            final_record = {**attempt_record, "archive": archive}
            destination = paths["done"] if error is None else paths["failed"]
            atomic_json(destination / f"{job_id}.json", final_record)
            (paths["claims"] / f"{job_id}.json").unlink(missing_ok=True)
            (paths["recovery"] / f"{job_id}.json").unlink(missing_ok=True)
            completed += 1
            if error is not None:
                # Scientific and implementation failures are not auto-retried.
                continue
    finally:
        lease.unlink(missing_ok=True)


def transitive_blocked(plan, failed):
    blocked = set()
    changed = True
    while changed:
        before = len(blocked)
        for job in plan["jobs"]:
            if any(dep in failed | blocked for dep in job["dependencies"]):
                blocked.add(job["id"])
        changed = len(blocked) != before
    return blocked - failed


def doctor(queue_path):
    queue, plan, site, paths = load_queue(queue_path)
    now = time.time()
    claims, leases = [], []
    for directory, output in ((paths["claims"], claims), (paths["leases"], leases)):
        for path in sorted(directory.glob("*.json")):
            record = read_json(path)
            output.append({
                "path": str(path), "record": record,
                "age_seconds": now - path.stat().st_mtime,
                "same_host_process_alive": (
                    process_exists(record.get("pid"))
                    if record.get("host") == os.uname().nodename else None
                ),
            })
    failed = {p.stem for p in paths["failed"].glob("*.json")}
    blocked = transitive_blocked(plan, failed)
    for job_id in blocked:
        path = paths["blocked"] / f"{job_id}.json"
        if not path.exists():
            atomic_json(path, {"job_id": job_id,
                               "reason": "failed_dependency",
                               "failed_dependencies": sorted(failed)})
    report = {
        "open_phase": read_json(queue / "control.json")["open_phase"],
        "done": len(list(paths["done"].glob("*.json"))),
        "failed": len(failed),
        "blocked": len(blocked),
        "claims": claims, "leases": leases,
        "queue": str(queue),
    }
    atomic_json(queue / "doctor_latest.json", report)
    print(json.dumps(report, indent=2))


def recover(queue_path, job_id, reason, confirm_process_stopped,
            same_host_only):
    require(reason == "external_interruption",
            "only verified external interruption is recoverable")
    require(confirm_process_stopped, "process-stopped confirmation is required")
    queue, plan, site, paths = load_queue(queue_path)
    job = next((x for x in plan["jobs"] if x["id"] == job_id), None)
    require(job is not None, "unknown job")
    done = paths["done"] / f"{job_id}.json"
    require(not done.exists(), "completed job cannot be recovered")
    claim_path = paths["claims"] / f"{job_id}.json"
    claim = read_json(claim_path) if claim_path.exists() else None
    if claim and claim.get("host") == os.uname().nodename:
        require(not process_exists(claim.get("pid")), "old worker process is alive")
    failure = paths["failed"] / f"{job_id}.json"
    history = paths["history"] / f"{job_id}__{time.time_ns()}"
    history.mkdir(parents=True)
    if claim_path.exists():
        shutil.move(claim_path, history / "claim.json")
    if failure.exists():
        shutil.move(failure, history / "failed.json")
    work = Path(site["scratch_root"]) / BATCH / job_id
    run_lock = work / "result" / "run" / ".run.lock"
    if run_lock.exists():
        run_lock.unlink()
    required_host = claim.get("host") if claim else os.uname().nodename
    record = {
        "job_id": job_id, "reason": reason,
        "approved_unix": time.time(),
        "same_host_only": bool(same_host_only and work.exists()),
        "required_host": required_host,
        "restart_if_scratch_missing": True,
    }
    atomic_json(paths["recovery"] / f"{job_id}.json", record)
    print(json.dumps(record, indent=2))


def materialize(queue_path, output):
    queue, plan, site, paths = load_queue(queue_path)
    output = Path(output)
    require(not output.exists(), "materialization directory already exists")
    output.mkdir(parents=True)
    inventory = []
    for job in plan["jobs"]:
        job_id = job["id"]
        record = None
        state = "missing_or_running"
        for candidate in ("done", "failed", "blocked"):
            path = paths[candidate] / f"{job_id}.json"
            if path.exists():
                record = read_json(path)
                state = candidate
                break
        if record and "archive" in record:
            require(record["job_sha256"] == fingerprint(job),
                    "materialized job identity differs")
            extract_verified(record["archive"]["path"],
                             record["archive"]["sha256"],
                             output / job_id)
        inventory.append({"job": job_id, "phase": job["phase"],
                          "state": state})
    atomic_json(output / "inventory.json", inventory)
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "frozen_site_identity.json", {
        "sha256": sha256_file(queue / "frozen_site.json")
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("init")
    command.add_argument("--site", required=True)
    command.add_argument("--test-report", required=True)
    command = sub.add_parser("worker")
    command.add_argument("--queue", required=True)
    command.add_argument("--gpu-uuid", required=True)
    command.add_argument("--max-jobs", type=int, default=52)
    command = sub.add_parser("doctor")
    command.add_argument("--queue", required=True)
    command = sub.add_parser("recover")
    command.add_argument("--queue", required=True)
    command.add_argument("--job", required=True)
    command.add_argument("--reason", required=True)
    command.add_argument("--confirm-process-stopped", action="store_true")
    command.add_argument("--same-host-only", action="store_true")
    command = sub.add_parser("materialize")
    command.add_argument("--queue", required=True)
    command.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.site, args.test_report)
    elif args.command == "worker":
        worker(args.queue, args.gpu_uuid, args.max_jobs)
    elif args.command == "doctor":
        doctor(args.queue)
    elif args.command == "recover":
        recover(args.queue, args.job, args.reason,
                args.confirm_process_stopped, args.same_host_only)
    else:
        materialize(args.queue, args.out)


if __name__ == "__main__":
    main()
