#!/usr/bin/env python3
"""Isolated COST1 queue; never allocates GPUs or accepts a measurement.

The command only operates inside an existing Slurm GPU step.  Release is a
separate, hash-bound action after the four-path timing gate has passed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dispatch_pcla_v7 import lock, read, sha, verify_source, write

QUEUE_NAME = "cost_queue"
RUN_ROOT_NAMES = {"pcla-global-budget-20260928-v7", "pcla_global_budget_20260928_v7"}
PLAN_SCHEMA = "pcla-cost1-bound-plan-1"
GATE_SCHEMA = "pcla-v7-cost-gpu-gate-1"
JOB_RE = re.compile(r"COST1_D\d{3}_s401\Z")
PAIR = ("COST1_D011_s401", "COST1_D035_s401")
HARDWARE = "NVIDIA GeForce RTX 4090 D"
PATHS = frozenset({"core01", "core02", "native03", "native04"})


def _expected_input_sha(plan: dict) -> str:
    matches = [value for name, value in plan["inputs"].items()
               if Path(name).name == "expected_cost_identities.json"]
    if len(matches) != 1 or not isinstance(matches[0], str) or len(matches[0]) != 64:
        raise ValueError("bound plan lacks one expected identities SHA")
    return matches[0]


def _checked_plan(path: Path) -> tuple[dict, dict[str, dict]]:
    plan = read(path)
    jobs = plan.get("jobs")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "not_dispatched"
            or plan.get("quality_results_replaced") is not False
            or not isinstance(jobs, list) or len(jobs) != 117):
        raise ValueError("expected unchanged 117-job COST1 bound plan")
    by_id = {}
    for job in jobs:
        jid = job.get("replay_id")
        if (not isinstance(jid, str) or not JOB_RE.fullmatch(jid) or jid in by_id
                or job.get("seed") != 401 or job.get("recipe_id") != jid[6:10]
                or job.get("dispatch_status") != "blocked_pending_gpu_timing_gate"
                or job.get("required_hardware") != "RTX4090D_exclusive"
                or job.get("end_control") not in {
                    "first_actual_budget_refusal_including_control", "at_cap_return_no_round201"}):
            raise ValueError(f"invalid or duplicate COST1 identity: {jid}")
        pro = job.get("pro_plan", {})
        if (pro.get("job_id") != jid or pro.get("recipe_id") != job["recipe_id"]
                or pro.get("original_job_id") != f"{job['recipe_id']}_s401"
                or int(pro.get("prefix_round", -1)) != job.get("expected_budget_return_round")
                or int(pro.get("expected_deployed_bytes", -1)) != job.get("expected_deployed_bytes")
                or pro.get("config_sha256") != job.get("original_config_sha256")
                or pro.get("source_manifest_sha256") != job.get("original_source_manifest_sha256")):
            raise ValueError(f"Pro identity differs in {jid}")
        by_id[jid] = job
    if not set(PAIR) <= set(by_id):
        raise ValueError("D011/D035 pair is absent")
    if set(by_id) != {f"COST1_D{i:03d}_s401" for i in range(1, 118)}:
        raise ValueError("COST1 matrix must cover D001 through D117 exactly")
    if (Counter(job["priority"] for job in jobs) != {0: 2, 10: 6, 30: 109}
            or any(by_id[jid]["priority"] != 0 for jid in PAIR)):
        raise ValueError("COST1 priority matrix differs")
    _expected_input_sha(plan)
    return plan, by_id


def checked_queue(queue: Path) -> tuple[Path, dict, dict[str, dict]]:
    if queue.is_symlink():
        raise ValueError("cost queue cannot be a symlink")
    queue = queue.resolve()
    if queue.name != QUEUE_NAME or queue.parent.name not in RUN_ROOT_NAMES:
        raise ValueError("expected run_root/cost_queue, isolated from D0/V")
    identity = read(queue / "matrix_identity.json")
    plan_path = Path(identity["plan"])
    if sha(plan_path) != identity.get("plan_sha256"):
        raise ValueError("COST1 bound plan mutated")
    plan, jobs = _checked_plan(plan_path)
    if (identity.get("schema") != "pcla-v7-cost-queue-1" or identity.get("count") != 117
            or identity.get("expected_identities_sha256") != _expected_input_sha(plan)
            or set(p.stem for p in (queue / "jobs").glob("*.json")) != set(jobs)):
        raise ValueError("cost queue matrix identity differs")
    return queue, identity, jobs


def _item(queue: Path, jid: str, jobs: dict[str, dict]) -> tuple[Path, dict]:
    if jid not in jobs:
        raise ValueError(f"unknown COST1 ID: {jid}")
    path = queue / "jobs" / f"{jid}.json"
    item = read(path)
    if item.get("job") != jobs[jid] or item.get("lane") != ("pair" if jid in PAIR else "general"):
        raise ValueError(f"cost queue job identity differs: {jid}")
    return path, item


def initialize(queue: Path, plan_path: Path) -> int:
    if queue.is_symlink() or queue.name != QUEUE_NAME or queue.parent.name not in RUN_ROOT_NAMES:
        raise ValueError("cost queue must be run_root/cost_queue")
    plan_path = plan_path.resolve(strict=True)
    plan, jobs = _checked_plan(plan_path)
    identity = {"schema": "pcla-v7-cost-queue-1", "count": 117,
                "plan": str(plan_path), "plan_sha256": sha(plan_path),
                "expected_identities_sha256": _expected_input_sha(plan)}
    with lock(queue):
        current = queue / "matrix_identity.json"
        if current.exists() and read(current) != identity:
            raise ValueError("existing cost queue identity differs")
        existing = {p.stem for p in (queue / "jobs").glob("*.json")}
        if existing - set(jobs):
            raise ValueError("cost queue contains foreign job IDs")
        for jid, job in jobs.items():
            path = queue / "jobs" / f"{jid}.json"
            if path.exists():
                _item(queue, jid, jobs)
            else:
                write(path, {"job": job, "lane": "pair" if jid in PAIR else "general",
                             "state": "blocked", "reason": "gpu_timing_gate_not_released",
                             "attempts": [], "created_at": time.time()})
        if not current.exists():
            write(current, identity)
        lane = queue / "pair_lane.json"
        if lane.exists():
            if read(lane).get("schema") != "pcla-v7-cost-pair-lane-1":
                raise ValueError("cost pair lane identity differs")
        else:
            write(lane, {"schema": "pcla-v7-cost-pair-lane-1", "gpu_uuid": None})
    return 117


def _checked_gate(path: Path, expected_sha: str, source_sha: str) -> dict:
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise ValueError("GPU timing gate receipt SHA differs")
    gate = read(path)
    if (gate.get("schema") != GATE_SCHEMA or gate.get("status") != "passed"
            or gate.get("source_manifest_sha256") != source_sha
            or gate.get("hardware") != HARDWARE or gate.get("mode") != "exclusive"
            or gate.get("technical_seeds") != [9001, 9002]
            or set(gate.get("passed_paths", [])) != PATHS):
        raise ValueError("four-path GPU timing gate is not passed for this source")
    return gate


def release(queue: Path, ids: list[str], source: Path, expected_path: Path,
            gate_path: Path, gate_sha256: str) -> list[str]:
    queue, identity, jobs = checked_queue(queue)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("release IDs must be unique and nonempty")
    source = source.resolve(strict=True)
    source_sha = verify_source(source)
    expected_path = expected_path.resolve(strict=True)
    if sha(expected_path) != identity["expected_identities_sha256"]:
        raise ValueError("Pro expected cost identities SHA differs")
    expected = read(expected_path)
    gate_path = gate_path.resolve(strict=True)
    _checked_gate(gate_path, gate_sha256, source_sha)
    changes = []
    with lock(queue):
        for jid in ids:
            path, item = _item(queue, jid, jobs)
            if item["state"] != "blocked":
                raise ValueError(f"{jid}: release requires blocked state")
            job = item["job"]
            config = Path(job["original_config"])
            old_source = Path(job["original_source_manifest"])
            if (sha(config) != job["original_config_sha256"] or
                    sha(old_source) != job["original_source_manifest_sha256"]):
                raise ValueError(f"{jid}: original config/source SHA differs")
            original = read(config)["job"]
            ref = expected.get(job["recipe_id"])
            if (original.get("job_id") != f"{job['recipe_id']}_s401" or
                    original.get("seed") != 401 or
                    not isinstance(ref, dict) or
                    ref.get("original_job_id") != original["job_id"] or
                    ref.get("config_sha256") != job["original_config_sha256"] or
                    ref.get("source_manifest_sha256") != job["original_source_manifest_sha256"] or
                    ref.get("budget_stop_round") != job["expected_budget_return_round"] or
                    ref.get("deployed_bytes") != job["expected_deployed_bytes"]):
                raise ValueError(f"{jid}: original endpoint identity differs")
            out = queue.parent / "COST1" / jid
            if out.is_symlink() or (out.exists() and any(out.iterdir())):
                raise ValueError(f"{jid}: COST1 output is not fresh")
            changes.append((path, dict(item, state="pending", reason=None,
                config=str(config), config_sha256=job["original_config_sha256"],
                old_source_manifest=str(old_source), old_source_manifest_sha256=job["original_source_manifest_sha256"],
                expected_identities=str(expected_path), expected_identities_sha256=sha(expected_path),
                source=str(source), source_manifest_sha256=source_sha,
                gate=str(gate_path), gate_sha256=gate_sha256,
                out=str(out), released_at=time.time())))
        for path, item in changes:
            write(path, item)
    return ids


def _verify_released(item: dict) -> None:
    if (sha(item["config"]) != item["config_sha256"] or
            sha(item["old_source_manifest"]) != item["old_source_manifest_sha256"] or
            sha(item["expected_identities"]) != item["expected_identities_sha256"] or
            sha(item["gate"]) != item["gate_sha256"] or
            verify_source(Path(item["source"])) != item["source_manifest_sha256"]):
        raise RuntimeError("released COST1 input mutated")
    _checked_gate(Path(item["gate"]), item["gate_sha256"], item["source_manifest_sha256"])


def claim(queue: Path, lane: str, worker_id: str, gpu_uuid: str) -> tuple[Path, dict] | None:
    queue, _, jobs = checked_queue(queue)
    if lane not in {"pair", "general"} or not gpu_uuid.startswith("GPU-"):
        raise ValueError("explicit lane and physical GPU UUID are required")
    with lock(queue):
        pair_state = read(queue / "pair_lane.json")
        pair_gpu = pair_state.get("gpu_uuid")
        if any(item["state"] == "running" and item.get("gpu_uuid") == gpu_uuid
               for _, item in (_item(queue, jid, jobs) for jid in jobs)):
            return None
        if lane == "general" and pair_gpu == gpu_uuid and (
                _item(queue, PAIR[0], jobs)[1]["state"] != "failed_needs_review" and
                _item(queue, PAIR[1], jobs)[1]["state"] not in
                {"executed_pending_audit", "failed_needs_review"}):
            return None
        if lane == "pair":
            first = _item(queue, PAIR[0], jobs)[1]
            second = _item(queue, PAIR[1], jobs)[1]
            if first["state"] == "pending" and pair_gpu is None:
                chosen = PAIR[0]
            elif (first["state"] == "executed_pending_audit" and
                  second["state"] == "pending" and pair_gpu == gpu_uuid):
                chosen = PAIR[1]
            else:
                return None
            path, item = _item(queue, chosen, jobs)
        else:
            candidates = [_item(queue, jid, jobs) for jid in jobs if jid not in PAIR]
            candidates = [(p, v) for p, v in candidates if v["state"] == "pending"]
            candidates.sort(key=lambda pair: (pair[1]["job"]["priority"], pair[0].stem))
            if not candidates:
                return None
            path, item = candidates[0]
        _verify_released(item)
        if Path(item["out"]).exists() and any(Path(item["out"]).iterdir()):
            raise RuntimeError("COST1 output contains files before first attempt")
        now = time.time()
        item.update(state="running", worker=worker_id, gpu_uuid=gpu_uuid, started_at=now)
        item["attempts"].append({"worker": worker_id, "gpu_uuid": gpu_uuid,
            "started_at": now, "slurm_job": os.getenv("SLURM_JOB_ID"),
            "slurm_step": os.getenv("SLURM_STEP_ID")})
        if lane == "pair" and chosen == PAIR[0]:
            pair_state.update(gpu_uuid=gpu_uuid, d011_claimed_at=now)
            write(queue / "pair_lane.json", pair_state)
        write(path, item)
        return path, item


def finish(queue: Path, path: Path, worker_id: str, exit_code: int) -> str:
    queue, _, jobs = checked_queue(queue)
    with lock(queue):
        verified, item = _item(queue, path.stem, jobs)
        if verified != path or item["state"] != "running" or item.get("worker") != worker_id:
            raise RuntimeError("lost COST1 job ownership")
        state = "executed_pending_audit" if exit_code == 0 else "failed_needs_review"
        now = time.time()
        item.update(state=state, finished_at=now, exit_code=exit_code)
        item["attempts"][-1].update(finished_at=now, exit_code=exit_code,
            raw_output=item["out"],
            worker_log=str(queue / "attempt_logs" / path.stem / "worker.log"))
        write(path, item)
        return state


def argv_for(item: dict, python: str) -> list[str]:
    job = item["job"]
    native = job["original_config"].split("/")[-2].startswith("native")
    script = "run_pcla_v7_native.py" if native else "run_pcla_v7.py"
    argv = [python, str(Path(item["source"]) / "scripts" / script),
            "--config", item["config"], "--out", item["out"], "--device", "cuda",
            "--cost-prefix", "--instrumented-source-manifest",
            str(Path(item["source"]) / "SOURCE_MANIFEST.json")]
    if not native:
        argv += ["--original-source-manifest", item["old_source_manifest"],
                 "--expected-cost-identities", item["expected_identities"]]
    return argv


def _gpu_uuid() -> str:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("COST1 worker requires exactly one visible GPU")
    result = subprocess.run(["nvidia-smi", "-i", visible, "--query-gpu=uuid",
                             "--format=csv,noheader"], capture_output=True, text=True,
                            check=True)
    values = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise RuntimeError("cannot verify one physical GPU UUID")
    return values[0]


def worker(queue: Path, lane: str, python: str = sys.executable,
           idle_seconds: int = 900) -> None:
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("COST1 worker requires an existing Slurm GPU step")
    uuid = _gpu_uuid()
    wid = f"{socket.gethostname()}:{os.getpid()}:{os.getenv('SLURM_JOB_ID')}.{os.getenv('SLURM_STEP_ID')}"
    idle_since = time.monotonic()
    while True:
        result = claim(queue, lane, wid, uuid)
        if result is None:
            if time.monotonic() - idle_since >= idle_seconds:
                return
            time.sleep(3)
            continue
        idle_since = time.monotonic()
        path, item = result
        out = Path(item["out"])
        attempt_dir = queue / "attempt_logs" / path.stem
        try:
            out.mkdir(parents=True, exist_ok=True)
            attempt_dir.mkdir(parents=True, exist_ok=False)
            argv = argv_for(item, python)
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       PCLA_GPU_SHARING="exclusive")
            write(attempt_dir / "dispatch_identity.json", {"job_id": item["job"]["replay_id"],
                  "worker": wid, "gpu_uuid": uuid, "argv": argv,
                  "plan_sha256": read(queue / "matrix_identity.json")["plan_sha256"],
                  "config_sha256": item["config_sha256"],
                  "old_source_manifest_sha256": item["old_source_manifest_sha256"],
                  "new_source_manifest_sha256": item["source_manifest_sha256"],
                  "expected_identities_sha256": item["expected_identities_sha256"],
                  "gate_sha256": item["gate_sha256"]})
            with (attempt_dir / "worker.log").open("ab") as log:
                code = subprocess.run(argv, cwd=item["source"], env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
        except BaseException as exc:
            out.mkdir(parents=True, exist_ok=True)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            with (attempt_dir / "worker.log").open("ab") as log:
                log.write(f"\nCOST1 worker launch failure: {exc!r}\n".encode())
            code = 127
        state = finish(queue, path, wid, code)
        print(json.dumps({"job": path.stem, "state": state, "exit": code}), flush=True)


def status(queue: Path) -> dict:
    queue, _, jobs = checked_queue(queue)
    with lock(queue):
        items = [_item(queue, jid, jobs)[1] for jid in sorted(jobs)]
        pair = read(queue / "pair_lane.json")
    return {"phase": "COST1", "count": len(items),
            "counts": dict(Counter(x["state"] for x in items)),
            "pair_gpu_uuid": pair.get("gpu_uuid"),
            "pair": {jid: next(x["state"] for x in items if x["job"]["replay_id"] == jid)
                     for jid in PAIR}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--plan", type=Path, required=True)
    r = sub.add_parser("release")
    r.add_argument("--ids", nargs="+", required=True)
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--expected-identities", type=Path, required=True)
    r.add_argument("--gpu-gate", type=Path, required=True)
    r.add_argument("--gpu-gate-sha256", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--lane", choices=["pair", "general"], required=True)
    w.add_argument("--python", default=sys.executable)
    w.add_argument("--idle-seconds", type=int, default=900)
    sub.add_parser("status")
    a = p.parse_args()
    if a.command == "init":
        print(json.dumps({"initialized": initialize(a.queue, a.plan)}))
    elif a.command == "release":
        print(json.dumps({"released": release(a.queue, a.ids, a.source,
            a.expected_identities, a.gpu_gate, a.gpu_gate_sha256)}))
    elif a.command == "worker":
        worker(a.queue, a.lane, a.python, a.idle_seconds)
    else:
        print(json.dumps(status(a.queue), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
