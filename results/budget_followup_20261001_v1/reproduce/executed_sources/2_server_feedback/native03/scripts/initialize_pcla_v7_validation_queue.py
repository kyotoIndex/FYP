#!/usr/bin/env python3
"""Initialize only the blocked, independent PCLA v7 validation queue.

The 13 fixed slots (156 jobs) come from the reviewed draft manifest. The three
unresolved slots remain dependencies, not runnable jobs. This script has no
release, claim, worker, or training command.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time


PROTOCOL = "pcla-global-budget-20260928-v7"
SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
FIXED_SLOTS = (1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
UNRESOLVED_STATES = {
    "3": "unresolved_cost_tie",
    "4": "depends_on_slot_03",
    "16": "depends_on_same_seed_slot_03_donor",
}
HARDWARE_FAMILIES = frozenset({"4090D", "A100"})
JOB_ID = re.compile(r"V_S(\d{2})_(D\d{3})_s(\d+)")
QUEUE_NAME = "validation_queue"


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temp.open("w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def locked(queue: Path):
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "queue.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def reject_d0_path(path: Path) -> None:
    if "d0" in (part.lower() for part in path.parts):
        raise ValueError(f"D0 path is forbidden: {path}")


def validate_jobs(jobs: list[dict], run_root: Path) -> None:
    if len(jobs) != len(FIXED_SLOTS) * len(SEEDS):
        raise ValueError("expected exactly 156 fixed V jobs, not a D0 or 192-job matrix")
    seen = set()
    pairs = set()
    for job in jobs:
        jid = job.get("job_id", "")
        match = JOB_ID.fullmatch(jid)
        if not match:
            raise ValueError(f"invalid V job ID: {jid!r}")
        slot, recipe, seed = int(match[1]), match[2], int(match[3])
        if jid in seen or (slot, seed) in pairs:
            raise ValueError(f"duplicate V job or slot/seed: {jid}")
        seen.add(jid)
        pairs.add((slot, seed))
        if slot not in FIXED_SLOTS or seed not in SEEDS:
            raise ValueError(f"unreleased slot or unexpected V seed: {jid}")
        if (job.get("phase") != "V" or job.get("protocol") != PROTOCOL
                or job.get("budget_id") != "B8" or job.get("status") != "draft_not_dispatched"
                or job.get("recipe_id") != recipe or job.get("seed") != seed
                or job.get("hardware_block") != f"V_s{seed}"):
            raise ValueError(f"V job identity is inconsistent: {jid}")
        if job.get("depends_on"):
            raise ValueError(f"fixed V slot unexpectedly depends on another job: {jid}")
        output = job.get("output_dir", "")
        target = PurePosixPath(output)
        expected = PurePosixPath("runs") / PROTOCOL / "V" / jid
        if target != expected or target.is_absolute() or "D0" in target.parts:
            raise ValueError(f"V output is not isolated: {jid}: {output}")
    if pairs != {(slot, seed) for slot in FIXED_SLOTS for seed in SEEDS}:
        raise ValueError("fixed V slot/seed matrix has a gap")


def validate_unresolved(slots: dict) -> None:
    if set(slots) != set(UNRESOLVED_STATES):
        raise ValueError("exactly V slots 3, 4, and 16 must remain unresolved")
    for slot, state in UNRESOLVED_STATES.items():
        if slots[slot].get("state") != state:
            raise ValueError(f"V slot {slot} dependency state differs")
        if slots[slot].get("recipe") is not None:
            raise ValueError(f"V slot {slot} cannot have a resolved recipe")
    if set(slots["3"].get("choices", [])) != {"D011", "D035"}:
        raise ValueError("V slot 3 cost-tie choices differ")
    if slots["4"].get("choices_by_slot_03") != {"D011": "D015", "D035": "D042"}:
        raise ValueError("V slot 4 must depend on the slot 3 representation")


def validate_pool_map(pool_map: dict) -> None:
    expected = {f"V_s{seed}" for seed in SEEDS}
    if set(pool_map) != expected:
        raise ValueError("hardware family map must explicitly cover all 12 V seed blocks")
    if any(pool not in HARDWARE_FAMILIES for pool in pool_map.values()):
        raise ValueError("unknown V hardware family")


def initialize(run_root: Path, manifest: Path, unresolved: Path,
               pool_map_path: Path, *, queue: Path | None = None) -> dict:
    run_root = run_root.resolve()
    queue = queue or run_root / QUEUE_NAME
    queue = Path(queue)
    for path in (run_root, manifest, unresolved, pool_map_path, queue):
        reject_d0_path(path)
    if queue.resolve() != (run_root / QUEUE_NAME).resolve() or queue.is_symlink():
        raise ValueError("V queue must be the independent run_root/validation_queue directory")
    if queue.exists() and not queue.is_dir():
        raise ValueError("V queue path is not a directory")
    if (run_root / "V").is_symlink():
        raise ValueError("V output directory cannot be a symlink")
    jobs = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    slots = read(unresolved)
    pool_map = read(pool_map_path)
    validate_jobs(jobs, run_root)
    validate_unresolved(slots)
    validate_pool_map(pool_map)
    identity = {
        "schema": "pcla-v7-validation-queue-1",
        "protocol": PROTOCOL,
        "phase": "V",
        "count": len(jobs),
        "fixed_slots": list(FIXED_SLOTS),
        "seeds": list(SEEDS),
        "manifest_sha256": digest(manifest),
        "unresolved_slots_sha256": digest(unresolved),
        "unresolved_slots": slots,
        "pool_map_sha256": digest(pool_map_path),
        "pool_map": pool_map,
        "release_state": "blocked; review and configuration gates require explicit release",
    }
    by_id = {job["job_id"]: job for job in jobs}
    identity_path = queue / "matrix_identity.json"
    # Reject an obvious foreign matrix before even creating queue.lock there.
    # Recheck under the lock below to handle concurrent initializers.
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("existing V matrix identity differs")
    with locked(queue):
        job_dir = queue / "jobs"
        if identity_path.exists():
            if read(identity_path) != identity:
                raise ValueError("existing V matrix identity differs")
            found = {path.name for path in job_dir.iterdir()}
            if found != {f"{jid}.json" for jid in by_id}:
                raise ValueError("existing V queue job set differs")
            for jid, job in by_id.items():
                item = read(job_dir / f"{jid}.json")
                if (item.get("job") != job
                        or item.get("pool") != pool_map[job["hardware_block"]]
                        or item.get("required_release_gates") != ["review", "configuration", "source"]):
                    raise ValueError(f"existing V queue job identity differs: {jid}")
            return identity
        if job_dir.exists() and any(job_dir.iterdir()):
            raise ValueError("partial or foreign queue exists without V matrix identity")
        if any(path.name not in {"queue.lock", "jobs"} for path in queue.iterdir()):
            raise ValueError("foreign queue files exist without V matrix identity")
        job_dir.mkdir(exist_ok=True)
        for jid, job in by_id.items():
            write(job_dir / f"{jid}.json", {
                "job": job,
                "pool": pool_map[job["hardware_block"]],
                "state": "blocked",
                "reason": "validation_review_and_configuration_not_released",
                "required_release_gates": ["review", "configuration", "source"],
                "attempts": [],
                "created_at": time.time(),
            })
        write(identity_path, identity)
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--pool-map", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize(args.run_root, args.manifest, args.unresolved,
                                args.pool_map), sort_keys=True))


if __name__ == "__main__":
    main()
