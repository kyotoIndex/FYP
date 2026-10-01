#!/usr/bin/env python3
"""Plan or launch exclusive V workers in four existing Slurm allocations.

Default is a live, read-only dry-run. --execute starts tmux-managed srun steps;
this program never submits, cancels, or enlarges an allocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dispatch_pcla_v7 import read, write
from scripts.dispatch_pcla_v7_validation import checked_queue

EXPECTED = {
    "173697": ("4090D", "4090", 4),
    "173698": ("4090D", "4090", 4),
    "172169": ("4090D", "4090", 2),
    "157325": ("A100", "a100", 2),
}


def command(argv: list[str], *, ok=(0,)) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, text=True, capture_output=True)
    if result.returncode not in ok:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return result


def fields(raw: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in raw.split() if "=" in token)


def allocations(path: Path) -> dict:
    value = read(path)
    if value.get("schema") != "pcla-v7-existing-allocations-1":
        raise ValueError("allocation map schema differs")
    requested = value.get("allocations")
    expected = {jid: {"hardware_family": family, "gpu_type": gpu_type, "count": count}
                for jid, (family, gpu_type, count) in EXPECTED.items()}
    if requested != expected:
        raise ValueError("allocation map must explicitly name the fixed four allocations and 12 GPUs")
    return requested


def parse_gpu_csv(raw: str, gpu_type: str) -> list[dict]:
    cards = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit() or not parts[3].removesuffix(" MiB").isdigit() or not parts[4].removesuffix(" %").isdigit():
            raise ValueError("unparseable nvidia-smi per-card evidence")
        index, uuid, name, memory, utilization = parts
        if gpu_type.lower() in name.lower():
            cards.append({"index": int(index), "uuid": uuid, "name": name,
                          "memory_mib": int(memory.removesuffix(" MiB")),
                          "utilization_pct": int(utilization.removesuffix(" %"))})
    return cards


def parse_indices(raw: str, count: int) -> list[int]:
    if not re.fullmatch(r"[0-9,\-]+", raw):
        raise RuntimeError("invalid Slurm GPU index list")
    indices = []
    for part in raw.split(","):
        ends = part.split("-")
        if len(ends) == 1:
            indices.append(int(ends[0]))
        elif len(ends) == 2 and int(ends[0]) <= int(ends[1]):
            indices.extend(range(int(ends[0]), int(ends[1]) + 1))
        else:
            raise RuntimeError("invalid Slurm GRES device range")
    if len(indices) != count or len(set(indices)) != count:
        raise RuntimeError("Slurm GRES device count differs")
    return sorted(indices)


def gres_indices(value: str, gpu_type: str, count: int) -> list[int]:
    match = re.fullmatch(rf"gpu:{re.escape(gpu_type)}:{count}\(IDX:([0-9,\-]+)\)", value)
    if not match:
        raise RuntimeError("Slurm GRES device indices are unavailable or differ")
    return parse_indices(match[1], count)


class LiveProbe:
    def __init__(self, run=command):
        self.run = run

    def allocation(self, jid: str, spec: dict) -> dict:
        live = self.run(["scontrol", "show", "job", "-dd", "-o", jid]).stdout
        parsed = fields(live)
        owner = parsed.get("UserId", "").split("(", 1)[0]
        tres = parsed.get("AllocTRES", parsed.get("TRES", ""))
        gpu_count = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        typed = parsed.get("TresPerNode", "").lower()
        node = parsed.get("NodeList")
        if (parsed.get("JobId") != jid or parsed.get("JobState") != "RUNNING"
                or owner != getpass.getuser() or not gpu_count
                or int(gpu_count[1]) != spec["count"]
                or f"gres:gpu:{spec['gpu_type']}:{spec['count']}" not in typed
                or not node or "[" in node or "," in node):
            raise RuntimeError(f"{jid}: live allocation identity, owner, type, count or node differs")
        record = {"job_id": jid, "node": node, "owner": owner, "job_state": "RUNNING",
                  "hardware_family": spec["hardware_family"], "gpu_type": spec["gpu_type"],
                  "gpu_count": spec["count"], "scontrol": live}
        record["gpu_indices"] = gres_indices(parsed.get("GRES", ""), spec["gpu_type"], spec["count"])
        return record

    def steps(self, jid: str) -> list[dict]:
        ids = self.run(["squeue", "--steps", "-j", jid, "-h", "-o", "%i"]).stdout.splitlines()
        found = []
        for step in ids:
            step = step.strip()
            if step in {f"{jid}.batch", f"{jid}.extern"}:
                continue
            if not re.fullmatch(re.escape(jid) + r"\.\d+", step):
                raise RuntimeError(f"{jid}: unknown Slurm step identity {step!r}")
            info = self.run(["scontrol", "show", "step", "-o", step]).stdout
            found.append({"step_id": step, "scontrol": info, "fields": fields(info)})
        return found

    def cards(self, node: str, gpu_type: str, allocated_indices: list[int],
              fallback_allocation: str) -> dict:
        prefix = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node]
        try:
            gpu = self.run(prefix + ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                                     "--format=csv,noheader"]).stdout
            apps = self.run(prefix + ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                      "--format=csv,noheader"]).stdout
            all_cards = parse_gpu_csv(gpu, gpu_type)
            cards = [card for card in all_cards if card["index"] in allocated_indices]
            evidence_source = "node_ssh"
            step_gpus = None
        except RuntimeError:
            # Some nodes reject SSH from the login host. This temporary, exact
            # step sees only the allocation's GPUs and runs read-only queries.
            query = ("import json,os,subprocess; "
                     "q=lambda x:subprocess.check_output(x,text=True); "
                     "print(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),"
                     "'node':os.environ.get('SLURMD_NODENAME'),"
                     "'step_gpus':os.environ.get('SLURM_STEP_GPUS'),"
                     "'gpu':q(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader']),"
                     "'apps':q(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])}))")
            result = self.run(["srun", f"--jobid={fallback_allocation}", "--nodes=1", "--exact",
                               "-n1", "-c1", f"--gres=gpu:{gpu_type}:{len(allocated_indices)}",
                               "--time=00:02:00", sys.executable, "-c", query])
            response = json.loads(result.stdout)
            step_gpus = parse_indices(response.get("step_gpus") or "", len(allocated_indices))
            if (response.get("job") != fallback_allocation or response.get("node") != node
                    or step_gpus != sorted(allocated_indices)):
                raise RuntimeError(f"{node}: read-only Slurm probe was not bound to expected GPUs")
            gpu, apps = response["gpu"], response["apps"]
            cards = parse_gpu_csv(gpu, gpu_type)
            evidence_source = "exact_srun_probe"
        if len(cards) != len(allocated_indices):
            raise RuntimeError(f"{node}: Slurm GRES indices do not match nvidia-smi cards")
        uuids = {card["uuid"] for card in cards}
        active = [line.strip() for line in apps.splitlines() if line.strip()
                  and line.split(",", 1)[0].strip() in uuids]
        if active or any(card["memory_mib"] > 512 or card["utilization_pct"] > 0 for card in cards):
            raise RuntimeError(f"{node}: requested GPU family has live compute or occupancy")
        return {"node": node, "gpu_type": gpu_type, "cards": cards, "compute_apps": active,
                "raw_gpu": gpu, "raw_compute_apps": apps,
                "evidence_source": evidence_source, "slurm_step_gpus": step_gpus}

    def sessions(self) -> list[str]:
        result = self.run(["tmux", "list-sessions", "-F", "#{session_name}"], ok=(0, 1))
        missing_server = any(term in result.stderr.lower() for term in
                             ("no server running", "failed to connect", "error connecting", "no such file"))
        if result.returncode == 1 and not missing_server:
            raise RuntimeError("tmux session inventory failed")
        return result.stdout.splitlines()


def build_plan(queue: Path, dispatcher: Path, python: Path, allocation_map: Path,
               generation: str, probe: LiveProbe) -> dict:
    queue, identity = checked_queue(queue)
    if not re.fullmatch(r"[a-zA-Z0-9_]{1,32}", generation):
        raise ValueError("generation must be 1-32 alphanumeric/underscore characters")
    if dispatcher.name != "dispatch_pcla_v7_validation.py" or not dispatcher.is_file():
        raise ValueError("explicit V dispatcher path is required")
    if not python.is_absolute() or not python.is_file():
        raise ValueError("explicit absolute Python interpreter is required")
    specs = allocations(allocation_map)
    sessions = probe.sessions()
    if any(name.startswith("pclav7_v_") for name in sessions):
        raise RuntimeError("existing V tmux worker session detected")
    records = {}
    occupancy = {}
    for jid, spec in specs.items():
        record = probe.allocation(jid, spec)
        steps = probe.steps(jid)
        if steps:
            raise RuntimeError(f"{jid}: existing Slurm step; training or worker ownership is uncertain")
        records[jid] = record
    for jid, spec in specs.items():
        record = records[jid]
        key = (record["node"], spec["gpu_type"])
        indices = set(record["gpu_indices"])
        for other_jid, other_spec in specs.items():
            if other_jid == jid or other_jid not in records:
                continue
            other = records[other_jid]
            if (other["node"], other_spec["gpu_type"]) == key and indices.intersection(other["gpu_indices"]):
                raise RuntimeError(f"{record['node']}: allocations overlap on Slurm GRES indices")
        if key not in occupancy:
            combined = sorted({i for other_jid, other_spec in specs.items()
                               if (records[other_jid]["node"], other_spec["gpu_type"]) == key
                               for i in records[other_jid]["gpu_indices"]})
            owners = [other_jid for other_jid, other_spec in specs.items()
                      if (records[other_jid]["node"], other_spec["gpu_type"]) == key]
            if len(owners) != 1:
                raise RuntimeError(f"{record['node']}: shared-node GPU probe needs explicit allocation ownership")
            occupancy[key] = probe.cards(*key, combined, owners[0])
    for (node, gpu_type), evidence in occupancy.items():
        required = sum(spec["count"] for jid, spec in specs.items()
                       if records[jid]["node"] == node and spec["gpu_type"] == gpu_type)
        if len(evidence["cards"]) < required:
            raise RuntimeError(f"{node}: fewer idle {gpu_type} cards than planned workers")
    worker_root = queue.parent / "v_workers"
    if (worker_root / f"launch_{generation}.json").exists():
        raise RuntimeError("V launch generation already recorded")
    workers = []
    for jid, spec in specs.items():
        for ordinal in range(spec["count"]):
            session = f"pclav7_v_{generation}_{jid}_{ordinal}"
            identity_path = worker_root / f"{session}.slurm.json"
            log = worker_root / f"{session}.log"
            if identity_path.exists() or log.exists():
                raise RuntimeError(f"{session}: prior worker evidence exists")
            argv = ["srun", f"--jobid={jid}", "--nodes=1", "--exact", "-n1", "-c4",
                    f"--gres=gpu:{spec['gpu_type']}:1", "--cpu-bind=cores",
                    str(python), str(Path(__file__).resolve()), "entry",
                    "--allocation", jid, "--identity", str(identity_path),
                    "--dispatcher", str(dispatcher.resolve()), "--queue", str(queue),
                    "--hardware-family", spec["hardware_family"], "--python", str(python)]
            workers.append({"session": session, "allocation": jid, "ordinal": ordinal,
                            "node": records[jid]["node"], "hardware_family": spec["hardware_family"],
                            "gpu_type": spec["gpu_type"], "mode": "exclusive", "cpu_threads": 4,
                            "identity": str(identity_path), "log": str(log), "argv": argv})
    return {"schema": "pcla-v7-validation-launch-1", "phase": "V", "queue": str(queue),
            "dispatcher": str(dispatcher.resolve()), "python": str(python),
            "allocation_map": str(allocation_map.resolve()), "generation": generation,
            "queue_manifest_sha256": identity["manifest_sha256"],
            "allocations": records, "occupancy": list(occupancy.values()), "workers": workers}


@contextmanager
def launch_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launch.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def launch(queue: Path, dispatcher: Path, python: Path, allocation_map: Path,
           generation: str, *, execute: bool = False, probe: LiveProbe | None = None,
           run=command) -> dict:
    probe = probe or LiveProbe(run)
    worker_root = queue.resolve().parent / "v_workers"
    if not execute:
        return build_plan(queue, dispatcher, python, allocation_map, generation, probe)
    with launch_lock(worker_root):
        plan = build_plan(queue, dispatcher, python, allocation_map, generation, probe)
        receipt = worker_root / f"launch_{generation}.json"
        plan.update(state="launching", started_at=time.time(), launched=[])
        write(receipt, plan)  # Persist intent before starting the first worker.
        for worker in plan["workers"]:
            shell = shlex.join(worker["argv"]) + " > " + shlex.quote(worker["log"]) + " 2>&1"
            try:
                run(["tmux", "new-session", "-d", "-s", worker["session"], shell])
                if worker["session"] not in probe.sessions():
                    raise RuntimeError(f"{worker['session']}: tmux launch not observable")
            except Exception as exc:
                plan.update(state="partial_launch_needs_review", error=repr(exc))
                write(receipt, plan)
                raise
            plan["launched"].append({"session": worker["session"], "allocation": worker["allocation"],
                                     "state": "tmux_started_step_unverified", "time": time.time(),
                                     "slurm_identity": worker["identity"]})
            write(receipt, plan)
        plan.update(state="tmux_started_steps_need_identity_check", finished_at=time.time())
        write(receipt, plan)
        return plan


def worker_entry(allocation: str, identity: Path, dispatcher: Path, queue: Path,
                 hardware_family: str, python: Path) -> None:
    if (os.getenv("SLURM_JOB_ID") != allocation or not os.getenv("SLURM_STEP_ID")
            or not os.getenv("CUDA_VISIBLE_DEVICES") or not os.getenv("SLURM_JOB_NODELIST")):
        raise RuntimeError("V worker entry requires the specified Slurm GPU step")
    if identity.exists():
        raise RuntimeError("V worker Slurm identity already exists")
    identity.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema": "pcla-v7-validation-worker-slurm-1", "time": time.time(),
             "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_step_id": os.environ["SLURM_STEP_ID"],
             "slurm_node": os.environ["SLURM_JOB_NODELIST"],
             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
             "hardware_family": hardware_family, "mode": "exclusive", "queue": str(queue),
             "dispatcher": str(dispatcher), "python": str(python)}
    with identity.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.execv(str(python), [str(python), str(dispatcher), "--queue", str(queue), "worker",
                           "--hardware-family", hardware_family, "--mode", "exclusive",
                           "--python", str(python)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--dispatcher", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--allocation-map", type=Path, required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--execute", action="store_true")
    e = sub.add_parser("entry")
    e.add_argument("--allocation", required=True)
    e.add_argument("--identity", type=Path, required=True)
    e.add_argument("--dispatcher", type=Path, required=True)
    e.add_argument("--queue", type=Path, required=True)
    e.add_argument("--hardware-family", choices=["4090D", "A100"], required=True)
    e.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "entry":
        worker_entry(args.allocation, args.identity, args.dispatcher, args.queue,
                     args.hardware_family, args.python)
    else:
        result = launch(args.queue, args.dispatcher, args.python, args.allocation_map,
                        args.generation, execute=args.execute)
        print(json.dumps({"state": result.get("state", "dry_run"),
                          "workers": len(result["workers"]),
                          "allocations": list(result["allocations"]),
                          "plan": result if not args.execute else str(args.queue.resolve().parent / "v_workers" / f"launch_{args.generation}.json")},
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
