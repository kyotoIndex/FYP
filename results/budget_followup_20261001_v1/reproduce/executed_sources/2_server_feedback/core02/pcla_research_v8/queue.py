"""Locked, append-preserving local queue for one v8 trajectory per GPU worker."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

STATES = {"blocked", "pending", "running", "executed_pending_audit", "completed", "failed"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _jobs(root):
    return [(p, read(p)) for p in sorted((root / "jobs").glob("*.json"))]


def initialize(root: Path, config_dir: Path):
    configs = sorted(config_dir.glob("DEV_*.json"))
    if len(configs) != 128:
        raise ValueError("queue requires complete 128-config DEV matrix")
    with locked(root):
        existing = {p.stem: value for p, value in _jobs(root)}
        if existing and set(existing) != {p.stem for p in configs}:
            raise ValueError("existing queue matrix identity differs")
        for config_path in configs:
            config = read(config_path)
            jid = config["job"]["job_id"]
            if jid != config_path.stem or config["job"]["phase"] != "DEV":
                raise ValueError("DEV config path/identity differs")
            old = existing.get(jid)
            if old:
                if old["config_sha256"] != sha(config_path):
                    raise ValueError(f"existing queue config changed: {jid}")
                continue
            write(root / "jobs" / f"{jid}.json", {
                "job_id": jid, "recipe_id": config["job"]["recipe_id"],
                "family": config["recipe"]["family"], "seed": config["job"]["seed"],
                "hardware_family": "RTX4090D", "state": "blocked", "reason": "family_gate_not_released",
                "config": str(config_path), "config_sha256": sha(config_path),
                "output": config["job"]["output_dir"], "attempts": [],
                "created_at": time.time(),
            })


def _verify_binding(config):
    binding = config["binding"]
    if binding["status"] != "bound_unverified":
        raise ValueError("config has no bound role/model/byte inputs")
    for field, key in (("manifest", "data_manifest_sha256"), ("b8_wire", "b8_wire_sha256"),
                       ("token_cache", "token_cache_sha256"), ("token_cache_metadata", "token_cache_metadata_sha256"),
                       ("source_manifest", "source_manifest_sha256")):
        path = Path(config[field])
        if not path.is_file() or sha(path) != binding[key]:
            raise ValueError(f"bound input changed: {field}")
    if not Path(config["model"]["repository"]).is_dir():
        raise ValueError("bound model absent")
    source_root = Path(config["source_manifest"]).parent
    for relative, expected in read(Path(config["source_manifest"]))["files"].items():
        source_file = source_root / relative
        if not source_file.is_file() or sha(source_file) != expected:
            raise ValueError(f"source closure differs: {relative}")
    if not Path(config["train_file"]).is_file():
        raise ValueError("bound SST-2 source absent")
    from .configuration import validate_data_manifest
    validate_data_manifest(Path(config["manifest"]), seed=int(config["job"]["seed"]),
                           train_file=Path(config["train_file"]), expected_hash=binding["data_manifest_sha256"])


def release_family(root: Path, family: str, gate_receipt: Path):
    gate = read(gate_receipt)
    if gate.get("accepted") is not True or gate.get("family") != family or not gate.get("checks"):
        raise ValueError("independent family gate receipt not accepted")
    with locked(root):
        changes = []
        for path, value in _jobs(root):
            if value["family"] != family:
                continue
            if value["state"] == "pending":
                continue
            if value["state"] != "blocked":
                raise ValueError(f"cannot release {value['job_id']} from {value['state']}")
            if sha(Path(value["config"])) != value["config_sha256"]:
                raise ValueError("config changed after enqueue")
            _verify_binding(read(value["config"]))
            value.update(state="pending", reason=None, gate_receipt=str(gate_receipt),
                         gate_receipt_sha256=sha(gate_receipt), released_at=time.time())
            changes.append((path, value))
        if not changes:
            raise ValueError("no blocked jobs for family")
        for path, value in changes:
            write(path, value)
    return len(changes)


def claim(root: Path, *, worker: str, gpu_uuid: str, slurm_job: str, slurm_step: str,
          cpu_cores: tuple[int, ...], cpu_limit: int = 4):
    if not gpu_uuid or not slurm_job or not slurm_step or not cpu_cores or len(cpu_cores) > cpu_limit:
        raise ValueError("claim requires GPU UUID, Slurm job/step and bounded CPU identity")
    with locked(root):
        records = _jobs(root)
        running = [v for _, v in records if v["state"] == "running"]
        if any(v["claim"]["gpu_uuid"] == gpu_uuid for v in running):
            return None  # one process per physical GPU, including cost seed 503
        if any(v["claim"]["slurm_job"] == slurm_job and v["claim"]["slurm_step"] == slurm_step for v in running):
            raise RuntimeError("Slurm step already owns a running job")
        if any(set(v["claim"]["cpu_cores"]) & set(cpu_cores) and v["claim"]["host"] == socket.gethostname() for v in running):
            return None
        pending = [(p, v) for p, v in records if v["state"] == "pending"]
        # Round-robin across recipes before second seed, while honoring seed-503 cost blocks.
        completed_counts = {rid: sum(v["recipe_id"] == rid and v["state"] in {"running", "executed_pending_audit", "completed"}
                                     for _, v in records) for rid in {v["recipe_id"] for _, v in records}}
        pending.sort(key=lambda pv: (completed_counts[pv[1]["recipe_id"]],
                                     0 if pv[1]["seed"] == 503 else 1,
                                     pv[1]["recipe_id"], pv[1]["seed"]))
        if not pending:
            return None
        path, item = pending[0]
        if sha(Path(item["config"])) != item["config_sha256"]:
            raise ValueError("released config mutated")
        _verify_binding(read(item["config"]))
        token = uuid.uuid4().hex
        claim_data = {"worker": worker, "token": token, "gpu_uuid": gpu_uuid,
                      "slurm_job": slurm_job, "slurm_step": slurm_step,
                      "cpu_cores": list(cpu_cores), "host": socket.gethostname(),
                      "claimed_at": time.time()}
        item.update(state="running", claim=claim_data, reason=None)
        item["attempts"].append(dict(claim_data))
        write(path, item)
        return item


def finish(root: Path, job_id: str, token: str, *, return_code: int, result_path: Path | None = None):
    with locked(root):
        path = root / "jobs" / f"{job_id}.json"
        item = read(path)
        if item["state"] != "running" or item["claim"]["token"] != token:
            raise RuntimeError("lost queue claim ownership")
        result = read(result_path) if result_path and result_path.is_file() else None
        success = (return_code == 0 and result is not None and result.get("job_id") == job_id
                   and result.get("state") == "executed_pending_audit")
        item.update(state="executed_pending_audit" if success else "failed",
                    result_path=str(result_path) if result_path else None,
                    result_sha256=sha(result_path) if success else None,
                    return_code=return_code, finished_at=time.time())
        write(path, item)
        return item["state"]


def complete_audit(root: Path, job_id: str, receipt_path: Path):
    receipt = read(receipt_path)
    with locked(root):
        path = root / "jobs" / f"{job_id}.json"
        item = read(path)
        if item["state"] != "executed_pending_audit":
            raise ValueError("job is not awaiting independent audit")
        if (receipt.get("accepted") is not True or receipt.get("job_id") != job_id
                or receipt.get("result_sha256") != item["result_sha256"]
                or not receipt.get("auditor")):
            raise ValueError("audit receipt does not bind executed result")
        item.update(state="completed", audit_receipt=str(receipt_path),
                    audit_receipt_sha256=sha(receipt_path), audited_at=time.time())
        write(path, item)


def gpu_identity() -> str:
    if not os.getenv("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("worker requires exactly one visible GPU")
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA GPU")
    prop = torch.cuda.get_device_properties(0)
    if "4090" not in prop.name:
        raise RuntimeError(f"wrong hardware family: {prop.name}")
    gpu_uuid = getattr(prop, "uuid", None)
    if gpu_uuid and str(gpu_uuid).startswith("GPU-"):
        return str(gpu_uuid)
    # CUDA driver ordinal respects CUDA_VISIBLE_DEVICES, unlike nvidia-smi -i.
    import ctypes
    import uuid as uuid_module
    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0):
        raise RuntimeError("cannot resolve visible CUDA device")
    get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", None) or cuda.cuDeviceGetUuid
    if get_uuid(ctypes.byref(raw), device):
        raise RuntimeError("cannot resolve visible physical GPU UUID")
    return "GPU-" + str(uuid_module.UUID(bytes=bytes(raw)))
