#!/usr/bin/env python3
"""Run one expanded PCLA v7 job in one process on one device.

Expanded JSON requires ``job``, ``manifest``, ``token_cache``, ``train_file``,
``model`` and ``runtime``. ``runtime.byte_cap`` is the frozen measured B8.
Only the explicitly integrated N methods are accepted by the runtime.
"""
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

from pcla_repair.core import BudgetLoRALinear, keyed_seed, tensor_digest
from pcla_repair.runtime import atomic_json, seed_train
from pcla_research_v7.data import TokenCache, digest
from pcla_research_v7.transport import b8_wire_manifest
from pcla_research_v7.runtime import run_experiment, SUPPORTED


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def prepare(expanded: dict, config_path: Path, device: torch.device):
    job = expanded["job"]
    runtime = expanded["runtime"]
    model_cfg = expanded["model"]
    seed = int(job["seed"])
    manifest_path = _path(expanded["manifest"], config_path)
    b8_path = _path(expanded["b8_wire"], config_path)
    b8 = json.loads(b8_path.read_text())
    expected_b8 = digest({k: v for k, v in b8.items() if k != "wire_manifest_sha256"})
    if b8.get("wire_manifest_sha256") != expected_b8:
        raise ValueError("B8 wire manifest digest differs")
    if b8["total_bytes"] != b8_wire_manifest(physically_encode=False)["total_bytes"]:
        raise ValueError("B8 codec schema differs from frozen measured budget")
    if int(runtime["byte_cap"]) != b8["total_bytes"]:
        raise ValueError("runtime byte cap differs from B8 wire manifest")
    source_manifest_path = _path(expanded.get("source_manifest", str(ROOT / "SOURCE_MANIFEST.json")), config_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    for relative, expected in source_manifest["files"].items():
        source_file = source_manifest_path.parent / relative
        if not source_file.is_file() or _sha(source_file) != expected:
            raise RuntimeError(f"frozen source identity differs: {relative}")
    cache_path = _path(expanded["token_cache"], config_path)
    train_path = _path(expanded["train_file"], config_path)
    if train_path.name != "train.tsv" or train_path.parent.name != "SST-2":
        raise ValueError("v7 first batch requires SST-2 train.tsv")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_sha256"] != _sha(train_path):
        raise ValueError("manifest/source hash mismatch")
    if str(seed) not in manifest["partitions"]:
        raise ValueError("job seed not present in frozen manifest")
    tokens = TokenCache(cache_path)
    if tokens.meta["identity"]["max_length"] != int(model_cfg.get("max_length", 64)):
        raise ValueError("token cache length differs from model protocol")
    from transformers import AutoModelForSequenceClassification
    model_name = model_cfg["repository"]
    revision = model_cfg.get("revision")
    local_only = bool(model_cfg.get("local_files_only", False))
    local_model = Path(model_name)
    if local_model.is_dir():
        for filename, expected in tokens.meta["identity"]["tokenizer_files"].items():
            item = local_model / filename
            if not item.is_file() or _sha(item) != expected:
                raise ValueError(f"model tokenizer differs from frozen token cache: {filename}")
    else:
        raise ValueError("v7 first-batch model must be a verified local directory")
    seed_train(keyed_seed(seed, "model_initialization"))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, num_labels=2, local_files_only=local_only,
        torch_dtype=torch.float32)
    pretrained_hash = tensor_digest(model.state_dict())
    rank = int(job.get("rank", 16))
    if rank not in {4, 8, 16}:
        raise ValueError("first-batch ranks are 4, 8, 16")
    targets = [(name, module) for name, module in model.named_modules()
               if name.endswith((".attention.self.query", ".attention.self.value"))
               and isinstance(module, torch.nn.Linear)]
    if len(targets) != 24:
        raise ValueError(f"expected 24 independent BERT Q/V projections, found {len(targets)}")
    for name, base in targets:
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        lora = BudgetLoRALinear(base, rank, float(rank), float(model_cfg.get("dropout", .1)))
        # Each projection draws the same 16-row primary matrix across ranks.
        generator = torch.Generator().manual_seed(keyed_seed(seed, "lora_initialization", name))
        primary = torch.empty(16, base.in_features, dtype=base.weight.dtype)
        bound = 1 / base.in_features ** .5
        primary.uniform_(-bound, bound, generator=generator)
        with torch.no_grad():
            lora.lora_a.copy_(primary[:rank])
            lora.lora_b.zero_()
        setattr(module, leaf, lora)
    clients = {cid: tokens.dataset(manifest, "client_train", "algorithm", seed, cid)
               for cid in range(50)}
    evaluation = tokens.dataset(manifest, "recipe_dev", "research_development")
    if len(evaluation) != 4096 or any(len(x) != 256 for x in clients.values()):
        raise ValueError("frozen v7 role sizes differ")
    sizes = runtime.get("capacity_by_tier", [96, 128, 192])
    capacities_list = [sizes[0]] * 30 + [sizes[1]] * 15 + [sizes[2]] * 5
    import random
    random.Random(keyed_seed(seed, "resource_assignment")).shuffle(capacities_list)
    capacities = dict(enumerate(capacities_list))
    model.to(device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hardware = {"name": props.name, "total_memory": props.total_memory,
                    "uuid": str(getattr(props, "uuid", "unavailable")),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    else:
        hardware = {"name": "cpu"}
    sources = {str(p.relative_to(ROOT)): _sha(p)
               for p in [ROOT / "pcla_research_v7" / "runtime.py",
                         ROOT / "pcla_research_v7" / "data.py",
                         ROOT / "pcla_research_v7" / "transport.py",
                         ROOT / "pcla_research_v7" / "selectors.py",
                         ROOT / "pcla_repair" / "core.py", Path(__file__)]}
    provenance = {"evaluation_role": "recipe_dev", "manifest_sha256": _sha(manifest_path),
                  "split_sha256": manifest["split_sha256"],
                  "source_sha256": manifest["source_sha256"],
                  "token_cache_sha256": tokens.meta["cache_sha256"],
                  "token_cache_identity_sha256": tokens.meta["identity_sha256"],
                  "pretrained_model_sha256": pretrained_hash,
                  "config_sha256": _sha(config_path), "sources": sources,
                  "source_manifest_sha256": _sha(source_manifest_path),
                  "b8_wire_manifest_sha256": _sha(b8_path),
                  "hardware": hardware, "torch": str(torch.__version__),
                  "transformers": __import__("transformers").__version__,
                  "python": platform.python_version()}
    return model, clients, capacities, evaluation, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config_path = args.config.resolve()
    expanded = json.loads(config_path.read_text())
    job = expanded["job"]
    if expanded.get("recipe", {}).get("required_adapter", "core") != "core":
        from scripts.run_pcla_v7_native import main as native_main
        return native_main()
    if "technical_stop_after" in expanded:
        if job.get("phase") != "TECH" or not expanded.get("technical_profile_only"):
            raise ValueError("Technical stop may only be used for a TECH job")
        if args.stop_after is None:
            args.stop_after = int(expanded["technical_stop_after"])
        if args.stop_after < 1:
            raise ValueError("Technical stop must be positive")
    rep = job.get("representation", "N")
    method = job["method"]
    valid_pair = (rep == "N" or
                  (rep in {"D", "S", "T"} and method in {"local", "cyclic"}) or
                  (rep == "S" and method in {"layer", "energy"}))
    if method not in SUPPORTED or not valid_pair:
        raise NotImplementedError(f"unsupported first-stage path {job['method']}/{job.get('representation')}")
    if args.out.exists() and (args.out / "latest.pt").exists() and not args.resume:
        raise ValueError("existing run requires --resume")
    if args.resume and not (args.out / "latest.pt").exists():
        raise ValueError("--resume requires latest.pt")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("run lock exists; inspect owner before retrying") from exc
    os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node()}).encode())
    os.close(fd)
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model, clients, capacities, evaluation, provenance = prepare(
            expanded, config_path, torch.device(args.device))
        rows = run_experiment(model, clients, evaluation, capacities,
                              job, expanded["runtime"], args.out, provenance,
                              stop_after=args.stop_after)
        atomic_json(args.out / "run_identity.json", {"job": job, "provenance": provenance})
        print(json.dumps({"job_id": job.get("job_id"), "rounds": len(rows),
                          "latest": str(args.out / "latest.pt")}), flush=True)
    except BaseException:
        (args.out / "exception.txt").write_text(traceback.format_exc())
        atomic_json(args.out / "status.json", {"state": "failed",
            "exception": traceback.format_exc().splitlines()[-1],
            "latest_checkpoint": str(args.out / "latest.pt") if (args.out / "latest.pt").exists() else None})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
