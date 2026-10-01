#!/usr/bin/env python3
"""Run one expanded PCLA v7 job in one process on one device.

Expanded JSON requires ``job``, ``manifest``, ``token_cache``, ``train_file``,
``model`` and ``runtime``. ``runtime.byte_cap`` is the frozen measured B8.
Only the explicitly integrated N methods are accepted by the runtime.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from pcla_research_v7.execution_timing import (
    COST_LABEL_CATEGORIES, ExecutionTimingLedger, required_breakdown,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def verify_instrumented_source_manifest(path: Path) -> str:
    """Bind every replayed source byte to a separately frozen timing manifest."""
    record = json.loads(path.read_text())
    files = record.get("files")
    required = {
        "pcla_research_v7/runtime.py", "pcla_research_v7/cohort_runtime.py",
        "pcla_research_v7/coordinates.py", "pcla_research_v7/timing_hooks.py",
        "pcla_research_v7/execution_timing.py", "pcla_research_v7/data.py",
        "pcla_research_v7/selectors.py", "pcla_research_v7/transport.py",
        "pcla_repair/core.py", "pcla_repair/runtime.py", "scripts/run_pcla_v7.py",
    }
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("instrumented source manifest lacks the core replay closure")
    for relative, expected in files.items():
        item = Path(relative)
        if (item.is_absolute() or ".." in item.parts or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"invalid instrumented source entry: {relative!r}")
        source_file = ROOT / item
        if not source_file.is_file() or _sha(source_file) != expected:
            raise RuntimeError(f"instrumented source identity differs: {relative}")
    return _sha(path)


def persist_failed_cost_timing(out_dir: Path, timing: ExecutionTimingLedger,
                               failed_operation: str) -> dict:
    """Write raw partial cost evidence, never an accepted receipt."""
    partial = timing.partial_snapshot()
    atomic_json(out_dir / "timing_events_partial.json", partial)
    atomic_json(out_dir / "cost_timing_failure.json", {
        "schema": "pcla-v7-cost-timing-failure-1", "accepted": False,
        "event_file_sha256": _sha(out_dir / "timing_events_partial.json"),
        "open_intervals": partial["open_intervals"],
        "tail_unmeasured": partial["tail_unmeasured"],
        "capture_sample_error": partial["capture_sample_error"],
        "failed_operation": failed_operation,
    })
    return partial


def prepare(expanded: dict, config_path: Path, device: torch.device, *, timing=None):
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
    if timing is not None:
        timing.begin_envelope()
        initialization_token = timing.start("task_initialization")
    seed_train(keyed_seed(seed, "model_initialization"))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, num_labels=2, local_files_only=local_only,
        torch_dtype=torch.float32)
    if timing is not None:
        timing.stop(initialization_token)
    with timing.span("provenance_and_hashing") if timing is not None else nullcontext():
        pretrained_hash = tensor_digest(model.state_dict())
    if timing is not None:
        initialization_token = timing.start("task_initialization")
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
    if timing is not None:
        timing.stop(initialization_token)
        provenance_token = timing.start("provenance_and_hashing")
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
    if timing is not None:
        sources.update({str(p.relative_to(ROOT)): _sha(p) for p in [
            ROOT / "pcla_research_v7" / "execution_timing.py",
            ROOT / "pcla_research_v7" / "cohort_runtime.py",
            ROOT / "pcla_research_v7" / "coordinates.py",
            ROOT / "pcla_research_v7" / "timing_hooks.py"]})
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
    if timing is not None:
        timing.stop(provenance_token)
    return model, clients, capacities, evaluation, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--cost-prefix", action="store_true",
                        help="one fresh seed-401 B8 cost replay; writes an unaccepted timing record")
    parser.add_argument("--instrumented-source-manifest", type=Path,
                        help="frozen SHA-256 manifest for the actual instrumented replay source")
    parser.add_argument("--original-source-manifest", type=Path,
                        help="original D0 source manifest used by the unchanged config")
    parser.add_argument("--expected-cost-identities", type=Path,
                        help="Pro's frozen recipe-to-original-endpoint identities JSON")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config_path = args.config.resolve()
    expanded = json.loads(config_path.read_text())
    job = expanded["job"]
    cost_inputs = (args.instrumented_source_manifest, args.original_source_manifest,
                   args.expected_cost_identities)
    if args.cost_prefix != all(item is not None for item in cost_inputs) or (
            not args.cost_prefix and any(item is not None for item in cost_inputs)):
        raise ValueError("cost-prefix requires new/old source manifests and expected identities together")
    if args.cost_prefix and (int(job["seed"]) != 401 or args.resume or args.stop_after is not None):
        raise ValueError("cost-prefix requires seed 401, a fresh run, and no technical stop")
    if args.cost_prefix and (not args.device.startswith("cuda") or
                             int(os.environ.get("OMP_NUM_THREADS", "4")) != 4 or
                             os.environ.get("PCLA_GPU_SHARING") != "exclusive"):
        raise ValueError("formal cost replay requires CUDA, four CPU threads, and exclusive GPU sharing")
    if expanded.get("recipe", {}).get("required_adapter", "core") != "core":
        if args.cost_prefix:
            raise ValueError("core cost-prefix runner does not time native paths")
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
    cost_source_sha = None
    expected_cost = None
    if args.cost_prefix:
        old_source_path = args.original_source_manifest.resolve()
        expected_path = args.expected_cost_identities.resolve()
        identities = json.loads(expected_path.read_text())
        expected_cost = identities.get(job.get("recipe_id"))
        if (not isinstance(expected_cost, dict) or
                expected_cost.get("original_job_id") != job.get("job_id") or
                expected_cost.get("seed") != 401 or
                expected_cost.get("config_sha256") != _sha(config_path) or
                expected_cost.get("source_manifest_sha256") != _sha(old_source_path)):
            raise ValueError("cost replay original config/source identity differs from Pro plan")
        cost_source_sha = verify_instrumented_source_manifest(
            args.instrumented_source_manifest.resolve())
        expanded = {**expanded, "source_manifest": str(old_source_path)}
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("run lock exists; inspect owner before retrying") from exc
    os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node()}).encode())
    os.close(fd)
    timing = None
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        timing = (ExecutionTimingLedger(COST_LABEL_CATEGORIES,
                  synchronize=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None)
                  if args.cost_prefix else None)
        model, clients, capacities, evaluation, provenance = prepare(
            expanded, config_path, device, timing=timing)
        if args.cost_prefix and provenance["hardware"]["name"] != "NVIDIA GeForce RTX 4090 D":
            raise ValueError("cost replay hardware family differs from the frozen plan")
        if cost_source_sha is not None:
            provenance["instrumented_source_manifest_sha256"] = cost_source_sha
        rows = run_experiment(model, clients, evaluation, capacities,
                              job, expanded["runtime"], args.out, provenance,
                              stop_after=args.stop_after, timing=timing,
                              cost_prefix=args.cost_prefix)
        if timing is not None:
            record = timing.snapshot()
            breakdown = required_breakdown(record)
            atomic_json(args.out / "timing_events.json", record)
            actual_round = rows[-1]["budget_stop_round"] if rows else None
            actual_bytes = rows[-1]["deployed_bytes"] if rows else 0
            endpoint_match = (actual_round == expected_cost["budget_stop_round"] and
                              actual_bytes == expected_cost["deployed_bytes"])
            atomic_json(args.out / "cost_timing_unaccepted.json", {
                "schema": "pcla-v7-cost-timing-unaccepted-1",
                "recipe_id": job.get("recipe_id"), "seed": job["seed"],
                "config_sha256": provenance["config_sha256"],
                "original_source_manifest_sha256": provenance["source_manifest_sha256"],
                "instrumented_sources": provenance["sources"],
                "instrumented_source_manifest_sha256": cost_source_sha,
                "hardware": provenance["hardware"],
                "budget_return_round": actual_round,
                "deployed_bytes": actual_bytes,
                "original_endpoint_match": endpoint_match,
                "expected_cost_identities_sha256": _sha(expected_path),
                "required_seconds": sum(breakdown.values()),
                "required_breakdown_seconds": breakdown,
                "coverage": timing.coverage("envelope"),
                "event_file_sha256": _sha(args.out / "timing_events.json"),
                "timing_source_sha256": _sha(ROOT / "pcla_research_v7" / "execution_timing.py"),
                "accepted": False,
                "pending_gates": ["parity", "coverage", "exclusive", "original_quality_retained"],
            })
            atomic_json(args.out / "status.json", {
                "state": "cost_timing_unaccepted", "budget_return_round": actual_round,
                "deployed_bytes": actual_bytes, "original_endpoint_match": endpoint_match,
                "coverage_pass": timing.coverage("envelope")["passes"],
            })
            if not endpoint_match:
                raise RuntimeError("cost replay B8 return differs from original endpoint")
        atomic_json(args.out / "run_identity.json", {"job": job, "provenance": provenance})
        print(json.dumps({"job_id": job.get("job_id"), "rounds": len(rows),
                          "latest": str(args.out / "latest.pt")}), flush=True)
    except BaseException:
        original_traceback = traceback.format_exc()
        timing_capture_error = None
        if timing is not None:
            try:
                persist_failed_cost_timing(args.out, timing,
                                           original_traceback.splitlines()[-1])
            except BaseException as exc:
                timing_capture_error = f"{type(exc).__name__}: {exc}"
        (args.out / "exception.txt").write_text(original_traceback)
        atomic_json(args.out / "status.json", {"state": "failed",
            "exception": original_traceback.splitlines()[-1],
            "timing_capture_error": timing_capture_error,
            "latest_checkpoint": str(args.out / "latest.pt") if (args.out / "latest.pt").exists() else None})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
