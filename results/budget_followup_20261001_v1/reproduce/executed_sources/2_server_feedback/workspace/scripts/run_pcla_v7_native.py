#!/usr/bin/env python3
"""Execute one expanded native Fed-SB, FedSVD, or Fed-PLoRA v7 job."""
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
from pcla_repair.runtime import atomic_json
from pcla_research_v7.data import TokenCache
from pcla_research_v7.native_runtime import (METHODS, NATIVE_TIMING_LABELS,
                                             initialize_fedsb, run_native)
from pcla_research_v7.execution_timing import (ExecutionTimingLedger, NECESSARY_ALGORITHM,
                                                required_breakdown)
from pcla_research_v7.native_fedsb import OFFICIAL_COMMIT as FEDSB_COMMIT
from pcla_research_v7.native_fedsvd import OFFICIAL_COMMIT as FEDSVD_COMMIT
from pcla_research_v7.native_fedplora import OFFICIAL_COMMIT as FEDPLORA_COMMIT
from scripts.run_pcla_v7 import prepare, _path, verify_instrumented_source_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--cost-prefix", action="store_true",
                   help="measure only the deployed B8 return prefix; never dispatch diagnostics")
    p.add_argument("--instrumented-source-manifest", type=Path,
                   help="SHA-256 manifest of the actual timing-instrumented source tree")
    args = p.parse_args()
    if args.cost_prefix != (args.instrumented_source_manifest is not None):
        raise ValueError("cost prefix requires exactly one instrumented source manifest")
    if args.cost_prefix and (args.resume or args.stop_after is not None):
        raise ValueError("cost prefix requires a fresh complete run without technical stop")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config_path = args.config.resolve()
    expanded = json.loads(config_path.read_text())
    job, recipe = expanded["job"], expanded["recipe"]
    if "technical_stop_after" in expanded:
        if job.get("phase") != "TECH" or not expanded.get("technical_profile_only"):
            raise ValueError("Technical stop may only be used for a TECH job")
        if args.stop_after is None:
            args.stop_after = int(expanded["technical_stop_after"])
        if args.stop_after < 1:
            raise ValueError("Technical stop must be positive")
    method = job["method"]
    if method not in METHODS or recipe["required_adapter"] != method or recipe["method"] != method:
        raise NotImplementedError("native method/adapter mismatch")
    if recipe["rank"] != job["rank"] or recipe["lr"] != job["lr"]:
        raise ValueError("native job/recipe differs")
    if args.out.exists() and (args.out / "latest.pt").exists() and not args.resume:
        raise ValueError("existing native run requires --resume")
    if args.resume and not (args.out / "latest.pt").exists():
        raise ValueError("native --resume requires latest.pt")
    if args.cost_prefix and args.out.exists() and any(args.out.iterdir()):
        raise ValueError("cost prefix requires a fresh output directory")
    cost_source_sha = None
    if args.cost_prefix:
        instrumented_path = args.instrumented_source_manifest.resolve()
        cost_source_sha = verify_instrumented_source_manifest(instrumented_path)
        instrumented_files = json.loads(instrumented_path.read_text())["files"]
        native_closure = {"pcla_research_v7/native_runtime.py", "scripts/run_pcla_v7_native.py",
                          "pcla_research_v7/native_fedsb.py", "pcla_research_v7/native_fedsvd.py",
                          "pcla_research_v7/native_fedplora.py"}
        if not native_closure <= set(instrumented_files):
            raise ValueError("instrumented source manifest lacks the native replay closure")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("native run lock exists") from exc
    os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node()}).encode())
    os.close(fd)
    timing = None
    provenance = None

    def persist_timing(*, failed: bool) -> None:
        if not args.cost_prefix:
            return
        if timing is None:
            atomic_json(args.out / "cost_prefix_draft.json", {
                "schema": "pcla-v7-native-cost-prefix-draft-1", "accepted": False,
                "failed": True, "reason": "failure before timing began; no required-time receipt"})
            return
        if timing.active_labels == ("envelope",):
            timing.end_envelope()
        elif timing.active_labels:
            atomic_json(args.out / "cost_prefix_draft.json", {
                "schema": "pcla-v7-native-cost-prefix-draft-1", "accepted": False,
                "failed": True, "reason": f"unclosed timing labels: {timing.active_labels}"})
            return
        record = timing.snapshot()
        if record["envelope_bounds"][1] is None:
            atomic_json(args.out / "cost_prefix_draft.json", {
                "schema": "pcla-v7-native-cost-prefix-draft-1", "accepted": False,
                "failed": True, "reason": "failure before complete cost envelope"})
            return
        coverage = timing.coverage("envelope")
        path = args.out / "exclusive_timing.json"
        atomic_json(path, record)
        status_path = args.out / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        atomic_json(args.out / "cost_prefix_draft.json", {
            "schema": "pcla-v7-native-cost-prefix-draft-1", "accepted": False,
            "failed": failed, "job_id": job["job_id"], "recipe_id": recipe["recipe_id"],
            "seed": job["seed"], "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "instrumented_source_manifest_sha256": cost_source_sha,
            "original_source_manifest_sha256": (provenance or {}).get("source_manifest_sha256"),
            "data_manifest_sha256": (provenance or {}).get("manifest_sha256"),
            "pretrained_model_sha256": (provenance or {}).get("pretrained_model_sha256"),
            "budget_return_round": status.get("budget_stop_round"),
            "deployed_bytes": status.get("deployed_bytes"),
            "interval_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timing_source_sha256": {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
                for source in (ROOT / "pcla_research_v7/execution_timing.py",
                               ROOT / "pcla_research_v7/native_runtime.py", Path(__file__))},
            "required_seconds_unaccepted": record["bucket_category_seconds"]["deployed_prefix"][NECESSARY_ALGORITHM],
            "required_breakdown_unaccepted": required_breakdown(record),
            "coverage": coverage,
            "reason": "scientific parity and hardware eligibility are not certified by this runner"})
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        # The shared loader verifies source/data/B8 and creates common BERT
        # head and client allocation. Fed-SB discards its temporary LoRA pair
        # before the full-weight task gradient, including for native rank32.
        generic = dict(expanded, job=dict(job, rank=16))
        device = torch.device(args.device)
        if args.cost_prefix:
            synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
            timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS, synchronize=synchronize)
        model, clients, capacities, evaluation, provenance = prepare(
            generic, config_path, device, timing=timing)
        if timing is not None and timing.active_labels != ("envelope",):
            raise RuntimeError("shared preparation did not leave the cost envelope active")
        if method == "fedsb":
            with timing.span("task_initialization") if timing is not None else nullcontext():
                targets = [(name, module) for name, module in model.named_modules()
                           if name.endswith((".attention.self.query", ".attention.self.value"))
                           and hasattr(module, "lora_a")]
                if len(targets) != 24:
                    raise ValueError("Fed-SB requires 24 temporary LoRA projections")
                for name, module in targets:
                    parent, leaf = name.rsplit(".", 1)
                    setattr(model.get_submodule(parent), leaf, module.base)
                tokens = TokenCache(_path(expanded["token_cache"], config_path))
                manifest = json.loads(_path(expanded["manifest"], config_path).read_text())
                public = tokens.dataset(manifest, "public_training", "algorithm")
                init = initialize_fedsb(model, public, int(job["rank"]), float(job["lr"]),
                                        batch_size=int(expanded["runtime"].get("batch_size", 16)),
                                        timing=timing)
            init_metadata = {k: v for k, v in init.items() if k not in {"cost", "projections"}}
            init_file = args.out / "native_initialization.json"
            with timing.span("checkpoint_and_audit") if timing is not None else nullcontext():
                if args.resume:
                    recorded = json.loads(init_file.read_text())
                    if {k: v for k, v in recorded.items() if k not in {"cost", "projections"}} != init_metadata:
                        raise ValueError("Fed-SB initialization recipe/role metadata changed on resume")
                else:
                    atomic_json(init_file, init)
                init_attempts_file = args.out / "native_initialization_attempts.json"
                init_attempts = json.loads(init_attempts_file.read_text()) if init_attempts_file.exists() else []
                init_attempts.append({"attempt": len(init_attempts) + 1, "cost": init["cost"]})
                atomic_json(init_attempts_file, init_attempts)
        else:
            init = None
            init_metadata = None
        with timing.span("provenance_and_hashing") if timing is not None else nullcontext():
            native_sources = [ROOT / "pcla_research_v7" / f"native_{name}.py" for name in
                              ("fedsb", "fedsvd", "fedplora")]
            native_sources += [ROOT / "pcla_research_v7" / "native_runtime.py", Path(__file__)]
            provenance["native_sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in native_sources}
        provenance["native_method"] = method
        provenance["official_source_commit"] = {"fedsb": FEDSB_COMMIT,
            "fedsvd": FEDSVD_COMMIT, "fedplora": FEDPLORA_COMMIT}[method]
        # Wall time, memory peaks, and numerical audit diagnostics remain in
        # native_initialization.json. Resume identity uses the source/data/
        # config hashes and exact initial tensor digest in run_native.
        provenance["native_initialization"] = init_metadata
        rows = run_native(model, clients, evaluation, capacities, job, recipe,
                          expanded["runtime"], args.out, provenance,
                          stop_after=args.stop_after, timing=timing,
                          cost_prefix=args.cost_prefix)
        persist_timing(failed=False)
        atomic_json(args.out / "run_identity.json", {"job": job, "recipe": recipe,
                                                      "provenance": provenance})
        print(json.dumps({"job_id": job.get("job_id"), "rounds": len(rows),
                          "latest": str(args.out / "latest.pt")}), flush=True)
    except BaseException:
        if args.cost_prefix:
            try:
                persist_timing(failed=True)
            except BaseException:
                (args.out / "cost_timing_exception.txt").write_text(traceback.format_exc())
        (args.out / "exception.txt").write_text(traceback.format_exc())
        attempted = 0
        committed = 0
        try:
            journal = json.loads((args.out / "wire_attempt_journal.json").read_text())
            attempted = sum(int(event["bytes"]) for event in journal["events"])
            latest = args.out / "latest.pt"
            if latest.exists():
                ck = torch.load(latest, map_location="cpu", weights_only=True)
                committed = int(ck["deployed_bytes"]) + int(ck["diagnostic_bytes"])
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            pass
        atomic_json(args.out / "status.json", {"state": "failed",
            "exception": traceback.format_exc().splitlines()[-1],
            "latest_checkpoint": str(args.out / "latest.pt") if (args.out / "latest.pt").exists() else None,
            "actual_attempted_wire_bytes": attempted,
            "failed_attempt_wire_bytes": max(0, attempted - committed)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
