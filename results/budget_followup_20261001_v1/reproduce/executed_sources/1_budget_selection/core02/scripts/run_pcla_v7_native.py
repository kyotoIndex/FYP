#!/usr/bin/env python3
"""Execute one expanded native Fed-SB, FedSVD, or Fed-PLoRA v7 job."""
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
from pcla_research_v7.data import TokenCache
from pcla_research_v7.native_runtime import METHODS, initialize_fedsb, run_native
from pcla_research_v7.native_fedsb import OFFICIAL_COMMIT as FEDSB_COMMIT
from pcla_research_v7.native_fedsvd import OFFICIAL_COMMIT as FEDSVD_COMMIT
from pcla_research_v7.native_fedplora import OFFICIAL_COMMIT as FEDPLORA_COMMIT
from scripts.run_pcla_v7 import prepare, _path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    args = p.parse_args()
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
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("native run lock exists") from exc
    os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node()}).encode())
    os.close(fd)
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
        model, clients, capacities, evaluation, provenance = prepare(
            generic, config_path, torch.device(args.device))
        if method == "fedsb":
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
                                    batch_size=int(expanded["runtime"].get("batch_size", 16)))
            if not args.resume:
                atomic_json(args.out / "native_initialization.json", init)
        else:
            init = None
        native_sources = [ROOT / "pcla_research_v7" / f"native_{name}.py" for name in
                          ("fedsb", "fedsvd", "fedplora")]
        native_sources += [ROOT / "pcla_research_v7" / "native_runtime.py", Path(__file__)]
        provenance["native_sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in native_sources}
        provenance["native_method"] = method
        provenance["official_source_commit"] = {"fedsb": FEDSB_COMMIT,
            "fedsvd": FEDSVD_COMMIT, "fedplora": FEDPLORA_COMMIT}[method]
        provenance["native_initialization"] = init
        rows = run_native(model, clients, evaluation, capacities, job, recipe,
                          expanded["runtime"], args.out, provenance,
                          stop_after=args.stop_after)
        atomic_json(args.out / "run_identity.json", {"job": job, "recipe": recipe,
                                                      "provenance": provenance})
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
