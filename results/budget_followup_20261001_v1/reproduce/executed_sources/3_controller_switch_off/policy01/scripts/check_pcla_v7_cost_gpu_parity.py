#!/usr/bin/env python3
"""Technical two-round GPU parity for old v7 training and new cost timing.

This is a TECH harness, never an accepted seed-401 cost receipt.  The parent
chooses the allocation and runs the CLI; importing this file performs no work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


CASES = {
    "D002": "core",       # N-local
    "D011": "core",       # D-local; R-local tie candidate and r2 coordinate refresh
    "D054": "core",       # JQ: actual shared information and coordination
    "D035": "core",       # T representation
    "D105": "native",     # Fed-SB
    "D107": "native",     # FedSVD repaired native04
    "D112": "native",     # Fed-PLoRA
}
TECH_SEEDS = (9001, 9002)
NATIVE_CLOSURE = {"pcla_research_v7/native_runtime.py", "scripts/run_pcla_v7_native.py",
                  "pcla_research_v7/native_fedsb.py", "pcla_research_v7/native_fedsvd.py",
                  "pcla_research_v7/native_fedplora.py"}
SCHEMA = "pcla-v7-cost-gpu-parity-tech-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_audited_runs(run_root: Path) -> dict[str, dict]:
    folder = run_root / "checks/summaries/full_20260928_01"
    hashes = json.loads((folder / "output_hashes.json").read_text())
    source = folder / "runs.json"
    if sha(source) != hashes["runs.json"]:
        raise ValueError("audited D0 run catalogue hash differs")
    rows = json.loads(source.read_text())
    selected = {row["job"]["recipe_id"]: row for row in rows
                if row["job"]["seed"] == 401 and row["job"]["recipe_id"] in CASES}
    if set(selected) != set(CASES):
        raise ValueError("seven D0 seed401 source identities are required")
    return selected


def original_path(run_root: Path, absolute_remote: str) -> Path:
    """Map an audited remote path into this run root, without trusting its prefix."""
    path = Path(absolute_remote)
    parts = path.parts
    if run_root.name not in parts:
        raise ValueError(f"path is outside the frozen run: {path}")
    at = parts.index(run_root.name)
    relative = Path(*parts[at + 1:])
    if ".." in relative.parts:
        raise ValueError("audited path traversal")
    return run_root / relative


def make_technical_config(original: dict, rid: str, seed: int,
                          source_manifest: Path) -> dict:
    if rid not in CASES or seed not in TECH_SEEDS or original["job"]["recipe_id"] != rid:
        raise ValueError("unapproved technical recipe or seed")
    value = json.loads(json.dumps(original))
    job = value["job"]
    job.update({"job_id": f"COSTTECH_{rid}_s{seed}", "phase": "TECH", "seed": seed,
                "hardware_block": f"COSTTECH_s{seed}", "status": "technical_only",
                "output_dir": f"checks/cost_gpu_parity/{rid}_s{seed}"})
    value["technical_profile_only"] = True
    value["technical_stop_after"] = 2
    value["source_manifest"] = str(source_manifest)
    value["expanded_sha256"] = digest({k: v for k, v in value.items()
                                        if k != "expanded_sha256"})
    return value


def shared_preparation_config(expanded: dict) -> dict:
    """Match native's temporary common LoRA loader without changing job identity."""
    if expanded["recipe"]["required_adapter"] == "core":
        return expanded
    return dict(expanded, job=dict(expanded["job"], rank=16))


def scientific_rows(rows: list[dict]) -> list[dict]:
    """Drop clocks, hardware peaks, and the later return annotation only."""
    result = []
    for row in rows:
        item = {k: v for k, v in row.items() if k not in {
            "cost", "cumulative_cost", "gpu_sharing", "budget_stop_round"}}
        item["client_training"] = [{k: v for k, v in child.items() if k != "train_seconds"}
                                   for child in item.get("client_training", [])]
        result.append(item)
    return result


def compare_parity(old: dict, cost: dict, returned: dict,
                   old_rows: list[dict], cost_rows: list[dict],
                   timing: dict, coverage: dict, breakdown: dict) -> dict:
    """Use exact tensor/RNG equality; never substitute a tolerance for parity."""
    import torch
    def same_tensors(a, b):
        return set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)
    checks = {
        "two_complete_rounds": old.get("round") == cost.get("round") == 2,
        "formal_200_round_cap": 1 <= len(old_rows) == len(cost_rows) <= 200 and
                                max(row["round"] for row in cost_rows) <= 200,
        "same_deployed_bytes": old.get("deployed_bytes") == cost.get("deployed_bytes"),
        "no_cost_diagnostic_bytes": cost.get("diagnostic_bytes") == 0,
        "return_round_two": cost.get("budget_stop_round") == returned.get("round") == 2,
        "adapter_exact": same_tensors(old["adapter"], cost["adapter"]),
        "returned_adapter_exact": same_tensors(old["adapter"], returned["adapter"]),
        "return_deployed_bytes": returned.get("deployed_bytes") == old.get("deployed_bytes"),
        "torch_rng_exact": torch.equal(old["torch_rng"], cost["torch_rng"]),
        "python_rng_exact": old["python_rng"] == cost["python_rng"],
        "cuda_rng_exact": len(old["cuda_rng"]) == len(cost["cuda_rng"]) and
                          all(torch.equal(a, b) for a, b in zip(old["cuda_rng"], cost["cuda_rng"])),
        "cache_exact": old["cache"] == cost["cache"],
        "scientific_round_rows_exact": scientific_rows(old_rows) == scientific_rows(cost_rows),
        "coverage": coverage.get("passes") is True,
        "seven_required_groups": set(breakdown) == {
            "task_initialization", "state_and_optimizer_setup", "client_training",
            "scoring_and_shared_information", "coordinate_and_server_decision",
            "message_processing", "budget_and_return_control"},
        "required_seconds_positive": timing["bucket_category_seconds"]["deployed_prefix"]
                                     ["necessary_algorithm"] > 0,
    }
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def worker(args) -> None:
    """Fresh process bound to the instrumented tree; bypasses formal CLI seed gate."""
    sys.path.insert(0, str(args.instrumented_source_root.resolve()))
    import torch
    from pcla_repair.runtime import atomic_json
    from pcla_research_v7.data import TokenCache
    from pcla_research_v7.execution_timing import (COST_LABEL_CATEGORIES,
        ExecutionTimingLedger, required_breakdown)
    from scripts.run_pcla_v7 import (prepare, _path,
        verify_instrumented_source_manifest)
    from pcla_research_v7.runtime import run_experiment
    from pcla_research_v7.native_runtime import run_native, initialize_fedsb
    from pcla_research_v7.native_fedsb import OFFICIAL_COMMIT as FEDSB_COMMIT
    from pcla_research_v7.native_fedsvd import OFFICIAL_COMMIT as FEDSVD_COMMIT
    from pcla_research_v7.native_fedplora import OFFICIAL_COMMIT as FEDPLORA_COMMIT

    root = args.instrumented_source_root.resolve()
    source_sha = verify_instrumented_source_manifest(args.instrumented_source_manifest.resolve())
    expanded = json.loads(args.config.read_text())
    job, recipe = expanded["job"], expanded["recipe"]
    native = recipe["required_adapter"] != "core"
    if native and not NATIVE_CLOSURE <= set(json.loads(args.instrumented_source_manifest.read_text())["files"]):
        raise ValueError("instrumented manifest lacks native replay closure")
    if job["phase"] != "TECH" or job["seed"] not in TECH_SEEDS or args.cap <= 0:
        raise ValueError("worker accepts only the approved technical matrix and positive cap")
    if args.out.exists():
        raise ValueError("technical cost output already exists")
    args.out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("GPU parity worker requires an allocated CUDA device")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timing = ExecutionTimingLedger(COST_LABEL_CATEGORIES,
                                   synchronize=lambda: torch.cuda.synchronize(device))
    provenance = None
    try:
        # Native's audited driver always loads a temporary rank-16 pair.
        # Fed-SB's actual native rank may be 32, after that pair is discarded.
        generic = shared_preparation_config(expanded)
        model, clients, capacities, evaluation, provenance = prepare(
            generic, args.config.resolve(), device, timing=timing)
        runtime = dict(expanded["runtime"], byte_cap=args.cap)
        if native:
            method = job["method"]
            if method == "fedsb":
                with timing.span("task_initialization"):
                    targets = [(name, module) for name, module in model.named_modules()
                               if name.endswith((".attention.self.query", ".attention.self.value"))
                               and hasattr(module, "lora_a")]
                    if len(targets) != 24:
                        raise ValueError("Fed-SB requires 24 temporary LoRA projections")
                    for name, module in targets:
                        parent, leaf = name.rsplit(".", 1)
                        setattr(model.get_submodule(parent), leaf, module.base)
                    tokens = TokenCache(_path(expanded["token_cache"], args.config.resolve()))
                    manifest = json.loads(_path(expanded["manifest"], args.config.resolve()).read_text())
                    public = tokens.dataset(manifest, "public_training", "algorithm")
                    init = initialize_fedsb(model, public, int(job["rank"]), float(job["lr"]),
                        batch_size=int(runtime.get("batch_size", 16)), timing=timing)
                provenance["native_initialization"] = {k: v for k, v in init.items()
                                                       if k not in {"cost", "projections"}}
                with timing.span("checkpoint_and_audit"):
                    atomic_json(args.out / "native_initialization.json", init)
                    atomic_json(args.out / "native_initialization_attempts.json",
                                [{"attempt": 1, "cost": init["cost"]}])
            else:
                provenance["native_initialization"] = None
            with timing.span("provenance_and_hashing"):
                sources = [root / "pcla_research_v7" / f"native_{name}.py" for name in
                           ("fedsb", "fedsvd", "fedplora")]
                sources += [root / "pcla_research_v7/native_runtime.py",
                            root / "scripts/run_pcla_v7_native.py"]
                provenance["native_sources"] = {str(path.relative_to(root)): sha(path)
                                                for path in sources}
                provenance["native_method"] = method
                provenance["official_source_commit"] = {
                    "fedsb": FEDSB_COMMIT, "fedsvd": FEDSVD_COMMIT,
                    "fedplora": FEDPLORA_COMMIT}[method]
            rows = run_native(model, clients, evaluation, capacities, job, recipe,
                runtime, args.out, provenance, timing=timing, cost_prefix=True)
        else:
            rows = run_experiment(model, clients, evaluation, capacities, job,
                runtime, args.out, provenance, timing=timing, cost_prefix=True)
        record = timing.snapshot()
        coverage = timing.coverage("envelope")
        breakdown = required_breakdown(record)
        atomic_json(args.out / "timing_events.json", record)
        atomic_json(args.out / "technical_timing.json", {
            "schema": SCHEMA, "accepted_cost_receipt": False,
            "recipe_id": recipe["recipe_id"], "seed": job["seed"],
            "instrumented_source_manifest_sha256": source_sha,
            "original_source_manifest_sha256": provenance["source_manifest_sha256"],
            "config_sha256": sha(args.config), "technical_cap_bytes": args.cap,
            "budget_return_round": 2, "completed_rounds": len(rows),
            "required_seconds_technical_only": sum(breakdown.values()),
            "required_breakdown": breakdown, "coverage": coverage,
            "timing_events_sha256": sha(args.out / "timing_events.json")})
        atomic_json(args.out / "run_identity.json", {"job": job, "recipe": recipe,
                                                       "provenance": provenance})
        if len(rows) != 2 or not coverage["passes"]:
            raise AssertionError("technical cost returned wrong prefix or timing coverage failed")
    except BaseException:
        (args.out / "exception.txt").write_text(traceback.format_exc())
        if timing.active_labels == ("envelope",):
            timing.end_envelope()
        if not timing.active_labels:
            atomic_json(args.out / "failed_timing_events.json", timing.snapshot())
        raise


def launch(args) -> None:
    if args.device != "cuda" or args.seeds != list(TECH_SEEDS):
        raise ValueError("GPU parity requires both fixed technical seeds and device cuda")
    if not args.cases or len(args.cases) != len(set(args.cases)) or any(rid not in CASES for rid in args.cases):
        raise ValueError("cases must be distinct approved recipes")
    if args.out.exists():
        raise ValueError("choose a fresh parity output directory")
    run_root = args.run_root.resolve()
    instrumented_root = args.instrumented_source_root.resolve()
    if not instrumented_root.is_dir() or not args.instrumented_source_manifest.is_file():
        raise ValueError("instrumented source and manifest must exist")
    manifest = json.loads((run_root / "data/pcla_v7/train_roles.json").read_text())
    if any(str(seed) not in manifest["partitions"] for seed in TECH_SEEDS):
        raise ValueError("technical seeds absent from frozen data manifest")
    audited = load_audited_runs(run_root)
    args.out.mkdir(parents=True)
    plan = []
    for rid in args.cases:
        family = CASES[rid]
        entry = audited[rid]
        original_config = original_path(run_root, entry["config"])
        if sha(original_config) != entry["config_sha256"]:
            raise ValueError(f"audited D0 config changed: {rid}")
        generation = ("validation02" if family == "core" else
                      Path(entry["source_manifest"]).parent.name)
        old_root = run_root / "sources" / generation
        old_manifest = old_root / "SOURCE_MANIFEST.json"
        if not old_manifest.is_file():
            raise ValueError(f"old scientific source absent: {rid}/{generation}")
        for seed in TECH_SEEDS:
            cfg = make_technical_config(json.loads(original_config.read_text()), rid, seed,
                                        old_manifest)
            condition = args.out / f"{rid}_s{seed}"
            condition.mkdir()
            config_path = condition / "technical_config.json"
            write(config_path, cfg)
            plan.append({"recipe_id": rid, "seed": seed, "family": family,
                         "original_config_sha256": entry["config_sha256"],
                         "audited_source_manifest_sha256": entry["source_manifest_sha256"],
                         "old_generation": generation, "old_source_manifest_sha256": sha(old_manifest),
                         "config": str(config_path), "old_source": str(old_root)})
    write(args.out / "plan.json", {"schema": SCHEMA, "accepted_cost_receipt": False,
                                   "conditions": plan, "instrumented_source": str(instrumented_root),
                                   "instrumented_source_manifest_sha256": sha(args.instrumented_source_manifest)})
    if args.prepare_only:
        print(json.dumps({"prepared": len(plan), "out": str(args.out)}), flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU parity requires a visible allocated CUDA device")
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for item in plan:
        rid, seed = item["recipe_id"], item["seed"]
        condition = args.out / f"{rid}_s{seed}"
        old_root = Path(item["old_source"])
        old_script = old_root / "scripts" / ("run_pcla_v7.py" if item["family"] == "core"
                                             else "run_pcla_v7_native.py")
        old_out = condition / "old"
        old_cmd = [sys.executable, str(old_script), "--config", item["config"],
                   "--out", str(old_out), "--device", args.device]
        with (condition / "old.log").open("wb") as log:
            subprocess.run(old_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        old_ck = torch.load(old_out / "latest.pt", map_location="cpu", weights_only=True)
        if old_ck["round"] != 2 or old_ck["deployed_bytes"] <= 0:
            raise ValueError(f"old technical prefix is not two complete rounds: {rid}/{seed}")
        cap = int(old_ck["deployed_bytes"])
        cost_out = condition / "cost"
        cost_cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--config", item["config"], "--out", str(cost_out),
                    "--instrumented-source-root", str(instrumented_root),
                    "--instrumented-source-manifest", str(args.instrumented_source_manifest),
                    "--cap", str(cap), "--device", args.device]
        with (condition / "cost.log").open("wb") as log:
            subprocess.run(cost_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        new_ck = torch.load(cost_out / "latest.pt", map_location="cpu", weights_only=True)
        returned = torch.load(cost_out / "budget_return.pt", map_location="cpu", weights_only=True)
        timing = json.loads((cost_out / "timing_events.json").read_text())
        technical = json.loads((cost_out / "technical_timing.json").read_text())
        checks = compare_parity(old_ck, new_ck, returned, old_ck["metrics"], new_ck["metrics"],
                                timing, technical["coverage"], technical["required_breakdown"])
        old_identity = json.loads((old_out / "run_identity.json").read_text())
        new_identity = json.loads((cost_out / "run_identity.json").read_text())
        old_hw = old_identity["provenance"]["hardware"]
        new_hw = new_identity["provenance"]["hardware"]
        checks["same_gpu_name"] = old_hw.get("name") == new_hw.get("name")
        checks["same_gpu_uuid"] = (old_hw.get("uuid") not in {None, "unavailable"} and
                                   old_hw.get("uuid") == new_hw.get("uuid"))
        if not all(checks.values()):
            raise AssertionError({rid: checks})
        write(condition / "parity.json", {"schema": SCHEMA, "accepted_cost_receipt": False,
                                          "recipe_id": rid, "seed": seed, "checks": checks,
                                          "technical_cap_bytes": cap,
                                          "old_checkpoint_sha256": sha(old_out / "latest.pt"),
                                          "cost_checkpoint_sha256": sha(cost_out / "latest.pt"),
                                          "timing_events_sha256": technical["timing_events_sha256"]})
        print(json.dumps({"recipe_id": rid, "seed": seed, "passed": True}), flush=True)
    write(args.out / "summary.json", {"schema": SCHEMA, "accepted_cost_receipt": False,
                                      "passed": len(plan), "conditions": len(plan),
                                      "device": torch.cuda.get_device_name()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--instrumented-source-root", type=Path, required=True)
    parser.add_argument("--instrumented-source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cap", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TECH_SEEDS))
    parser.add_argument("--cases", nargs="+", default=list(CASES),
                        choices=list(CASES), help="independent recipe shard; both seeds per case")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.config is None or args.cap is None:
            raise ValueError("worker requires config and cap")
        worker(args)
    else:
        if args.run_root is None:
            raise ValueError("run-root is required")
        launch(args)


if __name__ == "__main__":
    main()
