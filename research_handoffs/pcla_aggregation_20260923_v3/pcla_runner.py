from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import traceback

from common import (
    BATCH, CONFIG, FULL_RULES, SEEDS, SOURCES, atomic_json, bind_project,
    fingerprint, handoff_hashes, read_json, require, sha256_file,
)
from aggregation import accumulator_type

TRACE_DRIVER = "aggregation_fixed_support_v3"


def device_setup(torch, name: str, expected_uuid: str | None):
    """Validate, but do not alter, the source run's numerical environment."""
    device = torch.device(name)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        require(torch.cuda.device_count() == 1,
                "Slurm step must expose exactly one GPU")
        marker = torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        if expected_uuid:
            output = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ], text=True)
            uuids = set()
            for line in output.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] == str(os.getpid()):
                    uuids.add(parts[1])
            require(uuids == {expected_uuid},
                    "visible CUDA process does not match worker GPU UUID")
        del marker
    return device


def environment_snapshot(torch, device):
    return {
        "python": __import__("platform").python_version(),
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": (torch.backends.cuda.matmul.allow_tf32
                                  if torch.cuda.is_available() else None),
        "cudnn_allow_tf32": (torch.backends.cudnn.allow_tf32
                              if torch.cuda.is_available() else None),
        "cudnn_benchmark": (torch.backends.cudnn.benchmark
                            if torch.cuda.is_available() else None),
        "cudnn_deterministic": (torch.backends.cudnn.deterministic
                                if torch.cuda.is_available() else None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "host": os.uname().nodename,
    }


def original_run(site, runtime, seed: int, policy: str):
    root = Path(site["source_root"]) / f"seed_{seed}" / policy
    checkpoint = root / "checkpoint.pt"
    require(checkpoint.is_file(), f"missing complete source: {checkpoint}")
    value = runtime.load_checkpoint(checkpoint)
    require(value["policy"] == policy and value["round"] == 30,
            "source is not a complete 30-round run")
    require(value["identity"]["seed"] == seed, "source seed differs")
    require(value["identity_hash"] == runtime.fingerprint(value["identity"]),
            "source identity hash is invalid")
    require(value["identity"]["config"]["aggregation"] == "fixed",
            "source does not use the original fixed denominator")
    repo = Path(site["repo"]).resolve()
    for relative, digest in value["identity"]["provenance"]["code_files"].items():
        path = (repo / relative).resolve()
        require(repo in path.parents, "source path leaves repository")
        require(sha256_file(path) == digest,
                f"execution source differs: {relative}")
    return value, root


def prepare(site, seed: int, policy: str = "pcla",
            evaluation_split: str = "benchmark_dev"):
    torch, core, runtime, bridge = bind_project(site["repo"])
    device = device_setup(torch, "cuda", os.environ.get("PCLA_EXPECTED_GPU_UUID"))
    source, source_dir = original_run(site, runtime, seed, policy)
    plan = read_json(Path(site["repo"]) / CONFIG)
    require(plan["training"] == source["identity"]["config"],
            "base training configuration differs from source")
    plan["evaluation_split"] = evaluation_split
    model, layout, clients, capacities, evaluation, provenance, partition = bridge.prepare(
        Path(site["repo"]).resolve(), plan, seed, device
    )
    old = source["identity"]["provenance"]
    for key in (
        "legacy_blob", "base_config", "train_file_sha256", "dev_file_sha256",
        "partition_sha256", "initial_loaded_model_sha256",
    ):
        require(provenance[key] == old[key], f"preparation differs: {key}")
    require(core.tensor_digest(core.adapter_state(model)) ==
            source["identity"]["initial_adapter_sha256"],
            "adapter initialization differs")
    require(str(torch.__version__) == old["torch"], "PyTorch build differs")
    require(torch.version.cuda == old["cuda"], "CUDA build differs")
    return (torch, core, runtime, model, layout, clients, capacities,
            evaluation, provenance, partition, plan, source, source_dir,
            device, environment_snapshot(torch, device))


def validate_trace(torch, core, checkpoint, clients, capacities, cfg,
                   shape, seed):
    expected, result = set(), {}
    for round_id in range(1, cfg["rounds"] + 1):
        selected = random.Random(
            core.keyed_seed(seed, "participants", round_id)
        ).sample(sorted(clients), cfg["selected_clients"])
        for client in selected:
            key = f"{round_id}/{client}"
            expected.add(key)
            support = checkpoint["trace"][key].detach().cpu().clone()
            core.check_support(
                support, shape,
                capacities[client] * cfg["credits_per_layer"],
            )
            if checkpoint["policy"] == "random":
                regenerated = core.allocate(
                    "random", torch.zeros(shape), capacities[client],
                    cfg["credits_per_layer"],
                    core.keyed_seed(seed, "allocation", "random",
                                    round_id, client),
                )
                require(torch.equal(support, regenerated),
                        "random source trace is not seed-reproducible")
            result[key] = support
    require(set(checkpoint["trace"]) == expected,
            "source trace inventory differs")
    return result


def preserve_uncommitted(run_dir, runtime):
    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        return 0
    committed = runtime.load_checkpoint(checkpoint)["round"]
    recovery = run_dir.parent / "recovery" / str(time.time_ns())
    patterns = (
        r"aggregation_(\d{3})\.json$", r"logits_round_(\d{3})\.json$",
        r"predictions_round_(\d{3})\.json$", r"clients_round_(\d{3})\.json$",
        r"probe_round_(\d{3})\.pt$", r"evaluation_round_(\d{3})\.json$",
    )
    candidates = []
    for path in run_dir.iterdir():
        for pattern in patterns:
            match = re.match(pattern, path.name)
            if match and int(match.group(1)) > committed:
                candidates.append(path)
                break
    metrics = run_dir / "metrics.json"
    if metrics.exists() and len(read_json(metrics)) > committed:
        candidates.append(metrics)
    if candidates:
        recovery.mkdir(parents=True)
        for path in candidates:
            shutil.move(str(path), recovery / path.name)
        atomic_json(recovery / "record.json", {"committed_round": committed})
    return committed


def execute(torch, core, runtime, model, layout, clients, capacities,
            evaluation, cfg, seed, out_dir, provenance, partition,
            trace=None, donor_path=None):
    """Temporarily extend one runtime process; never call concurrently in threads."""
    original = (runtime.POLICIES, runtime.allocate,
                runtime.DeltaAccumulator, runtime.evaluate)
    scaling = {name: float(module.scaling)
               for name, module in core.lora_modules(model).items()}
    StudyAccumulator = accumulator_type(original[2], torch, scaling)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    committed = preserve_uncommitted(out_dir, runtime)
    evaluation_round = committed
    aggregation_round = committed
    lookup = {}
    driver = TRACE_DRIVER if trace is not None else (
        "replay_random_rank" if donor_path is not None else "pcla"
    )
    if trace is not None:
        for key, support in trace.items():
            round_id, client = map(int, key.split("/"))
            token = core.keyed_seed(seed, "allocation", TRACE_DRIVER,
                                    round_id, client)
            require(token not in lookup, "allocation-key collision")
            lookup[token] = support

    def allocate(policy, scores, q, m, allocation_seed, **kwargs):
        if policy != TRACE_DRIVER:
            return original[1](policy, scores, q, m, allocation_seed, **kwargs)
        require(allocation_seed in lookup, "unexpected fixed-support request")
        support = lookup[allocation_seed]
        core.check_support(support, tuple(scores.shape), q * m)
        return support.clone()

    class LoggedAccumulator(StudyAccumulator):
        def finish(self):
            nonlocal aggregation_round
            result = super().finish()
            aggregation_round += 1
            runtime.atomic_json(
                out_dir / f"aggregation_{aggregation_round:03d}.json",
                {"round": aggregation_round, **self.readout()},
            )
            return result

    def evaluate(*args, **kwargs):
        nonlocal evaluation_round
        kwargs["record_logits"] = True
        result = original[3](*args, **kwargs)
        evaluation_round += 1
        labels = result.pop("labels")
        logits = result.pop("logits")
        runtime.atomic_json(out_dir / f"logits_round_{evaluation_round:03d}.json",
                            {"labels": labels, "logits": logits})
        return result

    runtime.POLICIES = (*original[0], TRACE_DRIVER)
    runtime.allocate = allocate
    runtime.DeltaAccumulator = LoggedAccumulator
    runtime.evaluate = evaluate
    try:
        return runtime.run_experiment(
            model, layout, clients, capacities, evaluation, cfg, seed, driver,
            out_dir, provenance, partition=partition, donor_path=donor_path,
        )
    finally:
        runtime.POLICIES, runtime.allocate = original[:2]
        runtime.DeltaAccumulator, runtime.evaluate = original[2:]


def compare_source_parity(torch, runtime, source, source_dir, current, run_dir):
    require(len(source["metrics"]) == len(current["metrics"]) == 30,
            "parity run has incomplete metrics")
    common_metric_keys = set(source["metrics"][0]) & set(current["metrics"][0])
    ignored = {"simulator_round_wall_seconds", "cuda_peak_allocated",
               "cuda_peak_reserved"}
    common_metric_keys -= ignored
    metrics_equal = all(
        all(old[key] == new[key] for key in common_metric_keys)
        for old, new in zip(source["metrics"], current["metrics"])
    )
    predictions_equal = all(
        read_json(source_dir / f"predictions_round_{r:03d}.json") ==
        read_json(run_dir / f"predictions_round_{r:03d}.json")
        for r in range(1, 31)
    )
    clients_equal = all(
        [x["client"] for x in read_json(source_dir / f"clients_round_{r:03d}.json")] ==
        [x["client"] for x in read_json(run_dir / f"clients_round_{r:03d}.json")]
        for r in range(1, 31)
    )
    parity = {
        "parameters_equal": all(
            torch.equal(value, current["adapter"][name])
            for name, value in source["adapter"].items()
        ),
        "traces_equal": all(
            torch.equal(value, current["trace"][key])
            for key, value in source["trace"].items()
        ),
        "metrics_equal_on_common_fields": metrics_equal,
        "predictions_equal": predictions_equal,
        "selected_client_sequence_equal": clients_equal,
    }
    require(all(parity.values()), "full fixed replay differs from source")
    return parity


def run_job(site, job, output, dependency_roots):
    started = time.perf_counter()
    source_policy = job.get("source", "pcla")
    values = prepare(site, job["seed"], source_policy)
    (torch, core, runtime, model, layout, clients, capacities, evaluation,
     provenance, partition, plan, source, source_dir, device,
     environment) = values
    cfg = copy.deepcopy(plan["training"])
    cfg["aggregation"] = job["rule"]
    family = "trace" if job["kind"] == "trace" else "online"
    study = {
        "batch": BATCH,
        "family": family,
        "kind": job["kind"],
        "source_policy": source_policy,
        "rule": job["rule"],
        "seed": job["seed"],
        "original_identity": source["identity_hash"],
        "original_checkpoint_sha256": sha256_file(source_dir / "checkpoint.pt"),
        "handoff_sources": handoff_hashes(),
        "environment": environment,
    }
    provenance.update({
        "code_files": source["identity"]["provenance"]["code_files"],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "base_plan_sha256": source["identity"]["provenance"]["plan_sha256"],
        "aggregation_study": study,
    })
    updated_plan = copy.deepcopy(plan)
    updated_plan["name"] = BATCH
    updated_plan["training"] = cfg
    provenance["plan_sha256"] = fingerprint(updated_plan)

    trace = None
    donor = None
    shape = (max(layout.values()) + 1,
             next(iter(core.lora_modules(model).values())).rank)
    if family == "trace":
        trace = validate_trace(torch, core, source, clients, capacities,
                               cfg, shape, job["seed"])
    if job["kind"] == "online_replay":
        donor_root = Path(dependency_roots[job["donor_job"]])
        donor = donor_root / "result" / "run" / "checkpoint.pt"
        parent = runtime.load_checkpoint(donor)
        donor_study = parent["identity"]["provenance"].get("aggregation_study", {})
        require(parent["policy"] == "pcla" and parent["round"] == 30,
                "online donor is incomplete")
        require(donor_study.get("batch") == BATCH and
                donor_study.get("family") == "online" and
                donor_study.get("rule") == "contributors" and
                parent["identity"]["seed"] == job["seed"],
                "online donor has another study identity")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "run"
    run_dir.mkdir(exist_ok=True)
    metadata = {**study, "job_id": job["id"],
                "donor_sha256": sha256_file(donor) if donor else None}
    marker = output / "study.json"
    if marker.exists():
        require(read_json(marker) == metadata, "recovery identity changed")
    else:
        atomic_json(marker, metadata)

    lock = run_dir / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, json.dumps({
        "pid": os.getpid(), "job": job["id"],
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }).encode())
    os.close(descriptor)
    try:
        rows = execute(
            torch, core, runtime, model, layout, clients, capacities,
            evaluation, cfg, job["seed"], run_dir, provenance, partition,
            trace=trace, donor_path=donor,
        )
        current = runtime.load_checkpoint(run_dir / "checkpoint.pt")
        if job.get("check_source"):
            atomic_json(output / "source_parity.json", compare_source_parity(
                torch, runtime, source, source_dir, current, run_dir
            ))
        atomic_json(output / "result.json", {
            "state": "completed", "job_id": job["id"],
            "process_wall_seconds": time.perf_counter() - started,
            "last5_accuracy": sum(x["accuracy"] for x in rows[-5:]) / 5,
            "environment": environment,
        })
    except BaseException:
        (output / "exception.txt").write_text(traceback.format_exc(),
                                               encoding="utf-8")
        atomic_json(output / "result.json", {
            "state": "failed", "job_id": job["id"],
            "environment": environment,
        })
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dependencies", required=True)
    args = parser.parse_args()
    site = read_json(args.site)
    job = read_json(args.job)
    dependencies = read_json(args.dependencies)
    require(job["seed"] in SEEDS, "unexpected seed")
    if job["kind"] == "diagnose":
        from fixed_state import run_diagnostic
        run_diagnostic(site, job, args.out)
    else:
        require(job["rule"] in FULL_RULES and
                job.get("source", "pcla") in SOURCES,
                "unexpected trajectory job")
        run_job(site, job, args.out, dependencies)


if __name__ == "__main__":
    main()
