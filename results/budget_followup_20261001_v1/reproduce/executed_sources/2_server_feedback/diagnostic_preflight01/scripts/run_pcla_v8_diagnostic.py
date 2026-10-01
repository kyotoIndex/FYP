"""Execute one verified PostV diagnostic object; queue acceptance is external."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from pcla_research_v7.runtime import evaluate, load_state
from pcla_research_v8.diagnostic_feedback import evaluate_feedback
from pcla_research_v8.diagnostic_inputs import collect_rank8, prepare_context
from pcla_research_v8.server_control import _hash_state, _preserve_caller


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify_source(manifest, expected):
    manifest = Path(manifest).resolve()
    if manifest.parent != ROOT or sha(manifest) != expected:
        raise ValueError("executing source manifest identity differs")
    files = json.loads(manifest.read_text())["files"]
    for relative, digest in files.items():
        path = manifest.parent / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"executing source file differs: {relative}")
    driver = str(Path(__file__).resolve().relative_to(ROOT))
    if files.get(driver) != sha(__file__):
        raise ValueError("driver absent from frozen source")


def raw_readout(model, candidate, dataset, *, expected_samples):
    if len(dataset) != expected_samples:
        raise ValueError("readout role row count differs")
    with _preserve_caller(model), torch.no_grad():
        load_state(model, candidate)
        result = evaluate(model, dataset, batch_size=64, record_logits=True)
    logits = torch.tensor(result["logits"], dtype=torch.float32)
    labels = torch.tensor(result["labels"], dtype=torch.long)
    if len(labels) != expected_samples or logits.shape != (expected_samples, 2):
        raise ValueError("binary readout logits/labels shape differs")
    predictions = logits.argmax(1)
    result["loss"] = F.cross_entropy(logits.double(), labels, reduction="mean").item()
    result["accuracy"] = int((predictions == labels).sum()) / expected_samples
    result["prediction_counts"] = torch.bincount(predictions, minlength=2).tolist()
    result["state_hash"] = _hash_state(candidate)
    return result


def transfer_chain(context, update, reader):
    """Raw child improvements plus one equal weighting, with full readouts."""
    ids = context["participants"]
    if len(ids) != 10 or len(update["children"]) != 10 or update["weights"] != [0.1] * 10:
        raise ValueError("transfer chain requires ten raw equally weighted children")
    parent = context["parent_state"]
    global_parent = reader(parent)
    rows = []
    for cid, child, weight in zip(ids, update["children"], update["weights"]):
        local = context["local_readout_by_client"][cid]
        local_parent = raw_readout(context["model"], parent, local, expected_samples=64)
        local_child = raw_readout(context["model"], child, local, expected_samples=64)
        global_child = reader(child)
        global_gain = global_parent["loss"] - global_child["loss"]
        rows.append({"client_id": cid, "sample_weight": weight,
                     "child_state_hash": _hash_state(child),
                     "local_parent": local_parent, "local_child": local_child,
                     "individual_global": global_child,
                     "local_NLL_gain": local_parent["loss"] - local_child["loss"],
                     "individual_global_NLL_gain": global_gain,
                     "weighted_individual_global_NLL_gain": weight * global_gain})
    aggregate = reader(update["proposal"])
    return {"clients": rows, "global_parent": global_parent, "aggregate": aggregate,
            "weighted_individual_global_NLL_gain": sum(
                row["weighted_individual_global_NLL_gain"] for row in rows),
            "aggregate_global_NLL_gain": global_parent["loss"] - aggregate["loss"],
            "weighting": "raw child NLL gain weighted once by 0.1; aggregate separately evaluated"}


def restore_or_collect(context, config, output, config_sha):
    """Persist the new cohort before public selection and reuse on recovery."""
    path, seal = output / "new_cohort.pt", output / "new_cohort_seal.json"
    if path.exists() or seal.exists():
        if not path.is_file() or not seal.is_file():
            raise ValueError("partial cohort publication retained; recovery requires inspection")
        record = json.loads(seal.read_text())
        if (record["config_sha256"] != config_sha or record["pt_sha256"] != sha(path)
                or record["parent_state_hash"] != _hash_state(context["parent_state"])):
            raise ValueError("sealed cohort/config/parent differs")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["local_calls"] != 10 or value["optimizer_steps"] != 160:
            raise ValueError("sealed cohort training count differs")
        return value, {"reused": True, **record}
    value = collect_rank8(context, config)
    temporary = output / "new_cohort.partial.pt"
    torch.save(value, temporary)
    os.replace(temporary, path)
    record = {"config_sha256": config_sha, "parent_state_hash": _hash_state(context["parent_state"]),
              "pt_sha256": sha(path), "local_calls": value["local_calls"],
              "optimizer_steps": value["optimizer_steps"], "training_seconds": value["elapsed_seconds"]}
    write_json(seal, record)
    return value, {"reused": False, **record}


def execute(config_path, config_sha, source_manifest, source_sha, input_index, device):
    verify_source(source_manifest, source_sha)
    config_path = Path(config_path)
    if sha(config_path) != config_sha:
        raise ValueError("compiled diagnostic config differs")
    config = json.loads(config_path.read_text())
    config["expanded_sha256"] = config_sha  # in-memory metadata; original bytes retained
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {"job_id": config["job"]["job_id"], "config_sha256": config_sha,
                   "source_manifest_sha256": source_sha, "input_index_sha256": sha(input_index)}
        identity = output / "identity.json"
        if identity.exists() and json.loads(identity.read_text()) != binding:
            raise ValueError("output already belongs to different inputs/source")
        write_json(identity, binding)
        if (output / "result.json").exists():
            raise ValueError("completed diagnostic exists; use independent auditor")
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / str(len(list(attempts.glob("*.json"))) + 1)
        started = time.time()
        write_json(attempt.with_suffix(".json"), {"state": "running", "started": started, **binding})
        try:
            context = prepare_context(config, device, input_index_path=Path(input_index))
            write_json(output / "input_receipt.json", context["input_receipt"])
            if config["job"]["phase"] == "BANK_READOUT":
                updates, collection = context["aggregate_updates"], {"local_calls": 0, "optimizer_steps": 0}
            else:
                value, collection = restore_or_collect(context, config, output, config_sha)
                updates = value["aggregate_updates"]
            reports = {}
            for update in updates:
                support = update["support"]
                cached = {}
                def reader(candidate):
                    digest = _hash_state(candidate)
                    if digest not in cached:
                        cached[digest] = raw_readout(context["model"], candidate,
                                                    context["readout_dataset"], expected_samples=2048)
                    return cached[digest]
                _, feedback = evaluate_feedback(context["model"], context["parent_state"],
                    update["proposal"], update["children"], update["weights"], context["public"],
                    reader, config["job"]["seed"], config["job"]["anchor_round"] + 1, config=config)
                feedback["transfer_chain"] = transfer_chain(context, update, reader)
                reports[support] = feedback
                write_json(output / f"feedback_{support}.json", feedback)
            result = {"schema": "pcla-v8-diagnostic-result-1", **binding,
                      "state": "executed_pending_audit", "job": config["job"],
                      "input_receipt": context["input_receipt"], "collection": collection,
                      "old_reference": context["old_reference"], "reports": reports,
                      "elapsed_seconds": time.time() - started,
                      "evidence": {p.name: sha(p) for p in output.glob("feedback_*.json")}}
            write_json(output / "result.json", result)
            write_json(attempt.with_suffix(".json"), {"state": "executed_pending_audit",
                       "started": started, "ended": time.time(), "result_sha256": sha(output / "result.json"), **binding})
            return result
        except BaseException:
            write_json(attempt.with_suffix(".json"), {"state": "failed_preserved", "started": started,
                       "ended": time.time(), "traceback": traceback.format_exc(), **binding})
            raise


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "config-sha256", "source-manifest", "source-sha256", "input-index"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    execute(args.config, args.config_sha256, args.source_manifest, args.source_sha256,
            args.input_index, args.device)


if __name__ == "__main__":
    main()
