"""Verified old-bank and D066 input bridge for PostV diagnostic drivers.

Only the driver decides when to call these functions. No queue, output, or
checkpoint is mutated here. Old v7 runtime/data classes are imported directly;
their executing file digests are included in the input receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from pcla_repair.core import keyed_seed, tensor_digest
from pcla_research_v7.data import TokenCache, role_ids
from pcla_research_v7.diagnostic_configuration import participants as frozen_participants, verify_anchor
from pcla_research_v7.runtime import aggregate, full_support, load_state, local_train, state
from .diagnostic_configuration import file_sha256


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _checked_file(path: Path, expected: str | None = None) -> str:
    _require(path.is_file(), f"missing input: {path}")
    digest = file_sha256(path)
    if expected is not None:
        _require(digest == expected, f"SHA-256 differs: {path}")
    return digest


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _index_path(config: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("PCLA_V8_INPUT_INDEX"):
        return Path(os.environ["PCLA_V8_INPUT_INDEX"])
    saved = Path(config["binding"]["input_index_path"])
    if saved.is_file():
        return saved
    # Local compilation paths are not valid on the HPC compute node.
    return Path(config["output_dir"]).parents[2] / "checks/diagnostic_intake01/INPUT_INDEX.json"


def _check_intake_parent_binding(config: dict, intake: dict, bank_mode: bool) -> None:
    """The two frozen intake families use different names for the V config."""
    source = config["source"]
    config_key = "parent_config_path" if bank_mode else "config_path"
    sha_key = "parent_config_sha256_expected" if bank_mode else "config_sha256_expected"
    round_key = "anchor_round" if bank_mode else "parent_round"
    _require(source["parent_checkpoint"] == intake["parent_checkpoint"] and
             source["parent_config_path"] == intake[config_key] and
             source["parent_config_sha256_expected"] == intake[sha_key] and
             config["job"]["anchor_round"] == intake[round_key],
             "compiled parent path/metadata differs from sealed intake")


def _state_shape(values: dict, parent: dict | None = None) -> None:
    _require(isinstance(values, dict) and values and
             {"classifier.weight", "classifier.bias"} <= set(values) and
             any(k.endswith(".lora_a") for k in values) and
             any(k.endswith(".lora_b") for k in values), "incomplete adapter/head state")
    for key, value in values.items():
        _require(isinstance(key, str) and isinstance(value, torch.Tensor) and
                 value.device.type == "cpu" and value.is_floating_point() and
                 bool(torch.isfinite(value).all()), f"invalid tensor in state: {key}")
        if parent is not None:
            _require(key in parent and value.shape == parent[key].shape and
                     value.dtype == parent[key].dtype, f"state shape/dtype differs: {key}")
    if parent is not None:
        _require(set(values) == set(parent), "state tensor keys differ from parent")


def _role_rows(ids: list[str]) -> tuple[int, ...]:
    result = []
    for value in ids:
        _require(isinstance(value, str) and value.startswith("train:") and
                 value[6:].isdigit() and str(int(value[6:])) == value[6:],
                 "diagnostic role has invalid train source identity")
        result.append(int(value[6:]))
    _require(len(result) == len(set(result)), "diagnostic role duplicates source rows")
    return tuple(result)


def _role_views(manifest: dict, tokens: TokenCache, seed: int, selected: list[int]) -> dict:
    _require(len(manifest["roles"]["public_training"]) == 512 and
             len(manifest["roles"]["cohort_readout"]) == 2048,
             "public/readout role size differs")
    _require(len(selected) == len(set(selected)) == 10, "cohort needs ten distinct clients")
    public_ids = role_ids(manifest, "public_training", "algorithm")
    readout_ids = role_ids(manifest, "cohort_readout", "cohort_report")
    local_ids = {cid: role_ids(manifest, "client_local_readout", "local_diagnostic", seed, cid)
                 for cid in selected}
    train_ids = {cid: role_ids(manifest, "client_train", "algorithm", seed, cid)
                 for cid in selected}
    _require(all(len(v) == 64 for v in local_ids.values()) and
             all(len(v) == 256 for v in train_ids.values()), "old seed partition size differs")
    groups = [set(_role_rows(public_ids)), set(_role_rows(readout_ids))]
    groups += [set(_role_rows(ids)) for ids in local_ids.values()]
    for i, rows in enumerate(groups):
        _require(not any(rows & previous for previous in groups[:i]), "diagnostic data roles overlap")
    return {
        "public": tokens.dataset(manifest, "public_training", "algorithm"),
        "readout_dataset": tokens.dataset(manifest, "cohort_readout", "cohort_report"),
        "local_readout_by_client": {cid: tokens.dataset(manifest, "client_local_readout",
                                         "local_diagnostic", seed, cid) for cid in selected},
        "client_train_by_client": {cid: tokens.dataset(manifest, "client_train", "algorithm",
                                         seed, cid) for cid in selected},
        "role_source_rows": {"public": _role_rows(public_ids),
                             "readout": _role_rows(readout_ids),
                             "local_by_client": {cid: _role_rows(ids) for cid, ids in local_ids.items()}},
    }


def _bank_updates(bank: dict, parent: dict, config: dict) -> list[dict]:
    source = config["source"]
    selected = source["participants"]
    _require(bank.get("schema") == "pcla-v7-cohort-bank-1" and
             bank.get("object_id") == source["object_id"] and
             bank.get("factor") == source["factor"] and
             bank.get("participants") == selected and
             bank.get("anchor_sha256") == source["parent_checkpoint_sha256_historical_expected"],
             "old bank payload identity differs")
    updates = []
    for support in ("local", "cyclic"):
        prefixes = bank.get("prefix_states", {}).get(support)
        aggregates = bank.get("aggregate_states", {}).get(support)
        _require(isinstance(prefixes, dict) and set(prefixes) == set(selected) and
                 isinstance(aggregates, dict) and 16 in aggregates,
                 f"{support} lacks saved ten-client 16-step states")
        proposal = aggregates[16]
        _state_shape(proposal, parent)
        children = []
        for cid in selected:
            _require(isinstance(prefixes[cid], dict) and 16 in prefixes[cid],
                     f"{support} client {cid} lacks saved step 16")
            child = prefixes[cid][16]
            _state_shape(child, parent)
            children.append(child)
        # The v7 aggregate uses ten equal sample counts. Verify saved aggregate
        # once; downstream consumers receive raw children and one .1 weighting.
        recomputed = aggregate(parent, children, [1] * 10)
        _require(all(torch.equal(proposal[name], recomputed[name]) for name in parent),
                 f"{support} saved proposal differs from raw-child aggregation")
        updates.append({"support": support, "proposal": proposal, "children": children,
                        "weights": [0.1] * 10, "children_are_raw_unweighted": True,
                        "optimizer_steps_reused": 16})
    return updates


def prepare_context(config: dict, device: str | torch.device, *,
                    input_index_path: Path | None = None, model_prepare=None) -> dict:
    """Load verified old inputs and construct only the authorized data views.

    `model_prepare` is injectable for CPU tests; production uses the existing
    v7 model/data loader and verifies its provenance against run_identity.json.
    """
    _require(config.get("schema") == "pcla-v8-diagnostic-config-v1" and
             config.get("launch_status") == "blocked_pending_runtime_integration_and_on_node_preflight" and
             config["job"]["hardware_family"] == "RTX4090D", "diagnostic config identity differs")
    index_path = _index_path(config, input_index_path)
    index_sha = _checked_file(index_path, config["binding"]["input_index_sha256"])
    index = json.loads(index_path.read_text())
    bank_mode = config["job"]["phase"] == "BANK_READOUT"
    objects = index["cohort_bank_objects" if bank_mode else "d066_parent_checkpoints"]
    matching = [x for x in objects if x["object_id"] == config["source"]["object_id"]]
    _require(len(matching) == 1, "config object absent or duplicated in sealed intake")
    intake = matching[0]
    _require(intake["seed"] == config["job"]["seed"] and
             intake["parent_recipe_id"] == config["job"]["parent_recipe"],
             "input index/config identity mismatch")
    source = config["source"]
    parent_path = Path(source["parent_checkpoint"]["path"])
    _check_intake_parent_binding(config, intake, bank_mode)
    parent_stat = parent_path.stat()
    _require(parent_stat.st_size == source["parent_checkpoint"]["bytes"] and
             parent_stat.st_mtime_ns == source["parent_checkpoint"]["mtime_ns"],
             "parent checkpoint size or mtime differs from intake")
    historical = source.get("parent_checkpoint_sha256_historical_expected")
    if bank_mode:
        _require(historical == intake["parent_checkpoint_sha256_expected"],
                 "bank anchor historical SHA changed in compiled config")
    else:
        _require(source["parent_pt_sha256_historical_expected"] is None and
                 intake["parent_checkpoint_sha256_expected"] == "unknown_no_historical_pt_hash",
                 "D066 parent PT must not claim a historical seal")
    parent_sha = _checked_file(parent_path, historical)
    parent_config_path = Path(source["parent_config_path"])
    _checked_file(parent_config_path, source["parent_config_sha256_expected"])
    old_config = json.loads(parent_config_path.read_text())
    _require(old_config["job"]["job_id"] == source["parent_job_id"] and
             old_config["job"]["seed"] == config["job"]["seed"] and
             old_config["job"]["rank"] == config["model"]["rank"],
             "parent recipe/seed/rank differs")
    manifest_path = _resolve_path(old_config["manifest"], parent_config_path)
    manifest_sha = _checked_file(manifest_path, config["data"]["manifest_sha256_expected"])
    manifest = json.loads(manifest_path.read_text())
    selected_ids = sorted(map(int, manifest["partitions"][str(config["job"]["seed"])]["client_train"]))
    _require(selected_ids == list(range(50)), "old seed client population differs")
    anchor_round = config["job"]["anchor_round"]
    selected = frozen_participants(config["job"]["seed"], anchor_round + 1, selected_ids)
    if bank_mode:
        _require(selected == source["participants"], "old bank participants differ from frozen schedule")
    source_manifest = _resolve_path(old_config["source_manifest"], parent_config_path)
    source_manifest_sha = _checked_file(source_manifest)
    anchor = verify_anchor(parent_path, seed=config["job"]["seed"], anchor=anchor_round,
                           client_ids=selected_ids, parent_config=parent_config_path,
                           parent_config_sha256=source["parent_config_sha256_expected"],
                           source_manifest_sha256=source_manifest_sha)
    _require(anchor["state"] == "anchor_verified" and anchor["participants"] == selected and
             anchor["anchor_sha256"] == parent_sha, "old anchor or frozen next cohort differs")
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    _require(_checked_file(parent_path, parent_sha) == parent_sha and
             payload["round"] == anchor_round, "old anchor changed during load")
    parent = payload["adapter"]
    _state_shape(parent)
    bank = None
    updates = []
    bank_sha = None
    old_reference = None
    if bank_mode:
        bank_path = Path(source["bank_pt"]["path"])
        _require(source["bank_pt"] == intake["bank_pt"] and
                 bank_path.stat().st_size == source["bank_pt"]["bytes"] and
                 bank_path.stat().st_mtime_ns == source["bank_pt"]["mtime_ns"],
                 "old bank path/size/mtime differs from intake")
        _require(source["bank_sha256_historical_expected"] == intake["bank_pt_sha256_expected"],
                 "bank historical SHA changed in compiled config")
        bank_sha = _checked_file(bank_path, source["bank_sha256_historical_expected"])
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        _checked_file(bank_path, bank_sha)
        updates = _bank_updates(bank, parent, config)
        old_diag_config_path = Path(source["old_config_path"])
        _checked_file(old_diag_config_path, source["old_config_sha256_expected"])
        old_diag_config = json.loads(old_diag_config_path.read_text())
        _require(old_diag_config["job_id"] == source["object_id"],
                 "old diagnostic config identity differs")
        old_output = Path(old_diag_config["output"])
        stage_path = old_output / "bank_stage/bank_readout_stage.json"
        stage = json.loads(stage_path.read_text())
        result_path = Path(stage["result_path"])
        _require(result_path.parent == old_output / "bank_stage" and
                 stage["result_sha256"] == source["old_result_sha256_expected"],
                 "old finite-bank result identity differs")
        result_sha = _checked_file(result_path, source["old_result_sha256_expected"])
        old_reference = {"result_path": str(result_path),
                         "result_sha256_expected_and_current": result_sha,
                         "stage_receipt_path": str(stage_path),
                         "stage_receipt_sha256_current": _checked_file(stage_path),
                         "raw": json.loads(result_path.read_text()),
                         "selection_rule": "reference_only;never_reuse_old_readout_to_select_new_public_coefficient"}
    cache_path = _resolve_path(old_config["token_cache"], parent_config_path)
    cache_meta_path = (_resolve_path(old_config["token_cache_metadata"], parent_config_path)
                       if old_config.get("token_cache_metadata") else cache_path.with_suffix(".json"))
    train_path = _resolve_path(old_config["train_file"], parent_config_path)
    _require(_checked_file(train_path) == manifest["source_sha256"], "old train source differs")
    tokens = TokenCache(cache_path, cache_meta_path)
    _require(tokens.meta["identity"]["split_sha256"] == manifest["split_sha256"] and
             tokens.meta["identity"]["max_length"] == config["model"]["max_length"],
             "old token cache identity differs")
    model_repo = config["model"]["new_repository"] or os.environ.get("PCLA_V8_MODEL_REPOSITORY") or old_config["model"]["repository"]
    _require(Path(model_repo).is_dir(), "verified on-node model repository is absent")
    expanded = copy.deepcopy(old_config)
    expanded["model"]["repository"] = model_repo
    expanded["model"]["local_files_only"] = True
    if model_prepare is None:
        from scripts.run_pcla_v7 import prepare as model_prepare
    model, clients, capacities, unused_development, provenance = model_prepare(
        expanded, parent_config_path, torch.device(device))
    _require(set(selected) <= set(clients) and set(selected) <= set(capacities),
             "model loader omitted frozen clients")
    identity_path = parent_path.parent / "run_identity.json"
    identity = json.loads(identity_path.read_text())["provenance"]
    for key in ("manifest_sha256", "split_sha256", "source_sha256", "token_cache_sha256",
                "token_cache_identity_sha256", "pretrained_model_sha256", "b8_wire_manifest_sha256"):
        _require(provenance.get(key) == identity.get(key), f"old parent provenance differs: {key}")
    load_state(model, parent)
    views = _role_views(manifest, tokens, config["job"]["seed"], selected)
    import pcla_research_v7.data as old_data
    import pcla_research_v7.runtime as old_runtime
    import scripts.run_pcla_v7 as old_driver
    source_sha = {m.__name__: file_sha256(Path(m.__file__))
                  for m in (old_data, old_runtime, old_driver)}
    receipt = {"input_index_sha256": index_sha, "parent_pt_sha256_current": parent_sha,
               "parent_pt_sha256_historical_expected": historical,
               "parent_hash_provenance": "historical_and_current" if bank_mode else "current_only_no_historical_pt_hash",
               "bank_pt_sha256_current": bank_sha,
               "bank_pt_sha256_historical_expected": source.get("bank_sha256_historical_expected"),
               "data_manifest_sha256_current": manifest_sha,
               "token_cache_sha256_current": tokens.meta["cache_sha256"],
               "old_source_file_sha256": source_sha,
               "participants": selected, "anchor_verification": anchor}
    return {"model": model, "parent_state": parent, "public": views["public"],
            "readout_dataset": views["readout_dataset"],
            "local_readout_by_client": views["local_readout_by_client"],
            "client_train_by_client": views["client_train_by_client"],
            "role_source_rows": views["role_source_rows"],
            "provenance": provenance, "input_receipt": receipt,
            "participants": selected, "aggregate_updates": updates,
            "old_reference": old_reference,
            "old_bank": bank, "parent_checkpoint": payload,
            "anchor": anchor, "capacities": {cid: capacities[cid] for cid in selected}}


def collect_rank8(context: dict, config: dict) -> dict:
    """Run only one fresh D066 anchor+1 rank8 cohort with the v7 local closure."""
    _require(config["job"]["phase"] == "R8_TRANSFER" and
             config["job"]["new_client_training_calls"] == 10 and
             config["job"]["new_client_steps"] == 160 and
             config["model"]["rank"] == 8 and not context["aggregate_updates"],
             "rank8 collection requires a fresh D066 context")
    selected, parent, model = context["participants"], context["parent_state"], context["model"]
    _require(len(selected) == 10 and len(set(selected)) == 10, "rank8 cohort identity differs")
    _state_shape(parent)
    support = full_support(model, "both")
    _require(len(support) == 24 and all(all(bool(mask.all()) and len(mask) == 8
                                       for mask in factors.values()) for factors in support.values()),
             "rank8 full both-factor support differs")
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    modes = {name: module.training for name, module in model.named_modules()}
    children, details = [], []
    started = time.perf_counter()
    try:
        for cid in selected:
            load_state(model, parent)
            dataset = context["client_train_by_client"][cid]
            _require(len(dataset) == 256, f"client {cid} train role is not 256 rows")
            train_seed = keyed_seed(config["job"]["seed"], "train", config["job"]["anchor_round"] + 1, cid)
            outcome = local_train(model, dataset, support, seed=train_seed,
                                  lr=config["source"]["training"]["lr"],
                                  head_lr=config["source"]["training"]["head_lr"],
                                  batch_size=16, epochs=1, betas=(.9, .999), eps=1e-8,
                                  weight_decay=config["source"]["training"]["weight_decay"])
            _require(outcome["optimizer_steps"] == 16, f"client {cid} did not execute 16 steps")
            child = state(model)
            _state_shape(child, parent)
            children.append(child)
            details.append({"client_id": cid, "train_seed": train_seed,
                            "optimizer_steps": outcome["optimizer_steps"],
                            "train_seconds": outcome["train_seconds"],
                            "actual_child_state_sha256": tensor_digest(child)})
        proposal = aggregate(parent, children, [1] * 10)
        _state_shape(proposal, parent)
        elapsed = time.perf_counter() - started
        update = {"support": "rank8", "proposal": proposal, "children": children,
                  "weights": [0.1] * 10, "children_are_raw_unweighted": True,
                  "optimizer_steps_new": 160}
        return {"aggregate_updates": [update], "participants": selected,
                "local_calls": 10, "optimizer_steps": 160,
                "elapsed_seconds": elapsed, "client_details": details,
                "proposal_state_sha256": tensor_digest(proposal)}
    finally:
        try:
            load_state(model, parent)
            for name, module in model.named_modules():
                module.training = modes[name]
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
