"""Compile PostV diagnostic identities from the frozen Pro matrix and intake index.

This module only describes work. It never loads a checkpoint, dispatches a job,
or treats a historical D066 checkpoint hash as known.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


SEEDS = {101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307}
RULES = ("ordinary_mean", "scalar4", "scalar17", "block16", "separable", "old_finite_bank_reference")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compile_diagnostics(
    *, common_path: Path, jobs_path: Path, input_index_path: Path,
    authorization_path: Path, output_root: str, model_repository: str | None = None,
) -> list[dict]:
    """Return exactly 96 bank readouts and 24 rank-8 transfer descriptions."""
    common = json.loads(common_path.read_text())
    index = json.loads(input_index_path.read_text())
    auth = json.loads(authorization_path.read_text())
    with jobs_path.open(newline="") as f:
        jobs = list(csv.DictReader(f))
    _require(auth.get("resource_ceiling", {}).get("A100_for_this_run") == 0,
             "new diagnostic A100 allocation is not zero")
    _require(auth.get("resource_ceiling", {}).get("4090") == 14,
             "expected authorized 14 RTX4090D")
    _require("96 old-bank readouts +24 rank8 cohorts" in auth.get("scientific_scope", ""),
             "diagnostic scope not authorized")
    _require(common["data"]["counts"]["public_training"] == 512 and
             common["data"]["counts"]["cohort_readout"] == 2048,
             "Pro diagnostic data roles changed")
    _require(common["training"]["local_steps"] == 16 and
             common["federation"]["clients_per_round"] == 10,
             "Pro cohort training changed")
    _require(len(jobs) == 120 and len({r["job_id"] for r in jobs}) == 120,
             "diagnostic jobs must be 120 unique identities")
    banks = {(x["seed"], x["parent_recipe_id"], x["anchor_round"], x["factor"]): x
             for x in index["cohort_bank_objects"]}
    parents = {(x["seed"], x["parent_round"]): x for x in index["d066_parent_checkpoints"]}
    _require(len(banks) == 96 and len(parents) == 24, "intake object identity collision")
    _require({x["seed"] for x in index["cohort_bank_objects"]} == SEEDS and
             {x["seed"] for x in index["d066_parent_checkpoints"]} == SEEDS,
             "intake seed matrix differs")
    _require(index["counts"] == {"cohort_bank_objects": 96,
             "cohort_server_candidate_sets": 192, "d066_parent_checkpoints": 24},
             "intake matrix counts differ")
    _require(str(output_root).startswith("/") and not str(output_root).endswith("/"),
             "output root must be an absolute clean remote path")
    hashes = {"common_sha256": file_sha256(common_path), "jobs_sha256": file_sha256(jobs_path),
              "input_index_sha256": file_sha256(input_index_path),
              "authorization_sha256": file_sha256(authorization_path)}
    result = []
    for row in jobs:
        seed, round_ = int(row["seed"]), int(row["anchor_round"])
        _require(seed in SEEDS and round_ in (10, 60), f"bad job identity {row['job_id']}")
        _require(row["readout_role"] == "cohort_readout2048" and
                 row["primary_selection_role"] == "public512", "job role violation")
        _require(row["factor"] in ("a", "b", "both"), "unknown factor")
        bank_mode = row["phase"] == "BANK_READOUT"
        if bank_mode:
            key = (seed, row["parent_recipe"], round_, row["factor"])
            _require(key in banks and row["parent_recipe"] in ("D069", "D072"),
                     f"bank job has no exact input: {row['job_id']}")
            x = banks[key]
            _require(x["representation"] == {"D069": "full_r16", "D072": "full_alt"}[row["parent_recipe"]],
                     "recipe-to-bank representation mismatch")
            _require(row["job_id"] == f"BANK_{row['parent_recipe']}_r{round_}_{row['factor']}_s{seed}",
                     "bank job ID mismatch")
            _require(row["supports"] == "local;cyclic" and
                     (row["new_client_training_calls"], row["new_client_steps"]) == ("0", "0"),
                     "bank job must replay saved updates without retraining")
            _require(x["bank_pt"]["exists"] and len(x["participants"]) == 10,
                     "bank PT or participants missing")
            _require(x["bank_contents"]["local_16_step_aggregate"] == ["aggregate_states", "local", 16] and
                     x["bank_contents"]["cyclic_16_step_aggregate"] == ["aggregate_states", "cyclic", 16],
                     "bank lacks required actual 16-step aggregates")
            _require(len(x["bank_pt_sha256_expected"]) == 64 and
                     len(x["parent_checkpoint_sha256_expected"]) == 64,
                     "bank or old anchor lacks historical SHA")
            source = {
                "kind": "saved_cohort_bank", "object_id": x["object_id"],
                "bank_pt": x["bank_pt"], "bank_sha256_historical_expected": x["bank_pt_sha256_expected"],
                "bank_contents": x["bank_contents"], "supports": ["local", "cyclic"],
                "saved_client_training_calls_reused": 40, "saved_optimizer_steps_reused": 640,
                "participants": x["participants"], "parent_job_id": x["parent_job_id"],
                "parent_checkpoint": x["parent_checkpoint"],
                "parent_checkpoint_sha256_historical_expected": x["parent_checkpoint_sha256_expected"],
                "parent_config_path": x["parent_config_path"],
                "parent_config_sha256_expected": x["parent_config_sha256_expected"],
                "old_config_path": x["config_path"],
                "old_config_sha256_expected": x["config_sha256_expected"],
                "old_audit_path": x["audit_path"],
                "old_audit_sha256_expected": x["audit_sha256_expected"],
                "old_result_sha256_expected": x["result_sha256_expected"],
                "old_source_manifest_sha256_expected": x["source_manifest_sha256_expected"],
                "factor": x["factor"], "representation": x["representation"],
            }
            calls, steps, rank = 0, 0, 16
        else:
            _require(row["phase"] == "R8_TRANSFER" and row["parent_recipe"] == "D066" and
                     row["factor"] == "both" and row["supports"] == "full_r8",
                     "rank8 transfer identity mismatch")
            _require(row["job_id"] == f"R8_TRANSFER_r{round_}_s{seed}", "rank8 job ID mismatch")
            _require((row["new_client_training_calls"], row["new_client_steps"]) == ("10", "160"),
                     "rank8 cohort must train 10 clients x 16 steps")
            x = parents[(seed, round_)]
            _require(x["parent_checkpoint"]["exists"] and x["parent_rank"] == 8 and
                     x["parent_method"] == "full_both" and x["queue_state"] == "completed",
                     "D066 parent checkpoint or identity missing")
            _require(x["parent_checkpoint_sha256_expected"] == "unknown_no_historical_pt_hash",
                     "unexpected D066 hash provenance: review before compiling")
            source = {
                "kind": "rank8_parent_next_cohort", "object_id": x["object_id"],
                "parent_job_id": x["parent_job_id"], "parent_checkpoint": x["parent_checkpoint"],
                "parent_pt_sha256_historical_expected": None,
                "parent_pt_sha256_policy": "compute_current_sha256_on_compute_node_immediately_before_use;record_as_current_not_historical",
                "parent_config_path": x["config_path"],
                "parent_config_sha256_expected": x["config_sha256_expected"],
                "parent_run_identity_path": x["run_identity_path"],
                "parent_run_identity_sha256_expected": x["run_identity_sha256_at_export_and_current"],
                "parent_status_path": x["status_path"],
                "parent_status_sha256_expected": x["status_sha256_at_export_and_current"],
                "old_source_manifest_sha256_expected": x["source_manifest_sha256_expected"],
                "model_sha256_expected": x["model_sha256_expected"],
                "next_participants_policy": "exact frozen seed participation sequence at parent anchor_round + 1; resolve and record IDs before training",
                "training": {"rank": 8, "factors": ["a", "b"], "clients": 10,
                             "local_steps_each": 16, "optimizer": common["training"]["client_optimizer"],
                             "batch_size": common["training"]["batch_size"],
                             "optimizer_persistence": common["training"]["optimizer_persistence"],
                             "lr": 0.001, "head_lr": 0.001, "weight_decay": 0.01,
                             "schedule": "constant",
                             "old_D066_recipe_sha256": "cc30b197a53c9aadc8a1ff0d6244fb1f5fc1875417a95dbac5769027af3787d8"},
            }
            calls, steps, rank = 10, 160, 8
        _require(x["new_readout_hardware_family"] == "RTX4090D", "intake hardware override missing")
        config = {
            "schema": "pcla-v8-diagnostic-config-v1", "protocol": common["protocol"],
            "job": {"job_id": row["job_id"], "phase": row["phase"], "seed": seed,
                    "parent_recipe": row["parent_recipe"], "anchor_round": round_,
                    "factor": row["factor"], "hardware_family": "RTX4090D",
                    "old_hardware_family": x["old_execution_hardware_family"],
                    "new_client_training_calls": calls, "new_client_steps": steps},
            "source": source,
            "data": {"manifest_path": x.get("data_manifest_path") or
                     f"{index['remote_v7_root']}/data/pcla_v7/train_roles.json",
                     "manifest_sha256_expected": x["data_manifest_sha256_expected"],
                     "roles": x["roles"], "selection_role": "public_training_512",
                     "readout_role": "cohort_readout_2048",
                     "readout_must_not_select_or_fit": True,
                     "selection_objective": common["controller"]["objective"]},
            "model": {"old_repository": x["model"]["repository"],
                      "new_repository": model_repository,
                      "revision": common["model"]["revision"],
                      "dtype": common["model"]["dtype"],
                      "max_length": common["model"]["max_length"],
                      "targets": common["model"]["targets"], "rank": rank,
                      "base_weights": common["model"]["base_weights"]},
            "controller": {"candidate_rules": RULES if bank_mode else RULES[:-1],
                           "grid4": common["controller"]["grid4"],
                           "scalar17": common["controller"]["scalar17"],
                           "tie_abs_nll": common["controller"]["tie_abs_nll"],
                           "tie_rule": common["controller"]["tie_rule"],
                           "public_selection_then_independent_readout": True,
                           "posthoc_readout_best_only_diagnostic": True,
                           "preserve_actual_delta_and_single_sample_weighting": True,
                           "capture": ["ordinary_mean", "four_corners", "block_minus_scalar17_independent_nll",
                                       "block_minus_mean_independent_nll", "selected_alpha_beta",
                                       "public_to_independent_agreement", "necessary_server_time",
                                       "local_to_individual_global_to_aggregate"]},
            "binding": {"input_index_path": str(input_index_path.resolve()), **hashes,
                        "requires_on_node_input_hash_validation": True,
                        "requires_new_model_stage_validation": True,
                        "requires_runtime_and_scientific_audit": True},
            "output_dir": f"{output_root}/{row['job_id']}",
            "launch_status": "blocked_pending_runtime_integration_and_on_node_preflight",
        }
        result.append(config)
    _require(sum(c["job"]["phase"] == "BANK_READOUT" for c in result) == 96 and
             sum(c["job"]["phase"] == "R8_TRANSFER" for c in result) == 24 and
             sum(c["job"]["new_client_training_calls"] for c in result) == 240 and
             sum(c["job"]["new_client_steps"] for c in result) == 3840 and
             len({c["output_dir"] for c in result}) == 120,
             "compiled scientific matrix is incomplete or output collides")
    return result
