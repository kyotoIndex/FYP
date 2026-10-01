"""Strict compiler for the frozen v7 recipe and first-batch job matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import PROTOCOL, digest

RECIPE_KEYS = {"recipe_id", "protocol", "group", "method", "representation", "period", "lr",
               "rank", "xi", "native_ranks", "optimizer", "adapter_weight_decay",
               "adapter_clip_norm", "head_optimizer", "head_lr", "schedule", "head_lr_ratio",
               "factor_schedule", "action_unit", "selector_history", "sketch_h",
               "min_rounds", "max_rounds", "budget_multiplier", "budget_id",
               "uses_public_training", "evaluation_role", "full_forward", "base_policy",
               "required_adapter"}
JOB_KEYS = {"job_id", "protocol", "phase", "recipe_id", "seed", "group", "method",
            "representation", "lr", "period", "rank", "xi", "min_rounds", "max_rounds",
            "budget_id", "eval_role", "hardware_block", "preferred_pool", "requires",
            "depends_on", "priority", "status", "output_dir"}
METHODS = {"local", "cyclic", "JL", "JQ", "full_both", "full_alt", "tied_local",
           "local_both_half", "cyclic_both_half", "equal_quota_cyclic", "layer",
           "energy", "fedsb", "fedsvd", "fedplora"}
ADAPTERS = {"core", "fedsb", "fedsvd", "fedplora"}
COMMON_KEYS = {"protocol", "status", "base_commit", "model", "lora", "federation",
               "training", "scoring", "cohort", "budget", "evaluation", "data_roles"}
COMMON_NESTED_KEYS = {
    "model": {"repository", "revision", "max_length", "dtype"},
    "lora": {"targets", "rank", "scaling", "dropout", "forward", "base_policy"},
    "federation": {"clients", "clients_per_round", "client_samples", "local_readout_samples",
                   "dirichlet_alpha", "preserve_global_label_counts", "participation",
                   "population_weight", "aggregation_weight", "aggregation", "tier_counts",
                   "single_factor_projection_rank_positions"},
    "training": {"min_rounds", "max_rounds", "batch_size", "local_epochs",
                 "local_optimizer_steps", "optimizer", "optimizer_state", "betas", "eps",
                 "weight_decay", "clip_grad_norm", "factor_schedule", "learning_rates", "schedule"},
    "scoring": {"policy", "batch_size", "batches", "model_mode", "data", "history",
                "layer_or_projection_minimum_quota", "tie_break"},
    "cohort": {"sketch_h", "sketch", "packet_dtype", "orientation_packet_dtype",
               "sketch_seed_scope", "candidate_masks", "greedy_sweeps", "lambda_strength_grid",
               "coordination_periods", "head_in_surrogate", "head_in_training"},
    "budget": {"objective", "unit", "byte_cap", "byte_cap_status", "min_round_diagnostic_only_after_cap",
               "max_rounds", "include_control_handshakes", "charge_wire_but_return_previous_model_on_incomplete_round",
               "shared_pretrained_base", "cold_start"},
    "evaluation": {"D0_V", "primary_endpoint", "long_horizon_secondary", "early_secondary",
                   "ema", "no_test_access_before_final_freeze"},
    "data_roles": {"client_train", "client_local_readout", "public_training",
                   "cohort_select", "cohort_readout", "recipe_dev", "manifest_sha256",
                   "manifest_status"},
}


def load_common(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to parse handoff common.yaml") from exc
    value = yaml.safe_load(path.read_text())
    _strict_keys(value, COMMON_KEYS, "common")
    for key, expected in COMMON_NESTED_KEYS.items():
        _strict_keys(value[key], expected, f"common.{key}")
    if value["protocol"] != PROTOCOL or value["federation"]["clients"] != 50:
        raise ValueError("common protocol/federation mismatch")
    if value["data_roles"]["client_train"] != 12800 or value["data_roles"]["recipe_dev"] != 4096:
        raise ValueError("common role count mismatch")
    return value


def _strict_keys(value, expected, kind):
    missing, extra = expected - set(value), set(value) - expected
    if missing or extra:
        raise ValueError(f"{kind} missing={sorted(missing)} unused={sorted(extra)}")


def _validate_recipe(recipe):
    _strict_keys(recipe, RECIPE_KEYS, "recipe")
    if recipe["protocol"] != PROTOCOL or recipe["method"] not in METHODS:
        raise ValueError("unsupported protocol or method")
    if recipe["required_adapter"] not in ADAPTERS:
        raise ValueError("unsupported adapter")
    if recipe["method"] in {"fedsb", "fedsvd", "fedplora"} and recipe["required_adapter"] != recipe["method"]:
        raise ValueError("native method adapter mismatch")
    if recipe["method"] not in {"fedsb", "fedsvd", "fedplora"} and recipe["required_adapter"] != "core":
        raise ValueError("core method adapter mismatch")
    if recipe["representation"] not in {"N", "D", "S", "T", "native_SVD"}:
        raise ValueError("unsupported representation")
    if recipe["period"] not in {0, 1, 5} or recipe["rank"] not in {4, 8, 16, 32}:
        raise ValueError("unsupported period/rank")
    if recipe["optimizer"] not in {"adamw", "sgd"} or recipe["head_optimizer"] != "adamw":
        raise ValueError("unsupported optimizer")
    if recipe["factor_schedule"] not in {"alternating_B_first", "both", "R_only", "B_only"}:
        raise ValueError("unsupported factor schedule")
    if recipe["action_unit"] not in {"projection_rank", "paired_QV", "native"}:
        raise ValueError("unsupported action unit")
    if recipe["selector_history"] != "none_fresh" or recipe["schedule"] != "constant":
        raise ValueError("unsupported history/schedule")
    if recipe["min_rounds"] != 100 or recipe["max_rounds"] != 200 or recipe["budget_id"] != "B8" or recipe["budget_multiplier"] != 1.0:
        raise ValueError("unexpected horizon/budget")
    if recipe["evaluation_role"] != "recipe_dev" or recipe["full_forward"] is not True:
        raise ValueError("unsupported evaluation/forward")
    if recipe["base_policy"] not in {"immutable", "native"}:
        raise ValueError("unsupported base policy")
    if recipe["lr"] <= 0 or recipe["head_lr"] <= 0 or recipe["adapter_weight_decay"] < 0:
        raise ValueError("invalid learning rate/decay")
    if recipe["sketch_h"] not in {0, 32}:
        raise ValueError("unsupported sketch dimension")
    if recipe["native_ranks"] is not None and (not isinstance(recipe["native_ranks"], list) or not all(isinstance(x, int) and x > 0 for x in recipe["native_ranks"])):
        raise ValueError("invalid native ranks")
    return recipe


def compile_recipe(recipe, supported_adapters=None):
    """Return every supplied field verbatim plus a stable content identity.

    Runtime must dispatch `required_adapter`; requesting only a staged subset
    rejects native recipes rather than silently treating them as core.
    """
    _validate_recipe(recipe)
    if supported_adapters is not None and recipe["required_adapter"] not in supported_adapters:
        raise NotImplementedError(f"adapter {recipe['required_adapter']} unavailable")
    result = dict(recipe)
    result["recipe_sha256"] = digest(recipe)
    return result


def compile_job(job, recipes, supported_adapters=None):
    _strict_keys(job, JOB_KEYS, "job")
    if job["protocol"] != PROTOCOL or job["phase"] != "D0" or job["seed"] not in {401, 409, 419}:
        raise ValueError("job outside frozen D0 matrix")
    if job["status"] != "planned_not_dispatched" or job["eval_role"] != "recipe_dev":
        raise ValueError("job changed from frozen plan")
    if job["recipe_id"] not in recipes:
        raise ValueError("unknown recipe")
    recipe = compile_recipe(recipes[job["recipe_id"]], supported_adapters)
    for jk, rk in (("protocol", "protocol"), ("recipe_id", "recipe_id"),
                   ("group", "group"), ("method", "method"),
                   ("representation", "representation"), ("lr", "lr"),
                   ("period", "period"), ("rank", "rank"), ("xi", "xi"),
                   ("min_rounds", "min_rounds"), ("max_rounds", "max_rounds"),
                   ("budget_id", "budget_id"), ("eval_role", "evaluation_role")):
        if job[jk] != recipe[rk]:
            raise ValueError(f"job/recipe mismatch: {jk}")
    result = {"job": dict(job), "recipe": recipe,
              "config_sha256": digest({"job": job, "recipe": recipe})}
    return result


def compile_first_batch(recipe_path: Path, job_path: Path, supported_adapters=None,
                        common_path: Path | None = None):
    recipes_raw = json.loads(recipe_path.read_text())
    if len(recipes_raw) != 117:
        raise ValueError("frozen D0 requires 117 recipes")
    recipes = {r["recipe_id"]: r for r in recipes_raw}
    if len(recipes) != 117:
        raise ValueError("duplicate recipe ID")
    jobs_raw = [json.loads(line) for line in job_path.read_text().splitlines() if line]
    if len(jobs_raw) != 351 or len({j["job_id"] for j in jobs_raw}) != 351:
        raise ValueError("frozen D0 requires 351 unique jobs")
    jobs = [compile_job(job, recipes, supported_adapters) for job in jobs_raw]
    if common_path is not None:
        common = load_common(common_path)
        for item in jobs:
            item["common"] = common
            item["config_sha256"] = digest({"job": item["job"], "recipe": item["recipe"],
                                            "common": common})
    by_recipe = {}
    for item in jobs:
        by_recipe.setdefault(item["recipe"]["recipe_id"], set()).add(item["job"]["seed"])
    if set(by_recipe) != set(recipes) or any(seeds != {401, 409, 419} for seeds in by_recipe.values()):
        raise ValueError("D0 recipe/seed coverage incomplete")
    return jobs


STAGED_RUNTIME_METHODS = {"local", "cyclic", "full_both", "full_alt",
                          "local_both_half", "cyclic_both_half", "JL", "JQ",
                          "tied_local", "layer", "energy", "equal_quota_cyclic"}


def expand_for_runner(compiled, common, *, manifest_path: Path, train_file: Path,
                      token_cache: Path, model_path: Path, b8_bytes: int):
    """Compile one implemented recipe into the runner's exact JSON contract.

    Keep the complete recipe/common records for audit. Every non-default field
    which the first-stage runner cannot honor causes an error before dispatch.
    """
    job, recipe = compiled["job"], compiled["recipe"]
    method, rep = recipe["method"], recipe["representation"]
    valid_pair = (rep == "N" or
                  (rep in {"D", "S", "T"} and method in {"local", "cyclic"}) or
                  (rep == "S" and method in {"layer", "energy"}))
    if method not in STAGED_RUNTIME_METHODS or not valid_pair:
        raise NotImplementedError(f"runner method/representation {method}/{rep} unavailable")
    coordinated = method in {"JL", "JQ"} or rep == "T"
    if (recipe["sketch_h"] != (32 if coordinated else 0) or
            recipe["xi"] not in ({.25, 1.} if method == "JQ" else {0.}) or
            recipe["period"] not in ({1, 5} if coordinated or rep in {"D", "S"} else {0}) or
            recipe["action_unit"] != ("paired_QV" if method == "tied_local" else "projection_rank")):
        raise NotImplementedError("recipe coordination, period, or action field not implemented")
    if (recipe["required_adapter"] != "core" or recipe["optimizer"] != "adamw" or
            recipe["adapter_clip_norm"] is not None or recipe["uses_public_training"] or
            recipe["base_policy"] != "immutable" or recipe["head_lr"] != recipe["lr"] or
            recipe["head_lr_ratio"] != 1.0 or recipe["native_ranks"] is not None):
        raise NotImplementedError("recipe contains a field not honored by staged runner")
    if recipe["factor_schedule"] != ("both" if method in {"full_both", "local_both_half", "cyclic_both_half"}
                                     else "alternating_B_first"):
        raise NotImplementedError("runner factor schedule mismatch")
    if recipe["rank"] not in {4, 8, 16}:
        raise NotImplementedError("runner rank unsupported")
    if method not in {"full_both", "full_alt"} and recipe["rank"] != 16:
        raise NotImplementedError("partial-action first batch requires rank16")
    if common["protocol"] != PROTOCOL or b8_bytes <= 0:
        raise ValueError("invalid common/B8")
    runtime = {"byte_cap": int(b8_bytes), "clients_per_round": common["federation"]["clients_per_round"],
               "capacity_by_tier": common["federation"]["single_factor_projection_rank_positions"],
               "batch_size": common["training"]["batch_size"],
               "local_epochs": common["training"]["local_epochs"],
               "betas": common["training"]["betas"], "eps": common["training"]["eps"],
               "weight_decay": recipe["adapter_weight_decay"],
               "probe_batch_size": common["scoring"]["batch_size"],
               "probe_batches": common["scoring"]["batches"],
               "sketch_h": recipe["sketch_h"],
               "eval_batch_size": 64, "min_rounds": recipe["min_rounds"],
               "max_rounds": recipe["max_rounds"]}
    result = {"job": job, "recipe": recipe, "common": common,
              "manifest": str(manifest_path), "train_file": str(train_file),
              "token_cache": str(token_cache),
              "b8_wire": str(manifest_path.parent / "b8_wire.json"),
              "model": {"repository": str(model_path), "revision": None,
                        "local_files_only": True, "max_length": common["model"]["max_length"],
                        "dropout": common["lora"]["dropout"]},
              "runtime": runtime}
    result["expanded_sha256"] = digest(result)
    return result


def expand_for_native_runner(compiled, common, *, manifest_path: Path, train_file: Path,
                             token_cache: Path, model_path: Path, b8_bytes: int):
    """Compile exactly the 18 pinned native recipes; reject unhandled fields."""
    job, recipe = compiled["job"], compiled["recipe"]
    method = recipe["method"]
    if method not in {"fedsb", "fedsvd", "fedplora"} or recipe["required_adapter"] != method:
        raise NotImplementedError("native runner accepts only native adapter recipes")
    expected = {
        "fedsb": {"representation": "N", "factor_schedule": "R_only", "optimizer": "adamw",
                  "period": 0, "uses_public_training": True, "base_policy": "native",
                  "rank": {16, 32}, "native_ranks": None, "clip": None},
        "fedsvd": {"representation": "native_SVD", "factor_schedule": "B_only", "optimizer": "sgd",
                   "period": {1, 5}, "uses_public_training": False, "base_policy": "immutable",
                   "rank": {16}, "native_ranks": None, "clip": 2.},
        "fedplora": {"representation": "N", "factor_schedule": "both", "optimizer": "adamw",
                     "period": 0, "uses_public_training": False, "base_policy": "native",
                     "rank": {16}, "native_ranks": ([2, 3, 4], [4, 4, 4]), "clip": None},
    }[method]
    for key in ("representation", "factor_schedule", "optimizer", "uses_public_training", "base_policy"):
        if recipe[key] != expected[key]:
            raise NotImplementedError(f"native {method} unexpected {key}")
    for key in ("period", "rank"):
        allowed = expected[key]
        if recipe[key] not in (allowed if isinstance(allowed, set) else {allowed}):
            raise NotImplementedError(f"native {method} unexpected {key}")
    native_ranks = recipe["native_ranks"]
    if method == "fedplora":
        if native_ranks not in expected["native_ranks"]:
            raise NotImplementedError("unexpected PLoRA native tier ranks")
    elif native_ranks is not None:
        raise NotImplementedError("unexpected native ranks")
    if (recipe["adapter_clip_norm"] != expected["clip"] or
            recipe["head_optimizer"] != "adamw" or recipe["action_unit"] != "native" or
            recipe["sketch_h"] != 0 or recipe["xi"] != 0 or
            recipe["selector_history"] != "none_fresh" or recipe["schedule"] != "constant" or
            recipe["head_lr_ratio"] != (None if method == "fedsvd" else 1.) or
            recipe["evaluation_role"] != "recipe_dev" or recipe["full_forward"] is not True or
            recipe["budget_multiplier"] != 1. or recipe["budget_id"] != "B8"):
        raise NotImplementedError("native recipe contains unsupported control")
    if (common["protocol"] != PROTOCOL or b8_bytes <= 0 or
            common["training"]["batch_size"] != 16 or common["training"]["local_epochs"] != 1 or
            common["training"]["local_optimizer_steps"] != 16):
        raise ValueError("native common/B8 protocol differs")
    result = {"job": job, "recipe": recipe, "common": common,
              "manifest": str(manifest_path), "train_file": str(train_file),
              "token_cache": str(token_cache),
              "b8_wire": str(manifest_path.parent / "b8_wire.json"),
              "model": {"repository": str(model_path), "revision": None,
                        "local_files_only": True, "max_length": common["model"]["max_length"],
                        "dropout": common["lora"]["dropout"]},
              "runtime": {"byte_cap": int(b8_bytes),
                          "clients_per_round": common["federation"]["clients_per_round"],
                          "capacity_by_tier": common["federation"]["single_factor_projection_rank_positions"],
                          "batch_size": 16, "local_epochs": 1, "betas": common["training"]["betas"],
                          "eps": common["training"]["eps"], "weight_decay": recipe["adapter_weight_decay"],
                          "eval_batch_size": 64, "min_rounds": recipe["min_rounds"],
                          "max_rounds": recipe["max_rounds"]}}
    result["expanded_sha256"] = digest(result)
    return result
