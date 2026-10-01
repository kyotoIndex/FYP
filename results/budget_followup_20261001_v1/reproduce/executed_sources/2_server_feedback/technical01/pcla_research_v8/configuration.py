"""Compile the received v8 DEV matrix without converting proposals into launch approval."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verified_receipt(path: Path) -> dict:
    value = read_json(path)
    ceiling = value.get("resource_ceiling", {})
    if (value.get("public512") != "server algorithm training/selection approved"
            or ceiling.get("4090") != 14 or ceiling.get("A100_for_this_run") != 0
            or "all seeds all methods use4090" not in value.get("hardware_override", "")):
        raise ValueError("v8 scope authorization does not cover this matrix")
    return value


def verified_recipes(path: Path) -> dict[str, dict]:
    recipes = read_json(path)
    if len(recipes) != 32:
        raise ValueError("expected 32 v8 recipes")
    by_id = {}
    for recipe in recipes:
        identifier = recipe["recipe_id"]
        if identifier in by_id or digest({k: v for k, v in recipe.items() if k != "config_sha256"}) != recipe["config_sha256"]:
            raise ValueError(f"duplicate or altered recipe {identifier}")
        by_id[identifier] = recipe
    return by_id


def validate_data_manifest(path: Path, *, seed: int, train_file: Path, expected_hash: str | None = None) -> str:
    """Bind an existing generated role manifest. Never invent or bless one here."""
    if not path.is_file() or not train_file.is_file():
        raise FileNotFoundError("role manifest or SST-2 train.tsv absent")
    manifest = read_json(path)
    if manifest.get("source_sha256") != sha(train_file) or str(seed) not in manifest.get("partitions", {}):
        raise ValueError("role manifest does not bind train file and seed")
    if expected_hash is not None and sha(path) != expected_hash:
        raise ValueError("role manifest SHA256 differs from pinned identity")
    role = manifest["partitions"][str(seed)]
    for key in ("client_train", "local_readout", "public_training", "selection", "cohort_readout", "recipe_dev"):
        if key not in role and key not in manifest:
            raise ValueError(f"missing role {key}")
    return sha(path)


def compile_development(*, common_path: Path, recipes_path: Path, jobs_path: Path,
                        authorization_path: Path, data_dir: Path, train_file: Path,
                        token_cache: Path, model_path: Path, source_manifest: Path,
                        output_root: Path) -> list[dict]:
    common, receipt = read_json(common_path), verified_receipt(authorization_path)
    recipes = verified_recipes(recipes_path)
    jobs = rows(jobs_path)
    if len(jobs) != 128 or len({j["job_id"] for j in jobs}) != 128:
        raise ValueError("expected 128 unique DEV jobs")
    if common["budget"]["B8"] != 2382653750 or common["model"]["revision"] is None:
        raise ValueError("unexpected v8 common budget/model")
    source_record = read_json(source_manifest)
    if not source_record.get("files"):
        raise ValueError("missing source manifest files")
    source_manifest_hash = sha(source_manifest)
    role_manifest = data_dir / "train_roles.json"
    b8_wire = data_dir / "b8_wire.json"
    bound_hashes = {}
    for seed in {int(j["seed"]) for j in jobs}:
        try:
            bound_hashes[seed] = validate_data_manifest(role_manifest, seed=seed, train_file=train_file)
        except (FileNotFoundError, ValueError):
            bound_hashes[seed] = None
    if b8_wire.is_file():
        wire = read_json(b8_wire)
        if int(wire.get("total_bytes", -1)) != common["budget"]["B8"]:
            raise ValueError("B8 wire manifest differs")
        b8_hash = sha(b8_wire)
    else:
        b8_hash = None
    result = []
    for source_job in jobs:
        recipe = recipes[source_job["recipe_id"]]
        seed = int(source_job["seed"])
        if (source_job["phase"] != "DEV" or source_job["launch_status"] != "NOT_LAUNCHED"
                or source_job["recipe_sha256"] != recipe["config_sha256"]
                or source_job["family"] != recipe["family"]
                or int(source_job["rank"]) != recipe["rank"]
                or float(source_job["client_lr"]) != recipe["client_lr"]
                or float(source_job["head_lr"]) != recipe["head_lr"]
                or int(source_job["byte_cap"]) != common["budget"]["B8"]):
            raise ValueError(f"DEV row/recipe mismatch: {source_job['job_id']}")
        rule = recipe["server_rule"]
        method = "fedsb" if rule == "fedsb_r32" else "full_both"
        job = {
            "job_id": source_job["job_id"], "phase": "DEV", "recipe_id": recipe["recipe_id"],
            "seed": seed, "hardware_family": "RTX4090D", "method": method,
            "rank": recipe["rank"], "lr": recipe["client_lr"], "head_lr": recipe["head_lr"],
            "min_rounds": int(source_job["min_research_rounds"]),
            "max_rounds": int(source_job["max_rounds"]), "priority": int(source_job["priority"]),
            "output_dir": str(output_root / "dev" / recipe["recipe_id"] / f"s{seed}"),
        }
        runtime = {
            "byte_cap": common["budget"]["B8"],
            "clients_per_round": common["federation"]["clients_per_round"],
            "batch_size": common["training"]["batch_size"],
            "local_epochs": common["training"]["local_epochs"],
            "betas": common["training"]["betas"], "eps": common["training"]["eps"],
            "weight_decay": common["training"]["weight_decay"],
            "eval_batch_size": common["controller"]["eval_batch_size"],
            "min_rounds": job["min_rounds"], "max_rounds": job["max_rounds"],
            "server_rule": rule, "period": recipe["period"],
            "public_training": recipe["public_training"],
            "controller": common["controller"],
        }
        bound = bound_hashes[seed] is not None and b8_hash is not None and token_cache.is_file() and model_path.is_dir()
        item = {
            "schema": "pcla-v8-expanded-1", "job": job, "recipe": recipe, "common": common,
            "manifest": str(role_manifest), "train_file": str(train_file),
            "token_cache": str(token_cache), "b8_wire": str(b8_wire),
            "source_manifest": str(source_manifest),
            "model": {"repository": str(model_path), "revision": common["model"]["revision"],
                      "local_files_only": True, "max_length": common["model"]["max_length"],
                      "dropout": common["model"]["lora_dropout"]},
            "runtime": runtime,
            "binding": {"status": "bound_unverified" if bound else "blocked_missing_data_binding",
                        "data_manifest_sha256": bound_hashes[seed], "b8_wire_sha256": b8_hash,
                        "token_cache_sha256": sha(token_cache) if token_cache.is_file() else None,
                        "source_manifest_sha256": source_manifest_hash,
                        "common_sha256": sha(common_path), "recipes_sha256": sha(recipes_path),
                        "development_sha256": sha(jobs_path), "scope_authorization_sha256": sha(authorization_path),
                        "model_revision": common["model"]["revision"],
                        "source_file_hashes": source_record["files"]},
            "launch_status": "blocked_pending_integration_and_audit",
        }
        item["expanded_sha256"] = digest(item)
        result.append(item)
    by_recipe = {}
    for item in result:
        by_recipe.setdefault(item["job"]["recipe_id"], set()).add(item["job"]["seed"])
    if set(by_recipe) != set(recipes) or any(seeds != {503, 509, 521, 541} for seeds in by_recipe.values()):
        raise ValueError("DEV recipe/seed coverage differs")
    return result
