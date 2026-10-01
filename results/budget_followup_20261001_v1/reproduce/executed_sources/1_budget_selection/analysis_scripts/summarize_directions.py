#!/usr/bin/env python3
"""Fail-closed, CPU-only descriptive summary of the complete direction export."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile


SCHEMA = "pcla-v7-direction-descriptive-summary-1"
EXPORT_SCHEMA = "pcla-v7-diagnostic-json-export-1"
SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
LABELS = ("N", "D", "S", "T", "S_random_nonzero", "S_zero_removed")
POLICIES = ("local", "cyclic")
PREFIXES = (1, 4, 16)
PAIRS = (("D", "N"), ("S", "N"), ("T", "N"),
         ("S_random_nonzero", "S"), ("S_zero_removed", "S"))
TABLE_COUNTS = {"objects": 144, "cohort_readouts": 1536, "cohort_clients": 15360,
                "cohort_locks": 384, "cohort_evaluations": 34012,
                "direction_readouts": 1728, "direction_geometry": 9216}
WARNING = ("direction raw bank has no explicit readout source/row hash; "
           "role is bound by frozen driver/model source and manifest")
IDENTITY = ("job_id", "line", "seed", "parent_job_id", "parent_recipe_id",
            "anchor_round", "factor", "hardware_family", "config_sha256",
            "source_manifest_sha256", "anchor_sha256", "audit_receipt_sha256",
            "acceptance_receipt_sha256")


class SummaryError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SummaryError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def number(value: object, context: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"nonfinite or missing {context}")
    return float(value)


def roster() -> set[tuple[int, str, int, str, str]]:
    return {(seed, recipe, anchor, factor, line)
            for seed in SEEDS for anchor in (10, 60) for factor in ("a", "b")
            for recipe, line in (("D069", "cohort"), ("D072", "cohort"),
                                 ("D072", "direction"))}


def object_key(row: dict) -> tuple[int, str, int, str, str]:
    return (row["seed"], row["parent_recipe_id"], row["anchor_round"],
            row["factor"], row["line"])


def identity(row: dict) -> dict:
    return {key: row[key] for key in IDENTITY}


def check_identity(row: dict, obj: dict) -> None:
    require(all(row.get(key) == obj.get(key) for key in IDENTITY),
            f"object identity mismatch: {obj['job_id']}")


def unique_rows(rows: list[dict], name: str, fields: tuple[str, ...]) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(keys) == len(set(keys)), f"duplicate {name} key")


def load_export(export: Path) -> tuple[dict, dict[str, list[dict]], dict[str, str]]:
    manifest_path = export / "EXPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == EXPORT_SCHEMA and
            (manifest.get("object_count"), manifest.get("cohort_count"),
             manifest.get("direction_count")) == (144, 96, 48),
            "incomplete or wrong diagnostic export")
    require(set(manifest.get("tables", {})) == set(TABLE_COUNTS), "export table roster differs")
    hashes = {"EXPORT_MANIFEST.json": digest(manifest_path)}
    tables = {}
    for name, count in TABLE_COUNTS.items():
        path = export / f"{name}.jsonl"
        expected = manifest["tables"][name]
        require(expected.get("rows") == count and digest(path) == expected.get("sha256"),
                f"{name} count/hash differs")
        rows = records(path)
        require(len(rows) == count, f"{name} physical row count differs")
        tables[name] = rows
        hashes[path.name] = expected["sha256"]
    projection_path = export / "direction_projection_manifest.json"
    projection_sha = digest(projection_path)
    require(projection_sha == manifest.get("direction_projection_manifest_sha256") and
            hashes["direction_geometry.jsonl"] == manifest.get("direction_geometry_sha256"),
            "geometry projection hash chain differs")
    projection = json.loads(projection_path.read_text())
    require(projection.get("object_count") == 48 and projection.get("geometry_sha256") ==
            hashes["direction_geometry.jsonl"] and len(projection.get("objects", [])) == 48,
            "direction bank projection incomplete")
    hashes[projection_path.name] = projection_sha

    objects = tables["objects"]
    unique_rows(objects, "objects", ("job_id",))
    unique_rows(objects, "object roster", ("seed", "parent_recipe_id", "anchor_round", "factor", "line"))
    require({object_key(row) for row in objects} == roster(), "144-object configured roster differs")
    by_job = {row["job_id"]: row for row in objects}
    for row in objects:
        require(row.get("state") == "completed" and row.get("errors") == [] and
                row.get("source_manifest_sha256") == manifest.get("source_manifest_sha256") and
                row.get("hardware_family") == ("A100" if row["seed"] in (281, 307) else "4090D"),
                f"object status/source/hardware differs: {row['job_id']}")
        if row["line"] == "direction":
            require(row.get("warnings") == [WARNING] and
                    row.get("readout_source_limitation") == WARNING and
                    row.get("readout_source_id") is None and
                    row.get("direction_readout_source_row_hash") is None,
                    f"direction source warning differs: {row['job_id']}")
    projection_map = {row["job_id"]: row for row in projection["objects"]}
    require(len(projection_map) == 48 and set(projection_map) ==
            {row["job_id"] for row in objects if row["line"] == "direction"},
            "projected bank roster differs")

    keys = {"cohort_readouts": ("job_id", "choice", "prefix"),
            "cohort_clients": ("job_id", "choice", "prefix", "client_id"),
            "cohort_locks": ("job_id", "lock_kind"),
            "cohort_evaluations": ("job_id", "ordinal"),
            "direction_readouts": ("job_id", "representation", "policy", "prefix"),
            "direction_geometry": ("job_id", "record_type", "representation", "policy", "module", "client_id")}
    for name, rows in tables.items():
        if name == "objects":
            continue
        seen = set()
        for row in rows:
            job = row.get("job_id")
            require(job in by_job, f"orphan {name} row")
            check_identity(row, by_job[job])
            key = tuple(row.get(field) for field in keys[name])
            require(key not in seen, f"duplicate {name} key: {key}")
            seen.add(key)
            require(row["line"] == ("direction" if name.startswith("direction") else "cohort"),
                    f"wrong {name} line")
    for row in tables["direction_geometry"]:
        entry = projection_map[row["job_id"]]
        require(all(row.get(field) == entry[field] for field in
                    ("bank_sha256", "stage_sha256", "audit_receipt_sha256")),
                f"bank geometry identity differs: {row['job_id']}")
    for row in tables["direction_readouts"]:
        require(row.get("stage_sha256") == projection_map[row["job_id"]]["stage_sha256"] ==
                by_job[row["job_id"]]["result_sha256"],
                f"readout/stage identity differs: {row['job_id']}")
    expected_cells = {(row["job_id"], label, policy, prefix)
                      for row in objects if row["line"] == "direction"
                      for label in LABELS for policy in POLICIES for prefix in PREFIXES}
    require({(r["job_id"], r["representation"], r["policy"], r["prefix"])
             for r in tables["direction_readouts"]} == expected_cells,
            "six-representation readout matrix incomplete")
    for name, count_per_job in (("cohort_readouts", 16), ("cohort_clients", 160),
                                ("cohort_locks", 4)):
        counts = Counter(row["job_id"] for row in tables[name])
        require(set(counts) == {row["job_id"] for row in objects if row["line"] == "cohort"}
                and set(counts.values()) == {count_per_job}, f"{name} object coverage differs")
    lock_menu = {"JQ_pretraining", "selection1024", "prediction_h32_full",
                 "full_gradient_choice_readout"}
    require(all({row["lock_kind"] for row in tables["cohort_locks"]
                 if row["job_id"] == obj["job_id"]} == lock_menu
                for obj in objects if obj["line"] == "cohort"), "cohort lock menu differs")
    return manifest, tables, hashes


def stats(values: list[float]) -> dict:
    require(len(values) == 12, "12 complete seeds required")
    return {"n_seeds": 12, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values)}


def summarize_tables(tables: dict[str, list[dict]]) -> dict[str, list[dict]]:
    objects = {r["job_id"]: r for r in tables["objects"] if r["line"] == "direction"}
    geom = defaultdict(list)
    for row in tables["direction_geometry"]:
        geom[row["job_id"], row["record_type"]].append(row)
    module_rows, t_module_rows, pred_rows = [], [], []
    active = {}
    for job, obj in objects.items():
        modules = geom[job, "representation_module"]
        require(len(modules) == 6 * 24, f"24-module geometry incomplete: {job}")
        for label in LABELS:
            group = [r for r in modules if r["representation"] == label]
            require(len(group) == 24 and len({r["module"] for r in group}) == 24 and
                    {r["active"] for r in group} in ({True}, {False}),
                    f"module/activity coverage differs: {job}/{label}")
            active[job, label] = group[0]["active"]
            for row in group:
                require(isinstance(row["numerical_rank"], int) and row["numerical_rank"] >= 0,
                        f"invalid rank: {job}/{label}")
                number(row["rank_threshold"], "rank threshold")
                number(row["relative_error"], "module relative error")
                module_rows.append({**identity(obj), "stage_sha256": row["stage_sha256"],
                                    "bank_sha256": row["bank_sha256"],
                                    "representation": label, "module": row["module"],
                                    "active": row["active"], "numerical_rank": row["numerical_rank"],
                                    "rank_threshold": row["rank_threshold"],
                                    "relative_error": row["relative_error"],
                                    "random_rotation": row["random_rotation"]})
        require(len(geom[job, "direction_parent"]) == 1 and
                len(geom[job, "T_comparator_identity"]) == 1 and
                len(geom[job, "T_module_comparison"]) == 24 and
                len({r["module"] for r in geom[job, "T_module_comparison"]}) == 24 and
                len(geom[job, "T_prediction_aggregate"]) == 2 and
                len(geom[job, "T_prediction_client"]) == 20,
                f"T geometry coverage differs: {job}")
        module_set = {r["module"] for r in modules if r["representation"] == "N"}
        require(all({r["module"] for r in modules if r["representation"] == label} == module_set
                    for label in LABELS) and
                {r["module"] for r in geom[job, "T_module_comparison"]} == module_set,
                f"T/representation module roster differs: {job}")
        for row in geom[job, "T_module_comparison"]:
            number(row["gram_frobenius_absolute_error"], "T Gram absolute error")
            if row["gram_frobenius_relative_error"] is not None:
                number(row["gram_frobenius_relative_error"], "T Gram relative error")
            t_module_rows.append({**identity(obj), "stage_sha256": row["stage_sha256"],
                "bank_sha256": row["bank_sha256"], "module": row["module"],
                "gram_frobenius_absolute_error": row["gram_frobenius_absolute_error"],
                "gram_frobenius_relative_error": row["gram_frobenius_relative_error"],
                "training_status": "comparison_only_no_extra_training"})
        require(geom[job, "T_comparator_identity"][0]["status"] ==
                "diagnostic_only_no_extra_training", f"T comparator status differs: {job}")
        require({r["policy"] for r in geom[job, "T_prediction_aggregate"]} == set(POLICIES) and
                {(r["policy"], r["client_id"]) for r in geom[job, "T_prediction_client"]} ==
                {(policy, cid) for policy in POLICIES for cid in obj["participants"]},
                f"T prediction policy/client coverage differs: {job}")
        for row in geom[job, "T_prediction_aggregate"] + geom[job, "T_prediction_client"]:
            full = number(row["full"], "T full prediction")
            h32 = number(row["h32"], "T h32 prediction")
            pred_rows.append({**identity(obj), "stage_sha256": row["stage_sha256"],
                              "bank_sha256": row["bank_sha256"], "record_type": row["record_type"],
                              "policy": row["policy"], "client_id": row.get("client_id"),
                              "full": full, "h32": h32, "full_minus_h32": full - h32,
                              "relative_error_abs_full": (abs(full - h32) / abs(full)
                                                          if full != 0 else None),
                              "relative_error_status": ("defined" if full != 0 else "NA_zero_full"),
                              "training_status": "prediction_only_no_extra_training"})

    readouts = {(r["job_id"], r["representation"], r["policy"], r["prefix"]): r
                for r in tables["direction_readouts"]}
    original_rows, contrast_rows = [], []
    for job, obj in objects.items():
        for policy in POLICIES:
            for prefix in PREFIXES:
                for label in LABELS:
                    row = readouts[job, label, policy, prefix]
                    status = row["normmatch_status"]
                    require(status in ("matched", "NA_zero_reference", "NA_zero_candidate"),
                            f"invalid normmatch status: {job}")
                    for field in ("original_loss", "original_benefit", "reference_n_effective_norm",
                                  "candidate_effective_norm"):
                        number(row[field], field)
                    matched = status == "matched"
                    fields = ("normmatched_loss", "normmatched_benefit", "normmatch_multiplier")
                    require(all(row[field] is not None for field in fields) if matched else
                            all(row[field] is None for field in fields),
                            f"normmatch NA/values inconsistent: {job}")
                    if matched:
                        for field in fields:
                            number(row[field], field)
                    original_rows.append({**identity(obj), "stage_sha256": row["stage_sha256"],
                                          "representation": label, "policy": policy, "prefix": prefix,
                                          "readout_source_id": obj["readout_source_id"],
                                          "direction_readout_source_row_hash":
                                              obj["direction_readout_source_row_hash"],
                                          "active": active[job, label],
                                          "intervention_status": ("inactive_no_zero_direction"
                                                                  if label == "S_zero_removed" and
                                                                  not active[job, label] else "active"),
                                          **{field: row[field] for field in
                                             ("original_loss", "original_benefit", "normmatched_loss",
                                              "normmatched_benefit", "normmatch_status",
                                              "normmatch_multiplier", "reference_n_effective_norm",
                                              "candidate_effective_norm")},
                                          "readout_source_warning": obj["readout_source_limitation"]})
                for target, reference in PAIRS:
                    a = readouts[job, target, policy, prefix]
                    b = readouts[job, reference, policy, prefix]
                    available = a["normmatch_status"] == b["normmatch_status"] == "matched"
                    contrast_rows.append({**identity(obj), "stage_sha256": a["stage_sha256"],
                        "policy": policy, "prefix": prefix, "target": target, "reference": reference,
                        "readout_source_id": obj["readout_source_id"],
                        "direction_readout_source_row_hash": obj["direction_readout_source_row_hash"],
                        "original_delta_benefit": a["original_benefit"] - b["original_benefit"],
                        "normmatched_delta_benefit": (a["normmatched_benefit"] - b["normmatched_benefit"]
                                                       if available else None),
                        "normmatched_status": ("matched" if available else "NA_pair"),
                        "target_normmatch_status": a["normmatch_status"],
                        "reference_normmatch_status": b["normmatch_status"],
                        "target_active": active[job, target], "reference_active": active[job, reference],
                        "intervention_status": ("inactive_no_zero_direction"
                                                if target == "S_zero_removed" and not active[job, target]
                                                else "active"),
                        "readout_source_warning": obj["readout_source_limitation"]})

    readout_seed_rows, readout_overall_rows = [], []
    for label in LABELS:
        for policy in POLICIES:
            for prefix in PREFIXES:
                subset = [r for r in original_rows if (r["representation"], r["policy"], r["prefix"])
                          == (label, policy, prefix)]
                full_original, full_matched = [], []
                for seed in SEEDS:
                    group = [r for r in subset if r["seed"] == seed]
                    require(len(group) == 4, "four direction readouts required per seed")
                    matched = [r["normmatched_benefit"] for r in group
                               if r["normmatch_status"] == "matched"]
                    original_mean = statistics.mean(r["original_benefit"] for r in group)
                    matched_mean = statistics.mean(matched) if len(matched) == 4 else None
                    full_original.append(original_mean)
                    if matched_mean is not None:
                        full_matched.append(matched_mean)
                    readout_seed_rows.append({"seed": seed, "hardware_family": group[0]["hardware_family"],
                        "representation": label, "policy": policy, "prefix": prefix,
                        "n_objects_expected": 4, "n_original_objects": 4,
                        "original_mean_benefit": original_mean,
                        "n_matched_objects": len(matched),
                        "normmatched_mean_benefit": matched_mean,
                        "normmatched_status": "complete" if len(matched) == 4 else "incomplete_NA",
                        "n_active_objects": sum(r["active"] for r in group)})
                readout_overall_rows.append({"representation": label, "policy": policy,
                    "prefix": prefix, "original": stats(full_original),
                    "normmatched": stats(full_matched) if len(full_matched) == 12 else None,
                    "n_complete_matched_seeds": len(full_matched),
                    "n_active_objects_total": sum(r["n_active_objects"] for r in readout_seed_rows
                                                  if (r["representation"], r["policy"], r["prefix"])
                                                  == (label, policy, prefix))})

    module_seed_rows = []
    for label in LABELS:
        module_names = {r["module"] for r in module_rows if r["representation"] == label}
        require(len(module_names) == 24, "global module roster differs")
        for module in sorted(module_names):
            for seed in SEEDS:
                group = [r for r in module_rows if (r["representation"], r["module"], r["seed"])
                         == (label, module, seed)]
                require(len(group) == 4, "four module observations required per seed")
                module_seed_rows.append({"seed": seed, "hardware_family": group[0]["hardware_family"],
                    "representation": label, "module": module, "n_objects": 4,
                    "n_active_objects": sum(r["active"] for r in group),
                    "mean_numerical_rank": statistics.mean(r["numerical_rank"] for r in group),
                    "mean_rank_threshold": statistics.mean(r["rank_threshold"] for r in group),
                    "mean_relative_error": statistics.mean(r["relative_error"] for r in group)})

    seed_rows, overall_rows = [], []
    for policy in POLICIES:
        for prefix in PREFIXES:
            for target, reference in PAIRS:
                group = [r for r in contrast_rows if (r["policy"], r["prefix"], r["target"],
                                                     r["reference"]) == (policy, prefix, target, reference)]
                full_original, full_matched = [], []
                for seed in SEEDS:
                    subset = [r for r in group if r["seed"] == seed]
                    require(len(subset) == 4, "four direction objects required per seed")
                    matched = [r["normmatched_delta_benefit"] for r in subset
                               if r["normmatched_status"] == "matched"]
                    active_subset = [r for r in subset if r["intervention_status"] == "active"]
                    active_matched = [r["normmatched_delta_benefit"] for r in active_subset
                                      if r["normmatched_status"] == "matched"]
                    original_mean = statistics.mean(r["original_delta_benefit"] for r in subset)
                    matched_mean = statistics.mean(matched) if len(matched) == 4 else None
                    full_original.append(original_mean)
                    if matched_mean is not None:
                        full_matched.append(matched_mean)
                    seed_rows.append({"seed": seed, "hardware_family": subset[0]["hardware_family"],
                        "policy": policy, "prefix": prefix, "target": target, "reference": reference,
                        "n_objects_expected": 4, "n_original_objects": 4,
                        "original_mean_delta_benefit": original_mean,
                        "n_matched_objects": len(matched),
                        "normmatched_mean_delta_benefit": matched_mean,
                        "normmatched_status": ("complete" if len(matched) == 4 else "incomplete_NA"),
                        "n_active_objects": len(active_subset),
                        "active_conditional_original_mean":
                            (statistics.mean(r["original_delta_benefit"] for r in active_subset)
                             if active_subset else None),
                        "n_active_matched_objects": len(active_matched),
                        "active_conditional_normmatched_mean":
                            (statistics.mean(active_matched) if active_matched else None),
                        "active_conditional_scope": "within-seed active objects only; descriptive"})
                overall_rows.append({"policy": policy, "prefix": prefix, "target": target,
                    "reference": reference, "original": stats(full_original),
                    "normmatched": (stats(full_matched) if len(full_matched) == 12 else None),
                    "n_complete_matched_seeds": len(full_matched),
                    "n_active_objects_total": sum(r["n_active_objects"] for r in seed_rows
                                                  if (r["policy"], r["prefix"], r["target"],
                                                      r["reference"]) == (policy, prefix, target, reference)),
                    "n_active_seeds": sum(r["n_active_objects"] > 0 for r in seed_rows
                                          if (r["policy"], r["prefix"], r["target"],
                                              r["reference"]) == (policy, prefix, target, reference))})

    pred_seed_rows = []
    aggregate = [r for r in pred_rows if r["record_type"] == "T_prediction_aggregate"]
    for policy in POLICIES:
        for seed in SEEDS:
            subset = [r for r in aggregate if r["seed"] == seed and r["policy"] == policy]
            require(len(subset) == 4, "four T aggregate predictions required per seed")
            defined = [r["relative_error_abs_full"] for r in subset
                       if r["relative_error_status"] == "defined"]
            pred_seed_rows.append({"seed": seed, "hardware_family": subset[0]["hardware_family"],
                "policy": policy, "n_objects_expected": 4, "n_objects": 4,
                "mean_full_minus_h32": statistics.mean(r["full_minus_h32"] for r in subset),
                "n_nonzero_full_denominator": len(defined),
                "mean_relative_error_abs_full": statistics.mean(defined) if defined else None,
                "relative_error_status": "complete" if len(defined) == 4 else "incomplete_NA"})
    pred_overall = []
    for policy in POLICIES:
        subset = [r for r in pred_seed_rows if r["policy"] == policy]
        defined = [r["mean_relative_error_abs_full"] for r in subset
                   if r["relative_error_status"] == "complete"]
        pred_overall.append({"policy": policy,
            "full_minus_h32": stats([r["mean_full_minus_h32"] for r in subset]),
            "relative_error_abs_full": stats(defined) if len(defined) == 12 else None,
            "n_complete_relative_error_seeds": len(defined)})
    return {"direction_object_readouts.jsonl": original_rows,
            "direction_object_contrasts.jsonl": contrast_rows,
            "direction_seed_readouts.jsonl": readout_seed_rows,
            "direction_overall_readouts.jsonl": readout_overall_rows,
            "direction_seed_descriptive.jsonl": seed_rows,
            "direction_overall_descriptive.jsonl": overall_rows,
            "direction_geometry_modules.jsonl": module_rows,
            "direction_geometry_module_seed.jsonl": module_seed_rows,
            "direction_T_module_comparison.jsonl": t_module_rows,
            "direction_T_predictions.jsonl": pred_rows,
            "direction_T_prediction_seed.jsonl": pred_seed_rows,
            "direction_T_prediction_overall.jsonl": pred_overall}


def summarize(export: Path, out: Path) -> dict:
    export = export.resolve(strict=True)
    out = out.resolve(strict=False)
    require(not out.exists() and out.parent.is_dir() and not out.is_relative_to(export),
            "output must be a new directory outside the input export")
    source_manifest, tables, hashes = load_export(export)
    outputs = summarize_tables(tables)
    require(all(digest(export / name) == sha for name, sha in hashes.items()),
            "input changed during summary")
    manifest = {"schema": SCHEMA, "complete": True, "source_export_path": str(export),
        "source_export_schema": source_manifest["schema"], "input_sha256": hashes,
        "summarizer_sha256": digest(Path(__file__)), "output_sha256": {},
        "source_warnings_preserved": [WARNING],
        "source_roles": {"original": "already trained direction readout2048 at original update scale",
                         "normmatched": "read-only uniform LoRA effective update rescale; trained head unchanged",
                         "T_full_h32": "one-step linear predictions from fixed T mask; no new full-T training"},
        "definitions": {"fixed_pairs": [f"{a}-{b}" for a, b in PAIRS],
            "comparison_scope": "same parent object, policy, prefix, and readout role",
            "seed_estimator": "four configured direction objects equally weighted within each seed; 12 seeds equally weighted",
            "sample_sd": "sample standard deviation of 12 seed means, denominator 11",
            "normmatch_NA": "paired matched delta is null unless both readouts are matched; seed mean requires all four objects; overall requires all 12 complete seeds",
            "zero_removed": "inactive_no_zero_direction raw readout retained; active-only within-seed means report explicit object denominators and are conditional descriptions",
            "relative_error": "abs(full-h32)/abs(full), null when full is zero; seed mean requires all four defined objects",
            "module_scope": "24 modules per representation per object are long-table observations, never independent seeds",
            "original_protocol": "fixed comparisons only; weighting and NA aggregation are design candidates, not retroactive protocol changes",
            "inference": "descriptive only; no winner, correction A/B decision, threshold, confidence interval or hypothesis test"}}
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".building-", dir=out.parent))
    try:
        for name, rows in outputs.items():
            path = temporary / name
            with path.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            manifest["output_sha256"][name] = {"rows": len(rows), "sha256": digest(path)}
        require(all(digest(export / name) == sha for name, sha in hashes.items()),
                "input changed before publication")
        (temporary / "SUMMARY_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        require(not out.exists(), "output appeared during publication")
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.export, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
