#!/usr/bin/env python3
"""Fail-closed descriptive cohort summary of an audited diagnostic JSON export."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


SCHEMA = "pcla-v7-cohort-descriptive-summary-1"
EXPORT_SCHEMA = "pcla-v7-diagnostic-json-export-1"
SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
PARENTS = ("D069", "D072")
FACTORS = ("a", "b")
PREFIXES = (1, 4, 16)
SUPPORTS = ("local", "cyclic", "JL_h32", "public512_linear", "JQ_xi1")
RANK_SUPPORTS = ("local", "cyclic", "JL_h32")
CHOICES = SUPPORTS + ("finite_bank_search",)
LOCKS = ("JQ_pretraining", "selection1024", "prediction_h32_full",
         "full_gradient_choice_readout")
TABLES = ("objects", "cohort_readouts", "cohort_clients", "cohort_locks",
          "cohort_evaluations", "direction_readouts", "direction_geometry")
EXPECTED_ROWS = {"objects": 144, "cohort_readouts": 1536,
                 "cohort_clients": 15360, "cohort_locks": 384,
                 "direction_readouts": 1728, "direction_geometry": 9216}
RANK_COMPARISONS = ("h32_full_four", "h32_vs_16", "full_vs_16",
                    "actual1_vs_16", "actual4_vs_16", "actual4_minus_actual1")
SELECTION_CONTRAST_PAIRS = (("JL_h32", "local"), ("JQ_xi1", "JL_h32"),
                            ("JQ_xi1", "local"), ("public512_linear", "local"),
                            ("public512_linear", "JQ_xi1"),
                            ("finite_bank_search", "local"),
                            ("full_gradient_choice", "local"),
                            ("full_gradient_choice", "JQ_xi1"))


class SummaryError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/missing {name}")
    return float(value)


def read_jsonl(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {value}")))
            except ValueError as exc:
                raise SummaryError(f"invalid {path.name} line {number}: {exc}") from exc
            require(isinstance(row, dict), f"nonobject {path.name} line {number}")
            rows.append(row)
    require(len(rows) == count, f"{path.name} row count differs: {len(rows)} vs {count}")
    return rows


def key(row: dict) -> tuple[int, str, int, str]:
    return row["seed"], row["parent_recipe_id"], row["anchor_round"], row["factor"]


def identity_matches(row: dict, source: dict) -> bool:
    fields = ("job_id", "seed", "parent_job_id", "parent_recipe_id", "anchor_round",
              "factor", "hardware_family", "config_sha256", "source_manifest_sha256",
              "anchor_sha256", "audit_receipt_sha256", "acceptance_receipt_sha256")
    return all(row.get(field) == source.get(field) for field in fields)


def expected_keys(parents: tuple[str, ...]) -> set[tuple[int, str, int, str]]:
    return {(seed, parent, anchor, factor) for seed in SEEDS for parent in parents
            for anchor in (10, 60) for factor in FACTORS}


def load_export(export: Path):
    export = export.resolve(strict=True)
    manifest_path = export / "EXPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == EXPORT_SCHEMA and
            (manifest.get("object_count"), manifest.get("cohort_count"),
             manifest.get("direction_count")) == (144, 96, 48),
            "export manifest is not the complete 144/96/48 diagnostic export")
    listed = manifest.get("tables")
    require(isinstance(listed, dict) and set(listed) == set(TABLES),
            "diagnostic export table list differs")
    input_hashes = {str(manifest_path): digest(manifest_path)}
    for name in TABLES:
        path = export / f"{name}.jsonl"
        entry = listed[name]
        require(isinstance(entry, dict) and type(entry.get("rows")) is int and
                isinstance(entry.get("sha256"), str), f"invalid manifest entry: {name}")
        if name in EXPECTED_ROWS:
            require(entry["rows"] == EXPECTED_ROWS[name], f"{name} expected row count differs")
        actual = digest(path)
        require(actual == entry["sha256"], f"SHA256 differs: {name}")
        input_hashes[str(path)] = actual
    projection_path = export / "direction_projection_manifest.json"
    projection = json.loads(projection_path.read_text())
    require(projection.get("schema") == "pcla-v7-diagnostic-direction-projection-1" and
            projection.get("object_count") == 48 and
            projection.get("row_count") == listed["direction_geometry"]["rows"] and
            projection.get("geometry_sha256") == listed["direction_geometry"]["sha256"] and
            manifest.get("direction_projection_manifest_sha256") == digest(projection_path),
            "direction projection identity differs")
    input_hashes[str(projection_path)] = digest(projection_path)
    objects = read_jsonl(export / "objects.jsonl", 144)
    object_map = {}
    seen_jobs = set()
    for row in objects:
        k = key(row)
        expect_family = "A100" if row["seed"] in (281, 307) else "4090D"
        require(row.get("record_type") == "object" and row.get("state") == "completed" and
                row.get("line") in ("cohort", "direction") and
                row.get("hardware_family") == expect_family and
                row["seed"] in SEEDS and row["anchor_round"] in (10, 60) and
                row["factor"] in FACTORS and
                ((row["line"] == "cohort" and row["parent_recipe_id"] in PARENTS) or
                 (row["line"] == "direction" and row["parent_recipe_id"] == "D072")) and
                isinstance(row.get("participants"), list) and
                len(row["participants"]) == len(set(row["participants"])) == 10 and
                not row.get("errors"), f"object identity/state differs: {row.get('job_id')}")
        require(row["job_id"] not in seen_jobs, "duplicate object job ID")
        seen_jobs.add(row["job_id"])
        line_map = object_map.setdefault(row["line"], {})
        require(k not in line_map, f"duplicate object key: {row['line']} {k}")
        line_map[k] = row
    require(set(object_map["cohort"]) == expected_keys(PARENTS) and
            set(object_map["direction"]) == expected_keys(("D072",)),
            "complete object matrix differs")
    direction_by_job = {row["job_id"]: row for row in object_map["direction"].values()}
    projected = projection.get("objects")
    require(isinstance(projected, list) and len(projected) == 48 and
            {row.get("job_id") for row in projected} == set(direction_by_job) and
            all(row.get("audit_receipt_sha256") == direction_by_job[row["job_id"]]["audit_receipt_sha256"] and
                row.get("stage_sha256") == direction_by_job[row["job_id"]]["result_sha256"] and
                isinstance(row.get("bank_sha256"), str) and len(row["bank_sha256"]) == 64
                for row in projected), "direction projection object identity differs")

    def mapped(name: str, extension):
        result = {}
        for row in read_jsonl(export / f"{name}.jsonl", listed[name]["rows"]):
            k = key(row)
            require(row.get("line") == "cohort" and k in object_map["cohort"] and
                    identity_matches(row, object_map["cohort"][k]) and
                    row.get("readout_json_sha256") == object_map["cohort"][k]["result_sha256"],
                    f"{name} object/hash identity differs: {k}")
            full_key = (k, *extension(row))
            require(full_key not in result, f"duplicate {name} key: {full_key}")
            result[full_key] = row
        return result

    readouts = mapped("cohort_readouts", lambda row: (row["choice"], row["prefix"]))
    clients = mapped("cohort_clients", lambda row: (row["choice"], row["prefix"], row["client_id"]))
    locks = mapped("cohort_locks", lambda row: (row["lock_kind"],))
    expected_readouts = {(k, choice, prefix) for k in object_map["cohort"]
                         for choice in CHOICES
                         for prefix in ((16,) if choice == "finite_bank_search" else PREFIXES)}
    require(set(readouts) == expected_readouts, "cohort readout matrix differs")
    expected_clients = {(k, choice, prefix, cid) for k, object_row in object_map["cohort"].items()
                        for cid in object_row["participants"] for choice in CHOICES
                        for prefix in ((16,) if choice == "finite_bank_search" else PREFIXES)}
    require(set(clients) == expected_clients, "cohort client matrix differs")
    require(set(locks) == {(k, kind) for k in object_map["cohort"] for kind in LOCKS},
            "cohort lock matrix differs")
    for k, lock in locks.items():
        require(isinstance(lock.get("payload"), dict), f"lock payload missing: {k}")
    for k in object_map["cohort"]:
        participants = object_map["cohort"][k]["participants"]
        prediction = locks[k, "prediction_h32_full"]["payload"]
        require(prediction.get("candidate_order") == list(SUPPORTS[:4]) and
                set(prediction.get("per_client", {})) == {str(cid) for cid in participants},
                f"prediction menu/clients differ: {k}")
        full = locks[k, "full_gradient_choice_readout"]["payload"]
        finite(full.get("actual_bank_benefit"), "full selected actual bank benefit")
        for choice in CHOICES:
            for prefix in ((16,) if choice == "finite_bank_search" else PREFIXES):
                row = readouts[k, choice, prefix]
                require(row["parent_readout2048_loss"] ==
                        readouts[k, "local", 16]["parent_readout2048_loss"],
                        f"cohort parent loss differs: {k}")
    return manifest, object_map["cohort"], readouts, clients, locks, input_hashes


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end + 1) / 2
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def spearman(a: list[float], b: list[float]) -> tuple[float | None, str]:
    require(len(a) == len(b) and len(a) in (3, 4), "rank vector length differs")
    if any(type(v) not in (float, int) for v in (*a, *b)):
        return None, "NA_missing"
    if any(not math.isfinite(v) for v in (*a, *b)):
        return None, "NA_nonfinite"
    ra, rb = average_ranks(a), average_ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return None, "NA_constant"
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb), "measured"


def object_and_client_tables(objects, readouts, clients, locks):
    object_values, client_gains, selection, contrasts, client_ranks, object_ranks = [], [], [], [], [], []
    for k in sorted(objects):
        source = objects[k]
        base = {field: source[field] for field in ("job_id", "seed", "parent_job_id",
                "parent_recipe_id", "anchor_round", "factor", "hardware_family",
                "config_sha256", "anchor_sha256", "audit_receipt_sha256",
                "acceptance_receipt_sha256", "result_sha256")}
        participants = sorted(source["participants"])
        parent_loss = finite(readouts[k, "local", 16]["parent_readout2048_loss"], "parent loss")
        for choice in CHOICES:
            for prefix in ((16,) if choice == "finite_bank_search" else PREFIXES):
                row = readouts[k, choice, prefix]
                local = [finite(clients[k, choice, prefix, cid]["local64_benefit"],
                                "local64 benefit") for cid in participants]
                individual = [finite(clients[k, choice, prefix, cid]["weighted_readout2048_benefit"],
                                     "scaled individual benefit") for cid in participants]
                for cid, local_value, individual_value in zip(participants, local, individual):
                    client_gains.append({**base, "choice": choice, "prefix": prefix,
                        "client_id": cid, "local64_benefit": local_value,
                        "scaled_individual_readout2048_benefit": individual_value,
                        "scaled_weight_already_applied": True,
                        "readout_json_sha256": row["readout_json_sha256"]})
                joint = finite(row["aggregate_readout2048_benefit"], "joint benefit")
                nonadd = finite(row["nonadditivity"], "nonadditivity")
                independent = joint - math.fsum(individual)
                require(math.isclose(nonadd, independent, rel_tol=1e-9, abs_tol=1e-10),
                        f"nonadditivity independent formula differs: {k}/{choice}/{prefix}")
                status = row["cancellation_status"]
                require(status in ("measured", "NA_zero_denominator"), "cancellation status differs")
                cancellation = row["cancellation"]
                if status == "NA_zero_denominator":
                    require(cancellation is None and row["denominator_mean_effective_norm_sq"] == 0,
                            "cancellation zero denominator was filled")
                else:
                    finite(cancellation, "cancellation")
                object_values.append({**base, "choice": choice, "prefix": prefix,
                    "parent_readout2048_loss": parent_loss,
                    "local64_mean_benefit": statistics.mean(local),
                    "scaled_individual_readout2048_sum_benefit": math.fsum(individual),
                    "aggregate_readout2048_benefit": joint,
                    "nonadditivity_exported": nonadd,
                    "nonadditivity_independent": independent,
                    "head_only_readout2048_benefit": finite(row["head_only_readout2048_benefit"],
                                                             "head only benefit"),
                    "lora_only_readout2048_benefit": finite(row["lora_only_readout2048_benefit"],
                                                             "LoRA only benefit"),
                    "cancellation": cancellation, "cancellation_status": status,
                    "numerator_effective_norm_sq": finite(row["numerator_effective_norm_sq"], "numerator"),
                    "denominator_mean_effective_norm_sq": finite(row["denominator_mean_effective_norm_sq"], "denominator"),
                    "max_effective_linearity_relative_error": finite(row["max_effective_linearity_relative_error"], "linearity"),
                    "n_clients": 10, "readout_json_sha256": row["readout_json_sha256"]})
        local16 = finite(readouts[k, "local", 16]["aggregate_readout2048_benefit"], "local16")
        full = locks[k, "full_gradient_choice_readout"]["payload"]
        benefits = {choice: finite(readouts[k, choice, 16]["aggregate_readout2048_benefit"],
                                   f"{choice} benefit") for choice in CHOICES}
        benefits["full_gradient_choice"] = finite(full["actual_bank_benefit"], "full chosen benefit")
        for choice, benefit in benefits.items():
            delta = benefit - local16
            normalized = None if parent_loss <= 0 else delta / parent_loss
            selection.append({**base, "choice": choice, "prefix": 16,
                "actual_bank_aggregate_readout2048_benefit": benefit,
                "local_aggregate_readout2048_benefit": local16,
                "delta_vs_local_readout2048": delta, "parent_readout2048_loss": parent_loss,
                "delta_over_parent_loss": normalized,
                "normalization_status": "NA_parent_loss" if normalized is None else "measured",
                "source_role": "full_gradient_locked_choice_on_actual_bank" if choice == "full_gradient_choice"
                               else "finite_bank_selection_on_independent_readout" if choice == "finite_bank_search"
                               else "fixed_or_practical_bank_support_on_independent_readout",
                "source_sha256": locks[k, "full_gradient_choice_readout"]["readout_json_sha256"]
                                  if choice == "full_gradient_choice" else
                                  readouts[k, choice, 16]["readout_json_sha256"]})
        for target, reference in SELECTION_CONTRAST_PAIRS:
            delta = benefits[target] - benefits[reference]
            contrasts.append({**base, "target_choice": target, "reference_choice": reference,
                "prefix": 16, "target_benefit": benefits[target],
                "reference_benefit": benefits[reference], "delta": delta,
                "sign": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                "role": "same_parent_independent_readout2048_actual_trained_bank",
                "full_choice_is_locked_bank_selection": target == "full_gradient_choice"})
        prediction = locks[k, "prediction_h32_full"]["payload"]
        per_client = prediction["per_client"]
        by_comparison = defaultdict(list)
        for cid in participants:
            scores = per_client[str(cid)]["one_step_linear_gain"]
            require(set(scores) == {"h32", "full"} and
                    all(set(scores[direction]) == set(SUPPORTS[:4]) for direction in scores),
                    f"prediction support menu differs: {k}/{cid}")
            raw_vectors = {"prediction_h32_lora_only_one_step": [scores["h32"][choice] for choice in RANK_SUPPORTS],
                           "prediction_full_lora_only_one_step": [scores["full"][choice] for choice in RANK_SUPPORTS],
                           "actual_readout2048_scaled_step1": [clients[k, choice, 1, cid]["weighted_readout2048_benefit"] for choice in RANK_SUPPORTS],
                           "actual_readout2048_scaled_step4": [clients[k, choice, 4, cid]["weighted_readout2048_benefit"] for choice in RANK_SUPPORTS],
                           "actual_readout2048_scaled_step16": [clients[k, choice, 16, cid]["weighted_readout2048_benefit"] for choice in RANK_SUPPORTS]}
            for values in raw_vectors.values():
                for value in values:
                    finite(value, "ranking vector value")
            a16 = raw_vectors["actual_readout2048_scaled_step16"]
            four_h32 = [scores["h32"][choice] for choice in SUPPORTS[:4]]
            four_full = [scores["full"][choice] for choice in SUPPORTS[:4]]
            comparisons = {
                "h32_full_four": spearman(four_h32, four_full),
                "h32_vs_16": spearman(raw_vectors["prediction_h32_lora_only_one_step"], a16),
                "full_vs_16": spearman(raw_vectors["prediction_full_lora_only_one_step"], a16),
                "actual1_vs_16": spearman(raw_vectors["actual_readout2048_scaled_step1"], a16),
                "actual4_vs_16": spearman(raw_vectors["actual_readout2048_scaled_step4"], a16)}
            source_status = per_client[str(cid)]["spearman_status"]
            source_value = per_client[str(cid)]["spearman"]
            require(comparisons["h32_full_four"][1] == source_status and
                    ((source_value is None and comparisons["h32_full_four"][0] is None) or
                     (source_value is not None and math.isclose(
                         source_value, comparisons["h32_full_four"][0], rel_tol=1e-10, abs_tol=1e-10))),
                    f"source h32/full Spearman differs: {k}/{cid}")
            one, four = comparisons["actual1_vs_16"], comparisons["actual4_vs_16"]
            improvement = (four[0] - one[0], "measured") if one[0] is not None and four[0] is not None else (None, "NA_paired_rank")
            comparisons["actual4_minus_actual1"] = improvement
            for comparison, (value, status) in comparisons.items():
                by_comparison[comparison].append(value)
                client_ranks.append({**base, "client_id": cid, "comparison": comparison,
                    "spearman_or_difference": value, "status": status,
                    "support_order": list(RANK_SUPPORTS), "values": raw_vectors,
                    "source_h32_full_four_spearman": source_value,
                    "prediction_scope": "LoRA-only one-step linear; actual prefixes include trained head",
                    "prediction_lock_readout_sha256": locks[k, "prediction_h32_full"]["readout_json_sha256"]})
        for comparison in RANK_COMPARISONS:
            values = by_comparison[comparison]
            defined = [v for v in values if v is not None]
            object_ranks.append({**base, "comparison": comparison, "defined_clients": len(defined),
                "expected_clients": 10, "mean_defined_client_value": statistics.mean(defined) if defined else None,
                "status": "measured" if defined else "NA_defined_count"})
        four_values = by_comparison["h32_full_four"]
        require(prediction["spearman_defined_clients"] == sum(v is not None for v in four_values) and
                ((prediction["spearman_mean_defined"] is None and not any(v is not None for v in four_values)) or
                 (prediction["spearman_mean_defined"] is not None and
                  math.isclose(prediction["spearman_mean_defined"],
                               statistics.mean(v for v in four_values if v is not None),
                               rel_tol=1e-10, abs_tol=1e-10))),
                f"source h32/full defined-client mean differs: {k}")
    return object_values, client_gains, selection, contrasts, client_ranks, object_ranks


def seed_tables(object_values, selection, contrasts, object_ranks):
    groups = defaultdict(list)
    for row in object_values:
        for metric in ("local64_mean_benefit", "scaled_individual_readout2048_sum_benefit",
                       "aggregate_readout2048_benefit", "nonadditivity_exported", "cancellation"):
            groups[(row["seed"], "cohort_object", row["choice"], row["prefix"], metric)].append(row[metric])
    for row in selection:
        for metric in ("actual_bank_aggregate_readout2048_benefit", "delta_vs_local_readout2048",
                       "delta_over_parent_loss"):
            groups[(row["seed"], "selection16", row["choice"], 16, metric)].append(row[metric])
    for row in contrasts:
        groups[(row["seed"], "selection16_paired_contrast",
                f"{row['target_choice']}_minus_{row['reference_choice']}", 16,
                "delta")].append(row["delta"])
    for row in object_ranks:
        groups[(row["seed"], "client_rank", row["comparison"], None,
                "mean_defined_client_value")].append(row["mean_defined_client_value"])
    seed_rows = []
    for (seed, family, choice, prefix, metric), values in sorted(groups.items(), key=str):
        require(len(values) == 8, f"seed object denominator differs: {seed}/{family}/{choice}/{metric}")
        defined = [v for v in values if v is not None]
        seed_rows.append({"seed": seed, "family": family, "choice_or_comparison": choice,
                          "prefix": prefix, "metric": metric, "n_objects_expected": 8,
                          "n_objects_defined": len(defined),
                          "value": statistics.mean(defined) if len(defined) == 8 else None,
                          "status": "measured" if len(defined) == 8 else "NA_incomplete_object_metric"})
    # The two normalization definitions retain their separate denominators.
    finite_rows = [r for r in selection if r["choice"] == "finite_bank_search"]
    for seed in SEEDS:
        rows = [r for r in finite_rows if r["seed"] == seed]
        require(len(rows) == 8, f"finite bank seed denominator differs: {seed}")
        numerator = math.fsum(r["delta_vs_local_readout2048"] for r in rows)
        denominator = math.fsum(r["parent_readout2048_loss"] for r in rows)
        value = numerator / denominator if all(r["parent_readout2048_loss"] > 0 for r in rows) else None
        seed_rows.append({"seed": seed, "family": "correction_A_descriptive",
                          "choice_or_comparison": "finite_bank_search_minus_local", "prefix": 16,
                          "metric": "ratio_of_object_sums", "n_objects_expected": 8,
                          "n_objects_defined": 8 if value is not None else 0,
                          "value": value, "status": "measured" if value is not None else "NA_parent_loss",
                          "sum_delta": numerator, "sum_parent_loss": denominator})
    across = defaultdict(list)
    for row in seed_rows:
        across[(row["family"], row["choice_or_comparison"], row["prefix"], row["metric"])].append(row)
    across_rows = []
    for (family, choice, prefix, metric), rows in sorted(across.items(), key=str):
        require({r["seed"] for r in rows} == set(SEEDS), f"12-seed metric incomplete: {family}/{metric}")
        values = [r["value"] for r in rows if r["value"] is not None]
        complete = len(values) == 12
        across_rows.append({"family": family, "choice_or_comparison": choice, "prefix": prefix,
                            "metric": metric, "n_seeds_expected": 12, "n_seeds_defined": len(values),
                            "mean_of_seed_values": statistics.mean(values) if complete else None,
                            "sample_sd_seed_values": statistics.stdev(values) if complete else None,
                            "status": "measured" if complete else "NA_incomplete_seed_metric"})
    finite_seed = [r for r in seed_rows if r["family"] == "correction_A_descriptive"]
    if all(r["status"] == "measured" for r in finite_seed):
        pooled = math.fsum(r["sum_delta"] for r in finite_seed) / math.fsum(r["sum_parent_loss"] for r in finite_seed)
        across_rows.append({"family": "correction_A_descriptive", "choice_or_comparison": "finite_bank_search_minus_local",
                            "prefix": 16, "metric": "pooled_ratio_of_all_object_sums",
                            "n_seeds_expected": 12, "n_seeds_defined": 12,
                            "mean_of_seed_values": pooled, "sample_sd_seed_values": None,
                            "status": "measured_pooled_not_seed_mean"})
    return seed_rows, across_rows


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def summarize(export: Path, out: Path) -> dict:
    export = export.resolve(strict=True)
    out = out.resolve(strict=False)
    require(not out.exists() and out.parent.is_dir() and not out.is_relative_to(export),
            "output must be a new directory outside diagnostic export")
    source_manifest, objects, readouts, clients, locks, hashes = load_export(export)
    object_values, client_gains, selection, contrasts, client_ranks, object_ranks = object_and_client_tables(
        objects, readouts, clients, locks)
    seed_rows, across_rows = seed_tables(object_values, selection, contrasts, object_ranks)
    outputs = {"cohort_object_values.jsonl": object_values,
               "cohort_client_gains.jsonl": client_gains,
               "cohort_selection16.jsonl": selection,
               "cohort_selection16_contrasts.jsonl": contrasts,
               "cohort_client_ranks.jsonl": client_ranks,
               "cohort_object_rank_coverage.jsonl": object_ranks,
               "cohort_seed_descriptive.jsonl": seed_rows,
               "cohort_across_seed_descriptive.jsonl": across_rows}
    require(all(digest(Path(path)) == original for path, original in hashes.items()),
            "diagnostic input changed during summary")
    manifest = {"schema": SCHEMA, "complete": True, "source_export_schema": source_manifest["schema"],
                "source_export_path": str(export), "input_sha256": hashes,
                "summarizer_sha256": digest(Path(__file__)), "outputs": {},
                "independent_unit": "12 fixed seeds; 8 cohort objects per seed; 10 clients nested per object",
                "protocol_defined": ["three gain roles", "nonadditivity joint minus scaled individuals",
                                     "finite bank relative to local and parent-loss 0.5 percent with 8/12 sign",
                                     "same-path prefixes 1/4/16", "full vs h32 one-step fixed bank comparator"],
                "descriptive_choices_not_protocol_thresholds": [
                    "equal weight for 8 objects within each seed", "sample SD across 12 seed values",
                    "mean of object ratios and ratio of object sums both reported",
                    "three practical client supports for prefix ordering", "average ranks for ties",
                    "constant rank vectors are NA; incomplete object metric does not silently change weights"],
                "limits": "descriptive only; no A/B adjudication, winner, CI, or hypothesis test"}
    out.mkdir(exist_ok=False)
    try:
        for name, rows in outputs.items():
            path = out / name
            path.write_text(jsonl(rows))
            manifest["outputs"][name] = {"rows": len(rows), "sha256": digest(path)}
        require(all(digest(Path(path)) == original for path, original in hashes.items()),
                "diagnostic input changed before publication")
        (out / "SUMMARY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                              ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(out)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.export, args.out), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
