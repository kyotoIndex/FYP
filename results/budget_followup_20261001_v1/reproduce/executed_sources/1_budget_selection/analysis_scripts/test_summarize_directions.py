"""Synthetic, complete export fixtures; no formal diagnostic result is summarized here."""
from __future__ import annotations

import json

import pytest

import summarize_directions as s


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def fixture_export(tmp_path_factory):
    export = tmp_path_factory.mktemp("synthetic_direction_export")
    tables = {name: [] for name in s.TABLE_COUNTS}
    projection = []
    for seed in s.SEEDS:
        for recipe, line in (("D069", "cohort"), ("D072", "cohort"), ("D072", "direction")):
            for anchor in (10, 60):
                for factor in ("a", "b"):
                    job = f"{line}_{seed}_{recipe}_{anchor}_{factor}"
                    ident = dict(job_id=job, line=line, seed=seed, parent_job_id=f"parent_{seed}_{recipe}",
                                 parent_recipe_id=recipe, anchor_round=anchor, factor=factor,
                                 hardware_family="A100" if seed in (281, 307) else "4090D",
                                 config_sha256="c" * 64, source_manifest_sha256="s" * 64,
                                 anchor_sha256="a" * 64, audit_receipt_sha256="u" * 64,
                                 acceptance_receipt_sha256="e" * 64)
                    tables["objects"].append({**ident, "state": "completed", "errors": [],
                        "warnings": [s.WARNING] if line == "direction" else [],
                        "readout_source_id": None,
                        "readout_source_limitation": s.WARNING if line == "direction" else None,
                        "direction_readout_source_row_hash": None, "result_sha256": "t" * 64,
                        "participants": list(range(10))})
                    if line == "cohort":
                        for choice in ("local", "cyclic", "JL_h32", "public512_linear", "JQ_xi1"):
                            for prefix in (1, 4, 16):
                                tables["cohort_readouts"].append({**ident, "choice": choice, "prefix": prefix})
                                for client in range(10):
                                    tables["cohort_clients"].append({**ident, "choice": choice,
                                                                      "prefix": prefix, "client_id": client})
                        tables["cohort_readouts"].append({**ident, "choice": "finite_bank_search", "prefix": 16})
                        for client in range(10):
                            tables["cohort_clients"].append({**ident, "choice": "finite_bank_search",
                                                              "prefix": 16, "client_id": client})
                        for kind in ("JQ_pretraining", "selection1024", "prediction_h32_full",
                                     "full_gradient_choice_readout"):
                            tables["cohort_locks"].append({**ident, "lock_kind": kind})
                        # 28 objects have 355 evaluations; the rest have 354.
                        ordinal_count = 355 if len([r for r in tables["objects"] if r["line"] == "cohort"]) <= 28 else 354
                        for ordinal in range(ordinal_count):
                            tables["cohort_evaluations"].append({**ident, "ordinal": ordinal})
                        continue
                    projection.append({"job_id": job, "bank_sha256": "b" * 64,
                                       "stage_sha256": "t" * 64, "audit_receipt_sha256": "u" * 64})
                    base = {**ident, "bank_sha256": "b" * 64, "stage_sha256": "t" * 64}
                    tables["direction_geometry"].append({**base, "record_type": "direction_parent"})
                    for label in s.LABELS:
                        is_active = not (label == "S_zero_removed" and seed == 101 and anchor == 10 and factor == "a")
                        for module in range(24):
                            tables["direction_geometry"].append({**base, "record_type": "representation_module",
                                "representation": label, "module": f"m{module}", "active": is_active,
                                "numerical_rank": 2, "rank_threshold": 0.01,
                                "relative_error": 0.2, "random_rotation": None})
                    tables["direction_geometry"].append({**base, "record_type": "T_comparator_identity",
                        "status": "diagnostic_only_no_extra_training"})
                    for module in range(24):
                        tables["direction_geometry"].append({**base, "record_type": "T_module_comparison",
                            "module": f"m{module}", "gram_frobenius_absolute_error": 0.1,
                            "gram_frobenius_relative_error": 0.2})
                    for policy in s.POLICIES:
                        full = 0.0 if seed == 101 and anchor == 10 and factor == "a" else 2.0
                        tables["direction_geometry"].append({**base, "record_type": "T_prediction_aggregate",
                                                               "policy": policy, "full": full, "h32": 1.0})
                        for client in range(10):
                            tables["direction_geometry"].append({**base, "record_type": "T_prediction_client",
                                "policy": policy, "client_id": client, "full": full, "h32": 1.0})
                    values = {"N": 0.0, "D": 2.0, "S": 4.0, "T": 6.0,
                              "S_random_nonzero": 5.0, "S_zero_removed": 3.0}
                    for label in s.LABELS:
                        for policy in s.POLICIES:
                            for prefix in s.PREFIXES:
                                is_na = label == "D" and seed == 101 and anchor == 10 and factor == "a"
                                tables["direction_readouts"].append({**ident, "stage_sha256": "t" * 64,
                                    "representation": label, "policy": policy, "prefix": prefix,
                                    "original_loss": 10 - values[label], "original_benefit": values[label],
                                    "normmatched_loss": None if is_na else 10 - values[label],
                                    "normmatched_benefit": None if is_na else values[label],
                                    "normmatch_status": "NA_zero_candidate" if is_na else "matched",
                                    "normmatch_multiplier": None if is_na else 1.0,
                                    "reference_n_effective_norm": 1.0,
                                    "candidate_effective_norm": 0.0 if is_na else 1.0})
    assert {k: len(v) for k, v in tables.items()} == s.TABLE_COUNTS
    hashes = {}
    for name, rows in tables.items():
        path = export / f"{name}.jsonl"
        write_jsonl(path, rows)
        hashes[name] = {"rows": len(rows), "sha256": s.digest(path)}
    projection_record = {"object_count": 48, "objects": projection,
                         "geometry_sha256": hashes["direction_geometry"]["sha256"]}
    projection_path = export / "direction_projection_manifest.json"
    projection_path.write_text(json.dumps(projection_record))
    manifest = {"schema": s.EXPORT_SCHEMA, "object_count": 144, "cohort_count": 96,
                "direction_count": 48, "source_manifest_sha256": "s" * 64,
                "tables": hashes, "direction_geometry_sha256": hashes["direction_geometry"]["sha256"],
                "direction_projection_manifest_sha256": s.digest(projection_path)}
    (export / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
    return export


def test_complete_descriptive_fixture(fixture_export, tmp_path):
    out = tmp_path / "summary"
    manifest = s.summarize(fixture_export, out)
    assert manifest["complete"] is True
    contrasts = read_jsonl(out / "direction_object_contrasts.jsonl")
    assert len(contrasts) == 48 * 2 * 3 * 5
    d = next(r for r in contrasts if r["seed"] == 101 and r["anchor_round"] == 10 and
             r["factor"] == "a" and r["target"] == "D")
    assert d["original_delta_benefit"] == 2
    assert d["normmatched_delta_benefit"] is None
    seeds = read_jsonl(out / "direction_seed_descriptive.jsonl")
    d_seed = next(r for r in seeds if r["seed"] == 101 and r["target"] == "D")
    assert d_seed["n_matched_objects"] == 3 and d_seed["normmatched_mean_delta_benefit"] is None
    zero_seed = next(r for r in seeds if r["seed"] == 101 and r["target"] == "S_zero_removed")
    assert zero_seed["n_active_objects"] == 3
    assert zero_seed["active_conditional_original_mean"] == -1
    overall = read_jsonl(out / "direction_overall_descriptive.jsonl")
    d_total = next(r for r in overall if r["target"] == "D")
    assert d_total["original"]["mean"] == 2 and d_total["normmatched"] is None
    assert d_total["n_complete_matched_seeds"] == 11
    pred = read_jsonl(out / "direction_T_prediction_seed.jsonl")
    first = next(r for r in pred if r["seed"] == 101)
    assert first["n_nonzero_full_denominator"] == 3
    assert first["mean_relative_error_abs_full"] == 0.5
    assert not (tmp_path / "summary.building").exists()
    with pytest.raises(s.SummaryError, match="new directory"):
        s.summarize(fixture_export, out)


def test_hash_and_duplicate_gate(fixture_export, tmp_path):
    # Replacing a byte without updating the manifest fails before any output is published.
    copy = tmp_path / "tampered"
    copy.mkdir()
    for path in fixture_export.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    path = copy / "direction_readouts.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(s.SummaryError, match="count/hash"):
        s.summarize(copy, tmp_path / "bad_summary")
    assert not (tmp_path / "bad_summary").exists()
    # A duplicate object key remains invalid even if all table hashes are refreshed.
    path.write_bytes((fixture_export / path.name).read_bytes())
    rows = read_jsonl(path)
    rows[1] = rows[0]
    write_jsonl(path, rows)
    manifest_path = copy / "EXPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tables"]["direction_readouts"]["sha256"] = s.digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(s.SummaryError, match="duplicate direction_readouts"):
        s.summarize(copy, tmp_path / "bad_summary")
