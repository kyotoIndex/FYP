"""Synthetic 144-object diagnostic export fixtures; never use formal results."""
from __future__ import annotations

import json

import pytest

import summarize_cohort as summary


def fixture_export(tmp_path):
    export = tmp_path / "diagnostic_fixture"
    export.mkdir()
    rows = {name: [] for name in summary.TABLES}
    for seed in summary.SEEDS:
        for parent in summary.PARENTS:
            for anchor in (10, 60):
                for factor in summary.FACTORS:
                    jid = f"C_{parent}_{seed}_{anchor}_{factor}"
                    base = {"job_id": jid, "line": "cohort", "seed": seed,
                            "parent_job_id": f"V_{parent}_{seed}",
                            "parent_recipe_id": parent, "anchor_round": anchor,
                            "factor": factor, "hardware_family": "A100" if seed in (281, 307) else "4090D",
                            "config_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
                            "anchor_sha256": "c" * 64, "audit_receipt_sha256": "d" * 64,
                            "acceptance_receipt_sha256": "e" * 64}
                    rows["objects"].append({**base, "record_type": "object", "state": "completed",
                                            "participants": list(range(10)), "errors": [],
                                            "result_sha256": "f" * 64})
                    parent_loss = .5 + (anchor == 60) * .1
                    for choice in summary.CHOICES:
                        for prefix in ((16,) if choice == "finite_bank_search" else summary.PREFIXES):
                            # The individual gains already include aggregation weight.
                            individual = .001 + (choice == "cyclic") * .001
                            joint = .02 + prefix * .001 + (choice == "finite_bank_search") * .01
                            if choice in summary.RANK_SUPPORTS:
                                index = summary.RANK_SUPPORTS.index(choice)
                                individual = (3-index if prefix == 1 else index+1) * .001
                            row = {**base, "choice": choice, "prefix": prefix,
                                   "readout_json_sha256": "f" * 64,
                                   "parent_readout2048_loss": parent_loss,
                                   "aggregate_readout2048_benefit": joint,
                                   "nonadditivity": joint - 10 * individual,
                                   "head_only_readout2048_benefit": .001,
                                   "lora_only_readout2048_benefit": .002,
                                   "cancellation": .3, "cancellation_status": "measured",
                                   "numerator_effective_norm_sq": .7,
                                   "denominator_mean_effective_norm_sq": 1.0,
                                   "max_effective_linearity_relative_error": 0.0}
                            rows["cohort_readouts"].append(row)
                            for cid in range(10):
                                weighted = individual
                                if choice in summary.RANK_SUPPORTS:
                                    index = summary.RANK_SUPPORTS.index(choice)
                                    weighted = (3-index if prefix == 1 else index+1) * .001
                                rows["cohort_clients"].append({**base, "choice": choice,
                                    "prefix": prefix, "client_id": cid,
                                    "readout_json_sha256": "f" * 64,
                                    "parent_readout2048_loss": parent_loss,
                                    "local64_benefit": .01,
                                    "weighted_readout2048_benefit": weighted})
                    prediction = {"candidate_order": list(summary.SUPPORTS[:4]),
                                  "per_client": {str(cid): {
                                      "one_step_linear_gain": {"h32": {choice: i for i, choice in enumerate(summary.SUPPORTS[:4])},
                                                               "full": {choice: i for i, choice in enumerate(summary.SUPPORTS[:4])}},
                                      "spearman": 1.0, "spearman_status": "measured"}
                                                 for cid in range(10)},
                                  "spearman_defined_clients": 10, "spearman_mean_defined": 1.0}
                    for kind in summary.LOCKS:
                        payload = (prediction if kind == "prediction_h32_full" else
                                   {"actual_bank_benefit": .05} if kind == "full_gradient_choice_readout"
                                   else {"fixture": True})
                        rows["cohort_locks"].append({**base, "lock_kind": kind,
                                                      "readout_json_sha256": "f" * 64,
                                                      "payload": payload})
        for anchor in (10, 60):
            for factor in summary.FACTORS:
                rows["objects"].append({"job_id": f"D_{seed}_{anchor}_{factor}",
                    "line": "direction", "seed": seed, "parent_job_id": f"V_D072_{seed}",
                    "parent_recipe_id": "D072", "anchor_round": anchor, "factor": factor,
                    "hardware_family": "A100" if seed in (281, 307) else "4090D",
                    "config_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
                    "anchor_sha256": "c" * 64, "audit_receipt_sha256": "d" * 64,
                    "acceptance_receipt_sha256": "e" * 64, "record_type": "object",
                    "state": "completed", "participants": list(range(10)), "errors": [],
                    "result_sha256": "f" * 64})
    rows["direction_readouts"] = [{"fixture": i} for i in range(1728)]
    rows["direction_geometry"] = [{"fixture": i} for i in range(9216)]
    rows["cohort_evaluations"] = [{"fixture": True}]
    tables = {}
    for name, data in rows.items():
        path = export / f"{name}.jsonl"
        path.write_text(summary.jsonl(data))
        tables[name] = {"rows": len(data), "sha256": summary.digest(path)}
    projection = {"schema": "pcla-v7-diagnostic-direction-projection-1",
                  "object_count": 48, "row_count": tables["direction_geometry"]["rows"],
                  "geometry_sha256": tables["direction_geometry"]["sha256"],
                  "objects": [{"job_id": row["job_id"],
                               "audit_receipt_sha256": row["audit_receipt_sha256"],
                               "stage_sha256": row["result_sha256"],
                               "bank_sha256": "9" * 64}
                              for row in rows["objects"] if row["line"] == "direction"]}
    projection_path = export / "direction_projection_manifest.json"
    projection_path.write_text(json.dumps(projection))
    manifest = {"schema": summary.EXPORT_SCHEMA, "object_count": 144,
                "cohort_count": 96, "direction_count": 48, "tables": tables,
                "direction_projection_manifest_sha256": summary.digest(projection_path)}
    (export / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
    return export, rows


def rehash(export, name, rows):
    path = export / f"{name}.jsonl"
    path.write_text(summary.jsonl(rows))
    manifest_path = export / "EXPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tables"][name] = {"rows": len(rows), "sha256": summary.digest(path)}
    manifest_path.write_text(json.dumps(manifest))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_fixture_seed_units_and_rank_roles(tmp_path):
    export, _ = fixture_export(tmp_path)
    out = tmp_path / "summary"
    receipt = summary.summarize(export, out)
    assert receipt["complete"] is True
    object_values = read_jsonl(out / "cohort_object_values.jsonl")
    client_gains = read_jsonl(out / "cohort_client_gains.jsonl")
    selections = read_jsonl(out / "cohort_selection16.jsonl")
    contrasts = read_jsonl(out / "cohort_selection16_contrasts.jsonl")
    ranks = read_jsonl(out / "cohort_client_ranks.jsonl")
    seeds = read_jsonl(out / "cohort_seed_descriptive.jsonl")
    across = read_jsonl(out / "cohort_across_seed_descriptive.jsonl")
    assert (len(object_values), len(client_gains), len(selections), len(contrasts), len(ranks)) == (
        1536, 15360, 672, 768, 5760)
    local = next(r for r in object_values if r["seed"] == 101 and r["choice"] == "local" and r["prefix"] == 16)
    assert local["scaled_individual_readout2048_sum_benefit"] == pytest.approx(.01)
    assert local["nonadditivity_independent"] == pytest.approx(.026)
    finite = next(r for r in selections if r["seed"] == 101 and r["choice"] == "finite_bank_search")
    assert finite["delta_vs_local_readout2048"] == pytest.approx(.01)
    jq_jl = next(r for r in contrasts if r["seed"] == 101 and
                 r["target_choice"] == "JQ_xi1" and r["reference_choice"] == "JL_h32")
    assert jq_jl["delta"] == pytest.approx(0)
    assert jq_jl["sign"] == "tie"
    rank = next(r for r in ranks if r["seed"] == 101 and r["comparison"] == "actual4_minus_actual1")
    assert rank["spearman_or_difference"] == pytest.approx(2.0)
    ratio = next(r for r in seeds if r["seed"] == 101 and r["metric"] == "ratio_of_object_sums")
    assert ratio["n_objects_expected"] == 8
    assert ratio["value"] == pytest.approx(.01 / .55)
    mean_ratios = next(r for r in seeds if r["seed"] == 101 and
                       r["family"] == "selection16" and
                       r["choice_or_comparison"] == "finite_bank_search" and
                       r["metric"] == "delta_over_parent_loss")
    assert mean_ratios["value"] == pytest.approx((.01/.5 + .01/.6) / 2)
    assert mean_ratios["value"] != pytest.approx(ratio["value"])
    assert any(r["metric"] == "pooled_ratio_of_all_object_sums" for r in across)
    assert all(r["n_seeds_expected"] == 12 for r in across)
    with pytest.raises(summary.SummaryError, match="new directory"):
        summary.summarize(export, out)


def test_missing_object_and_bad_nonadd_fail_before_output(tmp_path):
    export, rows = fixture_export(tmp_path)
    rows["objects"].pop()
    rehash(export, "objects", rows["objects"])
    with pytest.raises(summary.SummaryError, match="expected row count"):
        summary.summarize(export, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_bad_nonadd_and_constant_rank_NA(tmp_path):
    export, rows = fixture_export(tmp_path)
    rows["cohort_readouts"][0]["nonadditivity"] += .1
    rehash(export, "cohort_readouts", rows["cohort_readouts"])
    with pytest.raises(summary.SummaryError, match="nonadditivity independent"):
        summary.summarize(export, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_spearman_ties_and_constant():
    assert summary.spearman([1, 1, 3], [1, 1, 3]) == (1.0, "measured")
    assert summary.spearman([1, 1, 1], [1, 2, 3]) == (None, "NA_constant")


def test_cancellation_NA_does_not_reweight_seed(tmp_path):
    export, rows = fixture_export(tmp_path)
    row = rows["cohort_readouts"][0]
    row["cancellation"] = None
    row["cancellation_status"] = "NA_zero_denominator"
    row["denominator_mean_effective_norm_sq"] = 0
    rehash(export, "cohort_readouts", rows["cohort_readouts"])
    out = tmp_path / "summary"
    summary.summarize(export, out)
    seed_row = next(r for r in read_jsonl(out / "cohort_seed_descriptive.jsonl")
                    if r["seed"] == 101 and r["family"] == "cohort_object" and
                    r["choice_or_comparison"] == "local" and r["prefix"] == 1 and
                    r["metric"] == "cancellation")
    assert seed_row["value"] is None
    assert seed_row["status"] == "NA_incomplete_object_metric"
    assert seed_row["n_objects_defined"] == 7


def test_constant_client_rank_keeps_coverage(tmp_path):
    export, rows = fixture_export(tmp_path)
    lock = next(row for row in rows["cohort_locks"] if row["lock_kind"] == "prediction_h32_full")
    prediction = lock["payload"]
    prediction["per_client"]["0"]["one_step_linear_gain"]["h32"] = {
        choice: 0 for choice in summary.SUPPORTS[:4]}
    prediction["per_client"]["0"]["spearman"] = None
    prediction["per_client"]["0"]["spearman_status"] = "NA_constant"
    prediction["spearman_defined_clients"] = 9
    prediction["spearman_mean_defined"] = 1.0
    rehash(export, "cohort_locks", rows["cohort_locks"])
    out = tmp_path / "summary"
    summary.summarize(export, out)
    ranks = read_jsonl(out / "cohort_client_ranks.jsonl")
    affected = next(r for r in ranks if r["job_id"] == lock["job_id"] and
                    r["client_id"] == 0 and r["comparison"] == "h32_full_four")
    assert affected["status"] == "NA_constant" and affected["spearman_or_difference"] is None
    coverage = next(r for r in read_jsonl(out / "cohort_object_rank_coverage.jsonl")
                    if r["job_id"] == lock["job_id"] and r["comparison"] == "h32_full_four")
    assert coverage["defined_clients"] == 9
