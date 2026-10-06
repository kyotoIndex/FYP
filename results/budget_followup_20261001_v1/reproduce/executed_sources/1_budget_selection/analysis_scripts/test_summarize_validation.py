"""Synthetic, complete-roster checks for the descriptive V summary."""
from __future__ import annotations

import csv
from io import StringIO
import json

import pytest

import summarize_validation as summary


def fixture_export(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    rows = []
    for slot in range(1, 17):
        for index, seed in enumerate(summary.SEEDS):
            # Slot 3 has a fixed 2 pp lead over slot 1. Replay loses to its
            # donor on even seeds and wins on odd seeds, retaining all 12 pairs.
            b8 = .50 + (slot - 1) * .01 + index * .001
            if slot == 3:
                b8 = .52 + index * .001
            if slot == 16:
                b8 = .52 + index * .001 + (.01 if index % 2 else -.01)
            rows.append({"job_id": f"V_S{slot:02d}_{summary.RECIPES[slot - 1]}_s{seed}",
                         "seed": seed, "slot": slot, "recipe_id": summary.RECIPES[slot - 1],
                         "hardware_family": "A100" if seed in (281, 307) else "4090D",
                         "half_accuracy": str(b8 - .05), "b8_accuracy": str(b8),
                         "r100_accuracy": str(b8 + .02)})
    endpoint = export / "v_endpoints.csv"
    endpoint.write_text(summary.csv_text(rows))
    manifest = {"schema": summary.EXPORT_SCHEMA, "complete": True, "job_count": 192,
                "fixed_count": 156, "conditional_count": 36, "replay_count": 12,
                "output_sha256": {"v_endpoints.csv": summary.digest(endpoint)}}
    (export / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
    return export, rows


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_full_descriptive_tables_and_donor_pair(tmp_path):
    export, _ = fixture_export(tmp_path)
    out = tmp_path / "summary"
    receipt = summary.summarize(export, out)
    assert receipt["complete"] is True
    slots = read_csv(out / "v_slot_descriptive.csv")
    seeds = read_csv(out / "v_seed_paired_differences.csv")
    pairs = read_csv(out / "v_pair_descriptive.csv")
    assert (len(slots), len(seeds), len(pairs)) == (48, 8640, 720)
    b8_slot3 = next(row for row in slots if row["endpoint"] == "b8" and row["slot"] == "3")
    assert float(b8_slot3["mean_accuracy_percent"]) == pytest.approx(52.55)
    assert float(b8_slot3["sample_sd_accuracy_pp"]) == pytest.approx(.3605551275)
    b8_3_vs_1 = next(row for row in pairs if row["endpoint"] == "b8" and
                     row["slot"] == "3" and row["reference_slot"] == "1")
    assert (float(b8_3_vs_1["mean_delta_pp"]), b8_3_vs_1["wins"],
            b8_3_vs_1["ties"], b8_3_vs_1["losses"]) == (2.0, "12", "0", "0")
    replay = next(row for row in pairs if row["endpoint"] == "b8" and
                  row["slot"] == "16" and row["reference_slot"] == "3")
    assert (replay["wins"], replay["ties"], replay["losses"], replay["comparison"]) == (
        "6", "0", "6", "donor_dependent_replay")
    assert float(replay["mean_delta_pp"]) == 0
    self_pair = next(row for row in pairs if row["endpoint"] == "b8" and
                     row["slot"] == "3" and row["reference_slot"] == "3")
    assert (self_pair["wins"], self_pair["ties"], self_pair["losses"]) == ("0", "12", "0")
    assert {row["hardware_family"] for row in seeds if row["seed"] == "281"} == {"A100"}
    for name, expected in receipt["output_sha256"].items():
        assert summary.digest(out / name) == expected
    with pytest.raises(summary.SummaryError, match="new directory"):
        summary.summarize(export, out)


def test_reject_incomplete_or_modified_export_before_output(tmp_path):
    export, rows = fixture_export(tmp_path)
    manifest_path = export / "EXPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(summary.SummaryError, match="full audited"):
        summary.summarize(export, tmp_path / "bad1")
    assert not (tmp_path / "bad1").exists()
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest))
    (export / "v_endpoints.csv").write_text(summary.csv_text(rows[:-1]))
    with pytest.raises(summary.SummaryError, match="SHA256"):
        summary.summarize(export, tmp_path / "bad2")
    assert not (tmp_path / "bad2").exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "192 rows"),
    (lambda rows: rows.__setitem__(-1, dict(rows[0])), "duplicate"),
    (lambda rows: rows[0].__setitem__("b8_accuracy", "nan"), "invalid accuracy"),
    (lambda rows: rows[0].__setitem__("hardware_family", "A100"), "hardware identity"),
])
def test_reject_bad_roster_even_with_matching_csv_hash(tmp_path, mutation, match):
    export, rows = fixture_export(tmp_path)
    mutation(rows)
    endpoint = export / "v_endpoints.csv"
    endpoint.write_text(summary.csv_text(rows))
    manifest_path = export / "EXPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["v_endpoints.csv"] = summary.digest(endpoint)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(summary.SummaryError, match=match):
        summary.summarize(export, tmp_path / "summary")
    assert not (tmp_path / "summary").exists()
