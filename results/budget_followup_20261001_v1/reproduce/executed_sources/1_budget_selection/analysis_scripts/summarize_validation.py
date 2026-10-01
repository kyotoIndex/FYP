#!/usr/bin/env python3
"""Descriptive, seed-paired summaries of a complete audited V endpoint export."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import shutil
import statistics


SCHEMA = "pcla-v7-validation-descriptive-summary-1"
EXPORT_SCHEMA = "pcla-v7-validation-export-1"
SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
RECIPES = ("D002", "D006", "D035", "D042", "D045", "D054", "D063", "D066",
           "D069", "D072", "D078", "D086", "D105", "D107", "D112", "D035")
ENDPOINTS = ("half", "b8", "r100")
JOB_RE = re.compile(r"V_S(\d{2})_(D\d{3})_s(\d+)")


class SummaryError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_text(rows: list[dict]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def accuracy(value: str, key: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError):
        raise SummaryError(f"invalid accuracy: {key}") from None
    if not result.is_finite() or not 0 <= result <= 1:
        raise SummaryError(f"invalid accuracy: {key}")
    return result


def mean_sd(values: list[Decimal]) -> tuple[str, str]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values, xbar=mean)
    return format(mean, ".10f"), format(sd, ".10f")


def load_export(export: Path) -> tuple[dict, dict[tuple[int, int], dict], dict[str, str]]:
    export = export.resolve(strict=True)
    manifest_path = export / "EXPORT_MANIFEST.json"
    endpoint_path = export / "v_endpoints.csv"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise SummaryError(f"invalid export manifest: {exc}") from exc
    if (manifest.get("schema") != EXPORT_SCHEMA or manifest.get("complete") is not True or
            manifest.get("job_count") != 192 or manifest.get("fixed_count") != 156 or
            manifest.get("conditional_count") != 36 or manifest.get("replay_count") != 12):
        raise SummaryError("export manifest does not certify full audited V roster")
    expected_hash = manifest.get("output_sha256", {}).get("v_endpoints.csv")
    actual_hash = digest(endpoint_path)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise SummaryError("v_endpoints.csv differs from export manifest SHA256")
    with endpoint_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if (fields is None or len(fields) != len(set(fields)) or
                not {"job_id", "seed", "slot", "recipe_id", "hardware_family",
                     *(f"{label}_accuracy" for label in ENDPOINTS)}.issubset(fields)):
            raise SummaryError("v_endpoints.csv required columns missing or duplicated")
        rows = list(reader)
    if len(rows) != 192:
        raise SummaryError("v_endpoints.csv must contain exactly 192 rows")
    mapped = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise SummaryError("malformed CSV row")
        match = JOB_RE.fullmatch(row["job_id"])
        if match is None:
            raise SummaryError("invalid V job ID")
        slot, seed = int(match[1]), int(match[3])
        if (slot not in range(1, 17) or seed not in SEEDS or
                row["slot"] != str(slot) or row["seed"] != str(seed) or
                row["recipe_id"] != RECIPES[slot - 1] or match[2] != RECIPES[slot - 1] or
                row["hardware_family"] != ("A100" if seed in (281, 307) else "4090D")):
            raise SummaryError(f"V roster or hardware identity differs: {row['job_id']}")
        key = (slot, seed)
        if key in mapped:
            raise SummaryError(f"duplicate V seed/slot: {key}")
        mapped[key] = {**row, "accuracy": {label: accuracy(row[f"{label}_accuracy"],
                                                       f"{row['job_id']} {label}")
                                             for label in ENDPOINTS}}
    if set(mapped) != {(slot, seed) for slot in range(1, 17) for seed in SEEDS}:
        raise SummaryError("V seed/slot matrix incomplete")
    # Guard against the export changing while this summary is calculated.
    hashes = {str(manifest_path): digest(manifest_path), str(endpoint_path): actual_hash}
    return manifest, mapped, hashes


def tables(mapped: dict[tuple[int, int], dict]) -> dict[str, str]:
    slot_rows, seed_rows, pair_rows = [], [], []
    for label in ENDPOINTS:
        for slot in range(1, 17):
            values = [mapped[slot, seed]["accuracy"][label] * 100 for seed in SEEDS]
            mean, sd = mean_sd(values)
            slot_rows.append({"endpoint": label, "slot": slot, "recipe_id": RECIPES[slot - 1],
                              "kind": "donor_dependent_replay" if slot == 16 else "non_replay",
                              "n_seeds": 12, "mean_accuracy_percent": mean,
                              "sample_sd_accuracy_pp": sd})
            for reference in range(1, 16):
                differences = []
                wins = ties = losses = 0
                for seed in SEEDS:
                    target = mapped[slot, seed]["accuracy"][label] * 100
                    baseline = mapped[reference, seed]["accuracy"][label] * 100
                    delta = target - baseline
                    differences.append(delta)
                    outcome = "win" if delta > 0 else "loss" if delta < 0 else "tie"
                    wins += outcome == "win"
                    ties += outcome == "tie"
                    losses += outcome == "loss"
                    seed_rows.append({"endpoint": label, "slot": slot, "reference_slot": reference,
                                      "seed": seed, "hardware_family": mapped[slot, seed]["hardware_family"],
                                      "target_accuracy_percent": format(target, ".10f"),
                                      "reference_accuracy_percent": format(baseline, ".10f"),
                                      "delta_pp": format(delta, ".10f"), "outcome": outcome})
                mean_delta, sd_delta = mean_sd(differences)
                pair_rows.append({"endpoint": label, "slot": slot, "reference_slot": reference,
                                  "n_paired_seeds": 12, "mean_delta_pp": mean_delta,
                                  "sample_sd_delta_pp": sd_delta, "wins": wins,
                                  "ties": ties, "losses": losses,
                                  "comparison": "donor_dependent_replay" if slot == 16 else "non_replay"})
    return {"v_slot_descriptive.csv": csv_text(slot_rows),
            "v_seed_paired_differences.csv": csv_text(seed_rows),
            "v_pair_descriptive.csv": csv_text(pair_rows)}


def summarize(export: Path, out: Path) -> dict:
    export = export.resolve(strict=True)
    out = out.resolve(strict=False)
    if out.exists() or not out.parent.is_dir() or out.is_relative_to(export):
        raise SummaryError("output must be a new directory outside the input export")
    original, mapped, hashes = load_export(export)
    outputs = tables(mapped)
    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise SummaryError("input changed during summary")
    manifest = {"schema": SCHEMA, "complete": True, "source_export_complete": True,
                "source_export_schema": original["schema"], "source_export_path": str(export),
                "input_sha256": hashes, "summarizer_sha256": digest(Path(__file__)),
                "output_sha256": {}, "endpoints": list(ENDPOINTS), "seed_count": 12,
                "slot_count": 16, "reference_slots": list(range(1, 16)),
                "definitions": {
                    "accuracy_percent": "raw endpoint accuracy fraction multiplied by 100",
                    "slot_mean": "unweighted arithmetic mean across the 12 fixed seeds",
                    "sample_sd": "sample standard deviation across 12 seeds, denominator n-1",
                    "paired_difference_pp": "same-seed target slot accuracy percent minus reference slot accuracy percent",
                    "wins_ties_losses": "counts by exact sign of the unrounded paired difference; ties equal zero",
                    "pair_mean_sd": "arithmetic mean and sample SD of 12 same-seed paired differences",
                    "reference_scope": "each of 16 target slots versus every one of 15 non-replay reference slots; self-pairs included",
                    "slot16": "donor-dependent same-seed slot3 quota replay; not independent deployable method",
                    "hardware": "seeds 281 and 307 A100; the other 10 seeds 4090D; no time aggregation",
                    "inference": "descriptive only; no winner selection, A/B adjudication, intervals, or hypothesis tests"}}
    out.mkdir(exist_ok=False)
    try:
        for name, content in outputs.items():
            path = out / name
            path.write_text(content)
            manifest["output_sha256"][name] = digest(path)
        for path, expected in hashes.items():
            if digest(Path(path)) != expected:
                raise SummaryError("input changed before summary publication")
        (out / "SUMMARY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                              ensure_ascii=False) + "\n")
    except BaseException:
        shutil.rmtree(out)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="complete validation export directory")
    parser.add_argument("--out", type=Path, required=True, help="new summary directory")
    args = parser.parse_args()
    print(json.dumps(summarize(args.export, args.out), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
