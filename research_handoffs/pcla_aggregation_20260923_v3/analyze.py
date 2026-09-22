from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

from common import (
    SEEDS, SOURCES, atomic_json, bind_project, fingerprint, read_json,
    require, sha256_file,
)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classification(logits, labels):
    require(len(logits) == len(labels) and len(labels) > 0,
            "logits/labels length mismatch")
    predictions, loss = [], 0.0
    confusion = [[0, 0], [0, 0]]
    for row, label in zip(logits, labels):
        require(len(row) == 2 and label in (0, 1), "invalid binary logits")
        require(all(math.isfinite(float(value)) for value in row),
                "nonfinite logits")
        maximum = max(row)
        exp0 = math.exp(row[0] - maximum)
        exp1 = math.exp(row[1] - maximum)
        denominator = exp0 + exp1
        probability = (exp0 if label == 0 else exp1) / denominator
        require(probability > 0 and math.isfinite(probability),
                "invalid true-class probability")
        loss -= math.log(probability)
        prediction = int(row[1] > row[0])
        predictions.append(prediction)
        confusion[label][prediction] += 1
    recalls = []
    for label in (0, 1):
        denominator = sum(confusion[label])
        recalls.append(confusion[label][label] / denominator if denominator else 0.0)
    return {
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "nll_float64": loss / len(labels),
        "negative_recall": recalls[0],
        "positive_recall": recalls[1],
        "balanced_accuracy": sum(recalls) / 2,
        "predictions": predictions,
        "confusion": confusion,
        "predicted_negative": predictions.count(0),
        "predicted_positive": predictions.count(1),
        "strict_one_class": len(set(predictions)) == 1,
    }


def adapters_equal(torch, left, right):
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def old_eval_labels(repo, bridge, source_dir, checkpoint):
    legacy = bridge.checked_legacy(Path(repo))
    old = checkpoint["identity"]["provenance"]["base_config"]
    dev_path = Path(old["dataset"]["dev_file"])
    if not dev_path.is_absolute():
        dev_path = Path(repo) / dev_path
    rows = legacy.read_text_rows(dev_path, old["dataset"])
    partition = read_json(source_dir / "partition_ids.json")
    ids = partition["benchmark_dev"]
    labels = []
    for source_id in ids:
        split, index = source_id.split(":")
        require(split == "dev", "benchmark source ID is not dev")
        labels.append(int(legacy.row_label(rows[int(index)])))
    return partition, ids, labels


def audit_old_fixed(site, torch, runtime, bridge, seed, policy):
    source_dir = Path(site["source_root"]) / f"seed_{seed}" / policy
    checkpoint = runtime.load_checkpoint(source_dir / "checkpoint.pt")
    require(checkpoint["round"] == 30 and checkpoint["policy"] == policy,
            "old fixed source is incomplete")
    require(checkpoint["identity_hash"] == fingerprint(checkpoint["identity"]),
            "old fixed identity hash differs")
    require(checkpoint["identity"]["config"]["aggregation"] == "fixed",
            "old fixed source has another aggregation")
    require(len(checkpoint["metrics"]) == 30,
            "old fixed metrics are incomplete")
    probe = runtime.load_checkpoint(source_dir / "probe_round_030.pt")
    require(probe["round"] == 30 and
            adapters_equal(torch, probe["adapter"], checkpoint["adapter"]),
            "old final checkpoint differs from round-30 probe")
    partition, source_ids, labels = old_eval_labels(
        site["repo"], bridge, source_dir, checkpoint
    )
    require(fingerprint(partition) ==
            checkpoint["identity"]["provenance"]["partition_sha256"],
            "old partition fingerprint differs")
    rows = []
    for round_id, saved in enumerate(checkpoint["metrics"], start=1):
        predictions = read_json(
            source_dir / f"predictions_round_{round_id:03d}.json"
        )
        require(len(predictions) == len(labels), "old prediction length differs")
        require(all(value in (0, 1) for value in predictions),
                "old prediction contains an invalid class")
        confusion = [[0, 0], [0, 0]]
        for label, prediction in zip(labels, predictions):
            confusion[label][prediction] += 1
        recalls = [
            confusion[label][label] / sum(confusion[label])
            if sum(confusion[label]) else 0.0
            for label in (0, 1)
        ]
        accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
        require(abs(accuracy - saved["accuracy"]) <= 1e-15,
                "old saved accuracy differs from predictions")
        rows.append({
            "round": round_id,
            "accuracy": accuracy,
            "gpu_nll": saved["loss"],
            "negative_recall": recalls[0],
            "positive_recall": recalls[1],
            "balanced_accuracy": sum(recalls) / 2,
            "predicted_negative": predictions.count(0),
            "predicted_positive": predictions.count(1),
            "strict_one_class": len(set(predictions)) == 1,
        })
    return {
        "checkpoint": checkpoint,
        "source_dir": source_dir,
        "partition": partition,
        "source_ids": source_ids,
        "labels": labels,
        "rows": rows,
        "nll_scope": "original saved GPU loss; logits were not retained",
    }


def stable_study(study):
    fields = (
        "batch", "family", "source_policy", "rule", "seed",
        "original_identity", "original_checkpoint_sha256", "handoff_sources",
    )
    return {key: study[key] for key in fields}


def audit_new_run(site, torch, runtime, root, job, source_audit):
    run_root = Path(root) / job / "result"
    study = read_json(run_root / "study.json")
    run = run_root / "run"
    status = read_json(run / "status.json")
    require(status["state"] == "completed" and status["completed_round"] == 30,
            f"new run is incomplete: {job}")
    checkpoint = runtime.load_checkpoint(run / "checkpoint.pt")
    source_checkpoint = source_audit["checkpoint"]
    require(checkpoint["round"] == 30,
            f"final checkpoint is incomplete: {job}")
    require(study["original_identity"] == source_checkpoint["identity_hash"],
            f"source identity differs: {job}")
    require(study["original_checkpoint_sha256"] ==
            sha256_file(source_audit["source_dir"] / "checkpoint.pt"),
            f"source checkpoint SHA differs: {job}")
    require(checkpoint["identity_hash"] == fingerprint(checkpoint["identity"]),
            f"checkpoint identity hash differs: {job}")
    require(checkpoint["identity"]["initial_adapter_sha256"] ==
            source_checkpoint["identity"]["initial_adapter_sha256"],
            f"initial adapter identity differs: {job}")
    require(checkpoint["identity"]["provenance"]["aggregation_study"] ==
            stable_study(study), f"checkpoint study metadata differs: {job}")
    partition = read_json(run / "partition_ids.json")
    require(partition == source_audit["partition"],
            f"partition/source IDs differ: {job}")
    require(fingerprint(partition) ==
            checkpoint["identity"]["provenance"]["partition_sha256"],
            f"partition fingerprint differs: {job}")

    if study["family"] == "fixed_trace":
        require(set(checkpoint["trace"]) == set(source_checkpoint["trace"]),
                f"trace inventory differs: {job}")
        require(all(torch.equal(value, checkpoint["trace"][key])
                    for key, value in source_checkpoint["trace"].items()),
                f"fixed-support trace differs: {job}")

    probe = runtime.load_checkpoint(run / "probe_round_030.pt")
    require(probe["round"] == 30 and
            probe["identity_hash"] == checkpoint["identity_hash"] and
            adapters_equal(torch, probe["adapter"], checkpoint["adapter"]),
            f"final checkpoint differs from round-30 probe: {job}")

    metrics = read_json(run / "metrics.json")
    require(len(metrics) == 30, f"metrics are incomplete: {job}")
    aggregation_files = sorted(run.glob("aggregation_*.json"))
    require(len(aggregation_files) == 30,
            f"aggregation audit count differs: {job}")
    rows = []
    labels_reference = None
    for round_id in range(1, 31):
        payload = read_json(run / f"logits_round_{round_id:03d}.json")
        readout = classification(payload["logits"], payload["labels"])
        if labels_reference is None:
            labels_reference = payload["labels"]
        require(payload["labels"] == labels_reference,
                f"evaluation labels/order changed: {job}")
        require(payload["labels"] == source_audit["labels"],
                f"evaluation labels differ from source benchmark: {job}")
        predictions = read_json(run / f"predictions_round_{round_id:03d}.json")
        require(predictions == readout["predictions"],
                f"predictions differ from logits: {job}/{round_id}")
        saved = metrics[round_id - 1]
        require(saved["round"] == round_id,
                f"saved metric round differs: {job}/{round_id}")
        require(abs(saved["accuracy"] - readout["accuracy"]) <= 1e-15,
                f"saved accuracy differs: {job}/{round_id}")
        require(abs(saved["loss"] - readout["nll_float64"]) <= 1e-5,
                f"saved GPU NLL differs unexpectedly: {job}/{round_id}")
        audit = read_json(run / f"aggregation_{round_id:03d}.json")
        require(audit["round"] == round_id and audit["rule"] == study["rule"],
                f"aggregation record differs: {job}/{round_id}")
        require(audit["classifier_increment_sha256"],
                f"classifier audit is missing: {job}/{round_id}")
        if study["rule"] == "function_norm_matched":
            require(audit["function_match_relative_error"] <= 1e-8,
                    f"function-norm matching failed: {job}/{round_id}")
        rows.append({
            "round": round_id,
            **{key: readout[key] for key in (
                "accuracy", "nll_float64", "negative_recall",
                "positive_recall", "balanced_accuracy",
                "predicted_negative", "predicted_positive", "strict_one_class",
            )},
            "recorded_gpu_nll": saved["loss"],
            "aggregation": audit,
        })

    selected_clients = [
        [x["client"] for x in read_json(
            run / f"clients_round_{round_id:03d}.json"
        )]
        for round_id in range(1, 31)
    ]
    source_clients = [
        [x["client"] for x in read_json(
            source_audit["source_dir"] / f"clients_round_{round_id:03d}.json"
        )]
        for round_id in range(1, 31)
    ]
    require(selected_clients == source_clients,
            f"selected-client sequence differs: {job}")

    if study["kind"] == "online_replay":
        donor_job = f"online_pcla_contributors_{study['seed']}"
        donor_root = Path(root) / donor_job / "result"
        donor_checkpoint_path = donor_root / "run" / "checkpoint.pt"
        require(sha256_file(donor_checkpoint_path) == study["donor_sha256"],
                f"online donor SHA differs: {job}")
        donor = runtime.load_checkpoint(donor_checkpoint_path)
        require(donor["round"] == 30 and donor["policy"] == "pcla",
                f"online donor is incomplete: {job}")
        require(donor["identity_hash"] == checkpoint["identity_hash"],
                f"online donor identity differs: {job}")
        require(set(donor["trace"]) == set(checkpoint["trace"]),
                f"online replay trace inventory differs: {job}")
        for key in donor["trace"]:
            donor_counts = donor["trace"][key].sum(1)
            replay_counts = checkpoint["trace"][key].sum(1)
            require(torch.equal(donor_counts, replay_counts),
                    f"online replay layer/count trajectory differs: {job}/{key}")

    return {"checkpoint": checkpoint, "study": study,
            "rows": rows, "labels": labels_reference,
            "nll_scope": "float64 recomputed from saved logits"}


def endpoint(rows):
    require([x["round"] for x in rows] == list(range(1, 31)),
            "endpoint needs 30 ordered rounds")
    tail = rows[-5:]
    return {
        "accuracy": statistics.mean(x["accuracy"] for x in tail),
        "nll": statistics.mean(
            x.get("nll_float64", x.get("gpu_nll")) for x in tail
        ),
        "tail_min": min(x["accuracy"] for x in tail),
        "tail_max": max(x["accuracy"] for x in tail),
        "final_accuracy": rows[-1]["accuracy"],
        "balanced_accuracy": statistics.mean(x["balanced_accuracy"] for x in tail),
        "strict_one_class_rounds": sum(bool(x["strict_one_class"]) for x in tail),
    }


def paired(name, values, seeds):
    numbers = [float(values[seed]) for seed in seeds]
    mean = statistics.mean(numbers)
    if len(numbers) > 1:
        critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
        require(len(numbers) in critical, "unsupported descriptive interval size")
        standard = statistics.stdev(numbers)
        half = critical[len(numbers)] * standard / math.sqrt(len(numbers))
        interval = [mean - half, mean + half]
    else:
        interval = [None, None]
    return {
        "comparison": name, "seeds": list(seeds),
        "per_seed_difference_pp": numbers,
        "mean_difference_pp": mean,
        "descriptive_t_interval_pp": interval,
        "positive_seeds": sum(x > 0 for x in numbers),
        "negative_seeds": sum(x < 0 for x in numbers),
    }


def fixed_state(queue, root, output):
    queue, root, output = Path(queue), Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inventory = read_json(root / "inventory.json")
    required = {f"diagnose_{seed}_{anchor}" for seed in SEEDS
                for anchor in (5, 15)}
    completed = {x["job"] for x in inventory if x["state"] == "done"}
    require(required <= completed, "all fixed-state jobs are required")
    rows, positions, coverage_cells = [], [], []
    max_match_error = 0.0
    for seed in SEEDS:
        for anchor in (5, 15):
            job = f"diagnose_{seed}_{anchor}"
            run = root / job / "result"
            result = read_json(run / "result.json")
            require(result["state"] == "completed" and
                    result["client_calls"] == 60 and
                    result["optimizer_steps"] == 960,
                    f"fixed-state execution count differs: {job}")
            summary = read_json(run / "summary.json")
            for row in summary:
                evaluation = read_json(
                    run / f"evaluation_{row['round']:03d}_{row['support']}_{row['rule']}.json"
                )
                readout = classification(evaluation["logits"], evaluation["labels"])
                require(abs(readout["accuracy"] - row["accuracy"]) <= 1e-15,
                        "fixed-state accuracy differs")
                require(abs(readout["nll_float64"] - row["nll"]) <= 1e-10,
                        "fixed-state NLL differs")
                rows.append({**row,
                             "recomputed_accuracy": readout["accuracy"],
                             "recomputed_nll": readout["nll_float64"]})
                if row["support"] != "full":
                    max_match_error = max(
                        max_match_error,
                        row["aggregation"]["function_match_relative_error"],
                    )
            for round_id in (anchor + 1, anchor + 2):
                for support in ("pcla", "replay_random_rank", "full"):
                    report = read_json(
                        run / f"coverage_{round_id:03d}_{support}.json"
                    )
                    covered = [x for x in report["positions"]
                               if x["coverage_fraction"] > 0]
                    for position in report["positions"]:
                        positions.append({"seed": seed, "anchor": anchor,
                                          "round": round_id, "support": support,
                                          **position})
                    coverage_cells.append({
                        "seed": seed, "anchor": anchor, "round": round_id,
                        "support": support,
                        "coverage_min": min(x["coverage_fraction"]
                                            for x in report["positions"]),
                        "coverage_max": max(x["coverage_fraction"]
                                            for x in report["positions"]),
                        "coverage_std": statistics.pstdev(
                            x["coverage_fraction"] for x in report["positions"]
                        ),
                        "covered_positions": len(covered),
                        "median_abs_positive_ratio_shift": (
                            statistics.median(abs(x["positive_ratio_shift"])
                                              for x in covered)
                            if covered else None
                        ),
                    })
    require(max_match_error <= 1e-8, "function matching exceeds tolerance")
    write_csv(output / "fixed_state_rows.csv", rows)
    write_csv(output / "coverage_positions.csv", positions)
    write_csv(output / "coverage_cells.csv", coverage_cells)

    lookup = {(x["seed"], x["anchor"], x["round"], x["support"], x["rule"]): x
              for x in rows}
    effects = []
    for seed in SEEDS:
        for anchor in (5, 15):
            for round_id in (anchor + 1, anchor + 2):
                for support in ("pcla", "replay_random_rank"):
                    fixed = lookup[(seed, anchor, round_id, support, "fixed")]
                    contributor = lookup[(seed, anchor, round_id, support,
                                          "contributors")]
                    matched = lookup[(seed, anchor, round_id, support,
                                      "function_norm_matched")]
                    effects.append({
                        "seed": seed, "anchor": anchor, "round": round_id,
                        "support": support,
                        "C_minus_F_accuracy_pp": 100 *
                            (contributor["accuracy"] - fixed["accuracy"]),
                        "G_minus_F_accuracy_pp": 100 *
                            (matched["accuracy"] - fixed["accuracy"]),
                        "C_minus_G_accuracy_pp": 100 *
                            (contributor["accuracy"] - matched["accuracy"]),
                        "C_minus_F_nll": contributor["nll"] - fixed["nll"],
                        "G_minus_F_nll": matched["nll"] - fixed["nll"],
                        "C_minus_G_nll": contributor["nll"] - matched["nll"],
                    })
    write_csv(output / "fixed_state_effects.csv", effects)
    nonzero = sum(abs(x["C_minus_F_nll"]) > 1e-6 or
                  abs(x["C_minus_F_accuracy_pp"]) > 1e-9 for x in effects)
    explained = []
    for row in effects:
        denominator = abs(row["C_minus_F_nll"])
        if denominator > 1e-8:
            explained.append(abs(row["G_minus_F_nll"]) / denominator)
    gate = {
        "cells": len(effects),
        "nonzero_C_vs_F_cells": nonzero,
        "max_function_match_relative_error": max_match_error,
        "median_abs_C_minus_F_accuracy_pp": statistics.median(
            abs(x["C_minus_F_accuracy_pp"]) for x in effects
        ),
        "median_abs_C_minus_F_nll": statistics.median(
            abs(x["C_minus_F_nll"]) for x in effects
        ),
        "median_G_fraction_of_C_nll_movement": (
            statistics.median(explained) if explained else None
        ),
        "median_coverage_std_pcla": statistics.median(
            x["coverage_std"] for x in coverage_cells if x["support"] == "pcla"
        ),
        "median_abs_positive_shift_pcla": statistics.median(
            x["median_abs_positive_ratio_shift"] for x in coverage_cells
            if x["support"] == "pcla" and
            x["median_abs_positive_ratio_shift"] is not None
        ),
        "requires_scientific_review": True,
    }
    atomic_json(output / "fixed_state_gate_summary.json", gate)
    atomic_json(output / "analysis_status.json", {"state": "completed"})


def complete(queue, root, output):
    queue, root, output = Path(queue), Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    site = read_json(queue / "frozen_site.json")
    inventory = read_json(root / "inventory.json")
    state_by_job = {x["job"]: x["state"] for x in inventory}
    for parity_job in ("parity_pcla_7", "parity_replay_random_rank_7"):
        require(state_by_job.get(parity_job) == "done",
                "parity runs are required before old F reuse")
        parity = read_json(root / parity_job / "result" / "source_parity.json")
        require(all(parity.values()), "source parity failed")

    torch, core, runtime, bridge = bind_project(site["repo"])
    endpoints, curves, source_cache = {}, [], {}
    for seed in SEEDS:
        for source in SOURCES:
            old = audit_old_fixed(site, torch, runtime, bridge, seed, source)
            source_cache[(seed, source)] = old
            endpoints[(seed, source, "fixed")] = endpoint(old["rows"])
            curves.extend({"seed": seed, "source": source, "rule": "fixed", **row}
                          for row in old["rows"])
            for rule in ("contributors", "function_norm_matched"):
                job = f"trace_{source}_{rule}_{seed}"
                if state_by_job.get(job) != "done":
                    continue
                audited = audit_new_run(site, torch, runtime, root, job, old)
                endpoints[(seed, source, rule)] = endpoint(audited["rows"])
                curves.extend({"seed": seed, "source": source, "rule": rule, **row}
                              for row in audited["rows"])

    available = set(
        seed for seed in SEEDS
        if all((seed, source, rule) in endpoints
               for source in SOURCES
               for rule in ("fixed", "contributors", "function_norm_matched"))
    )
    require(available in ({7, 13}, set(SEEDS)),
            f"unexpected partial seed set: {sorted(available)}")

    online = {}
    for seed in available:
        for kind in ("pcla", "replay"):
            job = f"online_{kind}_contributors_{seed}"
            require(state_by_job.get(job) == "done",
                    f"online pair is incomplete: {job}")
            audited = audit_new_run(site, torch, runtime, root, job,
                                    source_cache[(seed, "pcla")])
            online[(seed, kind)] = endpoint(audited["rows"])

    records = []
    for (seed, source, rule), value in sorted(endpoints.items()):
        records.append({"seed": seed, "family": "fixed_support",
                        "source": source, "rule": rule, **value})
    for (seed, kind), value in sorted(online.items()):
        records.append({"seed": seed, "family": "online",
                        "source": kind, "rule": "contributors", **value})
    write_csv(output / "per_seed_endpoints.csv", records)
    write_csv(output / "full_curves.csv", curves)

    definitions = []
    for source in SOURCES:
        for left, right in (("contributors", "fixed"),
                            ("function_norm_matched", "fixed"),
                            ("contributors", "function_norm_matched")):
            definitions.append((
                f"{source}: {left} - {right}",
                {seed: 100 * (endpoints[(seed, source, left)]["accuracy"] -
                              endpoints[(seed, source, right)]["accuracy"])
                 for seed in available},
            ))
    for rule in ("fixed", "contributors", "function_norm_matched"):
        for other in ("random", "replay_random_rank"):
            definitions.append((
                f"{rule}: pcla_support - {other}_support",
                {seed: 100 * (endpoints[(seed, "pcla", rule)]["accuracy"] -
                              endpoints[(seed, other, rule)]["accuracy"])
                 for seed in available},
            ))
    definitions.append((
        "online contributors: pcla - replay_random_rank",
        {seed: 100 * (online[(seed, "pcla")]["accuracy"] -
                      online[(seed, "replay")]["accuracy"])
         for seed in available},
    ))
    definitions.append((
        "selection-by-aggregation: (PCLA_C-Random_C)-(PCLA_F-Random_F)",
        {seed: 100 * (
            endpoints[(seed, "pcla", "contributors")]["accuracy"] -
            endpoints[(seed, "random", "contributors")]["accuracy"] -
            endpoints[(seed, "pcla", "fixed")]["accuracy"] +
            endpoints[(seed, "random", "fixed")]["accuracy"]
        ) for seed in available},
    ))

    groups = {
        "available": tuple(sorted(available)),
        "prior_lr_development": tuple(seed for seed in (7, 13)
                                      if seed in available),
        "not_used_for_prior_lr_ranking": tuple(seed for seed in (21, 42, 87)
                                                if seed in available),
    }
    contrast_records = {}
    for name, seeds in groups.items():
        if len(seeds) >= 2:
            contrast_records[name] = [
                paired(label, values, seeds) for label, values in definitions
                if all(seed in values for seed in seeds)
            ]
            atomic_json(output / f"contrasts_{name}.json",
                        contrast_records[name])
    gate = {
        "available_complete_seeds": sorted(available),
        "pilot_seeds": [7, 13],
        "pilot_contrasts": contrast_records.get("prior_lr_development", []),
        "project_materiality_guide": {
            "mean_accuracy_difference_pp": 1.0,
            "direction_consistency_full_matrix": "at least 4 of 5 seeds",
            "magnitude_explanation_fraction": 0.75,
            "residual_C_minus_G_accuracy_pp": 0.5,
            "status": "decision guide, not a universal statistical law"
        },
        "requires_scientific_review": True,
    }
    atomic_json(output / "trajectory_gate_summary.json", gate)
    atomic_json(output / "analysis_status.json", {
        "state": "completed", "independent_unit": "seed",
        "old_fixed_nll": "saved GPU loss",
        "new_nll": "float64 recomputed from saved logits",
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fixed-state", "complete"):
        command = sub.add_parser(name)
        command.add_argument("--queue", required=True)
        command.add_argument("--root", required=True)
        command.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "fixed-state":
        fixed_state(args.queue, args.root, args.out)
    else:
        complete(args.queue, args.root, args.out)


if __name__ == "__main__":
    main()
