from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import traceback

from common import (
    BATCH, DIAGNOSTIC_RULES, atomic_json, handoff_hashes, require, sha256_file,
)
from aggregation import accumulator_type, effective_distance, effective_inner
from pcla_runner import prepare, original_run, validate_trace


def cosine(torch, left, right, base, scaling, factor):
    left2 = effective_inner(torch, left, left, base, scaling, factor)
    right2 = effective_inner(torch, right, right, base, scaling, factor)
    cross = effective_inner(torch, left, right, base, scaling, factor)
    return cross / math.sqrt(left2 * right2) if left2 > 0 and right2 > 0 else None


def norm(torch, update, base, scaling, factor):
    return math.sqrt(max(0.0, effective_inner(
        torch, update, update, base, scaling, factor
    )))


def coverage_report(torch, client_rows, total_weight, shape):
    layers, rank = shape
    overall_positive = sum(x["positive_samples"] for x in client_rows)
    overall_examples = sum(x["sample_weight"] for x in client_rows)
    require(overall_examples == total_weight, "coverage total weight differs")
    result = []
    tiers = sorted({str(x["capacity"]) for x in client_rows})
    participant_sample_weights = {
        tier: sum(x["sample_weight"] for x in client_rows
                  if str(x["capacity"]) == tier)
        for tier in tiers
    }
    participant_client_counts = {
        tier: sum(1 for x in client_rows if str(x["capacity"]) == tier)
        for tier in tiers
    }
    participant_sample_shares = {
        tier: participant_sample_weights[tier] / overall_examples
        for tier in tiers
    }
    participant_client_shares = {
        tier: participant_client_counts[tier] / len(client_rows)
        for tier in tiers
    }
    for layer in range(layers):
        for component in range(rank):
            selected = [x for x in client_rows if x["support"][layer][component]]
            weight = sum(x["sample_weight"] for x in selected)
            positive = sum(x["positive_samples"] for x in selected)
            sample_tiers = {
                tier: sum(x["sample_weight"] for x in selected
                          if str(x["capacity"]) == tier)
                for tier in tiers
            }
            client_tiers = {
                tier: sum(1 for x in selected if str(x["capacity"]) == tier)
                for tier in tiers
            }
            sample_shares = (
                {tier: sample_tiers[tier] / weight for tier in tiers}
                if weight else None
            )
            client_shares = (
                {tier: client_tiers[tier] / len(selected) for tier in tiers}
                if selected else None
            )
            sample_tv = (
                0.5 * sum(abs(sample_shares[tier] -
                              participant_sample_shares[tier]) for tier in tiers)
                if sample_shares else None
            )
            client_tv = (
                0.5 * sum(abs(client_shares[tier] -
                              participant_client_shares[tier]) for tier in tiers)
                if client_shares else None
            )
            result.append({
                "layer": layer,
                "rank": component,
                "coverage_fraction": weight / total_weight,
                "contributor_samples": weight,
                "contributor_clients": len(selected),
                "contributor_positive_ratio": (positive / weight if weight else None),
                "participant_positive_ratio": overall_positive / overall_examples,
                "positive_ratio_shift": (
                    positive / weight - overall_positive / overall_examples
                    if weight else None
                ),
                "capacity_sample_weights": sample_tiers,
                "capacity_client_counts": client_tiers,
                "capacity_sample_shares": sample_shares,
                "capacity_client_shares": client_shares,
                "participant_capacity_sample_shares": participant_sample_shares,
                "participant_capacity_client_shares": participant_client_shares,
                "capacity_sample_total_variation": sample_tv,
                "capacity_client_total_variation": client_tv,
            })
    return result


def run_diagnostic(site, job, output):
    values = prepare(site, job["seed"], "pcla", "tuning")
    (torch, core, runtime, model, layout, clients, capacities, evaluation,
     provenance, partition, plan, pcla_source, pcla_dir, device,
     environment) = values
    replay_source, replay_dir = original_run(
        site, runtime, job["seed"], "replay_random_rank"
    )
    require(replay_source["identity_hash"] == pcla_source["identity_hash"],
            "original PCLA and replay identities differ")
    anchor_path = pcla_dir / f"probe_round_{job['anchor']:03d}.pt"
    anchor = runtime.load_checkpoint(anchor_path)
    require(anchor["round"] == job["anchor"] and
            anchor["identity_hash"] == pcla_source["identity_hash"],
            "anchor identity differs")
    cfg = copy.deepcopy(plan["training"])
    shape = (max(layout.values()) + 1,
             next(iter(core.lora_modules(model).values())).rank)
    scaling = {name: float(module.scaling)
               for name, module in core.lora_modules(model).items()}
    Accumulator = accumulator_type(runtime.DeltaAccumulator, torch, scaling)
    traces = {
        "pcla": validate_trace(torch, core, pcla_source, clients, capacities,
                               cfg, shape, job["seed"]),
        "replay_random_rank": validate_trace(
            torch, core, replay_source, clients, capacities,
            cfg, shape, job["seed"],
        ),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "identity.json", {
        "batch": BATCH,
        "job": job,
        "anchor_sha256": sha256_file(anchor_path),
        "pcla_checkpoint_sha256": sha256_file(pcla_dir / "checkpoint.pt"),
        "replay_checkpoint_sha256": sha256_file(replay_dir / "checkpoint.pt"),
        "source_identity": pcla_source["identity_hash"],
        "handoff_sources": handoff_hashes(),
        "evaluation_role": "train-derived tuning",
        "evaluation_source_ids": partition["tuning"],
        "provenance": provenance,
        "environment": environment,
    })
    parent = {name: value.clone() for name, value in anchor["adapter"].items()}
    rows = []
    client_calls = optimizer_steps = 0
    try:
        for round_id in (job["anchor"] + 1, job["anchor"] + 2):
            factor = "b" if round_id % 2 else "a"
            selected = random.Random(
                core.keyed_seed(job["seed"], "participants", round_id)
            ).sample(sorted(clients), cfg["selected_clients"])
            total_weight = sum(len(clients[c]) for c in selected)
            runtime.atomic_checkpoint(
                output / f"parent_{round_id:03d}.pt",
                {"adapter": parent, "round": round_id - 1,
                 "source_identity": pcla_source["identity_hash"]},
            )
            states, increments, readouts = {}, {}, {}
            for policy in ("pcla", "replay_random_rank", "full"):
                accumulator = Accumulator(parent, total_weight, "fixed")
                local_bank = []
                client_rows = []
                for client in selected:
                    support = (
                        torch.ones(shape, dtype=torch.bool)
                        if policy == "full"
                        else traces[policy][f"{round_id}/{client}"].clone()
                    )
                    core.load_adapter(model, parent)
                    telemetry = {}
                    masks, elapsed, _ = runtime.local_train(
                        model, layout, clients[client], support, factor,
                        core.keyed_seed(job["seed"], "train", round_id, client),
                        cfg, device, diagnostic=telemetry,
                    )
                    require(telemetry["optimizer_steps"] == 16,
                            "unexpected local optimizer-step count")
                    local = core.adapter_state(model)
                    accumulator.add(local, masks, len(clients[client]))
                    local_bank.append({
                        "client": client,
                        "weight": len(clients[client]),
                        "support": support.cpu(),
                        "masks": {name: mask.cpu() for name, mask in masks.items()},
                        "delta64": {
                            name: local[name].double() - parent[name].double()
                            for name in masks
                        },
                        "train_seconds": elapsed,
                    })
                    client_rows.append({
                        "client": client,
                        "sample_weight": len(clients[client]),
                        "positive_samples": int(
                            (clients[client].labels == 1).sum()
                        ),
                        "capacity": capacities[client],
                        "support": support.tolist(),
                    })
                    client_calls += 1
                    optimizer_steps += telemetry["optimizer_steps"]
                runtime.atomic_checkpoint(
                    output / f"local_{round_id:03d}_{policy}.pt",
                    {"round": round_id, "factor": factor,
                     "source_identity": pcla_source["identity_hash"],
                     "clients": local_bank},
                )
                readouts[policy] = accumulator.readout()
                atomic_json(output / f"coverage_{round_id:03d}_{policy}.json", {
                    "round": round_id,
                    "factor": factor,
                    "clients": client_rows,
                    "positions": coverage_report(
                        torch, client_rows, total_weight, shape
                    ),
                    "aggregation": readouts[policy],
                })
                applicable = ("fixed",) if policy == "full" else DIAGNOSTIC_RULES
                states[policy] = {
                    rule: accumulator.result_for(rule) for rule in applicable
                }
                increments[policy] = {
                    rule: {
                        name: state[name].double() - parent[name].double()
                        for name in parent
                    }
                    for rule, state in states[policy].items()
                }

            full = increments["full"]["fixed"]
            full_norm = norm(torch, full, parent, scaling, factor)
            for policy in ("pcla", "replay_random_rank", "full"):
                applicable = ("fixed",) if policy == "full" else DIAGNOSTIC_RULES
                for rule in applicable:
                    state = states[policy][rule]
                    core.load_adapter(model, state)
                    result = runtime.evaluate(
                        model, layout, evaluation, cfg["eval_batch_size"],
                        device, record_logits=True,
                    )
                    atomic_json(
                        output / f"evaluation_{round_id:03d}_{policy}_{rule}.json",
                        result,
                    )
                    delta = increments[policy][rule]
                    delta_norm = norm(torch, delta, parent, scaling, factor)
                    distance = effective_distance(
                        torch, delta, full, parent, scaling, factor
                    )
                    rows.append({
                        "seed": job["seed"],
                        "anchor": job["anchor"],
                        "round": round_id,
                        "factor": factor,
                        "support": policy,
                        "rule": rule,
                        "accuracy": result["accuracy"],
                        "nll": result["nll_recomputed_float64"],
                        "negative_recall": result["negative_recall"],
                        "positive_recall": result["positive_recall"],
                        "balanced_accuracy": result["balanced_accuracy"],
                        "effective_matrix_norm": delta_norm,
                        "effective_matrix_cosine_to_full": cosine(
                            torch, delta, full, parent, scaling, factor
                        ),
                        "effective_matrix_norm_ratio_to_full": (
                            delta_norm / full_norm if full_norm > 0 else None
                        ),
                        "effective_matrix_relative_error_to_full": (
                            distance / full_norm if full_norm > 0 else None
                        ),
                        "aggregation": readouts[policy],
                    })
            parent = {
                name: value.clone()
                for name, value in states["pcla"]["fixed"].items()
            }
            atomic_json(output / "summary.json", rows)
        require(client_calls == 60 and optimizer_steps == 960,
                "fixed-state work count differs")
        atomic_json(output / "result.json", {
            "state": "completed",
            "job_id": job["id"],
            "client_calls": client_calls,
            "optimizer_steps": optimizer_steps,
            "environment": environment,
        })
    except BaseException:
        (output / "exception.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        atomic_json(output / "result.json", {
            "state": "failed",
            "job_id": job["id"],
            "client_calls": client_calls,
            "optimizer_steps": optimizer_steps,
            "environment": environment,
        })
        raise
