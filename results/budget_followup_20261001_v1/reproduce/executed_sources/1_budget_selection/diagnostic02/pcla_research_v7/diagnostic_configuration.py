"""Compile the fixed 144 V diagnostic objects without dispatching training."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from pcla_research_v7.data import PROTOCOL

SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
PARENTS = {"full_alt": ("S10", "D072"), "full_r16": ("S09", "D069")}
ANCHORS = (10, 60)
FACTORS = ("a", "b")
DIAGNOSTIC_ID = re.compile(r"(Cohort|Direction)_s(\d+)(?:_(full_alt|full_r16))?_r(10|60)_([ab])")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def keyed_seed(*parts: object) -> int:
    """Exact pcla_repair.core.keyed_seed encoding, without importing GPU runtime."""
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def participants(seed: int, round_id: int, client_ids: list[int]) -> list[int]:
    return random.Random(keyed_seed(seed, "participants", round_id)).sample(sorted(client_ids), 10)


def verify_anchor(anchor_path: Path, *, seed: int, anchor: int, client_ids: list[int],
                  parent_config: Path, parent_config_sha256: str,
                  source_manifest_sha256: str) -> dict:
    """Read the atomically saved CPU checkpoint; never infer visits from live rows."""
    if not anchor_path.is_file():
        return {"state": "pending_parent_anchor", "missing": [str(anchor_path)]}
    import torch
    config = read(parent_config)
    if sha(parent_config) != parent_config_sha256 or config["job"]["seed"] != seed:
        raise ValueError("parent config identity differs")
    out = anchor_path.parent
    dispatch = read(out / "dispatch_identity.json")
    attempts = read(out / "execution_attempts.json")
    if (dispatch.get("config_sha256") != parent_config_sha256
            or dispatch.get("source_manifest_sha256") != source_manifest_sha256
            or not attempts):
        raise ValueError("anchor config/source dispatch binding differs")
    payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    identity = payload.get("identity")
    if (payload.get("format") != "pcla-v7-runtime-1" or payload.get("round") != anchor
            or not isinstance(identity, str) or not identity
            or any(attempt.get("identity") != identity for attempt in attempts)
            or not isinstance(payload.get("adapter"), dict) or not payload["adapter"]
            or not isinstance(payload.get("cache"), dict)
            or not isinstance(payload.get("torch_rng"), torch.Tensor)
            or not isinstance(payload.get("python_rng"), tuple)
            or payload.get("cuda_rng") is None):
        raise ValueError("anchor checkpoint identity, state, cache or RNG is incomplete")
    rows = payload.get("metrics")
    if not isinstance(rows, list) or len(rows) != anchor:
        raise ValueError("anchor metrics prefix length differs")
    for round_id, row in enumerate(rows, 1):
        if (row.get("round") != round_id
                or row.get("selected_clients") != participants(seed, round_id, client_ids)):
            raise ValueError(f"anchor participation prefix differs at round {round_id}")
    metrics_path = out / "metrics.json"
    if metrics_path.is_file() and read(metrics_path)[:anchor] != rows:
        raise ValueError("anchor and persisted metrics prefix differ")
    selected = participants(seed, anchor + 1, client_ids)
    visits = {str(cid): sum(cid in row["selected_clients"] for row in rows) for cid in selected}
    return {"state": "anchor_verified", "anchor_sha256": sha(anchor_path),
            "checkpoint_identity": identity, "participants": selected,
            "visits_from_anchor_metrics": visits,
            "metrics_rounds": anchor, "cache_entries": len(payload["cache"])}


def compile_diagnostics(csv_path: Path, run_root: Path, execution_root: Path,
                        manifest_path: Path, hardware_map_path: Path,
                        release_gate_path: Path, *, anchor_root: Path | None = None) -> tuple[list[dict], dict]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 144 or len({r["job_id"] for r in rows}) != 144:
        raise ValueError("expected 144 unique diagnostic objects")
    manifest, hardware_map, gate = read(manifest_path), read(hardware_map_path), read(release_gate_path)
    manifest_sha = sha(manifest_path)
    if (manifest.get("protocol") != PROTOCOL
            or not {str(s) for s in SEEDS} <= set(manifest.get("partitions", {}))):
        raise ValueError("frozen data manifest lacks V seed partitions")
    if any(len(manifest.get("roles", {}).get(role, [])) != size for role, size in
           (("public_training", 512), ("cohort_select", 1024), ("cohort_readout", 2048))):
        raise ValueError("frozen diagnostic role sizes differ")
    if (gate.get("schema") != "pcla-v7-release-gates-1" or gate.get("protocol") != PROTOCOL
            or gate.get("phase") != "V" or gate.get("review_gate") != "approved"
            or gate.get("configuration_gate") != "passed" or gate.get("source_gate") != "passed"):
        raise ValueError("V release gate identity differs")
    if set(hardware_map) != {f"V_s{s}" for s in SEEDS} or any(
            hardware_map[f"V_s{s}"] != ("A100" if s in (281, 307) else "4090D") for s in SEEDS):
        raise ValueError("V seed hardware assignment differs")
    expected = {(line, seed, parent, anchor, factor)
                for seed in SEEDS for anchor in ANCHORS for factor in FACTORS
                for line, parents in (("cohort", tuple(PARENTS)), ("direction", ("full_alt",)))
                for parent in parents}
    seen = set()
    configs = []
    anchor_cache = {}
    for row in rows:
        line, seed, anchor, factor = row["line"], int(row["seed"]), int(row["anchor"]), row["factor"]
        parent = row["parent"]
        parent_kind = parent.removeprefix("V_").removesuffix(f"_s{seed}")
        key = (line, seed, parent_kind, anchor, factor)
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate diagnostic object: {row['job_id']}")
        seen.add(key)
        expected_id = (f"Cohort_s{seed}_{parent_kind}_r{anchor}_{factor}" if line == "cohort"
                       else f"Direction_s{seed}_r{anchor}_{factor}")
        calls, steps, lr, eval_max = ((40, 640, .001, 80) if line == "cohort"
                                      else (120, 1920, .0003, 0))
        if (row["job_id"] != expected_id or row["protocol"] != PROTOCOL
                or row["clients"] != "10" or row["local_calls"] != str(calls)
                or row["optimizer_steps"] != str(steps)
                or row["selection_eval_max"] != str(eval_max)
                or row["eval_role"] != "cohort_readout"
                or row["participant_schedule_round"] != str(anchor + 1)
                or float(row["learning_rate"]) != lr or float(row["head_learning_rate"]) != lr
                or row["status"] != "parent_and_diagnostic_integration_pending"
                or row["requires"] != f"parent_anchor_atomic;{line}_integration"
                or row["hardware_block"] != expected_id):
            raise ValueError(f"diagnostic CSV contract differs: {row['job_id']}")
        slot, recipe = PARENTS[parent_kind]
        parent_id = f"V_{slot}_{recipe}_s{seed}"
        config_path = run_root / "configs/validation02" / f"{parent_id}.json"
        config = read(config_path)
        config_sha = sha(config_path)
        if (gate.get("config_sha256_by_job", {}).get(parent_id) != config_sha
                or config.get("job", {}).get("job_id") != parent_id
                or config["job"].get("phase") != "V" or config["job"].get("seed") != seed
                or config["job"].get("recipe_id") != recipe
                or config.get("source_manifest") != str(execution_root / "sources/validation02/SOURCE_MANIFEST.json")
                or config.get("common", {}).get("data_roles", {}).get("manifest_sha256") != manifest_sha):
            raise ValueError(f"frozen parent config binding differs: {parent_id}")
        part = manifest["partitions"][str(seed)]
        clients = sorted(int(cid) for cid in part["client_train"])
        if (clients != list(range(50)) or set(part["client_local_readout"]) != set(part["client_train"])
                or any(len(v) != 256 for v in part["client_train"].values())
                or any(len(v) != 64 for v in part["client_local_readout"].values())):
            raise ValueError(f"frozen V client partition differs: {seed}")
        selected = participants(seed, anchor + 1, clients)
        actual_parent_out = execution_root / "V" / parent_id
        local_anchor = (anchor_root or run_root / "V") / parent_id / f"anchor_round_{anchor:03d}.pt"
        if local_anchor not in anchor_cache:
            anchor_cache[local_anchor] = verify_anchor(local_anchor, seed=seed, anchor=anchor,
                client_ids=clients, parent_config=config_path, parent_config_sha256=config_sha,
                source_manifest_sha256=gate["source_manifest_sha256"])
        anchor_info = anchor_cache[local_anchor]
        if anchor_info["state"] == "anchor_verified" and anchor_info["participants"] != selected:
            raise ValueError("anchor and compiled participant schedule differ")
        common = {"optimizer": "adamw", "betas": [.9, .999], "eps": 1e-8,
                  "weight_decay": .01, "batch_size": 16, "local_examples": 256,
                  "local_optimizer_steps": 16, "dropout": .1,
                  "learning_rate": lr, "head_learning_rate": lr,
                  "prefix_steps": [1, 4, 16], "prefix_policy": "same_training_trajectory",
                  "paired_rng_key_scope": ["diagnostic_object_id", "client_id"]}
        item = {"schema": "pcla-v7-diagnostic-config-1", "protocol": PROTOCOL,
                "phase": "diagnostic", "status": "blocked_plan_only", "job_id": expected_id,
                "line": line, "seed": seed, "anchor_round": anchor, "factor": factor,
                "participant_schedule_round": anchor + 1, "participants": selected,
                "visits_source": "parent_anchor_checkpoint.metrics.rounds_1_through_anchor",
                "visits": anchor_info.get("visits_from_anchor_metrics"),
                "parent": {"job_id": parent_id, "recipe_id": recipe,
                           "config": str(execution_root / "configs/validation02" / f"{parent_id}.json"),
                           "config_sha256": config_sha,
                           "source_manifest_sha256": gate["source_manifest_sha256"],
                           "output": str(actual_parent_out),
                           "anchor": str(actual_parent_out / f"anchor_round_{anchor:03d}.pt"),
                           "anchor_check": anchor_info},
                "hardware_family": hardware_map[f"V_s{seed}"], "hardware_block": f"V_s{seed}",
                "output": str(execution_root / "diagnostics" / expected_id),
                "frozen_data_manifest_sha256": manifest_sha,
                "roles": {"client_train": "same_seed_client_train_256",
                          "client_local_readout": "same_seed_client_local_readout_64",
                          "public_direction": "public_training_512",
                          "bank_selection": "cohort_select_1024",
                          "independent_readout": "cohort_readout_2048"},
                "training": common,
                "calls": {"clients": 10, "local_calls": calls, "optimizer_steps": steps},
                "dependencies": {"parent_anchor": anchor_info["state"],
                                 "diagnostic_runner": "not_implemented"}}
        if line == "cohort":
            item["bank"] = {"supports": ["local", "cyclic", "JL_h32", "public512_linear"],
                            "JQ_xi": 1.0, "JQ_lock_before_training_outcomes": True,
                            "max_selection_evaluations": 80,
                            "selection_role": "cohort_select_1024",
                            "report_role": "cohort_readout_2048",
                            "exact_gradient_probe_examples": 32,
                            "aggregation_weight_per_client": .1}
        else:
            item["direction"] = {"representations": ["N", "D", "S", "T", "S_nonzero_nullspace_rotation", "S_zero_direction"],
                                 "support_policies": ["local", "cyclic"],
                                 "forced_coordinate_intervention": True,
                                 "selection_evaluations": 0,
                                 "norm_matched_readout": True}
        configs.append(item)
    if seen != expected:
        raise ValueError("diagnostic Cartesian matrix has a gap")
    receipt = {"schema": "pcla-v7-diagnostic-stage-1", "count": len(configs),
               "lines": dict(Counter(c["line"] for c in configs)),
               "csv_sha256": sha(csv_path), "manifest_sha256": manifest_sha,
               "hardware_map_sha256": sha(hardware_map_path),
               "release_gate_sha256": sha(release_gate_path),
               "parent_source_manifest_sha256": gate["source_manifest_sha256"],
               "anchor_states": dict(Counter(c["dependencies"]["parent_anchor"] for c in configs)),
               "status": "staged_not_dispatched"}
    return configs, receipt
