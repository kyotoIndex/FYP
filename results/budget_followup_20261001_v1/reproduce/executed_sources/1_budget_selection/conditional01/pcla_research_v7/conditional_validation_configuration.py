"""Stage the three cost-dependent V slots as a separate, blocked extension.

The Pro resolver owns recipe selection. This module verifies its resolution and
the accepted D0 cost identities, then compiles 36 draft configurations. It does
not change the fixed 156-job manifest, queue, or runtime replay semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .configuration import JOB_KEYS
from .data import PROTOCOL, digest
from .validation_configuration import (
    FIXED_SLOTS, V_SEEDS, expand_validation_job, load_attachment_plan,
    load_inputs, reference_paths, sha256,
)


CONDITIONAL_SLOTS = {3: "R_local", 4: "R_cyclic", 16: "R_replay"}
BRANCHES = {"D011": "D015", "D035": "D042"}
RECEIPT_SCHEMA = "pcla-v7-required-time-v1"
ACCEPTANCE_FLAGS = ("parity_passed", "interval_coverage_passed",
                    "exclusive_gpu", "original_quality_retained")
PLAN_KEYS = {"job_id", "protocol", "phase", "slot", "seed", "d0_recipe_id",
             "recipe", "common", "execution_policy", "hardware_block",
             "hardware_family", "depends_on", "priority", "requires",
             "output_dir", "plan_only"}


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify_cost_resolution(selection: Mapping[str, Any], receipts: Mapping[str, Any],
                           expected: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a Pro-produced R-local decision; never derive a recipe for output."""
    slots = selection.get("slots", {})
    local = slots.get("R_local", {})
    chosen = local.get("recipe_id")
    if (selection.get("protocol") != PROTOCOL or selection.get("selection_complete") is not True or
            selection.get("seeds") != list(V_SEEDS) or selection.get("job_count") != 192 or
            selection.get("aliases") != {} or chosen not in BRANCHES or
            local.get("status") != "cost_resolved" or
            local.get("eligible") != sorted(BRANCHES) or
            selection.get("selection_groups", {}).get("R_local") != local or
            slots.get("R_cyclic", {}).get("recipe_id") != BRANCHES[chosen] or
            slots.get("R_cyclic", {}).get("status") != "quality_resolved" or
            slots.get("R_cyclic", {}).get("eligible") != [BRANCHES[chosen]] or
            slots.get("R_replay", {}).get("recipe_id") != chosen or
            slots.get("R_replay", {}).get("donor_slot") != "R_local" or
            slots.get("R_replay", {}).get("execution_policy") != "quota_replay_v1" or
            slots.get("R_replay", {}).get("status") != "donor_resolved_integration_pending"):
        raise ValueError("Pro cost resolution is incomplete or branch-inconsistent")
    times = {}
    for rid in sorted(BRANCHES):
        receipt, identity = receipts.get(rid), expected.get(rid)
        if not isinstance(receipt, dict) or not isinstance(identity, dict):
            raise ValueError(f"accepted D0 cost receipt/identity missing: {rid}")
        if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("recipe_id") != rid or
                receipt.get("seed") != 401 or
                receipt.get("gpu_model") != "NVIDIA GeForce RTX 4090 D" or
                any(receipt.get(flag) is not True for flag in ACCEPTANCE_FLAGS) or
                any(receipt.get(field) != identity.get(field) for field in
                    ("budget_stop_round", "deployed_bytes", "source_manifest_sha256",
                     "config_sha256")) or
                any(not isinstance(receipt.get(field), str) or
                    re.fullmatch(r"[0-9a-f]{64}", receipt[field]) is None
                    for field in ("interval_log_sha256", "timing_implementation_sha256"))):
            raise ValueError(f"D0 cost receipt identity/acceptance differs: {rid}")
        seconds = receipt.get("required_seconds")
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or
                not math.isfinite(seconds) or seconds <= 0):
            raise ValueError(f"D0 cost receipt required_seconds invalid: {rid}")
        times[rid] = seconds
    if (local.get("required_seconds") != times or
            chosen != min(sorted(BRANCHES), key=lambda rid: (times[rid], rid))):
        raise ValueError("Pro selected R recipe differs from accepted cost receipts")
    return {"R_local_recipe_id": chosen,
            "R_cyclic_recipe_id": BRANCHES[chosen],
            "accepted_receipt_sha256": {rid: _sha_json(receipts[rid]) for rid in sorted(BRANCHES)},
            "expected_D0_identity": {rid: {key: expected[rid][key] for key in
                ("original_job_id", "seed", "budget_stop_round", "deployed_bytes",
                 "source_manifest_sha256", "config_sha256", "run_identity_sha256",
                 "audit_report_sha256")} for rid in sorted(BRANCHES)}}


def _validate_resolver_jobs(selection: Mapping[str, Any], rows: Sequence[dict],
                            recipes: Mapping[str, dict], common: dict,
                            chosen: str) -> dict[tuple[int, int], dict]:
    if len(rows) != 192 or len({row.get("job_id") for row in rows}) != 192:
        raise ValueError("Pro resolved V plan requires 192 unique jobs")
    known = {**FIXED_SLOTS, 3: chosen, 4: BRANCHES[chosen], 16: chosen}
    plan_common = json.loads(json.dumps(common))
    if not isinstance(plan_common["data_roles"]["manifest_sha256"], str):
        raise ValueError("local manifest must be bound before conditional compilation")
    plan_common["data_roles"]["manifest_sha256"] = None
    by_pair = {}
    for row in rows:
        slot = next((number for number, name in CONDITIONAL_SLOTS.items()
                     if row.get("slot") == name), None)
        if slot is None:
            continue
        seed, rid = row.get("seed"), known[slot]
        donor = f"V_R_local_s{seed}" if slot == 16 else None
        recipe = {key: value for key, value in recipes[rid].items()
                  if key != "recipe_sha256"}
        expected_requires = ["user_execution_scope", "slot_selection_sealed",
                             "V_compiler_checked", "attempt_ledger_reconciled",
                             "quota_replay_integration" if slot == 16 else "core_V_integration"]
        if (set(row) != PLAN_KEYS or type(seed) is not int or seed not in V_SEEDS or
                row.get("protocol") != PROTOCOL or row.get("phase") != "V" or
                row.get("d0_recipe_id") != rid or row.get("recipe") != recipe or
                row.get("common") != plan_common or row.get("plan_only") is not True or
                row.get("job_id") != f"V_{CONDITIONAL_SLOTS[slot]}_s{seed}" or
                row.get("output_dir") != f"runs/{PROTOCOL}/V/{CONDITIONAL_SLOTS[slot]}_s{seed}" or
                row.get("execution_policy") != ("quota_replay_v1" if slot == 16 else recipe["method"]) or
                row.get("hardware_block") != f"V_s{seed}" or
                row.get("hardware_family") != ("A100-PCIE-40GB" if seed in {281, 307} else "RTX4090D") or
                row.get("depends_on") != ([donor] if donor else []) or
                row.get("requires") != expected_requires or
                row.get("priority") != (40 if slot != 3 else 25)):
            raise ValueError(f"resolved conditional V row differs: {row.get('job_id')}")
        pair = (slot, seed)
        if pair in by_pair:
            raise ValueError("duplicate conditional slot/seed")
        by_pair[pair] = row
    if set(by_pair) != {(slot, seed) for slot in CONDITIONAL_SLOTS for seed in V_SEEDS}:
        raise ValueError("conditional V matrix is not three slots by twelve seeds")
    fixed = [row for row in rows if row.get("slot") not in CONDITIONAL_SLOTS.values()]
    if (len(fixed) != 156 or
            {row.get("job_id") for row in fixed} !=
            {f"V_{name}_s{seed}" for name in (
                "N_local", "N_cyclic", "JL", "JQ", "full_r4", "full_r8",
                "full_r16", "full_alt", "action", "layer_energy", "fedsb",
                "fedsvd", "fedplora") for seed in V_SEEDS}):
        raise ValueError("resolved plan changed fixed 156-job identity")
    return by_pair


def compile_conditional_matrix(*, selection: dict, plan_rows: Sequence[dict],
                               receipts: dict, expected_cost: dict,
                               recipes: dict[str, dict], common: dict,
                               references: dict[str, dict], b8_bytes: int) -> tuple[list[dict], dict]:
    """Produce 36 blocked drafts from a verified Pro resolution and D0 source."""
    resolution = verify_cost_resolution(selection, receipts, expected_cost)
    chosen = resolution["R_local_recipe_id"]
    by_pair = _validate_resolver_jobs(selection, plan_rows, recipes, common, chosen)
    ids = {chosen, BRANCHES[chosen]}
    if set(references) != ids or any(references[rid]["config"]["recipe"] != recipes[rid]
                                     for rid in ids):
        raise ValueError("conditional D0 recipe source differs")
    identity_keys = ("manifest", "train_file", "token_cache", "b8_wire", "model")
    identity = {key: references[chosen]["config"][key] for key in identity_keys}
    if any({key: references[rid]["config"][key] for key in identity_keys} != identity
           for rid in ids):
        raise ValueError("conditional D0 data/model identity differs")
    if common["protocol"] != PROTOCOL or common["budget"]["byte_cap"] != b8_bytes:
        raise ValueError("conditional common/B8 identity differs")
    configs = []
    for slot in sorted(CONDITIONAL_SLOTS):
        rid = chosen if slot in {3, 16} else BRANCHES[chosen]
        recipe = recipes[rid]
        for seed in V_SEEDS:
            row = by_pair[(slot, seed)]
            job_id = f"V_S{slot:02d}_{rid}_s{seed}"
            donor_job_id = f"V_S03_{chosen}_s{seed}" if slot == 16 else None
            job = {"job_id": job_id, "protocol": PROTOCOL, "phase": "V",
                   "recipe_id": rid, "seed": seed, "group": recipe["group"],
                   "method": recipe["method"], "representation": recipe["representation"],
                   "lr": recipe["lr"], "period": recipe["period"],
                   "rank": recipe["rank"], "xi": recipe["xi"],
                   "min_rounds": recipe["min_rounds"], "max_rounds": recipe["max_rounds"],
                   "budget_id": recipe["budget_id"], "eval_role": recipe["evaluation_role"],
                   "hardware_block": row["hardware_block"],
                   "preferred_pool": "VALIDATION_ASSIGN_COMPLETE_SEED_BLOCK",
                   "requires": ";".join(row["requires"]),
                   "depends_on": donor_job_id or "",
                   "priority": row["priority"], "status": "draft_blocked_dependencies",
                   "output_dir": f"runs/{PROTOCOL}/V/{job_id}"}
            if set(job) != JOB_KEYS or job["output_dir"].split("/")[-1] != job_id:
                raise ValueError("conditional V job identity differs")
            compiled = {"job": job, "recipe": recipe,
                        "config_sha256": digest({"job": job, "recipe": recipe})}
            reference = {**references[rid], "identity": identity}
            expanded = expand_validation_job(compiled, common, reference, b8_bytes)
            expanded["validation_plan"] = {key: value for key, value in row.items()
                                           if key not in {"recipe", "common"}}
            expanded["conditional_validation"] = {
                "slot": slot, "slot_name": CONDITIONAL_SLOTS[slot],
                "execution_policy": row["execution_policy"],
                "release_state": "blocked_not_enqueued",
                "R_local_recipe_id": chosen,
                "cost_resolution_receipt_sha256": resolution["accepted_receipt_sha256"],
                "D0_reference_config_sha256": sha256(references[rid]["path"]),
                "D0_executed_source_manifest_sha256": references[rid]["executed_source_manifest_sha256"],
                "donor": ({"job_id": donor_job_id,
                           "plan_job_id": f"V_R_local_s{seed}",
                           "seed": seed, "recipe_id": chosen,
                           "required_artifacts": ["source_manifest", "config",
                                                  "initialization_identity", "support_trace",
                                                  "support_trace_sha256"]}
                          if slot == 16 else None),
                "unmet_dependencies": (["same_seed_R_local_completed_and_audited",
                                          "quota_replay_v1_runtime_integration",
                                          "quota_replay_D_and_T_short_run_parity"]
                                         if slot == 16 else ["separate_conditional_V_queue_integration"])}
            expanded["expanded_sha256"] = digest({key: value for key, value in expanded.items()
                                                   if key != "expanded_sha256"})
            configs.append(expanded)
    if (len(configs) != 36 or len({item["job"]["job_id"] for item in configs}) != 36 or
            {item["job"]["seed"] for item in configs} != set(V_SEEDS)):
        raise ValueError("conditional V output matrix differs")
    return configs, {"schema": "pcla-v7-conditional-validation-draft-1",
                     "status": "blocked_not_enqueued", "conditional_slots": CONDITIONAL_SLOTS,
                     "job_count": 36, "seeds": list(V_SEEDS), "resolution": resolution,
                     "integration_gaps": ["separate extension queue (fixed queue requires exactly 156)",
                                          "same-seed audited donor support trace before replay",
                                          "quota_replay_v1 D/T runtime and short-run parity"]}


def build_conditional_plan(*, handoff: Path, run_root: Path, source_train: Path,
                           attachment: Path, resolution_dir: Path,
                           cost_receipts_path: Path) -> tuple[list[dict], dict]:
    """Load frozen local inputs and Pro resolver output; return drafts in memory."""
    inputs = load_inputs(handoff, run_root, source_train)
    common, attached = load_attachment_plan(attachment, inputs["common"], inputs["recipes"],
        inputs["seeds"], sha256(inputs["manifest_path"]), inputs["b8"]["total_bytes"])
    selection = json.loads((resolution_dir / "V_SELECTION.json").read_text())
    rows = [json.loads(line) for line in (resolution_dir / "V_JOBS.jsonl").read_text().splitlines()
            if line.strip()]
    receipts = json.loads(cost_receipts_path.read_text())
    expected_path = attachment / "configs/expected_cost_identities.json"
    expected = json.loads(expected_path.read_text())
    chosen = verify_cost_resolution(selection, receipts, expected)["R_local_recipe_id"]
    summary = run_root / "checks/summaries/full_20260928_01"
    hashes = json.loads((summary / "output_hashes.json").read_text())
    if sha256(summary / "runs.json") != hashes["runs.json"]:
        raise ValueError("full D0 run summary hash differs")
    audited = json.loads((summary / "runs.json").read_text())
    by_id = {item["job"]["job_id"]: item for item in audited}
    if len(audited) != 351 or len(by_id) != 351:
        raise ValueError("full D0 audited runs differ")
    ids = {chosen, BRANCHES[chosen]}
    references = {rid: reference_paths(run_root, by_id[f"{rid}_s401"], "core") for rid in ids}
    configs, receipt = compile_conditional_matrix(selection=selection, plan_rows=rows,
        receipts=receipts, expected_cost=expected, recipes=inputs["recipes"], common=common,
        references=references, b8_bytes=inputs["b8"]["total_bytes"])
    receipt["input_sha256"] = {str(path): sha256(path) for path in (
        resolution_dir / "V_SELECTION.json", resolution_dir / "V_JOBS.jsonl",
        cost_receipts_path, expected_path, inputs["manifest_path"], inputs["b8_path"])}
    receipt["original_attachment_input_sha256"] = attached["input_sha256"]
    return configs, receipt


def stage_conditional_plan(*, handoff: Path, run_root: Path, source_train: Path,
                           attachment: Path, resolution_dir: Path,
                           cost_receipts_path: Path, fixed_compilation_path: Path,
                           output_dir: Path) -> dict:
    """Write an independent 36-config draft only after checking fixed V identity."""
    if output_dir.exists():
        raise ValueError("conditional draft output already exists")
    fixed = json.loads(fixed_compilation_path.read_text())
    fixed_dir = fixed_compilation_path.parent / "draft_configs"
    expected_hashes = fixed.get("draft_config_sha256", {})
    if (fixed.get("fixed_job_count") != 156 or fixed.get("seeds") != list(V_SEEDS) or
            len(expected_hashes) != 156 or
            {path.name for path in fixed_dir.glob("*.json")} != set(expected_hashes) or
            any(sha256(fixed_dir / name) != expected for name, expected in expected_hashes.items())):
        raise ValueError("fixed 156-job draft receipt or files differ")
    configs, receipt = build_conditional_plan(handoff=handoff, run_root=run_root,
        source_train=source_train, attachment=attachment, resolution_dir=resolution_dir,
        cost_receipts_path=cost_receipts_path)
    identity = fixed["source_identity"]
    expected_paths = identity["D0_reference_paths"]
    if (identity["train_sha256"] != sha256(source_train) or
            identity["local_manifest_sha256"] != sha256(run_root / "data_manifest/train_roles.json") or
            identity["local_b8_sha256"] != sha256(run_root / "data_manifest/b8_wire.json") or
            identity["b8_bytes"] != configs[0]["runtime"]["byte_cap"] or
            any({key: config[key] for key in ("manifest", "train_file", "token_cache",
                                                      "b8_wire", "model")} != expected_paths
                for config in configs)):
        raise ValueError("conditional draft differs from fixed V data/model/wire identity")
    if set(expected_hashes) & {config["job"]["job_id"] + ".json" for config in configs}:
        raise ValueError("conditional/fixed job ID collision")
    receipt["fixed_compilation_sha256"] = sha256(fixed_compilation_path)
    receipt["fixed_config_hashes_sha256"] = _sha_json(expected_hashes)
    output_dir.mkdir(parents=True)
    drafts = output_dir / "draft_configs"
    drafts.mkdir()
    for config in configs:
        path = drafts / f"{config['job']['job_id']}.json"
        path.write_text(json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "conditional_jobs.jsonl").write_text("".join(
        json.dumps(config["job"], sort_keys=True, ensure_ascii=False) + "\n"
        for config in configs))
    receipt["draft_config_sha256"] = {path.name: sha256(path)
                                      for path in sorted(drafts.glob("*.json"))}
    (output_dir / "compilation.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    return receipt
