"""Compile the thirteen fixed v7 validation slots without releasing them.

The D0 compiler remains frozen. R-local, matched R-cyclic and quota replay
are dependency records until a new compliant cost result chooses D or T.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .configuration import (JOB_KEYS, compile_recipe, expand_for_native_runner,
                            expand_for_runner, load_common)
from .data import PROTOCOL, ROLE_SIZES, digest

FIXED_SLOTS = {
    1: "D002", 2: "D006", 5: "D045", 6: "D054", 7: "D063",
    8: "D066", 9: "D069", 10: "D072", 11: "D078", 12: "D086",
    13: "D105", 14: "D107", 15: "D112",
}
SLOT_NAMES = {
    1: "N_local", 2: "N_cyclic", 5: "JL", 6: "JQ", 7: "full_r4",
    8: "full_r8", 9: "full_r16", 10: "full_alt", 11: "action",
    12: "layer_energy", 13: "fedsb", 14: "fedsvd", 15: "fedplora",
}
V_SEEDS = (101, 109, 127, 149, 163, 181, 199, 223, 241, 263, 281, 307)
DEPENDENT_SLOTS = {
    3: {"state": "unresolved_cost_tie", "choices": ["D011", "D035"]},
    4: {"state": "depends_on_slot_03", "choices_by_slot_03": {"D011": "D015", "D035": "D042"}},
    16: {"state": "depends_on_same_seed_slot_03_donor", "recipe": None},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_attachment_plan(attachment: Path, original_common: dict, recipes: dict,
                         seeds: list[int], manifest_sha256: str, b8_bytes: int) -> tuple[dict, dict]:
    """Check the Pro plan exactly, then bind its intentionally absent local manifest hash."""
    common_path = attachment / "configs/common_resolved.json"
    recipe_path = attachment / "configs/recipe_catalogue.json"
    jobs_path = attachment / "jobs/resolved_without_cost/V_JOBS.jsonl"
    common = json.loads(common_path.read_text())
    expected_common = json.loads(json.dumps(original_common))
    expected_common["status"] = "V_plan_pending_executor_compiler_and_path_checks"
    expected_common["budget"]["byte_cap"] = b8_bytes
    expected_common["budget"]["byte_cap_status"] = "FROZEN_BEFORE_D0_RESULTS_REUSE_EXISTING_WIRE_IDENTITY"
    expected_common["data_roles"]["manifest_status"] = "REUSE_EXISTING_FROZEN_17_SEED_MANIFEST_BIND_EXACT_SHA_AT_COMPILE"
    if common != expected_common or common["data_roles"]["manifest_sha256"] is not None:
        raise ValueError("attachment common differs from frozen protocol or resolved B8")
    catalog = json.loads(recipe_path.read_text())
    if catalog != [{k: v for k, v in recipe.items() if k != "recipe_sha256"}
                   for recipe in recipes.values()]:
        raise ValueError("attachment recipe catalogue differs from frozen original")
    rows = [json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    if len(rows) != 156:
        raise ValueError("attachment fixed V plan requires 156 rows")
    allowed_fields = {"common", "d0_recipe_id", "depends_on", "execution_policy",
                      "hardware_block", "hardware_family", "job_id", "output_dir",
                      "phase", "plan_only", "priority", "protocol", "recipe",
                      "requires", "seed", "slot"}
    by_pair = {}
    for row in rows:
        if set(row) != allowed_fields:
            raise ValueError("attachment V plan field set differs")
        slot = next((number for number, name in SLOT_NAMES.items() if name == row["slot"]), None)
        seed = row["seed"]
        if (slot is None or type(seed) is not int or seed not in seeds or
                row["d0_recipe_id"] != FIXED_SLOTS[slot] or
                row["recipe"] != {k: v for k, v in recipes[FIXED_SLOTS[slot]].items()
                                  if k != "recipe_sha256"} or row["common"] != common or
                row["protocol"] != PROTOCOL or row["phase"] != "V" or
                row["plan_only"] is not True or row["depends_on"] != [] or
                row["execution_policy"] != row["recipe"]["method"] or
                row["job_id"] != f"V_{row['slot']}_s{seed}" or
                row["output_dir"] != f"runs/{PROTOCOL}/V/{row['slot']}_s{seed}" or
                row["hardware_block"] != f"V_s{seed}" or
                row["hardware_family"] != ("A100-PCIE-40GB" if seed in {281, 307} else "RTX4090D") or
                type(row["priority"]) is not int or
                row["priority"] != (10 if slot in {9, 10} else 20 if slot == 8 else 40) or
                not isinstance(row["requires"], list) or
                row["requires"][:4] != ["user_execution_scope", "slot_selection_sealed",
                                        "V_compiler_checked", "attempt_ledger_reconciled"] or
                row["requires"][4:] != [f"{row['recipe']['required_adapter']}_V_integration"]):
            raise ValueError(f"attachment V plan row differs: {row.get('job_id')}")
        pair = (slot, seed)
        if pair in by_pair:
            raise ValueError("duplicate attachment V slot/seed")
        by_pair[pair] = row
    if set(by_pair) != {(slot, seed) for slot in FIXED_SLOTS for seed in seeds}:
        raise ValueError("attachment V slot/seed coverage differs")
    common["data_roles"]["manifest_sha256"] = manifest_sha256
    return common, {"rows": by_pair, "input_sha256": {str(p): sha256(p) for p in
            (common_path, recipe_path, jobs_path)}}


def load_inputs(handoff: Path, run_root: Path, source_train: Path) -> dict:
    common = load_common(handoff / "configs/common.yaml")
    seeds_doc = json.loads((handoff / "configs/seeds.json").read_text())
    seeds = seeds_doc["V"]
    if seeds != list(V_SEEDS) or any(type(seed) is not int for seed in seeds):
        raise ValueError("frozen V seed list differs")
    if set(seeds) & set(seeds_doc["D0"]) or len(seeds) != len(set(seeds)):
        raise ValueError("V/D0 seeds overlap or V seeds repeat")
    raw = json.loads((handoff / "configs/recipes.json").read_text())
    if len(raw) != 117 or len({r["recipe_id"] for r in raw}) != 117:
        raise ValueError("frozen recipe catalogue changed")
    recipes = {r["recipe_id"]: compile_recipe(r) for r in raw}
    manifest_path = run_root / "data_manifest/train_roles.json"
    b8_path = run_root / "data_manifest/b8_wire.json"
    manifest, b8 = json.loads(manifest_path.read_text()), json.loads(b8_path.read_text())
    if (manifest["protocol"] != PROTOCOL or
            manifest["split_sha256"] != digest({k: v for k, v in manifest.items() if k != "split_sha256"}) or
            manifest["source_sha256"] != sha256(source_train)):
        raise ValueError("validation data manifest/source identity differs")
    if any(str(seed) not in manifest["partitions"] for seed in seeds):
        raise ValueError("validation seed partition absent from data manifest")
    if any(len(manifest["roles"][role]) != count for role, count in ROLE_SIZES.items()):
        raise ValueError("training-only role sizes differ")
    for seed in seeds:
        part = manifest["partitions"][str(seed)]
        if (len(part["client_train"]) != 50 or len(part["client_local_readout"]) != 50 or
                any(len(rows) != 256 for rows in part["client_train"].values()) or
                any(len(rows) != 64 for rows in part["client_local_readout"].values())):
            raise ValueError(f"validation client partition size differs: {seed}")
    if b8["total_bytes"] != 2382653750 or b8["wire_manifest_sha256"] != digest({
            k: v for k, v in b8.items() if k != "wire_manifest_sha256"}):
        raise ValueError("B8 measured wire budget differs")
    if common["evaluation"]["D0_V"] != "recipe_dev":
        raise ValueError("validation evaluation role differs")
    return {"common": common, "recipes": recipes, "seeds": seeds, "manifest": manifest,
            "b8": b8, "manifest_path": manifest_path, "b8_path": b8_path,
            "seeds_path": handoff / "configs/seeds.json",
            "recipes_path": handoff / "configs/recipes.json",
            "common_path": handoff / "configs/common.yaml"}


def compile_validation_job(slot: int, seed: int, recipes: dict, allowed_seeds: list[int],
                           slot_map: dict[int, str] = FIXED_SLOTS,
                           plan_row: dict | None = None) -> dict:
    if type(slot) is not int or slot not in FIXED_SLOTS or slot_map != FIXED_SLOTS:
        raise ValueError("unresolved or altered validation slot")
    if allowed_seeds != list(V_SEEDS) or type(seed) is not int or seed not in V_SEEDS:
        raise ValueError("seed outside frozen V list")
    recipe = recipes[FIXED_SLOTS[slot]]
    rid = recipe["recipe_id"]
    job_id = f"V_S{slot:02d}_{rid}_s{seed}"
    job = {"job_id": job_id, "protocol": PROTOCOL, "phase": "V", "recipe_id": rid,
           "seed": seed, "group": recipe["group"], "method": recipe["method"],
           "representation": recipe["representation"], "lr": recipe["lr"],
           "period": recipe["period"], "rank": recipe["rank"], "xi": recipe["xi"],
           "min_rounds": recipe["min_rounds"], "max_rounds": recipe["max_rounds"],
           "budget_id": recipe["budget_id"], "eval_role": recipe["evaluation_role"],
           "hardware_block": f"V_s{seed}",
           "preferred_pool": "VALIDATION_ASSIGN_COMPLETE_SEED_BLOCK",
           "requires": "V_source_freeze;V_runtime_gate;V_hardware_assignment",
           "depends_on": "", "priority": 30, "status": "draft_not_dispatched",
           "output_dir": f"runs/pcla-global-budget-20260928-v7/V/{job_id}"}
    if plan_row is not None:
        if (plan_row["slot"] != SLOT_NAMES[slot] or plan_row["seed"] != seed or
                plan_row["d0_recipe_id"] != rid):
            raise ValueError("attachment plan job identity differs")
        job.update({"hardware_block": plan_row["hardware_block"],
                    "requires": ";".join(plan_row["requires"]),
                    "depends_on": ";".join(plan_row["depends_on"]),
                    "priority": plan_row["priority"]})
    if set(job) != JOB_KEYS or job["output_dir"].split("/")[-1] != job_id:
        raise ValueError("V job schema/output identity differs")
    return {"job": job, "recipe": recipe,
            "config_sha256": digest({"job": job, "recipe": recipe})}


def validate_matrix(jobs: list[dict], seeds: list[int]) -> None:
    if len(jobs) != 156:
        raise ValueError("fixed V matrix requires 156 jobs")
    ids = [x["job"]["job_id"] for x in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate V job ID")
    observed = Counter()
    for entry in jobs:
        job = entry["job"]
        match = re.fullmatch(r"V_S(\d{2})_(D\d{3})_s(\d+)", job["job_id"])
        if match is None:
            raise ValueError("invalid V job ID")
        slot, rid, seed_from_id = int(match[1]), match[2], int(match[3])
        if (slot not in FIXED_SLOTS or rid != FIXED_SLOTS[slot] or
                rid != job["recipe_id"] or seed_from_id != job["seed"] or
                job["output_dir"].split("/")[-1] != job["job_id"]):
            raise ValueError("V job/slot/seed/output identity differs")
        observed[(slot, job["seed"])] += 1
    if observed != Counter({(slot, seed): 1 for slot in FIXED_SLOTS for seed in seeds}):
        raise ValueError("fixed V slot/seed coverage differs")
    if any(x["job"]["phase"] != "V" or x["job"]["status"] != "draft_not_dispatched"
           or "/D0/" in x["job"]["output_dir"] for x in jobs):
        raise ValueError("V matrix contains executable or D0 job")


def reference_paths(run_root: Path, audited_run: dict, required_adapter: str) -> dict:
    """Use the audited D0 config and executed source, which can differ in vintage."""
    recipe_id = audited_run["job"]["recipe_id"]
    if audited_run["job"]["seed"] != 401:
        raise ValueError("V reference must be D0 seed401")
    remote_config = Path(audited_run["config"])
    if remote_config.name != f"{recipe_id}_s401.json" or remote_config.parent.parent.name != "configs":
        raise ValueError("audited D0 config path differs")
    path = run_root / "configs" / remote_config.parent.name / remote_config.name
    if sha256(path) != audited_run["config_sha256"]:
        raise ValueError("local D0 config differs from audited config SHA")
    reference = json.loads(path.read_text())
    if reference["job"]["recipe_id"] != recipe_id or reference["job"]["seed"] != 401:
        raise ValueError("D0 reference configuration identity differs")
    if reference["recipe"]["required_adapter"] != required_adapter:
        raise ValueError("D0 reference native adapter differs")
    remote_source = Path(audited_run["source_manifest"])
    if remote_source.name != "SOURCE_MANIFEST.json" or remote_source.parent.parent.name != "sources":
        raise ValueError("audited executed source path differs")
    source_copy = run_root / "execution_sources" / remote_source.parent.name / remote_source.name
    if sha256(source_copy) != audited_run["source_manifest_sha256"]:
        raise ValueError("local executed source manifest differs from audit")
    return {"path": path, "config": reference,
            "audited_remote_config": str(remote_config),
            "executed_source": str(remote_source),
            "executed_source_local_copy": str(source_copy),
            "executed_source_manifest_sha256": audited_run["source_manifest_sha256"]}


def expand_validation_job(compiled: dict, common: dict, reference: dict,
                          b8_bytes: int) -> dict:
    source = reference["config"]
    if {k: source[k] for k in ("manifest", "train_file", "token_cache", "b8_wire", "model")} != {
            k: reference["identity"][k] for k in ("manifest", "train_file", "token_cache", "b8_wire", "model")}:
        raise ValueError("D0 data/model paths differ within fixed V slots")
    if source["recipe"] != compiled["recipe"]:
        raise ValueError("D0 recipe differs from canonical catalogue")
    expand = (expand_for_native_runner if compiled["recipe"]["required_adapter"] != "core"
              else expand_for_runner)
    result = expand(compiled, common, manifest_path=Path(source["manifest"]),
                    train_file=Path(source["train_file"]),
                    token_cache=Path(source["token_cache"]),
                    model_path=Path(source["model"]["repository"]), b8_bytes=b8_bytes)
    if result["model"] != source["model"] or result["runtime"]["byte_cap"] != b8_bytes:
        raise ValueError("V expanded model/B8 differs from D0 identity")
    return result


def build_plan(handoff: Path, run_root: Path, source_train: Path,
               attachment: Path | None = None) -> tuple[list[dict], dict]:
    inputs = load_inputs(handoff, run_root, source_train)
    attached = None
    if attachment is not None:
        inputs["common"], attached = load_attachment_plan(
            attachment, inputs["common"], inputs["recipes"], inputs["seeds"],
            sha256(inputs["manifest_path"]), inputs["b8"]["total_bytes"])
    jobs = [compile_validation_job(slot, seed, inputs["recipes"], inputs["seeds"],
                                   plan_row=attached["rows"][(slot, seed)] if attached else None)
            for slot in sorted(FIXED_SLOTS) for seed in inputs["seeds"]]
    validate_matrix(jobs, inputs["seeds"])
    summary_dir = run_root / "checks/summaries/full_20260928_01"
    hashes = json.loads((summary_dir / "output_hashes.json").read_text())
    if sha256(summary_dir / "runs.json") != hashes["runs.json"]:
        raise ValueError("full D0 run summary hash differs")
    audited_runs = json.loads((summary_dir / "runs.json").read_text())
    if len(audited_runs) != 351:
        raise ValueError("full D0 audited run count differs")
    by_job = {x["job"]["job_id"]: x for x in audited_runs}
    if len(by_job) != 351:
        raise ValueError("duplicate audited D0 run")
    references = {rid: reference_paths(run_root, by_job[f"{rid}_s401"],
                                       inputs["recipes"][rid]["required_adapter"])
                  for rid in set(FIXED_SLOTS.values())}
    d107_source = references["D107"]["executed_source_local_copy"]
    if Path(d107_source).parent.name != "native04":
        raise ValueError("D107 audited source is not repaired native04")
    audited_fedsvd = Path(d107_source).parent / "pcla_research_v7/native_fedsvd.py"
    current_fedsvd = Path(__file__).parent / "native_fedsvd.py"
    if sha256(audited_fedsvd) != sha256(current_fedsvd):
        raise ValueError("current FedSVD implementation differs from audited native04 repair")
    identities = [{k: v["config"][k] for k in ("manifest", "train_file", "token_cache", "b8_wire", "model")}
                  for v in references.values()]
    if any(x != identities[0] for x in identities):
        raise ValueError("D0 data/model paths differ across selected recipes")
    for value in references.values():
        value["identity"] = identities[0]
    expanded = [expand_validation_job(x, inputs["common"],
                                      references[x["recipe"]["recipe_id"]],
                                      inputs["b8"]["total_bytes"]) for x in jobs]
    if attached:
        for config in expanded:
            job = config["job"]
            slot = next(s for s, rid in FIXED_SLOTS.items() if rid == job["recipe_id"])
            plan_row = attached["rows"][(slot, job["seed"])]
            config["validation_plan"] = {
                k: v for k, v in plan_row.items() if k not in {"recipe", "common"}}
            config["expanded_sha256"] = digest({k: v for k, v in config.items()
                                                if k != "expanded_sha256"})
    for x in expanded:
        if x["job"]["phase"] != "V" or x["job"]["seed"] not in inputs["seeds"]:
            raise ValueError("expanded V job phase/seed changed")
    receipt = {"phase": "V", "status": "draft_not_dispatched", "fixed_slot_count": 13,
               "fixed_job_count": 156, "seeds": inputs["seeds"],
               "fixed_slots": {str(k): v for k, v in FIXED_SLOTS.items()},
               "dependent_slots": {str(k): v for k, v in DEPENDENT_SLOTS.items()},
               "source_identity": {"train_sha256": inputs["manifest"]["source_sha256"],
                                   "split_sha256": inputs["manifest"]["split_sha256"],
                                   "local_manifest_sha256": sha256(inputs["manifest_path"]),
                                   "local_b8_sha256": sha256(inputs["b8_path"]),
                                   "b8_bytes": inputs["b8"]["total_bytes"],
                                   "D0_reference_paths": identities[0]},
               "input_sha256": {str(p): sha256(p) for p in
                                (inputs["seeds_path"], inputs["recipes_path"], inputs["common_path"])},
               "D0_reference_config_sha256": {rid: sha256(v["path"]) for rid, v in sorted(references.items())},
               "D0_audited_execution": {
                   rid: {"remote_config": v["audited_remote_config"],
                         "executed_source_manifest": v["executed_source"],
                         "executed_source_manifest_sha256": v["executed_source_manifest_sha256"],
                         "local_source_copy": v["executed_source_local_copy"]}
                   for rid, v in sorted(references.items())},
               "D107_native04_fedsvd_sha256": sha256(audited_fedsvd),
               "limitations": ["R slots 3/4/16 remain unresolved until compliant cost replay and donor"]}
    if attached:
        receipt["attachment_input_sha256"] = attached["input_sha256"]
        receipt["attachment_plan_job_id_mapping"] = {
            config["job"]["job_id"]: config["validation_plan"]["job_id"] for config in expanded}
        receipt["source_identity"]["bound_manifest_sha256"] = sha256(inputs["manifest_path"])
    return expanded, receipt
