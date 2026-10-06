"""U005/U006 adapter to the audited v7 native Fed-SB training path.

The internal native runtime has a v7 protocol guard.  Its Fed-SB math, RNG,
wire and checkpoint path are reused unchanged; the external v8 identity is
bound in provenance and receipts.  No full_both client is created here.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

import torch

from pcla_repair.core import BudgetLoRALinear, tensor_digest
from pcla_repair.runtime import atomic_checkpoint, atomic_json
from pcla_research_v7.data import PROTOCOL as V7_PROTOCOL, TokenCache
from pcla_research_v7.native_fedsb import FedSBLinear, OFFICIAL_COMMIT
from pcla_research_v7.native_runtime import initialize_fedsb, native_modules, native_state, run_native
from scripts.run_pcla_v7 import prepare

V8_PROTOCOL = "pcla-realized-update-feedback-20260929-v8"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def native_contract(expanded: dict) -> tuple[dict, dict]:
    """Validate v8 Fed-SB identity and construct the v7 native call contract."""
    job, recipe, runtime = expanded["job"], expanded["recipe"], expanded["runtime"]
    recipe_digest = hashlib.sha256(json.dumps(
        {k: v for k, v in recipe.items() if k != "config_sha256"},
        sort_keys=True).encode()).hexdigest()
    if (expanded.get("schema") != "pcla-v8-expanded-1" or job.get("protocol") != V8_PROTOCOL
            or job.get("method") != "fedsb" or job.get("recipe_id") not in {"U005", "U006"}
            or recipe.get("recipe_id") != job["recipe_id"] or recipe.get("server_rule") != "fedsb_r32"
            or recipe.get("factor_schedule") != "R_only" or not recipe.get("public_training")
            or int(job.get("rank", 0)) != 32 or int(recipe.get("rank", 0)) != 32
            or float(recipe.get("client_lr", -1)) != float(job.get("lr", -2))
            or float(recipe.get("head_lr", -1)) != float(job.get("head_lr", -2))
            or float(recipe["head_lr"]) != float(recipe["client_lr"])
            or recipe.get("config_sha256") != recipe_digest
            or float(recipe["client_lr"]) != (0.0003 if job["recipe_id"] == "U005" else 0.001)):
        raise ValueError("U005/U006 requires native rank32 R-only Fed-SB and matched head LR")
    if (job.get("eval_role") != "recipe_dev" or int(job.get("min_rounds", 0)) != 100
            or int(job.get("max_rounds", 0)) != 200 or job.get("phase") not in {"DEV", "TECH"}
            or runtime.get("byte_cap") != 2382653750 or runtime.get("batch_size") != 16
            or runtime.get("clients_per_round") != 10 or runtime.get("weight_decay") != .01):
        raise ValueError("Fed-SB v8 runtime differs from frozen DEV/B8 protocol")
    internal_job = dict(job, protocol=V7_PROTOCOL, v8_protocol=V8_PROTOCOL)
    internal_recipe = {
        "method": "fedsb", "required_adapter": "fedsb", "representation": "N",
        "factor_schedule": "R_only", "rank": 32, "optimizer": "adamw",
        "head_optimizer": "adamw", "lr": float(recipe["client_lr"]),
        "head_lr": float(recipe["head_lr"]),
        "adapter_weight_decay": .01, "adapter_clip_norm": None,
        "period": 0, "uses_public_training": True,
        "recipe_id": recipe["recipe_id"], "v8_recipe_sha256": recipe["config_sha256"],
        "protocol": V8_PROTOCOL,
    }
    return internal_job, internal_recipe


def _install_fedsb(model, public, job, runtime, timing):
    targets = [(name, module) for name, module in model.named_modules()
               if name.endswith((".attention.self.query", ".attention.self.value"))
               and isinstance(module, BudgetLoRALinear)]
    if len(targets) != 24:
        raise ValueError("Fed-SB requires 24 temporary BERT Q/V LoRA projections")
    with timing.span("task_initialization") if timing is not None else nullcontext():
        for name, module in targets:
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, module.base)
        init = initialize_fedsb(model, public, 32, float(job["lr"]),
                                batch_size=int(runtime["batch_size"]), timing=timing)
    modules = native_modules(model, "fedsb")
    if len(modules) != 24 or not all(isinstance(m, FedSBLinear) for m in modules.values()):
        raise RuntimeError("native Fed-SB installation incomplete")
    if init["public_rows"] != 512 or init["full_weight_gradient_targets"] != 24:
        raise RuntimeError("Fed-SB public-gradient initialization incomplete")
    return init


def _save_mean_records(out: Path, rows: list[dict], timing) -> None:
    """Attach mean-controller semantics without altering native client math."""
    with timing.span("checkpoint_and_audit") if timing is not None else nullcontext():
        for row in rows:
            hashes = row.get("native_state_hashes")
            if not isinstance(hashes, dict) or not all(hashes.get(k) for k in
                    ("parent_hash", "proposal_hash", "accepted_hash", "state_role")):
                raise RuntimeError("v8 native round lacks committed state hashes")
            row["server_controller"] = {
                "rule": "fedsb_r32", "alpha": 1.0, "beta": 1.0,
                "selection_public_access": False, "initialization_public_role": "public_training512",
                "grid_public_NLL": [], "features_count": 0, "encoder_batches": 0,
                "head_evals": 0, "public_dataset_passes_for_selection": 0,
                "parent_hash": hashes["parent_hash"],
                "proposal_hash": hashes["proposal_hash"],
                "committed_state_hash": hashes["accepted_hash"],
                "state_role": hashes["state_role"],
            }
        latest = out / "latest.pt"
        if latest.exists():
            ck = torch.load(latest, map_location="cpu", weights_only=True)
            if ck.get("format") != "pcla-v7-native-1" or len(ck["metrics"]) != len(rows):
                raise RuntimeError("native checkpoint differs from returned rows")
            ck["metrics"] = rows
            atomic_checkpoint(latest, ck)
        atomic_json(out / "metrics.json", rows)
        atomic_json(out / "native_controller.json", [
            {"round": row["round"], **row["server_controller"]} for row in rows])


def execute_native_fedsb(expanded: dict, config_path: Path, out: Path, device: torch.device,
                         *, timing, stop_after: int | None = None, resume: bool = False) -> tuple[list[dict], dict]:
    """Run a v8 Fed-SB trajectory through the audited v7 native runtime."""
    job, recipe = native_contract(expanded)
    if resume != (out / "latest.pt").exists():
        raise ValueError("Fed-SB resume flag must match latest checkpoint")
    if stop_after is not None and (expanded["job"]["phase"] != "TECH" or stop_after < 1):
        raise ValueError("technical stop requires a TECH identity")
    # prepare() deliberately constructs rank16 temporary LoRA.  Native Fed-SB
    # discards it before taking full-weight public gradients, exactly as D105.
    generic = dict(expanded, job=dict(job, rank=16))
    model, clients, capacities, evaluation, provenance = prepare(generic, config_path, device, timing=timing)
    manifest_path = _path(expanded["manifest"], config_path)
    manifest = json.loads(manifest_path.read_text())
    tokens = TokenCache(_path(expanded["token_cache"], config_path),
                        _path(expanded["token_cache_metadata"], config_path))
    public = tokens.dataset(manifest, "public_training", "algorithm")
    if len(public) != 512 or public.role != "public_training" or public.actor != "algorithm":
        raise ValueError("Fed-SB public initialization role differs")
    init = _install_fedsb(model, public, job, expanded["runtime"], timing)
    init_metadata = {k: v for k, v in init.items() if k not in {"cost", "projections"}}
    out.mkdir(parents=True, exist_ok=True)
    with timing.span("checkpoint_and_audit") if timing is not None else nullcontext():
        init_file = out / "native_initialization.json"
        if resume:
            recorded = json.loads(init_file.read_text())
            if {k: v for k, v in recorded.items() if k not in {"cost", "projections"}} != init_metadata:
                raise ValueError("Fed-SB initialization changed on resume")
        else:
            atomic_json(init_file, init)
        attempts_file = out / "native_initialization_attempts.json"
        attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
        attempts.append({"attempt": len(attempts) + 1, "cost": init["cost"]})
        atomic_json(attempts_file, attempts)
    workspace = Path(__file__).resolve().parents[1]
    source_files = [workspace / item for item in (
        "pcla_research_v7/native_fedsb.py", "pcla_research_v7/native_runtime.py",
        "pcla_research_v8/native_fedsb_runtime.py", "scripts/run_pcla_v8_native.py")]
    with timing.span("provenance_and_hashing") if timing is not None else nullcontext():
        provenance["native_sources"] = {str(path.relative_to(workspace)): _sha(path) for path in source_files}
        provenance["v8_protocol"] = V8_PROTOCOL
        provenance["v8_job"] = expanded["job"]
        provenance["v8_recipe_sha256"] = recipe["v8_recipe_sha256"]
        provenance["official_source_commit"] = OFFICIAL_COMMIT
        provenance["native_method"] = "fedsb"
        provenance["native_initialization"] = init_metadata
        provenance["native_initial_state_hash"] = tensor_digest(native_state(model, "fedsb"))
        provenance["v8_source_manifest_sha256"] = _sha(workspace / "SOURCE_MANIFEST.json")
    rows = run_native(model, clients, evaluation, capacities, job, recipe,
                      expanded["runtime"], out, provenance,
                      stop_after=stop_after, timing=timing)
    _save_mean_records(out, rows, timing)
    with timing.span("checkpoint_and_audit") if timing is not None else nullcontext():
        atomic_json(out / "run_identity.json", {"job": expanded["job"],
            "native_internal_job": job, "recipe": expanded["recipe"],
            "native_internal_recipe": recipe, "provenance": provenance})
    return rows, provenance
