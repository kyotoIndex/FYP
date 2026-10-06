#!/usr/bin/env python3
"""Compile all 128 DEV configs and 108 unresolved V slots; no dispatch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcla_research_v8.configuration import compile_development, rows


def write_once(path, value):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"immutable config differs: {path}")
    else:
        path.write_text(payload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--loader-template", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    template = json.loads(a.loader_template.read_text())
    out = a.out or a.root / "configs" / "development01"
    jobs = compile_development(
        common_path=a.package / "configs/common.json",
        recipes_path=a.package / "configs/recipes.json",
        jobs_path=a.package / "jobs/development.csv",
        authorization_path=a.root / "scope_authorization.json",
        data_dir=Path(template["manifest"]).parent,
        train_file=Path(template["train_file"]),
        token_cache=Path(template["token_cache"]),
        token_cache_metadata=Path(template["token_cache_metadata"]),
        model_path=Path(template["model"]["repository"]),
        source_manifest=a.source_root / "SOURCE_MANIFEST.json",
        output_root=a.root / "results",
    )
    # The old encoded-wire manifest and old model/cache are shared immutable inputs.
    for config in jobs:
        config["b8_wire"] = template["b8_wire"]
        config["runtime"]["capacity_by_tier"] = template["runtime"]["capacity_by_tier"]
        config["runtime"]["probe_batch_size"] = template["runtime"]["probe_batch_size"]
        config["runtime"]["probe_batches"] = template["runtime"]["probe_batches"]
        config["binding"]["b8_wire_sha256"] = None
        from pcla_research_v8.configuration import sha, digest
        b8 = Path(template["b8_wire"])
        if b8.is_file():
            if json.loads(b8.read_text()).get("total_bytes") != config["runtime"]["byte_cap"]:
                raise ValueError("template wire differs from B8")
            config["binding"]["b8_wire_sha256"] = sha(b8)
        if config["binding"]["data_manifest_sha256"] and config["binding"]["b8_wire_sha256"] and config["binding"]["token_cache_sha256"] and config["binding"]["token_cache_metadata_sha256"] and Path(config["model"]["repository"]).is_dir():
            config["binding"]["status"] = "bound_unverified"
        else:
            config["binding"]["status"] = "blocked_missing_data_binding"
        config.pop("expanded_sha256")
        config["expanded_sha256"] = digest(config)
        write_once(out / (config["job"]["job_id"] + ".json"), config)
    slots = rows(a.package / "jobs/validation_slots.csv")
    if len(slots) != 108 or any(s["recipe_id"] != "RESOLVE_AFTER_COMPLETE_DEV" for s in slots):
        raise ValueError("V slot matrix was preselected or incomplete")
    for slot in slots:
        item = dict(slot, hardware_family="RTX4090D", state="blocked_unresolved_dev_selection")
        write_once((a.root / "configs" / "validation_slots01" / (slot["job_id"] + ".json")), item)
    print(json.dumps({"DEV_configs": len(jobs), "bound_unverified": sum(j["binding"]["status"] == "bound_unverified" for j in jobs), "V_slots_unresolved": len(slots)}))


if __name__ == "__main__":
    main()
