#!/usr/bin/env python3
"""Compile 120 diagnostic configs locally; no checkpoint read or dispatch."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcla_research_v8.diagnostic_configuration import compile_diagnostics, file_sha256


def write_once(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"existing immutable artifact differs: {path}")
    else:
        path.write_text(payload)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--input-index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--remote-output-root", required=True)
    p.add_argument("--model-repository", help="New verified on-node model staging path; optional until runtime integration")
    a = p.parse_args()
    configs = compile_diagnostics(
        common_path=a.package / "configs/common.json",
        jobs_path=a.package / "jobs/diagnostics.csv",
        input_index_path=a.input_index,
        authorization_path=a.run_root / "scope_authorization.json",
        output_root=a.remote_output_root,
        model_repository=a.model_repository,
    )
    checksums = {}
    for config in configs:
        path = a.out / "configs" / (config["job"]["job_id"] + ".json")
        write_once(path, config)
        checksums[config["job"]["job_id"]] = file_sha256(path)
    manifest = {
        "schema": "pcla-v8-diagnostic-config-manifest-v1",
        "status": "compiled_blocked_pending_runtime_integration_and_on_node_preflight",
        "counts": {"jobs": len(configs), "bank_readouts": 96, "bank_candidate_sets": 192,
                   "rank8_transfers": 24, "new_client_training_calls": 240,
                   "new_client_optimizer_steps": 3840},
        "hardware_family": "RTX4090D",
        "input_index_sha256": file_sha256(a.input_index),
        "pro_jobs_sha256": file_sha256(a.package / "jobs/diagnostics.csv"),
        "pro_common_sha256": file_sha256(a.package / "configs/common.json"),
        "authorization_sha256": file_sha256(a.run_root / "scope_authorization.json"),
        "remote_output_root": a.remote_output_root,
        "model_repository": a.model_repository,
        "config_sha256": checksums,
        "preflight": [
            "Rehash each old bank PT against its historical expected SHA on the compute node before use.",
            "Hash each D066 parent PT on the compute node before use and label it current, not historical.",
            "Resolve D066 frozen next-round participant IDs and verify old data-role manifest on node.",
            "Stage and verify the real model repository on the compute node.",
            "Audit the diagnostic runtime and independent public/readout data path before dispatch.",
        ],
    }
    write_once(a.out / "CONFIG_MANIFEST.json", manifest)
    print(json.dumps({"configs": len(configs), "manifest": str(a.out / "CONFIG_MANIFEST.json"),
                      "launch_status": manifest["status"]}))


if __name__ == "__main__":
    main()
