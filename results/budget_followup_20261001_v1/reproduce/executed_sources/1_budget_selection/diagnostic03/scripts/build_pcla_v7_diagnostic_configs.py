#!/usr/bin/env python3
"""Stage fixed V diagnostic configs; never initialize a queue or run a diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcla_research_v7.diagnostic_configuration import compile_diagnostics


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n")


def stage(csv_path: Path, run_root: Path, execution_root: Path, manifest: Path,
          hardware_map: Path, release_gate: Path, out: Path,
          *, anchor_root: Path | None = None) -> dict:
    if out.exists():
        raise ValueError("diagnostic stage output already exists")
    configs, receipt = compile_diagnostics(csv_path, run_root, execution_root,
        manifest, hardware_map, release_gate, anchor_root=anchor_root)
    out.mkdir(parents=True)
    config_dir = out / "diagnostic_configs"
    config_dir.mkdir()
    hashes = {}
    for config in configs:
        path = config_dir / f"{config['job_id']}.json"
        write(path, config)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt["config_sha256"] = hashes
    write(out / "compilation.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="local frozen V configs and data manifest root")
    parser.add_argument("--execution-root", type=Path, required=True,
                        help="physical HPC root recorded in V configs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hardware-map", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path,
                        help="optional local mirror of V parent output directories")
    args = parser.parse_args()
    receipt = stage(args.csv, args.run_root, args.execution_root, args.manifest,
                    args.hardware_map, args.release_gate, args.out,
                    anchor_root=args.anchor_root)
    print(json.dumps({"count": receipt["count"], "lines": receipt["lines"],
                      "anchor_states": receipt["anchor_states"], "output": str(args.out)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
