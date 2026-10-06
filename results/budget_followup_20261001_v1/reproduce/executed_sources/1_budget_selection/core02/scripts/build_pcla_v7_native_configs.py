#!/usr/bin/env python3
"""Expand exactly the 54 frozen native D0 jobs or 18 seed9001 technical jobs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcla_research_v7.configuration import compile_first_batch, load_common, expand_for_native_runner
from pcla_research_v7.data import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--technical", action="store_true")
    args = p.parse_args()
    handoff, data = args.root / "handoff", args.root / "data/pcla_v7"
    common = load_common(handoff / "configs/common.yaml")
    compiled = compile_first_batch(handoff / "configs/recipes.json", handoff / "jobs/first_batch.jsonl")
    b8 = json.loads((data / "b8_wire.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    ids = []
    for item in compiled:
        if item["recipe"]["required_adapter"] not in {"fedsb", "fedsvd", "fedplora"}:
            continue
        if args.technical and item["job"]["seed"] != 401:
            continue
        expanded = expand_for_native_runner(item, common,
            manifest_path=data / "train_roles.json", train_file=args.train,
            token_cache=data / "train_tokens.npz", model_path=args.model,
            b8_bytes=b8["total_bytes"])
        if args.technical:
            job = expanded["job"] = dict(expanded["job"])
            job.update(seed=9001, phase="TECH", job_id="TECH_" + item["job"]["recipe_id"] + "_s9001")
            expanded["technical_profile_only"] = True
        expanded.pop("expanded_sha256", None)
        expanded["expanded_sha256"] = digest(expanded)
        path = args.out / (expanded["job"]["job_id"] + ".json")
        payload = json.dumps(expanded, sort_keys=True, indent=2) + "\n"
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"existing native config differs: {path}")
        path.write_text(payload)
        ids.append(expanded["job"]["job_id"])
    if len(ids) != (18 if args.technical else 54):
        raise AssertionError(f"native matrix coverage {len(ids)}")
    (args.out / "compilation.json").write_text(json.dumps({"jobs": ids,
        "count": len(ids), "b8_bytes": b8["total_bytes"]}, indent=2) + "\n")
    print(json.dumps({"count": len(ids), "out": str(args.out)}))


if __name__ == "__main__":
    main()
