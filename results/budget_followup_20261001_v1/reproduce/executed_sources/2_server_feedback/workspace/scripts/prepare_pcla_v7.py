#!/usr/bin/env python3
"""Prepare the frozen SST-2 train-only v7 roles and B8 wire budget.

Example on HPC:
  python scripts/prepare_pcla_v7.py --train data/SST-2/train.tsv \
      --output data/pcla_v7 --seeds 401 409 419
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcla_research_v7.data import make_manifest, digest, build_token_cache, FIRST_BATCH_SEEDS
from pcla_research_v7.transport import b8_wire_manifest


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FIRST_BATCH_SEEDS))
    parser.add_argument("--tokenizer", type=Path, help="Existing local BERT tokenizer directory")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    manifest = make_manifest(args.train, args.seeds)
    b8 = b8_wire_manifest()
    b8["wire_manifest_sha256"] = digest(b8)
    atomic_json(args.output / "train_roles.json", manifest)
    atomic_json(args.output / "b8_wire.json", b8)
    cache = None
    if args.tokenizer is not None:
        cache = build_token_cache(args.train, manifest, args.tokenizer,
                                  args.output / "train_tokens.npz", args.max_length)
    print(json.dumps({"train_roles": str(args.output / "train_roles.json"),
                      "split_sha256": manifest["split_sha256"],
                      "b8_bytes": b8["total_bytes"],
                      "b8_wire": str(args.output / "b8_wire.json"),
                      "token_cache": str(args.output / "train_tokens.npz") if cache else None,
                      "token_cache_sha256": cache["cache_sha256"] if cache else None,
                      "seeds": args.seeds}, sort_keys=True))


if __name__ == "__main__":
    main()
