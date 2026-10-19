#!/usr/bin/env python3
"""Prepare the fixed models, MultiNLI partition and token caches used by AmpLoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen
import zipfile

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "federated_finetuning"
EXPERIMENT = CODE / "experiments/scra_diagnostics_20261017"
MODELS = {
    "roberta": ("FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b", "roberta-base"),
    "qwen": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987", "qwen2.5-0.5b"),
}
MNLI_URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
SOURCE_FILES = ("multinli_1.0_train.jsonl", "multinli_1.0_dev_matched.jsonl",
                "multinli_1.0_dev_mismatched.jsonl")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def download_models(root: Path, families: list[str]) -> None:
    from huggingface_hub import snapshot_download
    for family in families:
        repo_id, revision, dirname = MODELS[family]
        output = root / "models" / dirname
        print(f"Downloading {repo_id} at {revision} to {output}", flush=True)
        snapshot_download(repo_id, revision=revision, local_dir=output,
                          allow_patterns=["config.json", "generation_config.json", "*.safetensors",
                                          "*.safetensors.index.json", "tokenizer*", "vocab.json",
                                          "merges.txt", "special_tokens_map.json", "LICENSE*", "README.md"])
        (output / "REVISION.txt").write_text(revision + "\n")


def prepare_data(root: Path, source: Path | None) -> None:
    from fedft.data import prepare
    output = root / "data/mnli_r1_seed2026"
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if (manifest.get("seed"), manifest.get("scenario"), manifest.get("partition"),
                len(manifest.get("clients", []))) != (2026, "R1", "genre", 50):
            raise ValueError(f"An unrelated partition already exists at {output}")
        if not all((output / info["path"]).is_file() for info in manifest["files"].values()):
            raise FileNotFoundError(f"Partition files are missing under {output}")
        if (manifest["files"]["train"]["rows"], manifest["files"]["validation"]["rows"]) != (372557, 19681):
            raise ValueError(f"Existing partition row counts differ from the study: {output}")
        print(f"Using existing partition {output}", flush=True)
        return
    if source is None:
        source = root / "data/raw/multinli_1.0"
        if not all((source / name).exists() for name in SOURCE_FILES):
            source.mkdir(parents=True, exist_ok=True)
            archive = source.parent / "multinli_1.0.zip"
            print(f"Downloading official MultiNLI 1.0 (227 MB) from {MNLI_URL}", flush=True)
            partial = archive.with_suffix(".zip.part")
            try:
                with urlopen(MNLI_URL, timeout=120) as response, partial.open("wb") as dest:
                    shutil.copyfileobj(response, dest)
                partial.replace(archive)
                with zipfile.ZipFile(archive) as bundle:
                    # Extract just the three known JSONL files, without trusting ZIP paths.
                    for filename in SOURCE_FILES:
                        target = source / filename
                        temporary = target.with_suffix(".jsonl.part")
                        with bundle.open(f"multinli_1.0/{filename}") as src, temporary.open("wb") as dest:
                            shutil.copyfileobj(src, dest)
                        temporary.replace(target)
            finally:
                partial.unlink(missing_ok=True)
                archive.unlink(missing_ok=True)
    source = source.expanduser().resolve()
    for name in SOURCE_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    manifest = prepare("multinli", source, output, partition="genre", clients=50, alpha=0.3,
                       seed=2026, validation_fraction=0.05, min_samples=8, strong_fraction=0.4,
                       scenario="R1", cross_genre_policy="exclude")
    counts = {split: meta["rows"] for split, meta in manifest["files"].items()}
    print(json.dumps({"partition": str(output), "rows": counts}), flush=True)
    if (counts["train"], counts["validation"]) != (372557, 19681):
        raise ValueError(f"Partition counts differ from the recorded study: {counts}")


def prepare_roberta_cache(root: Path) -> None:
    import torch
    from transformers import AutoTokenizer
    from fedft.data import TokenizedRows, load_jsonl

    plan = json.loads((EXPERIMENT / "plans_batch/00_amp_s2040.json").read_text())
    output = root / plan["cache_directory"]
    data = root / "data/mnli_r1_seed2026"
    manifest = json.loads((data / "manifest.json").read_text())
    complete = output / "cache_complete.json"
    spec = {"model_revision": MODELS["roberta"][1], "max_length": 256,
            "data_seed": 2026, "train_rows": manifest["files"]["train"]["rows"],
            "validation_rows": manifest["files"]["validation"]["rows"]}
    if complete.exists():
        if json.loads(complete.read_text()).get("spec") != spec:
            raise ValueError(f"Existing RoBERTa cache has another specification: {output}")
        if all((output / f"{split}.pt").is_file() for split in ("train", "validation")):
            print(f"Using existing RoBERTa token cache {output}", flush=True)
            return
    tokenizer = AutoTokenizer.from_pretrained(root / "models/roberta-base", local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "completed", "spec": spec, "files": {}}
    task = {"name": "multinli", "kind": "classification", "num_labels": 3, "max_length": 256}
    for split in ("train", "validation"):
        rows = load_jsonl(data / manifest["files"][split]["path"])
        encoded = TokenizedRows(rows, tokenizer, task)
        tokens, offsets, labels = [], [0], []
        for item in encoded.items:
            tokens.extend(item["input_ids"])
            offsets.append(len(tokens))
            labels.append(item["labels"])
        value = {"tokens": torch.tensor(tokens, dtype=torch.int32),
                 "offsets": torch.tensor(offsets, dtype=torch.long),
                 "labels": torch.tensor(labels, dtype=torch.long), "pad_id": tokenizer.pad_token_id,
                 "rows": [{k: r[k] for k in ("domain", "client", "label")} for r in rows],
                 "truncated": encoded.truncated}
        dest = output / f"{split}.pt"
        temporary = dest.with_suffix(".pt.tmp")
        torch.save(value, temporary)
        temporary.replace(dest)
        report["files"][split] = {"rows": len(rows), "tokens": len(tokens), "truncated": encoded.truncated}
        print(json.dumps({"cache": split, **report["files"][split]}), flush=True)
        del rows, encoded, tokens, offsets, labels, value
    write_json(complete, report)


def prepare_qwen_cache(root: Path) -> None:
    from scra.llm import build_prompt_cache
    plan = json.loads((EXPERIMENT / "plan_c2.json").read_text())
    meta = build_prompt_cache(root / "data/mnli_r1_seed2026/manifest.json",
                             str(root / "models/qwen2.5-0.5b"), root / plan["cache_directory"], 256)
    print(json.dumps({"cache": "qwen", "splits": meta["splits"], "label_ids": meta["label_token_ids"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["models", "data", "cache", "all"])
    parser.add_argument("--family", choices=["roberta", "qwen", "both"], default="both")
    parser.add_argument("--project-root", type=Path, default=REPO / ".reproduction",
                        help="Asset and new-run directory; defaults to the ignored .reproduction folder")
    parser.add_argument("--mnli-source", type=Path,
                        help="Existing official JSONL directory; omit to download the official ZIP")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    families = ["roberta", "qwen"] if args.family == "both" else [args.family]
    sys.path[:0] = [str(CODE), str(EXPERIMENT)]
    if args.stage in ("models", "all"):
        download_models(root, families)
    if args.stage in ("data", "all"):
        prepare_data(root, args.mnli_source)
    if args.stage in ("cache", "all"):
        for family in families:
            {"roberta": prepare_roberta_cache, "qwen": prepare_qwen_cache}[family](root)


if __name__ == "__main__":
    main()
