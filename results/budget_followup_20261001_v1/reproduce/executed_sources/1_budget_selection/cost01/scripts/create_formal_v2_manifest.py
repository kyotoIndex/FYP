"""Record exact code, data, model, configuration, and environment provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(paths)
    }


def main() -> None:
    tracked = [
        ROOT / "src" / "fedhello_sst2.py",
        ROOT / "scripts" / "run_full_experiments.py",
        ROOT / "scripts" / "run_formal_v2_matrix.py",
        ROOT / "scripts" / "validate_setup.py",
        ROOT / "scripts" / "validate_results.py",
        ROOT / "scripts" / "generate_formal_v2_configs.py",
        ROOT / "scripts" / "prepare_external_datasets.py",
        ROOT / "tests" / "test_allocations.py",
        ROOT / "data" / "DATASET_MANIFEST.json",
        ROOT / "data" / "SST-2" / "train.tsv",
        ROOT / "data" / "SST-2" / "dev.tsv",
        ROOT / "data" / "QNLI" / "train.tsv",
        ROOT / "data" / "QNLI" / "dev.tsv",
        ROOT / "data" / "AG_NEWS" / "train.tsv",
        ROOT / "data" / "AG_NEWS" / "dev.tsv",
        ROOT / "models" / "roberta-base" / "config.json",
        ROOT / "models" / "roberta-base" / "model.safetensors",
        *sorted((ROOT / "configs" / "formal_v2").glob("*.json")),
    ]
    missing = [path for path in tracked if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "formal-v2 manifest inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "files": relative_hashes(tracked),
    }
    output = ROOT / "results" / "formal_v2" / "protocol_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
