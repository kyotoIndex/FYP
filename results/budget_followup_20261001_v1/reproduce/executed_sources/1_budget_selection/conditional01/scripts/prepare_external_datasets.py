"""Prepare QNLI and AG News from preserved raw downloads.

The script is deterministic and writes a checksum manifest for every raw and
processed file used by the formal cross-dataset experiments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_qnli() -> dict:
    archive = RAW / "QNLIv2.zip"
    output = ROOT / "data" / "QNLI"
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        for split in ("train", "dev"):
            member = f"QNLI/{split}.tsv"
            if member not in names:
                raise RuntimeError(f"{archive} is missing {member}")
            (output / f"{split}.tsv").write_bytes(handle.read(member))
    rows = {}
    labels = {}
    for split in ("train", "dev"):
        path = output / f"{split}.tsv"
        split_labels: Counter[str] = Counter()
        row_count = 0
        with path.open("r", encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    raise ValueError(f"malformed QNLI row in {path}")
                split_labels[parts[-1]] += 1
                row_count += 1
        rows[split] = row_count
        labels[split] = dict(split_labels)
    return {
        "source_url": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
        "raw_sha256": sha256(archive),
        "rows": rows,
        "labels": labels,
        "processed_sha256": {
            split: sha256(output / f"{split}.tsv")
            for split in ("train", "dev")
        },
    }


def convert_ag_news(source: Path, target: Path) -> tuple[int, dict[str, int]]:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    labels: Counter[str] = Counter()
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.reader(input_handle)
        with target.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.writer(
                output_handle,
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow(["title", "description", "label"])
            for raw_label, title, description in reader:
                label = int(raw_label) - 1
                if label not in range(4):
                    raise ValueError(f"unexpected AG News label: {raw_label}")
                writer.writerow([title, description, label])
                labels[str(label)] += 1
                count += 1
    return count, dict(labels)


def prepare_ag_news() -> dict:
    output = ROOT / "data" / "AG_NEWS"
    raw_train = RAW / "ag_news_train.csv"
    raw_test = RAW / "ag_news_test.csv"
    train_rows, train_labels = convert_ag_news(
        raw_train, output / "train.tsv"
    )
    dev_rows, dev_labels = convert_ag_news(raw_test, output / "dev.tsv")
    return {
        "source_url": (
            "https://github.com/mhjabreel/CharCnn_Keras/"
            "tree/master/data/ag_news_csv"
        ),
        "raw_sha256": {
            "train": sha256(raw_train),
            "test": sha256(raw_test),
        },
        "rows": {"train": train_rows, "dev": dev_rows},
        "labels": {"train": train_labels, "dev": dev_labels},
        "processed_sha256": {
            "train": sha256(output / "train.tsv"),
            "dev": sha256(output / "dev.tsv"),
        },
    }


def main() -> None:
    manifest = {
        "QNLI": prepare_qnli(),
        "AG_NEWS": prepare_ag_news(),
    }
    path = ROOT / "data" / "DATASET_MANIFEST.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
