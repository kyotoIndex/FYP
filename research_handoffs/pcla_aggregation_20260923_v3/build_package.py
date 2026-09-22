from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


EXCLUDE = {"MANIFEST_SHA256.json", "BUILD_STATUS.json"}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True,
                        help="Destination ZIP; must not already exist")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(output)

    syntax_checked = []
    for path in sorted(root.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        syntax_checked.append(path.name)

    status = {
        "syntax_checked": syntax_checked,
        "numerical_tests": "not_run_by_builder",
        "BERT_CUDA_tests": "not_run_by_builder",
        "scientific_experiments": "not_run",
    }
    (root / "BUILD_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in EXCLUDE:
            payload = path.read_bytes()
            manifest[path.name] = {"bytes": len(payload),
                                   "sha256": sha256(payload)}
    (root / "MANIFEST_SHA256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_payload = (root / "MANIFEST_SHA256.json").read_bytes()
    status_payload = (root / "BUILD_STATUS.json").read_bytes()

    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(manifest):
            archive.write(root / name, f"pcla_aggregation_20260923_v3/{name}")
        archive.writestr("pcla_aggregation_20260923_v3/MANIFEST_SHA256.json",
                         manifest_payload)
        archive.writestr("pcla_aggregation_20260923_v3/BUILD_STATUS.json",
                         status_payload)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for name, meta in manifest.items():
            payload = archive.read(f"pcla_aggregation_20260923_v3/{name}")
            if len(payload) != meta["bytes"] or sha256(payload) != meta["sha256"]:
                raise RuntimeError(f"ZIP manifest mismatch: {name}")
    print(json.dumps({
        "zip": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output.read_bytes()),
        "syntax_check": "passed",
        "numerical_and_GPU_tests": "not run by builder",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
