from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile

BATCH = "aggregation-causal-20260923-v3"
BASE_COMMIT = "e53078b638ee30a633bad98195c1e2d4476523f5"
SEEDS = (7, 13, 21, 42, 87)
SOURCES = ("pcla", "replay_random_rank", "random")
FULL_RULES = ("fixed", "contributors", "function_norm_matched")
DIAGNOSTIC_RULES = FULL_RULES + ("parameter_norm_matched",)
CONFIG = "configs/pcla_repair/core_shared_hh_20260918_v3.json"

EXPECTED_SHA256 = {
    "pcla_repair/__init__.py":
        "d8ab3147577b2acc5cf5a55955459d7859711491355ac06695376cbad24f23da",
    "pcla_repair/core.py":
        "dde28d7b7a1838c7a9d113142ce9fffb60583ccbd41aa18599b63f153e73bcdd",
    "pcla_repair/runtime.py":
        "577e3be7cf56d2708638eac75bba0962709a9f5a038c0ccf3b82777ccb902283",
    "pcla_repair/bridge.py":
        "583b749f5e8a5f8e545d7dea3bd22129ad6eff93204d47f34544d88d797f586d",
    "pcla_repair/diagnostics.py":
        "20f20e25beb50d4dc647d4ef517bac39974e7193c1332e346a8d904b29ad72df",
    "pcla_repair/optimization_dev.py":
        "b890c28d5ffe0858312dc2eded62bc29bd7be079d2ded224ccdd964f083e84dc",
    "pcla_repair/testing.py":
        "edee243fc0ab52066d5e5700cdd9f98f16c14c10fd67b3a47c93686cd91f7acb",
    "scripts/run_pcla_repaired.py":
        "4dbca4ae63f778bc9aca2a472b66623f4f7014e7b4f775e548ae91551fc6d0a1",
}

HANDOFF_CODE = (
    "common.py", "aggregation.py", "pcla_runner.py", "fixed_state.py",
    "dispatch.py", "gate.py", "analyze.py", "discovery_math.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def handoff_root() -> Path:
    return Path(__file__).resolve().parent


def handoff_hashes() -> dict[str, str]:
    root = handoff_root()
    return {name: sha256_file(root / name) for name in HANDOFF_CODE}


def bind_project(repo: str | Path):
    repo = Path(repo).resolve()
    for relative, expected in EXPECTED_SHA256.items():
        path = repo / relative
        require(path.is_file(), f"missing inspected source: {relative}")
        require(sha256_file(path) == expected, f"inspected source changed: {relative}")
    sys.path.insert(0, str(repo))
    import torch
    from pcla_repair import core, runtime, bridge
    require(Path(core.__file__).resolve() == repo / "pcla_repair/core.py",
            "a different PCLA checkout is already imported")
    return torch, core, runtime, bridge


def resolve_site(raw: dict) -> dict:
    required = ("repo", "source_root", "queue_root", "scratch_root", "python_pcla")
    for key in required:
        require(key in raw, f"site is missing {key}")
    result = dict(raw)
    for key in required:
        result[key] = str(Path(raw[key]).expanduser().resolve())
    require(Path(result["repo"]).is_dir(), "repo path is missing")
    require(Path(result["source_root"]).is_dir(), "source_root is missing")
    require(Path(result["python_pcla"]).is_file(), "python_pcla is missing")
    result.setdefault("environment", {})
    result.setdefault("require_slurm", True)
    result.setdefault("worker_idle_seconds", 4)
    result.setdefault("full_run_timeout_seconds", 1800)
    result.setdefault("diagnostic_timeout_seconds", 900)
    result.setdefault("claim_stale_seconds", 7200)
    return result


def write_tree_manifest(root: str | Path, output: str | Path) -> dict:
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"manifest refuses symlink: {path}")
        if path.is_file() and path.resolve() != Path(output).resolve():
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    atomic_json(output, files)
    return files


def archive_tree(root: str | Path, target: str | Path) -> dict:
    root, target = Path(root), Path(target)
    require(root.is_dir(), "archive source is missing")
    require(not target.exists(), "archive target already exists")
    manifest = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"archive refuses symlink: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            require(name != "ARCHIVE_MANIFEST.json", "reserved archive name")
            manifest[name] = {"bytes": path.stat().st_size,
                              "sha256": sha256_file(path)}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    require(not partial.exists(), "partial archive already exists")
    try:
        with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for name in manifest:
                zf.write(root / name, name)
            zf.writestr("ARCHIVE_MANIFEST.json",
                        json.dumps(manifest, sort_keys=True, indent=2))
        with zipfile.ZipFile(partial) as zf:
            require(zf.testzip() is None, "archive CRC failed")
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"path": str(target.resolve()), "sha256": sha256_file(target),
            "bytes": target.stat().st_size, "files": len(manifest)}


def extract_verified(archive: str | Path, expected_sha: str,
                     destination: str | Path) -> Path:
    archive, destination = Path(archive), Path(destination)
    require(sha256_file(archive) == expected_sha, "archive SHA mismatch")
    if destination.exists():
        marker = destination / ".extracted.json"
        require(marker.is_file(), "existing extraction lacks identity")
        require(read_json(marker)["archive_sha256"] == expected_sha,
                "existing extraction has another archive identity")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-",
                                dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as zf:
            raw_manifest = zf.read("ARCHIVE_MANIFEST.json")
            manifest = json.loads(raw_manifest)
            members = [x for x in zf.infolist() if not x.is_dir()]
            names = [x.filename for x in members]
            require(len(names) == len(set(names)), "duplicate archive members")
            require(set(names) == set(manifest) | {"ARCHIVE_MANIFEST.json"},
                    "archive inventory differs from manifest")
            for info in members:
                if info.filename == "ARCHIVE_MANIFEST.json":
                    continue
                name = Path(info.filename)
                require(not name.is_absolute() and ".." not in name.parts,
                        "unsafe archive path")
                mode = info.external_attr >> 16
                require(not stat.S_ISLNK(mode), "archive symlink refused")
                data = zf.read(info)
                meta = manifest[info.filename]
                require(len(data) == meta["bytes"] and
                        hashlib.sha256(data).hexdigest() == meta["sha256"],
                        f"archive member mismatch: {info.filename}")
                path = temp / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        atomic_json(temp / ".extracted.json", {"archive_sha256": expected_sha})
        os.replace(temp, destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination
