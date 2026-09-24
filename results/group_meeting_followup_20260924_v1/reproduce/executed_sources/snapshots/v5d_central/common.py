"""Identity, atomic persistence and archive utilities. No implicit remote writes."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
VERSION = '4.1'
BATCH = 'pcla-v5d-central-replay-20260924'
SEEDS = (7, 13, 21, 42, 87)
COMMIT = 'e53078b638ee30a633bad98195c1e2d4476523f5'
CONFIG = 'configs/pcla_repair/core_shared_hh_20260918_v3.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path); sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def tensor_load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def source_files():
    """Full execution and analysis source identity, not only selected modules."""
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted(ROOT.glob('*.py'))}


def verify_repo(repo):
    repo = Path(repo).resolve()
    for rel, expected in read(ROOT / 'SOURCE_LOCK.json')['files'].items():
        require(sha256(repo / rel) == expected['sha256'], 'source differs: ' + rel)
    return repo


def bind_repo(repo=None):
    repo = verify_repo(repo or ROOT / 'reference_snapshot')
    if 'pcla_repair.core' in sys.modules:
        require(Path(sys.modules['pcla_repair.core'].__file__).resolve() ==
                repo / 'pcla_repair/core.py', 'another PCLA checkout already imported')
    sys.path.insert(0, str(repo))
    import torch
    from pcla_repair import core, runtime, bridge
    return torch, core, runtime, bridge


@contextlib.contextmanager
def exclusive(path, metadata=None):
    """Fail closed on stale locks; recovery is an explicit separate operation."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    value = {'token': token, 'pid': os.getpid(), 'host': socket.gethostname(),
             'time': time.time(), **(metadata or {})}
    try:
        os.write(fd, json.dumps(value).encode()); os.fsync(fd)
    finally:
        os.close(fd)
    try:
        yield value
    finally:
        if path.exists() and read(path).get('token') == token:
            path.unlink()


def file_manifest(root, exclude=()):
    root = Path(root)
    values = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'symlink refused: ' + str(p))
        if p.is_file() and p.relative_to(root).as_posix() not in exclude:
            require(not p.name.endswith('.lock'), 'active lock inside archive')
            values[p.relative_to(root).as_posix()] = {
                'size': p.stat().st_size, 'sha256': sha256(p)}
    return values


def make_archive(root, target):
    root, target = Path(root), Path(target)
    require(not target.exists(), 'archive already exists')
    require(not (root / 'ARCHIVE_MANIFEST.json').exists(), 'reserved archive filename')
    manifest = file_manifest(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.partial-' + uuid.uuid4().hex)
    with zipfile.ZipFile(tmp, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel in manifest:
            z.write(root / rel, rel)
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, sort_keys=True))
    verify_archive(tmp)
    # Exclusive link prevents two workers from publishing to the same destination.
    os.link(tmp, target); tmp.unlink(); sync_directory(target.parent)
    return {'path': str(target.resolve()), 'sha256': sha256(target),
            'bytes': target.stat().st_size, 'files': len(manifest)}


def verify_archive(path, expected_sha=None, max_bytes=64 << 30):
    if expected_sha is not None:
        require(sha256(path) == expected_sha, 'archive SHA mismatch')
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist()]
        require(len(names) == len(set(names)), 'duplicate archive entry')
        require(sum(i.file_size for i in z.infolist()) <= max_bytes, 'archive too large')
        for i in z.infolist():
            n = PurePosixPath(i.filename)
            require(not n.is_absolute() and '..' not in n.parts and '\\' not in i.filename,
                    'unsafe archive path')
            require(not stat.S_ISLNK(i.external_attr >> 16), 'archive symlink')
        m = json.loads(z.read('ARCHIVE_MANIFEST.json'))
        require(set(names) == set(m) | {'ARCHIVE_MANIFEST.json'}, 'archive inventory mismatch')
        for rel, info in m.items():
            h = hashlib.sha256(); count = 0
            with z.open(rel) as f:
                for b in iter(lambda: f.read(8 << 20), b''):
                    h.update(b); count += len(b)
            require(count == info['size'] and h.hexdigest() == info['sha256'],
                    'corrupt archive member: ' + rel)
    return m


def extract_archive(path, expected_sha, destination):
    import shutil
    dest = Path(destination)
    require(not dest.exists(), 'extraction destination exists')
    manifest = verify_archive(path, expected_sha)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(path) as z:
        for rel in manifest:
            p = dest / rel; p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(rel) as src, p.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
    require(file_manifest(dest) == manifest, 'extracted files differ')
    atomic_json(dest / 'EXTRACTION_RECEIPT.json', {'archive_sha256': expected_sha})
    return dest
