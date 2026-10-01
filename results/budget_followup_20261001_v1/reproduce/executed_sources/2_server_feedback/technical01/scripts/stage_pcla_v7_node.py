#!/usr/bin/env python3
"""Verify and cache the frozen local BERT files on an allocated compute node."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Stage inside an existing allocation')
    files = ['config.json', 'model.safetensors', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.txt']
    expected = {name: digest(args.source / name) for name in files}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists():
        actual = {name: digest(args.cache / name) for name in files}
        if actual != expected:
            raise ValueError('Existing cache differs; retain it and use another cache path')
    else:
        staging = Path(tempfile.mkdtemp(prefix=args.cache.name + '.staging-', dir=args.cache.parent))
        for name in files:
            shutil.copyfile(args.source / name, staging / name)
        actual = {name: digest(staging / name) for name in files}
        if actual != expected:
            raise ValueError('Node cache copy hash mismatch')
        staging.rename(args.cache)
    receipt = {'host': socket.gethostname(), 'allocation': os.environ['SLURM_JOB_ID'],
               'cache': str(args.cache), 'files': actual,
               'revision': 'google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594'}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
