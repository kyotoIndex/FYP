#!/usr/bin/env python3
"""Copy execution source into a new content-identified immutable release directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-commit', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Never overwrite a source release')
    a.output.mkdir(parents=True)
    names = ['pcla_repair', 'pcla_research_v7', 'scripts', 'tests_pcla_v7', 'tests_pcla_research_v7']
    manifest = {'base_commit': a.base_commit, 'files': {}}
    for name in names:
        for source in sorted((a.source / name).rglob('*.py')):
            if '__pycache__' in source.parts:
                continue
            relative = source.relative_to(a.source)
            dest = a.output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            manifest['files'][str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if not manifest['files']:
        raise ValueError('Source release is empty')
    target = a.output / 'SOURCE_MANIFEST.json'
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    for file in a.output.rglob('*'):
        if file.is_file():
            file.chmod(0o444)
    print(json.dumps({'source': str(a.output), 'files': len(manifest['files']),
                      'manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
