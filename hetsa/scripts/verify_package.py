"""Verify all published HetSA/literature files against their packaged hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
manifest=ROOT/'hetsa/reproduce/PACKAGED_FILE_MANIFEST.json'

def main():
    records=json.loads(manifest.read_text(encoding='utf-8'))['files']
    errors=[]
    for record in records:
        path=ROOT/record['path']
        if not path.is_file():
            errors.append(f'Missing: {record["path"]}')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            errors.append(f'Hash mismatch: {record["path"]}')
    if errors:
        raise RuntimeError('\n'.join(errors))
    print(f'{len(records)} published files match packaged SHA-256 hashes.')

if __name__=='__main__':
    main()
