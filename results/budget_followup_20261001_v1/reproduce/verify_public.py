#!/usr/bin/env python3
"""Check every published file against PUBLIC_MANIFEST.json."""
import gzip, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
manifest = json.load(open(os.path.join(ROOT, 'reproduce', 'PUBLIC_MANIFEST.json'), encoding='utf-8'))
listed = set()
bad = []
for entry in manifest['files']:
    path = os.path.join(ROOT, entry['published'])
    listed.add(entry['published'])
    if not os.path.isfile(path):
        bad.append(f"missing {entry['published']}")
        continue
    raw = open(path, 'rb').read()
    content = gzip.decompress(raw) if entry['gzip'] else raw
    if hashlib.sha256(content).hexdigest() != entry['content_sha256']:
        bad.append(f"hash mismatch {entry['published']}")
    if not entry['redacted'] and entry['content_sha256'] != entry['source_sha256']:
        bad.append(f"unredacted file differs from source {entry['published']}")

own = {'README.md', 'METHODS.md', 'SOURCES.md', 'reproduce/README.md', 'reproduce/PUBLIC_MANIFEST.json',
       'reproduce/verify_tables.py', 'reproduce/verify_public.py'}
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d != '__pycache__']
    for f in fn:
        rel = os.path.relpath(os.path.join(dp, f), ROOT)
        if rel not in listed and rel not in own:
            bad.append(f'unlisted {rel}')

if bad:
    print('FAILED:\n  ' + '\n  '.join(bad[:50]))
    sys.exit(1)
print(f'PASS: {len(listed)} published files match PUBLIC_MANIFEST.json')
