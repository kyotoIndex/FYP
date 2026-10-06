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

# Runtime entry points verify SOURCE_MANIFEST.json, rather than PUBLIC_MANIFEST.
# Keep the executed manifest separate from the hashes of the published sources.
source_root = os.path.join(ROOT, 'reproduce', 'executed_sources')
source_count = 0
for dp, _, fn in os.walk(source_root):
    if 'SOURCE_MANIFEST.json' not in fn:
        continue
    source_count += 1
    source = json.load(open(os.path.join(dp, 'SOURCE_MANIFEST.json'), encoding='utf-8'))
    for relative, expected in source['files'].items():
        path = os.path.join(dp, relative)
        if not os.path.isfile(path) or hashlib.sha256(open(path, 'rb').read()).hexdigest() != expected:
            bad.append(f"runtime source mismatch {os.path.relpath(path, ROOT)}")
    publication = source.get('publication')
    if publication:
        original_path = os.path.join(dp, publication['original_manifest'])
        if not os.path.isfile(original_path) or hashlib.sha256(open(original_path, 'rb').read()).hexdigest() != publication['original_manifest_sha256']:
            bad.append(f"original manifest mismatch {os.path.relpath(original_path, ROOT)}")

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
print(f'PASS: {len(listed)} published files match PUBLIC_MANIFEST.json; {source_count} runtime source manifests match published sources')
