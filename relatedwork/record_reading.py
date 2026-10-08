"""Record actual reading scope; metadata only, no changes to downloaded sources."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
SCOPES = {
 'FedSA-LoRA': 'PDF 1-10; selected appendix A.3/A.4/A.7 and theoretical assumptions',
 'Fed-HeLLo': 'PDF 1-13', 'LEGEND': 'PDF 1-12',
 'Fed-pilot': 'PDF 1-10; 15-17 resource/implementation settings',
 'Fed-PLoRA': 'PDF 1-10; algorithm on p16',
 'FedAS-LoRA': 'PDF 1-8, 11-12, 14-15; active TeX projection formulas',
 'FedSAIL': 'PDF 1-11, 19-21; targeted scans of 22-26; TeX compact/Gram/objective formulas; rendered p21 table',
 'AS-LoRA': 'PDF 1-9; appendix proofs not fully audited',
 'RoLoRA': 'PDF 1-10; long appendix not fully read',
 'FDLoRA': 'PDF 1-10', 'HAFLQ': 'PDF 1-12',
 'HetLoRA': 'PDF 1-8', 'FlexLoRA': 'PDF 1-9', 'FLoRA': 'PDF 1-9',
 'FFA-LoRA': 'PDF 1-9', 'SLoRA': 'PDF 1-8',
 'LoRA': 'PDF 3-11', 'AdaLoRA': 'PDF 3-9', 'FedIT': 'PDF 3-10', 'FedAvg': 'PDF 2-8',
}
manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
assert set(SCOPES) == {p['method'] for p in manifest}
records = []
for p in manifest:
    closest = p['priority'] == '特别接近'
    depth = '精读核心方法/关键公式/主要实验及定向附录' if closest else '阅读核心方法/主要实验/局限'
    p['reading_depth'] = depth+'；未复现实验，未逐行验证全部证明'
    p['reading_scope'] = SCOPES[p['method']]
    p['reading_report'] = 'result.md'
    records.append(dict(method=p['method'], arxiv_version=p['arxiv_id']+p['version'],
        pdf_file=p['pdf_file'], pdf_sha256=p['pdf_sha256'], depth=depth,
        scope=SCOPES[p['method']], limitations='No training replication; not all appendices/references read; no full proof re-derivation', report='result.md'))
(ROOT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(ROOT/'reading_log.json').write_text(json.dumps(dict(date='2026-09-30', papers=records), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with (ROOT/'paper_index.csv').open(encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
if 'reading_depth' in fields:
    for row in rows:
        if row.get('method') in SCOPES:
            row['reading_depth'] = next(p['reading_depth'] for p in manifest if p['method'] == row['method'])
    with (ROOT/'paper_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
print('Recorded 20 paper reading scopes; 11 close-paper deep reviews; original PDF/TeX retained.')
