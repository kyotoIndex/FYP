#!/usr/bin/env python3
"""Read-only real-schema and mutation probe for repaired V exporter."""
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
source = (here / 'export_validation.py').read_text().split('\nif __name__ == "__main__":')[0]
probe = '''
ROOT = Path('$HPC_HOME/pcla_global_budget_20260928_v7')
examples = (
    ('validation_queue', 'V_S12_D086_s101'),
    ('validation_queue', 'V_S13_D105_s101'),
    ('conditional_validation_queue', 'V_S04_D042_s181'),
    ('conditional_validation_queue', 'V_S03_D035_s101'),
    ('conditional_validation_queue', 'V_S03_D035_s149'),
    ('conditional_validation_queue', 'V_S03_D035_s163'),
    ('conditional_validation_queue', 'V_S03_D035_s181'),
    ('conditional_validation_queue', 'V_S16_D035_s163'),
)
donor = read(ROOT / 'conditional_validation_queue/jobs/V_S03_D035_s163.json')
roster = {(163, 3): donor}
rows = []
for queue, jid in examples:
    item = read(ROOT / queue / 'jobs' / (jid + '.json'))
    try:
        result = extract_one(ROOT, item, roster, {})
        rows.append({'job_id': jid, 'passed': True,
                     'cost_rounds': len(result['costs']),
                     'endpoint': {k: result['endpoint'][k] for k in
                                  ('half_round', 'b8_round', 'r100_round')}})
    except Exception as error:
        rows.append({'job_id': jid, 'passed': False, 'error': str(error)})
original = read(ROOT / 'conditional_validation_queue/jobs/V_S04_D042_s181.json')
mutations = {}
for name in ('missing_gates', 'wrong_labels'):
    item = dict(original)
    if name == 'missing_gates':
        for key in ('release_receipt', 'release_receipt_sha256',
                    'technical_receipt', 'technical_receipt_sha256',
                    'draft_config', 'draft_config_sha256'):
            item.pop(key, None)
    else:
        item['job'] = dict(original['job'])
        item['job']['method'] = 'fedsb'
        item['job']['representation'] = 'U'
    try:
        extract_one(ROOT, item, roster, {})
        mutations[name] = {'rejected': False}
    except ExportError as error:
        mutations[name] = {'rejected': True, 'error': str(error)}
print(json.dumps({'real_samples': rows, 'mutations': mutations}, sort_keys=True))
'''
process = subprocess.run(
    ['ssh', 'hpc-bnbu', '$HPC_HOME/pcla_repair_20260918/venv/bin/python -'],
    input=source + probe, text=True, capture_output=True,
)
if process.returncode:
    raise SystemExit(f'probe exited {process.returncode}: {process.stderr[-2000:]}')
result = json.loads(process.stdout)
(here / 'REVIEWS_02_VALIDATION_PROBE.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(result)
