#!/usr/bin/env python3
"""Read-only real-schema probe; sends exporter source through SSH stdin."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
source = (HERE / 'export_validation.py').read_text()
main_guard = '\nif __name__ == "__main__":'
if source.count(main_guard) != 1:
    raise SystemExit('exporter main guard changed')
probe = '''
ROOT = Path('$HPC_HOME/pcla_global_budget_20260928_v7')
examples = (
    ('validation_queue', 'V_S12_D086_s101'),
    ('validation_queue', 'V_S13_D105_s101'),
    ('conditional_validation_queue', 'V_S04_D042_s181'),
    ('conditional_validation_queue', 'V_S03_D035_s101'),
    ('conditional_validation_queue', 'V_S16_D035_s163'),
)
donor_id = 'V_S03_D035_s163'
donor = read(ROOT / 'conditional_validation_queue/jobs' / (donor_id + '.json'))
roster = {(163, 3): donor}
results = []
for queue, jid in examples:
    item = read(ROOT / queue / 'jobs' / (jid + '.json'))
    hashes = {}
    try:
        result = extract_one(ROOT, item, roster, hashes)
        results.append({'job_id': jid, 'ok': True,
                        'endpoint': result['endpoint'],
                        'provenance': result['provenance'],
                        'cost_summary': {
                            'rows': len(result['costs']),
                            'missing_cost_rounds': [r['round'] for r in result['costs'] if r['cost'] is None],
                            'missing_cumulative_rounds': [r['round'] for r in result['costs'] if r['cumulative_cost'] is None],
                            'first_cost_keys': sorted(result['costs'][0]['cost'] or {}),
                            'last_cumulative_keys': sorted(result['costs'][-1]['cumulative_cost'] or {}),
                        },
                        'hashed_input_count': len(hashes)})
    except Exception as error:
        results.append({'job_id': jid, 'ok': False,
                        'error_type': type(error).__name__, 'error': str(error)})
print(json.dumps(results, sort_keys=True, default=str))
'''
payload = source.split(main_guard)[0] + probe
process = subprocess.run(
    ['ssh', 'hpc-bnbu', '$HPC_HOME/pcla_repair_20260918/venv/bin/python -'],
    input=payload, text=True, capture_output=True,
)
if process.returncode:
    raise SystemExit(f'remote CPU probe exited {process.returncode}: {process.stderr[-2000:]}')
results = json.loads(process.stdout)
(HERE / 'INDEPENDENT_REAL_SAMPLE_PROBE.json').write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
print([(row['job_id'], row['ok'], row.get('error')) for row in results])
