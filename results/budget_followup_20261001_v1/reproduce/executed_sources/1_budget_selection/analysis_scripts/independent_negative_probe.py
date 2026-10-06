#!/usr/bin/env python3
"""Read-only mutation probe against real completed queue records."""
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
source = (here / 'export_validation.py').read_text().split('\nif __name__ == "__main__":')[0]
probe = '''
ROOT = Path('$HPC_HOME/pcla_global_budget_20260928_v7')
original = read(ROOT / 'conditional_validation_queue/jobs/V_S04_D042_s181.json')
missing_gates = dict(original)
for key in ('release_receipt', 'release_receipt_sha256', 'technical_receipt',
            'technical_receipt_sha256', 'draft_config', 'draft_config_sha256'):
    missing_gates.pop(key, None)
gate_result = extract_one(ROOT, missing_gates, {}, {})
wrong_job = dict(original)
wrong_job['job'] = dict(original['job'])
wrong_job['job']['method'] = 'fedsb'
wrong_job['job']['representation'] = 'U'
wrong_result = extract_one(ROOT, wrong_job, {}, {})
print(json.dumps({'job_id': original['job']['job_id'],
                  'missing_gates_accepted': gate_result['endpoint']['job_id'] == original['job']['job_id'],
                  'wrong_job_accepted': wrong_result['endpoint']['job_id'] == original['job']['job_id'],
                  'original_method': original['job']['method'],
                  'exported_wrong_method': wrong_result['endpoint']['method'],
                  'original_representation': original['job']['representation'],
                  'exported_wrong_representation': wrong_result['endpoint']['representation']}))
'''
process = subprocess.run(
    ['ssh', 'hpc-bnbu', '$HPC_HOME/pcla_repair_20260918/venv/bin/python -'],
    input=source + probe, text=True, capture_output=True,
)
if process.returncode:
    raise SystemExit(f'negative probe exited {process.returncode}: {process.stderr[-2000:]}')
result = json.loads(process.stdout)
(here / 'INDEPENDENT_NEGATIVE_PROBE.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(result)
