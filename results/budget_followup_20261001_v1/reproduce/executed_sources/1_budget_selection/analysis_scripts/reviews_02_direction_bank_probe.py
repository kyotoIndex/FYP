#!/usr/bin/env python3
"""Read-only CPU projection of one audited direction bank, without output export."""
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
source = (here / 'export_diagnostics.py').read_text().split("\nif __name__ == '__main__':")[0]
probe = '''
root = Path('$HPC_HOME/pcla_global_budget_20260928_v7')
jid = 'Direction_s101_r10_a'
out = root / 'diagnostics' / jid
queue = read(root / 'diagnostic_queue/jobs' / (jid + '.json'))
config = read(root / 'checks/diagnostic_ready01/diagnostic_configs' / (jid + '.json'))
stage_path = out / 'direction_stage.json'
stage = read(stage_path)
audit_path = Path(queue['audit']['receipt'])
bank_path = out / 'direction_bank.pt'
require(digest(bank_path) == stage['bank_sha256'], 'bank/stage SHA differs')
bank = _load_pt(bank_path, root / 'sources/diagnostic03')
acceptance = queue['acceptance'].get('receipt', queue['acceptance'].get('path'))
item = {'job_id': jid, 'line': 'direction', 'config': config, 'job': queue,
        'config_sha256': queue['config_sha256'], 'stage': stage,
        'stage_sha256': digest(stage_path), 'audit_sha256': digest(audit_path),
        'acceptance_sha256': digest(Path(acceptance))}
rows = project_bank(bank, item)
from collections import Counter
print(json.dumps({'job_id': jid, 'slurm_job_id': __import__('os').environ.get('SLURM_JOB_ID'),
                  'slurm_step_id': __import__('os').environ.get('SLURM_STEP_ID'),
                  'bank_sha256': digest(bank_path), 'row_count': len(rows),
                  'record_types': dict(Counter(row['record_type'] for row in rows)),
                  'na_count': sum(v['normmatch_status'].startswith('NA_')
                                  for p in stage['readouts'].values()
                                  for q in p.values() for v in q.values())}, sort_keys=True))
'''
process = subprocess.run([
    'ssh', 'hpc-bnbu',
    'srun --jobid=172165 --exclusive --exact --immediate=60 -n1 -c1 --gres=none '
    '$HPC_HOME/pcla_repair_20260918/venv/bin/python -',
], input=source + probe, text=True, capture_output=True, timeout=180)
if process.returncode:
    raise SystemExit(f'direction CPU probe exited {process.returncode}: {process.stderr[-3000:]}')
result = json.loads(process.stdout)
(here / 'REVIEWS_02_DIRECTION_BANK_PROBE.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(result)
