#!/usr/bin/env python3
"""One-object read-only CPU probe of repaired diagnostic PT/JSON comparisons."""
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
source = (here / 'export_diagnostics.py').read_text().split("\nif __name__ == '__main__':")[0]
probe = '''
root = Path('$HPC_HOME/pcla_global_budget_20260928_v7')
jid = 'Cohort_s101_full_alt_r10_a'
out = root / 'diagnostics' / jid
stage = out / 'bank_stage'
source_root = root / 'sources/diagnostic03'
job = read(root / 'diagnostic_queue/jobs' / (jid + '.json'))
audit = read(Path(job['audit']['receipt']))
paths = {
    'supports': out / 'supports.pt',
    'prediction': stage / 'prediction_lock.pt',
    'selection': stage / 'selection_lock.pt',
}
for key, path in paths.items():
    expected = audit['hashes'][key + '_lock_pt' if key != 'supports' else 'supports_pt']
    require(digest(path) == expected, key + ' PT SHA changed')
supports = _load_pt(paths['supports'], source_root)
prediction = _load_pt(paths['prediction'], source_root)
selection = _load_pt(paths['selection'], source_root)
jq = read(stage / 'jq_pretraining_lock.json')
pred = read(stage / 'prediction_lock.json')
sel = read(stage / 'selection_lock.json')
result = {
    'slurm_job_id': __import__('os').environ.get('SLURM_JOB_ID'),
    'slurm_step_id': __import__('os').environ.get('SLURM_STEP_ID'),
    'job_id': jid,
    'supports_jq_equal': _plain(supports['jq_lock']) == jq,
    'prediction_equal': _plain(prediction) == pred['prediction'],
    'selection_equal': _plain(selection) == {k: v for k, v in sel.items() if k != 'file_sha256'},
    'pt_sha': {k: digest(v) for k, v in paths.items()},
}
print(json.dumps(result, sort_keys=True))
'''
command = [
    'ssh', 'hpc-bnbu',
    'srun --jobid=172165 --exclusive --exact --immediate=60 -n1 -c1 --gres=none '
    '$HPC_HOME/pcla_repair_20260918/venv/bin/python -',
]
process = subprocess.run(command, input=source + probe, text=True, capture_output=True, timeout=150)
if process.returncode:
    raise SystemExit(f'CPU probe exited {process.returncode}: {process.stderr[-2000:]}')
result = json.loads(process.stdout)
(here / 'REVIEWS_02_DIAGNOSTIC_REAL_PROBE.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(result)
