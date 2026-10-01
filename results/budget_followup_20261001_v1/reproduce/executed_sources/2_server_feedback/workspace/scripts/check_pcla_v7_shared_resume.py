#!/usr/bin/env python3
"""Exercise two independent restart checks concurrently on one GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
procs = []
for i in range(2):
    argv = [sys.executable, str(a.source / 'scripts/check_pcla_v7_resume.py'),
            '--source', str(a.source), '--config', str(a.config), '--out', str(a.out / str(i))]
    procs.append(subprocess.Popen(argv, env=dict(os.environ, PCLA_GPU_SHARING='shared')))
codes = [p.wait() for p in procs]
print(json.dumps({'exit_codes': codes}), flush=True)
raise SystemExit(max(codes))
