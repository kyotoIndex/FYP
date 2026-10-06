#!/usr/bin/env python3
"""Sequential path-level restart checks on a single allocated technical GPU."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--recipes',nargs='+',required=True)
a=p.parse_args()
if not os.getenv('SLURM_JOB_ID') or not os.getenv('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Requires allocated GPU')
a.out.mkdir(parents=True,exist_ok=False)
for rid in a.recipes:
 for mode in ('single','shared'):
  script='check_pcla_v7_resume.py' if mode=='single' else 'check_pcla_v7_shared_resume.py'
  subprocess.run([sys.executable,str(a.source/'scripts'/script),'--source',str(a.source),'--config',str(a.config_dir/(f'TECH_{rid}_s9001.json')),'--out',str(a.out/rid/mode)],check=True)
(a.out/'status.json').write_text(json.dumps({'state':'completed','recipes':a.recipes,'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID')},indent=2))
