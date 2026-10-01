#!/usr/bin/env python3
"""Use one drained, allocated GPU for path-specific bounded concurrency checks."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from dispatch_pcla_v7 import lock, read, write

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--config-dir',type=Path,required=True);p.add_argument('--recipes',nargs='+',required=True);p.add_argument('--generation',required=True);p.add_argument('--restore-reservation',action='store_true')
a=p.parse_args()
if not os.getenv('SLURM_JOB_ID') or not os.getenv('CUDA_VISIBLE_DEVICES'):raise RuntimeError('An allocated GPU is required')
base=a.root/'checks/path_profiles'/a.generation;base.mkdir(parents=True,exist_ok=False)
write(base/'allocation.json',{'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID'),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'time':time.time()})
if a.restore_reservation:
 restored=[]
 with lock(a.root/'queue'):
  for path in (a.root/'queue/jobs').glob('*.json'):
   v=read(path)
   if v['state']=='profile_gpu_reservation':
    v.update(state='pending',reason=None);write(path,v);restored.append(v['job']['job_id'])
 write(base/'reservation_restored.json',restored)
for recipe in a.recipes:
 cfg=a.config_dir/(f'TECH_{recipe}_s9001.json')
 if not cfg.is_file():raise FileNotFoundError(cfg)
 out=base/recipe
 for n in (1,2):
  argv=[sys.executable,str(a.root/'repo/scripts/profile_pcla_v7_concurrency.py'),'--source',str(a.source),'--config',str(cfg),'--out',str(out/f'p{n}'),'--processes',str(n),'--rounds','4']
  subprocess.run(argv,check=True)
 write(out/'execution_finished.json',{'time':time.time(),'state':'awaiting_independent_profile_audit'})
write(base/'status.json',{'state':'executed_pending_audit','recipes':a.recipes})
