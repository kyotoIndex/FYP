#!/usr/bin/env python3
"""Validate a newly frozen FedSVD numeric repair, then consume admitted D0 work."""
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--generation',required=True)
a=p.parse_args(); r=a.root.resolve(); source=r/'sources'/a.generation
# Scientific TECH config remains byte-identical to native03.
cfg=r/'configs/technical_native03'
subprocess.run([sys.executable,str(r/'repo/scripts/profile_pcla_v7_paths.py'),'--root',str(r),'--source',str(source),'--config-dir',str(cfg),'--generation',a.generation,'--recipes','D107'],check=True)
subprocess.run([sys.executable,str(r/'repo/scripts/check_pcla_v7_path_resumes.py'),'--source',str(source),'--config-dir',str(cfg),'--out',str(r/'checks/path_resumes'/a.generation),'--recipes','D107'],check=True)
# No FedSVD D0 is admitted by this script: the parent verifies its evidence.
subprocess.run([sys.executable,str(r/'repo/scripts/dispatch_pcla_v7.py'),'--queue',str(r/'queue'),'worker','--pool','4090D_single','--python',sys.executable,'--mode','exclusive','--idle-seconds','3'],check=True)
subprocess.run([sys.executable,str(r/'repo/scripts/dispatch_pcla_v7.py'),'--queue',str(r/'queue'),'worker','--pool','4090D','--python',sys.executable,'--mode','exclusive'],check=True)
