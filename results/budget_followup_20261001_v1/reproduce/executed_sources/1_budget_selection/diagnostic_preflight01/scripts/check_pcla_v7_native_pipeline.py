#!/usr/bin/env python3
"""Bounded native GPU integration, throughput and restart checks, then queue work."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--generation',required=True)
a=p.parse_args();r=a.root.resolve();source=r/'sources'/a.generation;cfg=r/'configs'/('technical_'+a.generation)
recipes=['D104','D107','D116']
subprocess.run([sys.executable,str(r/'repo/scripts/profile_pcla_v7_paths.py'),'--root',str(r),'--source',str(source),'--config-dir',str(cfg),'--generation',a.generation,'--recipes',*recipes],check=True)
subprocess.run([sys.executable,str(source/'scripts/check_pcla_v7_path_resumes.py'),'--source',str(source),'--config-dir',str(cfg),'--out',str(r/'checks/path_resumes'/a.generation),'--recipes',*recipes],check=True)
# GPU evidence is audited by the controller; this worker can immediately
# continue already admitted exclusive D0 work within the same allocation.
subprocess.run([sys.executable,str(r/'repo/scripts/dispatch_pcla_v7.py'),'--queue',str(r/'queue'),'worker','--pool','4090D_single','--python',sys.executable,'--mode','exclusive','--idle-seconds','3'],check=True)
subprocess.run([sys.executable,str(r/'repo/scripts/dispatch_pcla_v7.py'),'--queue',str(r/'queue'),'worker','--pool','4090D','--python',sys.executable,'--mode','exclusive'],check=True)
