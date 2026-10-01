#!/usr/bin/env python3
"""Resume independent native checks after the preserved FedSVD TECH failure."""
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
a=p.parse_args(); r=a.root.resolve(); source=r/'sources/native03'; cfg=r/'configs/technical_native03'
# FedSB's existing four-round profile is retained. Only its restart check is missing.
subprocess.run([sys.executable,str(r/'repo/scripts/check_pcla_v7_path_resumes.py'),'--source',str(source),'--config-dir',str(cfg),'--out',str(r/'checks/path_resumes/native03_sb_plora'),'--recipes','D104','D116'],check=True)
subprocess.run([sys.executable,str(r/'repo/scripts/profile_pcla_v7_paths.py'),'--root',str(r),'--source',str(source),'--config-dir',str(cfg),'--generation','native03_plora_continuation','--recipes','D116'],check=True)
