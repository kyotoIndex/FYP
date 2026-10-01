#!/usr/bin/env python3
"""Keep one allocated GPU across the remaining bounded checks and admitted jobs."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
subprocess.run([sys.executable,str(r/'repo/scripts/check_pcla_v7_path_resumes.py'),'--source',str(r/'sources/core02'),'--config-dir',str(r/'configs/technical_core02'),'--out',str(r/'checks/path_resumes/core02'),'--recipes','D002','D020','D032','D050','D077'],check=True)
subprocess.run([sys.executable,str(r/'repo/scripts/check_pcla_v7_native_pipeline.py'),'--root',str(r),'--generation','native03'],check=True)
