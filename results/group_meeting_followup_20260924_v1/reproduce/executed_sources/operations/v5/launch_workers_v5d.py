"""Launch the two V5d queues on ten free GPUs inside existing 4090 allocations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path('$HPC_HOME/pcla_v5_20260924')
PYTHON='$HPC_HOME/pcla_repair_20260918/venv/bin/python'
SLOTS={'central':[('172164',4),('172168',1)], 'svd':[('172165',4),('172169',1)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('variant',choices=SLOTS);a=p.parse_args()
    variant=a.variant
    handoff=ROOT/'handoff_v5d'/variant
    queue=ROOT/'queue_v5d'/variant
    logs=ROOT/'workers_v5d'/variant
    assert (queue/'FROZEN.json').is_file()
    assert not logs.exists(), 'new launch directory required'
    needed={job for job,_ in SLOTS[variant]}
    state=subprocess.check_output(['squeue','-u',subprocess.check_output(['id','-un'],text=True).strip(),
                                   '-h','-o','%i %T %b'],text=True)
    allocations={}
    for line in state.splitlines():
        parts=line.split(maxsplit=2)
        if len(parts)==3 and parts[0] in needed:allocations[parts[0]]=parts[1:]
    assert needed==set(allocations) and all(x[0]=='RUNNING' and '4090' in x[1] for x in allocations.values()), \
           'assigned 4090 allocations not all running'
    steps=subprocess.check_output(['squeue','-s','-u',subprocess.check_output(['id','-un'],text=True).strip(),
                                   '-h','-o','%i %j'],text=True)
    busy=[line for line in steps.splitlines() if line.split('.')[0] in needed and
          not any(x in line for x in ('.batch ','.extern '))]
    assert not busy, f'existing steps on assigned allocations: {busy}'
    logs.mkdir(parents=True)
    manifest=[]
    for jobid,count in SLOTS[variant]:
        for i in range(count):
            session=f'pcla_v5d_{variant}_{jobid}_{i}'
            assert subprocess.run(['tmux','has-session','-t','='+session],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
            argv=['srun',f'--jobid={jobid}','--nodes=1','--ntasks=1','--gpus-per-task=1',
                  '--cpus-per-task=4','--exact','--cpu-bind=cores',PYTHON,
                  str(handoff/'dispatch.py'),'worker','--queue',str(queue),'--pool','4090',
                  '--max-jobs','1']
            log=logs/f'{session}.log'
            cmd=f'{shlex.join(argv)} > {shlex.quote(str(log))} 2>&1; echo "exit=$?" >> {shlex.quote(str(log))}'
            subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
            manifest.append({'session':session,'allocation':jobid,'argv':argv,'log':str(log),
                             'launched_at':time.time()})
            time.sleep(.2)
    (logs/'launch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'variant':variant,'launched':len(manifest),'manifest':str(logs/'launch_manifest.json')}))


if __name__=='__main__':main()
