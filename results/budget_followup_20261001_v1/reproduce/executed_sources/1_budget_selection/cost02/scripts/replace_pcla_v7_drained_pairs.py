#!/usr/bin/env python3
"""Replace eight old paired steps only after each has naturally finished."""
import getpass,json,shlex,subprocess,sys,time
from pathlib import Path
from dispatch_pcla_v7 import read,write
r=Path('$HPC_HOME/pcla_global_budget_20260928_v7')
old={'172165':['233','234','235','236'],'172168':['322','323'],'172169':['172','173']}
receipt=r/'workers/pair_refresh_core02.json'
state=read(receipt) if receipt.exists() else {'replacements':{},'created_at':time.time()}
while len(state['replacements'])<8:
 for allocation,steps in old.items():
  live=subprocess.check_output(['scontrol','show','job','-o',allocation],text=True)
  fields=dict(s.split('=',1) for s in live.split() if '=' in s)
  if fields.get('JobState')!='RUNNING' or fields.get('UserId','').split('(')[0]!=getpass.getuser():raise RuntimeError('Approved allocation no longer active/owned')
  lines=subprocess.check_output(['squeue','--steps','-j',allocation,'-h'],text=True)
  present={line.split()[0] for line in lines.splitlines() if line.strip()}
  for step in steps:
   old_id=allocation+'.'+step
   if old_id in state['replacements'] or old_id in present:continue
   session='pclav7_pairrefresh_'+allocation+'_'+step
   if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0:raise RuntimeError('Unrecorded replacement session exists')
   argv=['srun','--jobid='+allocation,'--nodes=1','--ntasks=1','--gpus-per-task=1','--cpus-per-task=8','--exact','--cpu-bind=cores',sys.executable,str(r/'repo/scripts/dispatch_pcla_v7.py'),'--queue',str(r/'queue'),'pair','--pool','4090D_shared','--python',sys.executable]
   log=r/'workers'/(session+'.log')
   subprocess.run(['tmux','new-session','-d','-s',session,shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1'],check=True)
   state['replacements'][old_id]={'session':session,'argv':argv,'log':str(log),'time':time.time()};write(receipt,state)
   print(json.dumps({'old_step_finished':old_id,'replacement_session':session}),flush=True)
 time.sleep(10)
