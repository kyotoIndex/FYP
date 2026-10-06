#!/usr/bin/env python3
"""One authorized off-policy continuation using the complete original r60 state."""
import argparse,hashlib,json,os,sys,traceback,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_pcla_v7 import prepare,persist_failed_cost_timing
from pcla_repair.runtime import atomic_json
from pcla_research_v7.runtime import run_experiment
from pcla_research_v7.execution_timing import ExecutionTimingLedger,COST_LABEL_CATEGORIES,required_breakdown,COST_LABEL_GROUPS,REQUIRED_GROUPS
from pcla_research_v8.policy_switch import sha

def require(v,w):
 if not v:raise RuntimeError(w)
def main():
 p=argparse.ArgumentParser();p.add_argument('--branch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--technical-first-round',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 b=json.loads(a.branch.read_text());policy=dict(b['policy']);cfgpath=Path(b['parent_config_path']);cfg=json.loads(cfgpath.read_text());out=a.out
 require(sha(cfgpath)==b['parent_config_sha256'],'parent config changed')
 require(sha(ROOT/'SOURCE_MANIFEST.json')==b['new_source_manifest_sha256'],'source manifest changed')
 for rel,s in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['files'].items():require(sha(ROOT/rel)==s,'new frozen source differs '+rel)
 require(cfg['recipe']['server_rule'] in {'scalar4','block16'} and cfg['job']['recipe_id']==b['parent_recipe'] and cfg['job']['seed']==b['seed'],'parent identity differs')
 require(torch.cuda.is_available() and os.environ.get('PCLA_GPU_SHARING')=='exclusive' and len(os.sched_getaffinity(0))==4,'requires exclusive GPU and four CPUs')
 require(all(os.environ.get(k)=='4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'four-thread numerical identity differs')
 if a.technical_first_round:policy['first_round_technical_only']=61
 out.mkdir(parents=True,exist_ok=True)
 require((out/'latest.pt').exists()==a.resume,'resume must match existing branch checkpoint')
 lock=out/'.policy_run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID')}).encode());os.close(fd)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda:0');timing=ExecutionTimingLedger(COST_LABEL_CATEGORIES,synchronize=lambda:torch.cuda.synchronize(device));started=time.time()
 physical=out/'physical_attempts'/uuid.uuid4().hex;physical.mkdir(parents=True,exist_ok=False)
 atomic_json(physical/'START.json',{'started_at':started,'branch_config_sha256':sha(a.branch),'driver_sha256':sha(__file__),'resume':a.resume,'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID')})
 try:
  origin={'branch_config_sha256':sha(a.branch),'policy':policy}
  if (out/'BRANCH_ORIGIN.json').exists():require(json.loads((out/'BRANCH_ORIGIN.json').read_text())==origin,'existing branch origin differs')
  else:atomic_json(out/'BRANCH_ORIGIN.json',origin)
  model,clients,caps,evalset,prov=prepare(cfg,cfgpath,device,timing=timing)
  prov['v8_source_manifest_sha256']=sha(ROOT/'SOURCE_MANIFEST.json');prov['server_rule']='policy_switch_ordinary_mean_r8';prov['branch_config_sha256']=sha(a.branch)
  if not a.resume:
   for key in ('parent_journal','parent_labels'):
    path=Path(b[key+'_path']);require(sha(path)==b[key+'_sha256'],key+' changed')
   journal=json.loads(Path(b['parent_journal_path']).read_text());journal['entries']=[e for e in journal['entries'] if e['round']<=60]
   if not (out/'communication_journal.json').exists():atomic_json(out/'communication_journal.json',journal)
   else:
    current=json.loads((out/'communication_journal.json').read_text())
    require([e for e in current['entries'] if e['round']<=60]==journal['entries'],'inherited journal prefix changed')
   if not (out/'evaluation_labels.json').exists():atomic_json(out/'evaluation_labels.json',json.loads(Path(b['parent_labels_path']).read_text()))
   else:require(sha(out/'evaluation_labels.json')==sha(Path(b['parent_labels_path'])),'existing labels changed')
  runtime=dict(cfg['runtime'])
  if cfg['recipe'].get('head_lr')!=cfg['recipe'].get('client_lr'):runtime['head_lr']=cfg['recipe']['head_lr']
  rows=run_experiment(model,clients,evalset,caps,cfg['job'],runtime,out,prov,timing=timing,v8_mode=True,server_controller=None,policy_switch=policy)
  timing.end_envelope();record=timing.snapshot();atomic_json(physical/'TIMING.json',record);attempts=json.loads((out/'execution_attempts.json').read_text());atomic_json(out/f'timing_attempt_{len(attempts):03d}.json',record)
  new=[r for r in rows if r['round']>60];target=61 if a.technical_first_round else 80
  require([r['round'] for r in new]==list(range(61,target+1)),'fixed continuation horizon incomplete')
  require(all(r['server_controller']['alpha']==r['server_controller']['beta']==1 and r['server_controller']['accepted_hash']==r['server_controller']['proposal_hash'] and len(r['client_training'])==10 and all(c['optimizer_steps']==16 for c in r['client_training']) for r in new),'ordinary policy/client steps differ')
  all_events=[]
  physical_records=[]
  for directory in sorted((out/'physical_attempts').iterdir()):
   candidate=directory/'timing_events_partial.json' if (directory/'timing_events_partial.json').exists() else directory/'TIMING.json'
   if candidate.exists():physical_records.append(json.loads(candidate.read_text()))
  require(len(physical_records)==len(list((out/'physical_attempts').iterdir())),'physical attempt timing missing')
  for t in physical_records:all_events.extend(t['events'])
  measured_cost={group:sum(sum(t['bucket_label_seconds']['deployed_prefix'][label] for label,g in COST_LABEL_GROUPS.items() if g==group) for t in physical_records) for group in REQUIRED_GROUPS}
  cost_complete=not any(t.get('partial') for t in physical_records)
  atomic_json(out/'PHYSICAL_COST.json',{'attempts':len(physical_records),'complete':cost_complete,'measured_required_lower_bound':measured_cost,'required_cost_breakdown':measured_cost if cost_complete else None,'partial_timing_paths':[str(p.relative_to(out)) for p in (out/'physical_attempts').glob('*/timing_events_partial.json')]})
  ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
  for row in new:
   events=[e for e in all_events if e['round']==row['round']]
   wire=[e for e in ck['cache']['events'] if e['round']==row['round']]
   row['policy_switch_research_cost']={'necessary_seconds':sum(e['seconds'] for e in events if e['category']=='necessary_algorithm'),'research_seconds':sum(e['seconds'] for e in events if e['category']=='research_observation'),'upload_bytes':sum(e['bytes'] for e in wire if e['direction']=='upload'),'download_bytes':sum(e['bytes'] for e in wire if e['direction']=='download')}
  atomic_json(out/'metrics.json',rows)
  atomic_json(out/'run_identity.json',{'branch':b,'provenance':prov,'parent_prefix_reused':True})
  atomic_json(out/'result.json',{'job_id':b['job_id'],'parent_recipe':b['parent_recipe'],'seed':b['seed'],'state':'technical_complete' if a.technical_first_round else 'executed_pending_audit','first_new_round':61,'last_new_round':target,'new_rounds':len(new),'client_calls':len(new)*10,'optimizer_steps':len(new)*160,'parent_state_sha256':policy['parent_state_sha256'],'parent_checkpoint_file_sha256':policy['parent_checkpoint_file_sha256'],'new_source_manifest_sha256':b['new_source_manifest_sha256'],'branch_config_sha256':sha(a.branch),'prefix_deployed_bytes':b['prefix_deployed_bytes'],'prefix_diagnostic_bytes':b['prefix_diagnostic_bytes'],'additional_bytes':rows[-1]['deployed_bytes']+rows[-1]['diagnostic_after_budget_bytes']-b['prefix_deployed_bytes']-b['prefix_diagnostic_bytes'],'required_cost_breakdown':measured_cost if cost_complete else None,'measured_required_lower_bound':measured_cost,'physical_cost_complete':cost_complete,'physical_attempts':len(physical_records),'cost_scope':'new research branch only; inherited prefix excluded','elapsed_seconds_this_attempt':time.time()-started,'hardware':prov['hardware']})
  atomic_json(physical/'END.json',{'ended_at':time.time(),'state':'executed','result_sha256':sha(out/'result.json')})
  print(json.dumps({'job_id':b['job_id'],'new_rounds':len(new),'out':str(out)}),flush=True)
 except BaseException:
  trace=traceback.format_exc();(physical/'exception.txt').write_text(trace);persist_failed_cost_timing(physical,timing,trace.splitlines()[-1]);atomic_json(physical/'END.json',{'ended_at':time.time(),'state':'failed'});raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
