"""Join V quality summaries to separately measured COST1 recipe times."""
import csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(root,out):
 sources=[root/'validation_export02/EXPORT_MANIFEST.json',root/'validation_report01/MANIFEST.json',root/'cost_export02/EXPORT_MANIFEST.json']
 for p in sources:
  d=json.loads(p.read_text())
  for n,h in d['output_sha256'].items():assert sha(p.parent/n)==h,n
 rows=list(csv.DictReader((root/'validation_export02/v_endpoints.csv').open()));recipes={int(x['slot']):x['recipe_id'] for x in rows}
 costs={x['recipe_id']:x for x in csv.DictReader((root/'cost_export02/cost_endpoints.csv').open())}
 quality=[x for x in csv.DictReader((root/'validation_report01/slot_endpoints.csv').open()) if x['endpoint']=='b8'];assert len(quality)==16
 result=[]
 for q in quality:
  s=int(q['slot']);c=None if s==16 else costs[recipes[s]]; selected=[x for x in rows if int(x['slot'])==s]
  result.append({'slot':s,'method_zh':q['method_zh'],'recipe_id':recipes[s],'V_mean_B8_accuracy_percent':q['mean_percent'],'V_sample_sd_accuracy_pp':q['sample_sd_pp'],'V_n_seeds':12,'V_mean_return_round':sum(int(x['b8_round']) for x in selected)/12,'V_mean_deployed_bytes':sum(int(x['b8_deployed_bytes']) for x in selected)/12,'COST1_job_id':None if c is None else c['job_id'],'COST1_seed':None if c is None else c['seed'],'COST1_required_seconds':None if c is None else c['required_seconds'],'COST1_return_round':None if c is None else c['budget_return_round'],'COST1_deployed_bytes':None if c is None else c['deployed_bytes'],'cost_scope':'no_separate_COST1_replay_measurement' if c is None else 'separate_seed401_exclusive_4090D; not paired V timing','COST1_cache_unknown_clients':None if c is None else c['digest_unverifiable_clients'],'COST1_rng_status':None if c is None else c['rng_status']})
 assert not out.exists();out.mkdir()
 f=out/'quality_cost_join.csv'
 with f.open('x',newline='') as w:
  writer=csv.DictWriter(w,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
 manifest={'schema':'pcla-v-quality-cost-separate-runs-1','source_manifest_sha256':{str(p):sha(p) for p in sources},'script_sha256':sha(Path(__file__)),'output_sha256':{f.name:sha(f)},'notes':['V accuracy across12seeds; COST1 latency from one separate fixed seed401','slot16 donor-dependent replay has no separate COST1 recipe; donor time not imputed','return may stop at200round cap before exhausting B8; actual bytes/round retained']}
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
