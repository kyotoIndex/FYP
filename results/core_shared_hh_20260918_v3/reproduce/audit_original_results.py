"""Read-only audits and predeclared secondary readouts for the fixed v3 matrix.

Core NLL is recorded GPU loss, not recomputed from absent logits. Step counts
are derived from checked sample sizes and fixed batch/epoch configuration.
"""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
def read(p):return json.loads(p.read_text())
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def csvsave(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 jobs=[json.loads(l) for l in a.jobs.read_text().splitlines()];assert len(jobs)==40
 plan=read(a.repo/'configs/pcla_repair/core_shared_hh_20260918_v3.json');cfg=plan['training']
 assert plan['evaluation_split']=='benchmark_dev' and not cfg['record_optimization_diagnostics']
 labels={}
 for split in ('train','dev'):
  labels[split]=[int(l.rsplit('\t',1)[1]) for l in (a.repo/f'data/SST-2/{split}.tsv').read_text().splitlines()[1:] if l]
 assert len(labels['dev'])==872
 def label(s):split,i=s.split(':');return labels[split][int(i)]
 paired={};protocols=set();donors={};endpoints=[];curves=[];budgets=[];replays=[];inventory=[];checkpoint_count=0
 for j in jobs:
  seed,policy=j['seed'],j['policy'];q=a.root/f'seed_{seed}'/policy
  status=read(q/'status.json');ident=read(q/'identity.json');part=read(q/'partition_ids.json');ms=read(q/'metrics.json');prov=ident['provenance']
  assert status=={'state':'completed','completed_round':30} and len(ms)==30
  assert ident['seed']==seed and ident['policy']==policy
  assert fp({k:v for k,v in ident.items() if k not in ('identity_hash','policy')})==ident['identity_hash']
  assert ident['config']==cfg and prov['plan_sha256']==fp(plan) and prov['evaluation_split']=='benchmark_dev'
  assert fp(part)==prov['partition_sha256']
  for split in ('train','dev'):assert sha(a.repo/f'data/SST-2/{split}.tsv')==prov[f'{split}_file_sha256']
  for name,h in prov['code_files'].items():assert sha(a.repo/name)==h
  protocols.add(fp({k:v for k,v in prov.items() if k not in ('partition_sha256','initial_loaded_model_sha256')}))
  population=[x for xs in part['clients'].values() for x in xs]
  assert len(population)==len(set(population))==12800 and len(part['clients'])==50
  assert all(len(xs)==256 for xs in part['clients'].values())
  groups=[set(population),set(part['proxy']),set(part['tuning']),set(part['benchmark_dev'])]
  assert [len(s) for s in groups]==[12800,512,1024,872]
  assert all(not groups[i]&groups[k] for i in range(4) for k in range(i))
  assert groups[3]=={f'dev:{i}' for i in range(872)}
  y=np.asarray([label(x) for x in part['benchmark_dev']]);assert np.array_equal(np.bincount(y,minlength=2),[428,444])
  zero=torch.load(q/'probe_round_000.pt',map_location='cpu',weights_only=True)
  final=torch.load(q/'checkpoint.pt',map_location='cpu',weights_only=True)
  assert final['identity_hash']==ident['identity_hash'] and final['policy']==policy and final['round']==30 and final['metrics']==ms
  assert len(final['trace'])==300
  if seed in paired:
   other=paired[seed];assert other['identity']==ident['identity_hash'] and other['part']==part
   assert all(torch.equal(v,other['initial'][k]) for k,v in zero['adapter'].items())
   assert other['participants']==[m['selected_clients'] for m in ms]
  else:paired[seed]={'identity':ident['identity_hash'],'part':part,'initial':zero['adapter'],'participants':[m['selected_clients'] for m in ms]}
  for r in (0,5,15,30):
   cp=torch.load(q/f'probe_round_{r:03d}.pt',map_location='cpu',weights_only=True)
   assert cp['format']=='pcla-repair-v1' and cp['round']==r and cp['policy']==policy and cp['identity_hash']==ident['identity_hash']
   assert cp['metrics']==ms[:r] and len(cp['trace'])==10*r
   assert all(torch.isfinite(v).all() for v in cp['adapter'].values())
   if policy=='head_only':assert all(torch.equal(v,zero['adapter'][k]) for k,v in cp['adapter'].items() if not k.startswith('classifier.'))
   checkpoint_count+=1
  assert all(torch.equal(v,final['adapter'][k]) for k,v in cp['adapter'].items())
  for key,value in cp['trace'].items():assert torch.equal(value,final['trace'][key])
  if policy=='pcla':donors[seed]={'trace':final['trace'],'identity':ident['identity_hash'],'sha256':sha(q/'checkpoint.pt')}
  if policy=='replay_random_rank':
   assert seed in donors and donors[seed]['identity']==ident['identity_hash']
   assert j['donor']==f'results/core_shared_hh_20260918_v3/seed_{seed}/pcla/checkpoint.pt'
  head_parameters=sum(v.numel() for k,v in zero['adapter'].items() if k.startswith('classifier.'));assert head_parameters==1538
  totals={'credits':0,'head_parameter_updates':0,'logical_parameter_updates':0,'optimizer_tensor_parameter_calls':0,'client_calls':0,'derived_optimizer_steps':0,'download_bytes':0,'upload_bytes':0,'probe_seconds':0.,'train_seconds':0.,'probe_calls':0}
  own_curves=[];replay_changed=0
  for expected_round,m in enumerate(ms,1):
   r=m['round'];assert r==expected_round and m['samples']==872 and math.isfinite(m['loss'])
   pred=np.asarray(read(q/f'predictions_round_{r:03d}.json'));assert pred.shape==(872,) and np.isin(pred,[0,1]).all()
   acc=float((pred==y).mean());assert acc==m['accuracy']
   recall=[float((pred[y==c]==c).mean()) for c in (0,1)];counts=np.bincount(pred,minlength=2)
   row={'seed':seed,'policy':policy,'round':r,'accuracy':acc,'nll':m['loss'],'negative_recall':recall[0],'positive_recall':recall[1],'balanced_accuracy':sum(recall)/2,'predicted_positive_count':int(counts[1]),'constant_prediction':int(max(counts)==872)}
   curves.append(row);own_curves.append(row)
   details=read(q/f'clients_round_{r:03d}.json');assert len(details)==10 and [d['client'] for d in details]==m['selected_clients'] and len(set(m['selected_clients']))==10
   traffic=0;probe_time=0.;train_time=0.
   for d in details:
    cid=d['client'];support=np.asarray(d['support']);assert support.shape==(12,16) and support.dtype==bool
    trace=final['trace'][f'{r}/{cid}'];assert np.array_equal(support,trace.numpy())
    credits=int(support.sum());target=0 if policy=='head_only' else 192 if policy=='full_alt' else part['capacities'][str(cid)]*8
    assert credits==d['credits']==target
    assert d['factor']==('none' if policy=='head_only' else 'b' if r%2 else 'a')
    assert d['sample_weight']==256 and d['logical_update_parameters']==1538+credits*1536
    active_layers=int(support.any(axis=1).sum());assert d['optimizer_tensor_parameters']==1538+active_layers*16*1536
    if policy in ('layer_only','random_layer'):assert set(support.sum(axis=1).tolist())<={0,16}
    if policy=='uniform_cyclic':
     low,rem=divmod(credits,12);assert set(support.sum(axis=1).tolist())<={low,low+int(rem>0)}
    if policy=='replay_random_rank':
     donor=donors[seed]['trace'][f'{r}/{cid}'].numpy();assert np.array_equal(support.sum(axis=1),donor.sum(axis=1))
     replay_changed+=int(not np.array_equal(support,donor))
    if policy not in ('pcla','layer_only'):assert d['probe_seconds']==0
    assert d['probe_seconds']>=0 and d['train_seconds']>=0
    totals['credits']+=credits;totals['head_parameter_updates']+=1538;totals['logical_parameter_updates']+=d['logical_update_parameters'];totals['optimizer_tensor_parameter_calls']+=d['optimizer_tensor_parameters'];totals['client_calls']+=1
    totals['derived_optimizer_steps']+=math.ceil(d['sample_weight']/cfg['batch_size'])*cfg['local_epochs']
    for k in ('download_bytes','upload_bytes','probe_seconds','train_seconds'):totals[k]+=d[k]
    totals['probe_calls']+=int(d['probe_seconds']>0);traffic+=d['download_bytes']+d['upload_bytes'];probe_time+=d['probe_seconds'];train_time+=d['train_seconds']
   assert traffic==m['serialized_traffic_bytes'] and abs(probe_time-m['probe_seconds'])<1e-8 and abs(train_time-m['train_seconds'])<1e-8
  assert totals['client_calls']==300 and totals['derived_optimizer_steps']==4800
  tail=own_curves[-5:]
  endpoints.append({'seed':seed,'policy':policy,'development_exposure':j['development_exposure'],
   'last5_accuracy':statistics.mean(m['accuracy'] for m in tail),'last5_nll':statistics.mean(m['nll'] for m in tail),
   'last5_balanced_accuracy':statistics.mean(m['balanced_accuracy'] for m in tail),'last5_negative_recall':statistics.mean(m['negative_recall'] for m in tail),'last5_positive_recall':statistics.mean(m['positive_recall'] for m in tail),
   'last5_min_accuracy':min(m['accuracy'] for m in tail),'last5_max_accuracy':max(m['accuracy'] for m in tail),
   'last5_constant_rounds':sum(m['constant_prediction'] for m in tail),'final_accuracy':ms[-1]['accuracy'],'all30_accuracy_mean':statistics.mean(m['accuracy'] for m in ms)})
  budgets.append({'seed':seed,'policy':policy,**totals,'serialized_traffic_bytes':totals['download_bytes']+totals['upload_bytes'],'evaluation_seconds':sum(m['eval_seconds'] for m in ms),'simulator_round_wall_seconds':sum(m['simulator_round_wall_seconds'] for m in ms)})
  if policy=='replay_random_rank':replays.append({'seed':seed,'donor_sha256':donors[seed]['sha256'],'all_300_client_layer_quotas_match':True,'calls_with_different_rank_ids':replay_changed,'donor_cost_not_included_in_replay_own_cost':True})
  inventory.append({'id':j['id'],'state':'completed','rounds':30,'identity_hash':ident['identity_hash'],'checkpoint_sha256':sha(q/'checkpoint.pt'),'train_dev_duplicate_text_count':part['train_dev_duplicate_text_count']})
 assert len(protocols)==1
 for seed in paired:
  matched=[b['credits'] for b in budgets if b['seed']==seed and b['policy'] not in ('head_only','full_alt')];assert len(set(matched))==1
 assert len(endpoints)==40 and len(curves)==1200 and sum(b['client_calls'] for b in budgets)==12000
 csvsave(a.out/'per_seed_endpoints.csv',endpoints);save(a.out/'per_seed_endpoints.json',endpoints);csvsave(a.out/'per_round.csv',curves);csvsave(a.out/'budget_and_costs.csv',budgets);save(a.out/'budget_and_costs.json',budgets);save(a.out/'replay_audit.json',replays);save(a.out/'job_inventory.json',inventory)
 groups=[]
 for name,seeds in [('all_five',[7,13,21,42,87]),('development_exposed',[7,13]),('not_used_for_this_lr',[21,42,87])]:
  for policy in dict.fromkeys(j['policy'] for j in jobs):
   rs=[e for e in endpoints if e['seed'] in seeds and e['policy']==policy]
   groups.append({'group':name,'policy':policy,'seeds':seeds,**{k:statistics.mean(x[k] for x in rs) for k in ('last5_accuracy','last5_nll','last5_balanced_accuracy','last5_negative_recall','last5_positive_recall','final_accuracy','all30_accuracy_mean')},'constant_rounds':sum(x['last5_constant_rounds'] for x in rs),'tail_observations':5*len(rs)})
 save(a.out/'group_descriptive.json',groups)
 checks={'pass':True,'planned':40,'completed':40,'failed':0,'blocked':0,'rounds':1200,'client_calls':12000,'derived_optimizer_steps':sum(b['derived_optimizer_steps'] for b in budgets),'probe_checkpoints_loaded':checkpoint_count,'final_checkpoints_loaded':40,'matched_credits_within_seed':True,'replay_all_layer_quotas_match':True,'same_seed_initial_parameters_partition_participants':True,'all_1200_prediction_metrics_recomputed':True,'nll_source':'recorded GPU loss; finite checked; no logits-based NLL recomputation','step_count_source':'derived from actual completed client details and fixed batch16/epoch1, not additional optimization instrumentation'}
 save(a.out/'verification.json',checks);print(json.dumps(checks))
if __name__=='__main__':main()
