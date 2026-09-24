#!/usr/bin/env python3
"""Summarize only the registered 14-job development matrix, not benchmark dev."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))

def fingerprint(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=REPO)
 p.add_argument('--jobs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();repo=a.repo.resolve()
 jobs=[json.loads(x) for x in a.jobs.read_text().splitlines() if x.strip()]
 expected={(t,s) for t in ('P_HH','H_H','R_HH','R_LH','R_HL','R_LL','H_L') for s in (7,13)}
 if {(j['trial'],j['seed']) for j in jobs}!=expected or len(jobs)!=14:raise ValueError('not the registered matrix')
 records=[];failures=[];common=set();per_seed={};labels_hashes=set()
 for j in jobs:
  try:
   plan_path=repo/j['config'];raw=plan_path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=j['config_file_sha256']:raise ValueError('registered config file changed')
   plan=json.loads(raw)
   if plan['evaluation_split']!='tuning':raise ValueError('benchmark dev must not enter optimization development')
   q=repo/j['out_dir'];ident=json.loads((q/'identity.json').read_text());status=json.loads((q/'status.json').read_text())
   metrics=json.loads((q/'metrics.json').read_text());ref=json.loads((q/'diagnostic_references.json').read_text())
   if ident['seed']!=j['seed'] or ident['policy']!=j['policy']:raise ValueError('seed/policy identity differs')
   payload={k:v for k,v in ident.items() if k not in ('identity_hash','policy')}
   if fingerprint(payload)!=ident['identity_hash']:raise ValueError('identity hash differs')
   if ident['config']!=plan['training'] or ident['provenance']['plan_sha256']!=fingerprint(plan):raise ValueError('plan/training differs')
   if ident['provenance']['evaluation_split']!='tuning':raise ValueError('wrong evaluation split')
   if status!={'state':'completed','completed_round':30}:raise ValueError('not complete at round30')
   if [m['round'] for m in metrics]!=list(range(1,31)):raise ValueError('wrong fixed horizon')
   labels=json.loads((q/'evaluation_labels.json').read_text());labels_hashes.add(fingerprint(labels))
   part=json.loads((q/'partition_ids.json').read_text())
   if fingerprint(part)!=ident['provenance']['partition_sha256']:raise ValueError('partition hash differs')
   if len(labels)!=1024 or len(part['tuning'])!=1024:raise ValueError('wrong tuning population')
   common.add(fingerprint({k:v for k,v in ident['provenance'].items()
                   if k not in ('partition_sha256','initial_loaded_model_sha256','plan_sha256')}))
   seed_pair=(ident['initial_adapter_sha256'],ident['provenance']['partition_sha256'],ident['provenance']['initial_loaded_model_sha256'])
   per_seed.setdefault(j['seed'],set()).add(seed_pair)
   for m in metrics:
    if m['samples']!=1024 or not math.isfinite(m['loss']) or not math.isfinite(m['accuracy']) or not 0<=m['accuracy']<=1:raise ValueError('invalid metrics')
    counts=m['predicted_class_counts']
    if sum(counts)!=1024 or any(v<0 for v in counts):raise ValueError('invalid prediction counts')
    logits=np.asarray(json.loads((q/f'logits_round_{m["round"]:03d}.json').read_text()),dtype=np.float64)
    target=np.asarray(labels,dtype=np.int64)
    if logits.shape!=(1024,2) or not np.isfinite(logits).all():raise ValueError('invalid raw tuning logits')
    pred=logits.argmax(1)
    if np.bincount(pred,minlength=2).tolist()!=counts:raise ValueError('raw logits disagree with prediction counts')
    if float((pred==target).mean())!=m['accuracy']:raise ValueError('raw logits disagree with accuracy')
    nll=float((np.logaddexp.reduce(logits,axis=1)-logits[np.arange(1024),target]).mean())
    if abs(nll-m['loss'])>1e-5:raise ValueError('raw logits disagree with reported NLL')
   tail=metrics[-5:]
   records.append({'trial':j['trial'],'policy':j['policy'],'seed':j['seed'],'head_lr':j['head_lr'],'lora_lr':j['lora_lr'],
      'last5_accuracy':statistics.mean(m['accuracy'] for m in tail),'last5_nll':statistics.mean(m['loss'] for m in tail),
      'last5_balanced_accuracy':statistics.mean(m['balanced_accuracy'] for m in tail),
      'last5_all_one_class_rounds':sum(max(m['predicted_class_counts'])==1024 for m in tail),
      'final_accuracy':metrics[-1]['accuracy'],'final_nll':metrics[-1]['loss'],
      'round4_accuracy':metrics[3]['accuracy'],'round4_nll':metrics[3]['loss'],
      'reference_majority_accuracy':ref['constant_training_majority_accuracy'],
      'reference_prior_nll':ref['constant_training_prior_nll'],
      'traffic_bytes':sum(m['serialized_traffic_bytes'] for m in metrics),
      'training_gpu_seconds':sum(m['train_seconds']+m['probe_seconds'] for m in metrics),
      'evaluation_identity':fingerprint(part['tuning'])})
  except Exception as exc:failures.append({'job':j['id'],'reason':str(exc)})
 if len(common)>1 or any(len(v)>1 for v in per_seed.values()) or len(labels_hashes)>1:
  failures.append({'job':'matrix','reason':'source/data/environment or within-seed model/partition mismatch'})
 table=[]
 for trial in ('P_HH','H_H','R_HH','R_LH','R_HL','R_LL','H_L'):
  rows=[x for x in records if x['trial']==trial]
  if len(rows)!=2:continue
  table.append({'trial':trial,'seeds':[x['seed'] for x in rows],
    'mean_last5_accuracy':statistics.mean(x['last5_accuracy'] for x in rows),
    'mean_last5_nll':statistics.mean(x['last5_nll'] for x in rows),
    'mean_last5_balanced_accuracy':statistics.mean(x['last5_balanced_accuracy'] for x in rows),
    'tail_collapsed_rounds_out_of_10':sum(x['last5_all_one_class_rounds'] for x in rows),
    'each_seed_has_directional_signal':all(x['last5_accuracy']>x['reference_majority_accuracy'] and x['last5_nll']<x['reference_prior_nll'] for x in rows)})
 # This ordering is descriptive, fixed in advance, and never uses PCLA margins.
 ordered=sorted([x for x in table if x['trial'].startswith('R_')],key=lambda x:(-x['mean_last5_accuracy'],x['mean_last5_nll'],x['trial']))
 result={'scope':'train-only optimization development, seeds7/13; not confirmatory evidence',
      'complete_registered_batch':not failures,'planned_jobs':14,'completed_valid_jobs':len(records),
      'failures':failures,'records':records,'trial_table':table,'random_layer_candidate_order':ordered,
      'selection_status':'Descriptive proposal only. No automatic hyperparameter adoption or core-job launch.',
      'note':'Rounds are repeated observations, not independent seeds. Non-significance is not equivalence. Do not pool distinct learning-rate cells.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 if failures:raise SystemExit('Registered batch incomplete or inconsistent; inventory saved, no automatic selection.')
if __name__=='__main__':main()
