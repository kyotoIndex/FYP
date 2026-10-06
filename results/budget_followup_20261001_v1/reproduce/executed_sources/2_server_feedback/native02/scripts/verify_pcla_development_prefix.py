#!/usr/bin/env python3
"""Compare round-4 prefixes already needed by the new 30-round HH runs.

Do not reuse an old four-round checkpoint as a 30-round resume checkpoint.
This comparison deliberately checks values, not equal protocol identities.
"""
import argparse,json,sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pcla_repair.runtime import load_checkpoint,atomic_json

def compare(old_run,new_run):
 a=load_checkpoint(old_run/'probe_round_004.pt');b=load_checkpoint(new_run/'probe_round_004.pt')
 if a['round']!=4 or b['round']!=4 or a['policy']!=b['policy'] or a['identity']['seed']!=b['identity']['seed']:raise ValueError('wrong prefixes')
 oldcfg=dict(a['identity']['config']);newcfg=dict(b['identity']['config'])
 if newcfg.pop('head_learning_rate')!=oldcfg['learning_rate'] or newcfg.pop('lora_learning_rate')!=oldcfg['learning_rate']:raise ValueError('only HH is comparable')
 newcfg.pop('record_optimization_diagnostics')
 for c in (oldcfg,newcfg):
  c.pop('rounds');c.pop('probe_rounds')
 if oldcfg!=newcfg:raise ValueError('training semantics changed beyond the allowed horizon and readouts')
 for k in ('partition_sha256','initial_loaded_model_sha256'):
  if a['identity']['provenance'][k]!=b['identity']['provenance'][k]:raise ValueError('data/model changed')
 if a['identity']['initial_adapter_sha256']!=b['identity']['initial_adapter_sha256']:raise ValueError('adapter initialization changed')
 if a['adapter'].keys()!=b['adapter'].keys() or a['trace'].keys()!=b['trace'].keys():raise ValueError('parameter/trace keys differ')
 exact=all(torch.equal(a['adapter'][k],b['adapter'][k]) for k in a['adapter'])
 trace=all(torch.equal(a['trace'][k],b['trace'][k]) for k in a['trace'])
 pred=all(json.loads((old_run/f'predictions_round_{r:03d}.json').read_text())==json.loads((new_run/f'predictions_round_{r:03d}.json').read_text()) for r in range(1,5))
 loss=[x['loss'] for x in a['metrics']]==[x['loss'] for x in b['metrics']]
 return {'policy':a['policy'],'seed':a['identity']['seed'],'adapter_bitwise_equal':exact,'trace_equal':trace,'predictions_equal':pred,'loss_equal':loss,'pass':exact and trace and pred and loss}

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=REPO)
 p.add_argument('--old-root',default='results/pcla_integration/seed_7');p.add_argument('--new-root',default='results/optimization_dev_20260918_v2')
 p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.repo.resolve()
 records=[]
 for trial,policy in [('P_HH','pcla'),('H_H','head_only')]:
  records.append(compare(r/a.old_root/policy,r/a.new_root/trial/'seed_7'/policy))
 result={'scope':'same-rate real prefix compatibility, not another training-efficacy gate','records':records,'pass':all(x['pass'] for x in records)}
 atomic_json(a.out,result)
 if not result['pass']:raise SystemExit('prefix changed; stop new development dispatch and inspect instrumentation')
if __name__=='__main__':main()
