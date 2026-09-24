#!/usr/bin/env python3
"""Five head-only, fixed-checkpoint interventions; tuning only.

This is a privileged offline diagnostic, not a deployable federated policy or
an equal-cost benchmark. It reuses existing round-2 and round-4 artifacts.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from pcla_repair.bridge import prepare
from pcla_repair.core import adapter_state,load_adapter,tensor_digest,keyed_seed,DeltaAccumulator,lora_modules
from pcla_repair.runtime import load_checkpoint,local_train,evaluate,atomic_json,atomic_checkpoint
from pcla_repair.optimization_dev import dataset_labels,adapter_readout,constant_references


class RemixedDataset(Dataset):
    def __init__(self, clients, refs):
        self.clients,self.refs=clients,list(refs)
    def __len__(self):return len(self.refs)
    def __getitem__(self,index):
        cid,i=self.refs[index]
        return self.clients[cid][i]


def remix_refs(clients, selected, seed):
    """Spread binary labels almost uniformly, preserving exact union and sizes."""
    if not selected or len(set(selected))!=len(selected):raise ValueError("invalid selected clients")
    sizes=[len(clients[c]) for c in selected]
    if len(set(sizes))!=1:raise ValueError("this registered intervention requires equal client sizes")
    buckets={0:[],1:[]}
    for cid in selected:
        for i,label in enumerate(dataset_labels(clients[cid]).tolist()):
            if label not in buckets:raise ValueError("binary SST-2 diagnostic only")
            buckets[label].append((cid,i))
    rng=random.Random(seed)
    for values in buckets.values():rng.shuffle(values)
    q,rem=divmod(len(buckets[0]),len(selected));start={0:0,1:0};result={}
    for j,cid in enumerate(selected):
        n0=q+int(j<rem);counts={0:n0,1:sizes[j]-n0};group=[]
        for label in (0,1):
            group.extend(buckets[label][start[label]:start[label]+counts[label]])
            start[label]+=counts[label]
        rng.shuffle(group);result[cid]=group
    union=[x for group in result.values() for x in group]
    expected=[x for group in buckets.values() for x in group]
    if len(union)!=len(set(union)) or set(union)!=set(expected):raise RuntimeError("remix lost/reused source IDs")
    if any(len(result[c])!=len(clients[c]) for c in selected):raise RuntimeError("remix changed client size")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pcla_repair/integration.json')
    p.add_argument('--existing-run',type=Path,default=Path('results/pcla_integration/seed_7/head_only'))
    p.add_argument('--device',default='cuda');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('diagnostic output exists; preserve it and use a new attempt directory')
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan_path=Path(a.config);plan_path=plan_path if plan_path.is_absolute() else REPO/plan_path
    plan=json.loads(plan_path.read_text())
    if plan['evaluation_split']!='tuning':raise ValueError('benchmark dev is prohibited here')
    oldrun=a.existing_run if a.existing_run.is_absolute() else REPO/a.existing_run
    cp_path=oldrun/'probe_round_002.pt';cp=load_checkpoint(cp_path)
    if cp['round']!=2 or cp['policy']!='head_only' or cp['identity']['seed']!=7:raise ValueError('wrong diagnostic parent')
    model,layout,clients,_,evaluation,prov,partition=prepare(REPO,plan,7,torch.device(a.device))
    for k in ('partition_sha256','initial_loaded_model_sha256','plan_sha256'):
        if prov[k]!=cp['identity']['provenance'][k]:raise ValueError(f'parent identity differs: {k}')
    if tensor_digest(adapter_state(model))!=cp['identity']['initial_adapter_sha256']:raise ValueError('initial adapter differs')
    oldmetrics=json.loads((oldrun/'metrics.json').read_text())
    selected3=oldmetrics[2]['selected_clients'];selected4=oldmetrics[3]['selected_clients']
    cfg=dict(plan['training']);cfg.update(head_learning_rate=.001,lora_learning_rate=.001)
    rank=next(iter(lora_modules(model).values())).rank
    support=torch.zeros(max(layout.values())+1,rank,dtype=torch.bool)
    base=cp['adapter'];a.out.mkdir(parents=True)
    def assess(name,state,ids,role,extra=None):
        load_adapter(model,state)
        result=evaluate(model,layout,evaluation,cfg['eval_batch_size'],torch.device(a.device),record_logits=True)
        logits=result.pop('logits');labels=result.pop('labels');pred=result.pop('predictions')
        atomic_json(a.out/f'{name}_logits.json',{'logits':logits,'labels':labels,'predictions':pred})
        result.update(name=name,participants=ids,role=role,adapter_diagnostics=adapter_readout(state,base))
        if extra:result.update(extra)
        atomic_checkpoint(a.out/f'{name}_head.pt',{'format':'pcla-head-intervention-v1',
                              'head':{k:v for k,v in state.items() if k.startswith('classifier.')},'parent_identity':cp['identity_hash']})
        return result,pred
    records=[]
    before,_=assess('before_update',base,[], 'unchanged parent reference');records.append(before)
    groups={'selected_round3':selected3,'selected_round4':selected4,'all_50':sorted(clients)}
    acc={name:DeltaAccumulator(base,sum(len(clients[c]) for c in ids),'fixed') for name,ids in groups.items()}
    localrecords=[]
    # Every update starts from the SAME round-2 state. Same keyed randomness for
    # a given client in all reaggregations; round4 here is not a new trajectory.
    for cid in sorted(clients):
        load_adapter(model,base);diag={}
        masks,elapsed,_=local_train(model,layout,clients[cid],support,'none',keyed_seed(7,'train',3,cid),cfg,torch.device(a.device),diagnostic=diag)
        local=adapter_state(model)
        localrecords.append({'client':cid,'optimizer':diag,'adapter_update':adapter_readout(local,base),'train_seconds':elapsed})
        for name,ids in groups.items():
            if cid in ids:acc[name].add(local,masks,len(clients[cid]))
    for name,ids in groups.items():
        result,pred=assess(name,acc[name].finish(),ids,'offline reaggregation; all_50 has 5x client training cost')
        records.append(result)
        if name=='selected_round3':
            original_pred=json.loads((oldrun/'predictions_round_003.json').read_text())
            original_loss=oldmetrics[2]['loss']
            if pred!=original_pred or abs(result['loss']-original_loss)>1e-6:
                atomic_json(a.out/'factual_mismatch.json',{'old_loss':original_loss,'new_loss':result['loss'],'predictions_equal':pred==original_pred})
                raise RuntimeError('factual round3 is not reproduced; stop interpreting interventions')
    remixed=remix_refs(clients,selected4,keyed_seed(7,'registered-remix',20260918))
    conditions=[('selected_round4_remixed',.001,True),('selected_round4_head_lr_low',.0001,False)]
    for name,rate,mix in conditions:
        conf=dict(cfg);conf['head_learning_rate']=rate
        aggregator=DeltaAccumulator(base,sum(len(clients[c]) for c in selected4),'fixed')
        for cid in selected4:
            ds=RemixedDataset(clients,remixed[cid]) if mix else clients[cid]
            load_adapter(model,base);diag={}
            masks,elapsed,_=local_train(model,layout,ds,support,'none',keyed_seed(7,'train',3,cid),conf,torch.device(a.device),diagnostic=diag)
            local=adapter_state(model);aggregator.add(local,masks,len(ds))
            localrecords.append({'condition':name,'client':cid,'optimizer':diag,'adapter_update':adapter_readout(local,base),'train_seconds':elapsed})
        result,_=assess(name,aggregator.finish(),selected4,'offline intervention from SAME parent',{'head_learning_rate':rate,'remixed':mix})
        records.append(result)
    atomic_json(a.out/'remixed_source_ids.json',{str(cid):[partition['clients'][str(c)][i] for c,i in refs] for cid,refs in remixed.items()})
    atomic_json(a.out/'local_updates.json',localrecords)
    source_files={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((REPO/'pcla_repair').glob('*.py'))}
    source_files[str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(a.out/'interventions.json',{'scope':'seed7 head-only fixed-round2 diagnosis; not independent seeds or rank evidence',
          'parent_checkpoint_sha256':hashlib.sha256(cp_path.read_bytes()).hexdigest(),'parent_identity_hash':cp['identity_hash'],
          'new_sources':source_files,'reference':constant_references(clients,evaluation),
          'local_training_calls':70,'optimizer_steps':sum(x['optimizer']['optimizer_steps'] for x in localrecords),
          'factual_round3_reproduced':True,'conditions':records,
          'interpretation':'Changing selected sets changes client composition as a whole; it does not isolate label ratio alone. Remix changes within-client composition/order but preserves the selected source union and total steps. All50 is a higher-cost reference.'})
    print(json.dumps({'output':str(a.out),'conditions':len(records),'local_training_calls':70}))
if __name__=='__main__':main()
