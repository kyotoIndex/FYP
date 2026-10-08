"""Bounded local exploratory pilot, paired data/init/participation/dropout seeds.

No validation-set tuning. No overwrite/resume of historical experiments.
CPU stores only private adapters/heads, not 20 copies of the backbone.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import fedhello_sst2 as core
from hetsa_research import transport_private_b, suffix_layers, shared_a_update_round

METHODS = ['hetsa','repair_half','repair_full','slow_a','local_only','fixed_a_local_b']

def dump(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def append(path, row):
    existed = path.exists()
    with path.open('a',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row))
        if not existed: writer.writeheader()
        writer.writerow(row)

def state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters()
            if 'lora_' in n or n.startswith('classifier.')}

@torch.no_grad()
def apply(model, values):
    parameters=dict(model.named_parameters())
    for n,v in values.items(): parameters[n].copy_(v.to(parameters[n].device))

def sync():
    torch.cuda.synchronize()

@torch.no_grad()
def evaluate(model,loader):
    model.eval()
    core.set_active_layers(model,range(len(core.encoder_layers(model))))
    total=correct=0; loss=0.
    for batch in loader:
        batch={k:v.to('cuda') for k,v in batch.items()}
        output=model(**batch)
        n=batch['labels'].numel(); total+=n
        loss+=float(output.loss)*n
        correct+=int((output.logits.argmax(-1)==batch['labels']).sum())
    return dict(loss=loss/max(total,1),accuracy=correct/max(total,1),samples=total)

def suffix_key(name,layers):
    return any(f'.encoder.layer.{i}.' in name for i in layers)

def personal_values(method,private,global_state):
    if method=='local_only': return private
    return {n:(private[n] if n.endswith('lora_b') else global_state[n]) for n in global_state}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/hetsa_lora_confirm_extension_v1.json')
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--rounds',type=int,default=30)
    parser.add_argument('--methods',default=','.join(METHODS))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    out=Path(args.out_dir).resolve()
    if out.exists(): raise FileExistsError(f'New output directory required: {out}')
    out.mkdir(parents=True)
    start=time.perf_counter()
    status=dict(stage='preflight',seed=args.seed,methods=args.methods.split(','),pid=os.getpid())
    dump(out/'status.json',status)
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; refuse silent CPU training')
        torch.set_num_threads(4)
        cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
        cfg['federated']['rounds']=args.rounds
        cfg['federated']['seeds']=[args.seed]
        cfg['name']='hetsa_research_development_20261008'
        cfg['stage']='exploratory_smoke' if args.smoke else 'development_pilot'
        cfg['baselines']=args.methods.split(',')
        cfg['output_dir']=str(out)
        cfg['research']=dict(ridge_relative=1e-3,repair_strengths=[.5,1.],slow_a_warmup=3,
            slow_a_period=3,common_eval_samples=64 if args.smoke else 256,
            pairing='per-client-round minibatch and dropout RNG',cost='actual serialized parameter tensor size proxy; cached downlinks; no wire metadata')
        if args.smoke:
            cfg['federated'].update(total_clients=6,selected_clients=3,samples_per_client=128)
            cfg['dataset']['round_eval_samples']=64
        dump(out/'config.json',cfg)
        files=[ROOT/args.config,Path(__file__),ROOT/'src/hetsa_research.py',ROOT/'src/fedhello_sst2.py']
        files += [ROOT/cfg['dataset'][k] for k in ['train_file','dev_file']]
        dump(out/'provenance.json',dict(date_hkt='2026-10-08',python=sys.version,executable=sys.executable,
            torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),
            platform=platform.platform(),seed=args.seed,stage='exploratory_smoke' if args.smoke else 'development_pilot',
            files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
        core.set_seed(args.seed)
        tokenizer=AutoTokenizer.from_pretrained(cfg['dataset']['model'],local_files_only=True)
        train=core.read_text_rows(ROOT/cfg['dataset']['train_file'],cfg['dataset'])
        dev=core.read_text_rows(ROOT/cfg['dataset']['dev_file'],cfg['dataset'])
        train,proxy,test=core.prepare_data_splits(train,dev,cfg,args.seed)
        ratios=[sum(core.row_label(r)==label for r in train)/len(train) for label in range(4)]
        clients=core.make_clients(cfg,12,ratios[1],args.seed,ratios)
        rows=core.build_client_rows(train,clients,args.seed)
        training,holdout=core.split_client_holdout(rows,.2,args.seed)
        def dataset(rows): return core.SST2Dataset(rows,tokenizer,cfg['dataset']['max_length'])
        datasets={c.client_id:dataset(training[c.client_id]) for c in clients}
        hloaders={c.client_id:DataLoader(dataset(holdout[c.client_id]),batch_size=32,shuffle=False) for c in clients}
        balanced=core.stratified_subset(test,64 if args.smoke else 256,20261008)
        common=DataLoader(dataset(balanced),batch_size=32,shuffle=False)
        dump(out/'partition.json',dict(clients=[vars(c) for c in clients],
            hashes={str(cid):dict(train=hashlib.sha256(json.dumps(training[cid],ensure_ascii=False).encode()).hexdigest(),
                                 holdout=hashlib.sha256(json.dumps(holdout[cid],ensure_ascii=False).encode()).hexdigest()) for cid in training},
            common_eval_counts=[sum(core.row_label(r)==label for r in balanced) for label in range(4)]))
        core.set_seed(args.seed)
        model=AutoModelForSequenceClassification.from_pretrained(cfg['dataset']['model'],num_labels=4,local_files_only=True)
        core.inject_lora(model,16,16.,.1)
        model.to('cuda')
        initial=state(model)
        dump(out/'initialization.json',{n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in initial.items()})
        ids=[c.client_id for c in clients]; specs={c.client_id:c for c in clients}
        summaries=[]
        for method in args.methods.split(','):
            if method not in METHODS: raise ValueError(method)
            apply(model,initial)
            global_state={n:v.clone() for n,v in initial.items()}
            private={cid:{n:v.clone() for n,v in initial.items()} for cid in ids}
            last_train={cid:0 for cid in ids}
            values=[]; cumulative_train=cumulative_repair=cumulative_payload=0.
            method_start=time.perf_counter()
            for rnd in range(1,args.rounds+1):
                round_start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                selected=random.Random(args.seed*10000+rnd).sample(ids,cfg['federated']['selected_clients'])
                update_a=shared_a_update_round(method,rnd)
                uploads=[]; train_seconds=repair_seconds=payload=0.
                round_diagnostics=[]; local_scores=[]; balance_scores=[]
                for cid in selected:
                    active=suffix_layers(12,specs[cid].capacity)
                    current=personal_values(method,private[cid],global_state)
                    apply(model,current)
                    core.set_active_layers(model,active)
                    core.set_trainable_layers(model,active)
                    if not update_a and method!='local_only':
                        for n,p in model.named_parameters():
                            if n.endswith('lora_a'): p.requires_grad=False
                    generator=torch.Generator().manual_seed(args.seed*1000003+rnd*1009+cid)
                    loader=DataLoader(datasets[cid],batch_size=cfg['federated']['batch_size'],shuffle=True,generator=generator)
                    core.set_seed(args.seed*1000003+rnd*1009+cid)
                    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
                    model.train(); sync(); t=time.perf_counter()
                    for batch in loader:
                        optimizer.zero_grad(set_to_none=True)
                        batch={k:v.to('cuda') for k,v in batch.items()}
                        model(**batch).loss.backward(); optimizer.step()
                    sync(); train_seconds+=time.perf_counter()-t
                    trained=state(model)
                    private[cid]=trained; last_train[cid]=rnd
                    uploads.append((cid,active,trained))
                    if method!='local_only':
                        for n,p in trained.items():
                            if n.startswith('classifier.') or (update_a and n.endswith('lora_a') and suffix_key(n,active)):
                                payload+=p.numel()*p.element_size() # uplink
                    del optimizer
                if method!='local_only':
                    for n in global_state:
                        if n.startswith('classifier.'):
                            contributing=uploads
                        elif update_a and n.endswith('lora_a'):
                            contributing=[u for u in uploads if suffix_key(n,u[1])]
                        else: continue
                        if contributing:
                            total=sum(specs[cid].samples for cid,_,_ in contributing)
                            global_state[n]=sum(s[n]*(specs[cid].samples/total) for cid,_,s in contributing)
                # All clients receive current A for evaluation. Count actual cached-A changes,
                # including nonparticipants; this differs from old selected-only payload proxy.
                for cid in ids:
                    before=private[cid]
                    after=personal_values(method,before,global_state)
                    if method!='local_only':
                        active=suffix_layers(12,specs[cid].capacity)
                        for n,p in global_state.items():
                            communicated = n.startswith('classifier.') or (n.endswith('lora_a') and suffix_key(n,active))
                            if communicated and not torch.equal(before[n],p):
                                payload+=p.numel()*p.element_size() # one updated download per client
                    apply(model,before)
                    # Hold classifier head fixed to post-aggregation value in all 3 diagnostic states.
                    if method!='local_only': apply(model,{n:p for n,p in global_state.items() if n.startswith('classifier.')})
                    pre=evaluate(model,hloaders[cid])
                    apply(model,after); post=evaluate(model,hloaders[cid])
                    if method.startswith('repair_'):
                        strength=.5 if method=='repair_half' else 1.
                        sync(); t=time.perf_counter()
                        for n in after:
                            if n.endswith('lora_b'):
                                akey=n[:-len('lora_b')]+'lora_a'
                                after[n]=transport_private_b(before[akey],after[akey],before[n],strength)
                        apply(model,after); sync(); repair_seconds+=time.perf_counter()-t
                    repaired=evaluate(model,hloaders[cid])
                    after={n:v.clone() for n,v in after.items()}
                    private[cid]=after
                    common_score=evaluate(model,common)
                    record=dict(method=method,round=rnd,client_id=cid,capability=specs[cid].capability,
                        selected=cid in selected,missed_rounds=rnd-last_train[cid],
                        pre_loss=pre['loss'],post_loss=post['loss'],repair_loss=repaired['loss'],
                        pre_accuracy=pre['accuracy'],post_accuracy=post['accuracy'],repair_accuracy=repaired['accuracy'],
                        common_balanced_accuracy=common_score['accuracy'])
                    append(out/'client_diagnostics.csv',record); round_diagnostics.append(record)
                    local_scores.append(repaired['accuracy']); balance_scores.append(common_score['accuracy'])
                sync(); elapsed=time.perf_counter()-round_start
                cumulative_train+=train_seconds; cumulative_repair+=repair_seconds; cumulative_payload+=payload
                row=dict(method=method,round=rnd,seed=args.seed,local_accuracy=sum(local_scores)/len(ids),
                    balanced_accuracy=sum(balance_scores)/len(ids),a_updated=update_a,
                    mean_replacement_loss_change=sum(d['post_loss']-d['pre_loss'] for d in round_diagnostics)/len(ids),
                    mean_repair_loss_change=sum(d['repair_loss']-d['post_loss'] for d in round_diagnostics)/len(ids),
                    actual_serial_round_seconds=elapsed,train_seconds=train_seconds,repair_seconds=repair_seconds,
                    cumulative_train_seconds=cumulative_train,cumulative_repair_seconds=cumulative_repair,
                    parameter_payload_mib=payload/1024**2,cumulative_parameter_payload_mib=cumulative_payload/1024**2,
                    gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
                append(out/'round_metrics.csv',row); values.append(row)
                for cid in selected: append(out/'participation.csv',dict(method=method,round=rnd,client_id=cid))
                status.update(stage='running',method=method,round=rnd,rounds=args.rounds,
                    elapsed_seconds=time.perf_counter()-start,last_round=row)
                dump(out/'status.json',status)
                print(f'{method} {rnd}/{args.rounds} local={row["local_accuracy"]:.4f} balanced={row["balanced_accuracy"]:.4f} shock={row["mean_replacement_loss_change"]:+.4f} repair={row["mean_repair_loss_change"]:+.4f} round_s={elapsed:.1f}',flush=True)
                torch.save(dict(method=method,round=rnd,global_state=global_state,private=private,last_train=last_train),out/'latest_state.pt')
            summary=dict(method=method,seed=args.seed,last5_local_accuracy=sum(v['local_accuracy'] for v in values[-5:])/len(values[-5:]),
                last5_balanced_accuracy=sum(v['balanced_accuracy'] for v in values[-5:])/len(values[-5:]),
                actual_total_seconds=time.perf_counter()-method_start,train_seconds=cumulative_train,
                repair_seconds=cumulative_repair,parameter_payload_mib=cumulative_payload/1024**2)
            summaries.append(summary); dump(out/'method_summary.json',summaries)
            torch.save(dict(global_state=global_state,private=private),out/f'{method}_final.pt')
        status.update(stage='completed',elapsed_seconds=time.perf_counter()-start)
        dump(out/'status.json',status)
    except BaseException:
        status.update(stage='failed',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        dump(out/'status.json',status)
        raise

if __name__=='__main__': main()
