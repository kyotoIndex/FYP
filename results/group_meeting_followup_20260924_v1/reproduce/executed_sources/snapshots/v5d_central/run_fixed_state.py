"""Common-parent interventions with retained local factors and compact coverage."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import random
import time
import traceback
import torch

from common import (SEEDS, require, read, atomic_json, atomic_torch, sha256,
                    fingerprint, exclusive, tensor_load)
from context import prepare_real, load_original, base_study, write_evaluation_identity
from aggregation import AggregationSnapshot, compare_updates, RULES
from run_aggregation import check_trace


def coverage_table(clients, total_weight, shape):
    """Weighted composition at each protocol-defined layer/rank position."""
    weights = torch.tensor([c['weight'] for c in clients], dtype=torch.float64)
    positive = torch.tensor([c['positive_samples'] for c in clients], dtype=torch.float64)
    masks = torch.stack([c['support'].double() for c in clients])
    capacities = sorted({c['capacity'] for c in clients})
    cov = (masks*weights[:,None,None]).sum(0)
    pos = (masks*positive[:,None,None]).sum(0)
    count = masks.sum(0)
    w2 = (masks*weights.square()[:,None,None]).sum(0)
    full_positive = float(positive.sum()/weights.sum())
    rows = []
    for l in range(shape[0]):
        for j in range(shape[1]):
            covered = float(cov[l,j]); ratio = float(pos[l,j]/cov[l,j]) if covered else None
            tier = {str(k): float(sum(c['weight'] for c in clients if c['capacity']==k and bool(c['support'][l,j])))/covered
                    if covered else None for k in capacities}
            population = {str(k): sum(c['weight'] for c in clients if c['capacity']==k)/total_weight for k in capacities}
            rows.append({'layer':l,'rank_id':j,'coverage_fraction':covered/total_weight,
                'contributor_count':int(count[l,j]), 'covered_weight':covered,
                'effective_contributor_count':covered**2/float(w2[l,j]) if covered else None,
                'positive_fraction':ratio, 'participants_positive_fraction':full_positive,
                'positive_shift':ratio-full_positive if ratio is not None else None,
                'capacity_weight_fractions':tier,'participant_capacity_weight_fractions':population})
    return rows


def run_bundle(bundle, out, source, replay, anchor, parent=None, stop_after_steps=None,
               resume=False, recovery_reason=None):
    from pcla_repair import core, runtime as rt
    from pcla_repair.optimization_dev import dataset_labels
    require(bundle.provenance['evaluation_split']=='tuning', 'fixed-state uses train-derived tuning only')
    require(source['identity_hash']==replay['identity_hash'], 'PCLA and replay original identity differ')
    require(anchor+2<=bundle.cfg['rounds'], 'anchor is too late')
    traces={'pcla':check_trace(source,bundle),'replay_random_rank':check_trace(replay,bundle)}
    if parent is None:
        require(bundle.source_path is not None, 'anchor path missing')
        p=rt.load_checkpoint(bundle.source_path.parent/f'probe_round_{anchor:03d}.pt')
        require(p['identity_hash']==source['identity_hash'] and p['round']==anchor, 'anchor differs')
        parent=p['adapter']
    out=Path(out);require(not out.exists() or resume,'use a new diagnostic directory')
    out.mkdir(parents=True,exist_ok=True)
    identity={**base_study(bundle),'kind':'fixed_state','anchor':anchor,'config':bundle.cfg,
              'parent_sha256':core.tensor_digest(parent),'replay_trace_sha256':core.tensor_digest(replay['trace']),
              'pcla_trace_sha256':core.tensor_digest(source['trace'])}
    ident=fingerprint(identity)
    scalings={n:float(m.scaling) for n,m in core.lora_modules(bundle.model).items()}
    parent={n:v.clone() for n,v in parent.items()}
    checkpoint=out/'checkpoint.pt'; completed=0; rows=[]
    with exclusive(out/'.run.lock',{'identity_hash':ident}):
        if resume:
            require(recovery_reason and checkpoint.exists(),'diagnostic resume needs reason/checkpoint')
            cp=tensor_load(checkpoint); require(cp['identity_hash']==ident,'diagnostic resume identity differs')
            for step in cp['commits']:
                for rel,digest in step.items():require(sha256(out/rel)==digest,'committed G0 evidence differs')
            parent,completed,rows,commits=cp['adapter'],cp['completed_steps'],cp['rows'],cp['commits']
            for d in (out/'rounds').glob('*'):
                if d.is_dir() and int(d.name)>anchor+completed:
                    dest=out/'recovery'/str(time.time_ns())/d.name;dest.parent.mkdir(parents=True,exist_ok=True);d.rename(dest)
        else:
            commits=[]
            atomic_json(out/'study.json',{'kind':'fixed_state','identity':identity,'identity_hash':ident})
            write_evaluation_identity(out,bundle)
            atomic_torch(out/'support_reference.pt', {
                'pcla':source['trace'], 'replay_random_rank':replay['trace']})
            atomic_torch(checkpoint,{'identity':identity,'identity_hash':ident,'adapter':parent,
                'completed_steps':0,'rows':[],'commits':[]})
        try:
            for step in range(completed,2):
                r=anchor+step+1; factor='b' if r%2 else 'a'
                participants=random.Random(core.keyed_seed(bundle.seed,'participants',r)).sample(
                    sorted(bundle.clients),bundle.cfg['selected_clients'])
                total=sum(len(bundle.clients[c]) for c in participants)
                round_dir=out/'rounds'/f'{r:03d}';round_dir.mkdir(parents=True,exist_ok=False)
                atomic_torch(round_dir/'parent.pt',{'adapter':parent,'scalings':scalings,'factor':factor,'round':r})
                snapshots={}; costs={}; full_bank={}
                for policy in ('pcla','replay_random_rank','full'):
                    snap=AggregationSnapshot(parent,total,scalings,factor)
                    bank=[]; client_seconds=0.
                    for cid in participants:
                        support=torch.ones(bundle.shape,dtype=torch.bool) if policy=='full' else traces[policy][f'{r}/{cid}']
                        core.load_adapter(bundle.model,parent); observation={}
                        masks,seconds,allocated=rt.local_train(bundle.model,bundle.layout,bundle.clients[cid],support,factor,
                            core.keyed_seed(bundle.seed,'train',r,cid),bundle.cfg,bundle.device,diagnostic=observation)
                        local=core.adapter_state(bundle.model)
                        snap.add(local,masks,len(bundle.clients[cid]),cid)
                        labels=dataset_labels(bundle.clients[cid])
                        bank.append({'client':cid,'weight':len(bundle.clients[cid]),'capacity':bundle.capacities[cid],
                            'positive_samples':int((labels==1).sum()),'support':support.clone(),
                            'active_state':{n:local[n] for n in masks},
                            'optimizer_steps':observation['optimizer_steps'],
                            'training_examples_seen':observation['training_examples_seen'],
                            'source_ids':bundle.partition['clients'][str(cid)],'train_seconds':seconds,
                            'optimizer_tensor_parameters':allocated})
                        require(observation['training_examples_seen']==len(labels)*bundle.cfg['local_epochs'],
                                'unexpected diagnostic data exposure')
                        client_seconds+=seconds
                    atomic_torch(round_dir/(policy+'_local.pt'),{'clients':bank,'layout':bundle.layout,'factor':factor,'shape':bundle.shape})
                    atomic_json(round_dir/(policy+'_coverage.json'),coverage_table(bank,total,bundle.shape))
                    snapshots[policy]=snap;costs[policy]=client_seconds;full_bank[policy]=bank
                reference,_=snapshots['full'].increments(); reference=reference['fixed']
                factual=None
                for policy,snap in snapshots.items():
                    increments,_=snap.increments()
                    for rule in (('fixed',) if policy=='full' else RULES):
                        state,metadata=snap.state(rule)
                        if policy=='pcla' and rule=='fixed':factual=state
                        core.load_adapter(bundle.model,state)
                        ev=rt.evaluate(bundle.model,bundle.layout,bundle.evaluation,bundle.cfg['eval_batch_size'],bundle.device,record_logits=True)
                        require(ev['labels']==bundle.labels,'tuning identity changed')
                        record={'seed':bundle.seed,'anchor':anchor,'round':r,'support':policy,'rule':rule,
                            'aggregation':snap.readout(rule),'evaluation':ev,
                            'to_full':compare_updates(increments[rule],reference,parent,scalings,factor),
                            'to_same_support_F':compare_updates(increments[rule],increments['fixed'],parent,scalings,factor),
                            'client_train_seconds':costs[policy]}
                        atomic_json(round_dir/f'{policy}_{rule}.json',record)
                        rows.append({'round':r,'support':policy,'rule':rule,'accuracy':ev['accuracy'],
                                     'nll':ev['nll_recomputed_float64'],'function_match_defined':metadata['function_match_defined']})
                require(factual is not None,'common factual parent missing')
                parent=factual
                files={p.relative_to(out).as_posix():sha256(p) for p in round_dir.iterdir() if p.is_file()}
                commits.append(files)
                atomic_torch(checkpoint,{'identity':identity,'identity_hash':ident,'adapter':parent,
                    'completed_steps':step+1,'rows':rows,'commits':commits})
                if stop_after_steps is not None and step+1>=stop_after_steps:break
            cp=tensor_load(checkpoint)
            atomic_json(out/'status.json',{'state':'completed' if cp['completed_steps']==2 else 'paused',
                        'completed_steps':cp['completed_steps']})
            return cp
        except BaseException:
            (out/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
            atomic_json(out/'status.json',{'state':'failed'});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--source-root',required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--anchor',type=int,choices=(5,15),required=True)
    p.add_argument('--out',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--reason')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();b=prepare_real(a.repo,a.source_root,a.seed,'pcla','tuning',threads=a.threads)
    replay,_=load_original(a.source_root,a.seed,'replay_random_rank',a.repo)
    run_bundle(b,a.out,b.source,replay,a.anchor,resume=a.resume,recovery_reason=a.reason)

if __name__=='__main__':main()
