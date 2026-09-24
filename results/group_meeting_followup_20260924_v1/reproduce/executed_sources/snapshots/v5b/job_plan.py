"""V5b open-question batch: AG_NEWS controls (D), 100-round SST-2 PCLA (L2), SVD-refactored online PCLA (V4).
Technical S0 jobs gate only their own type; no accuracy-threshold selection."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import BATCH,SEEDS,CONFIG,DATASETS,atomic_json,read,require

CONTROLS=('central_iid_both','fed_both_mean','fed_both_exact')
RUNTIME=(('pcla','pcla'),('full_alt','full_alt'),('replay_random_rank','replay'))


def build():
    jobs=[]
    def add(id,**kw):jobs.append(dict(id=id,**kw))
    add('parity_online_pcla_fixed_s7',family='aggregation',kind='online',source='pcla',rule='fixed',parity=True,
        seed=7,stage='S0',pool='4090',depends=[],priority=1)
    for mode in CONTROLS:
        add(f'smoke_agnews_{mode}_s7',family='lora_control',kind='control',mode=mode,dataset='ag_news',seed=7,
            stop_after=5,stage='S0',pool='technical',depends=[],priority=2)
    for policy,tag in RUNTIME:
        extra=dict(donor='smoke_agnews_pcla_s7') if policy=='replay_random_rank' else {}
        add(f'smoke_agnews_{tag}_s7',family='runtime',kind='runtime_policy',policy=policy,dataset='ag_news',rounds=5,
            seed=7,stage='S0',pool='technical',depends=list(extra.values()),priority=2,**extra)
    add('smoke_agnews_online_C_s7',family='aggregation',kind='online',rule='contributors',dataset='ag_news',seed=7,
        stop_after=5,stage='S0',pool='technical',depends=[],priority=2)
    add('smoke_pcla_svd_s7',family='aggregation',kind='online',source='pcla',rule='fixed',refactor='svd',seed=7,
        stop_after=5,stage='S0',pool='technical',depends=[],priority=2)
    for seed in SEEDS:
        for mode in CONTROLS:
            add(f'D_agnews_{mode}_s{seed}',family='lora_control',kind='control',mode=mode,dataset='ag_news',seed=seed,
                stage='S1',pool='4090',depends=[f'smoke_agnews_{mode}_s7'],priority=4)
        for policy,tag in RUNTIME:
            extra=dict(donor=f'D_agnews_pcla_s{seed}') if policy=='replay_random_rank' else {}
            add(f'D_agnews_{tag}_s{seed}',family='runtime',kind='runtime_policy',policy=policy,dataset='ag_news',seed=seed,
                stage='S1',pool='4090',depends=[f'smoke_agnews_{tag}_s7',*extra.values()],
                priority=3 if policy=='pcla' else 4,**extra)
        add(f'D_agnews_online_C_s{seed}',family='aggregation',kind='online',rule='contributors',dataset='ag_news',seed=seed,
            stage='S1',pool='4090',depends=['smoke_agnews_online_C_s7'],priority=4)
        add(f'L2_pcla_r100_s{seed}',family='runtime',kind='runtime_policy',policy='pcla',dataset='sst2',rounds=100,
            prefix_parity=True,seed=seed,stage='S1',pool='4090',depends=['smoke_agnews_pcla_s7'],priority=3)
        add(f'V4_pcla_svd_s{seed}',family='aggregation',kind='online',source='pcla',rule='fixed',refactor='svd',seed=seed,
            stage='S1',pool='4090',depends=['smoke_pcla_svd_s7','parity_online_pcla_fixed_s7'],priority=4)
    require(len(jobs)==54 and len({j['id'] for j in jobs})==54,'job inventory differs')
    return jobs


STAGE_DEPENDENCIES={'S0':[],'S1':[]}


def parity_files(job):
    """Technical equalities a job must pass before it counts as completed."""
    return ((['parity.json'] if job.get('parity') or job.get('mode')=='bridge_full_alt' else [])+
            (['prefix_parity.json'] if job.get('prefix_parity') else []))


def job_inputs(job,site):
    """Files the job actually reads outside the locked code: dataset config/TSVs, old SST-2 PCLA sources."""
    repo,root=Path(site['repo']),Path(site['source_root'])
    spec=DATASETS[job.get('dataset') or 'sst2'];ds=read(repo/spec['config'])['dataset']
    paths=[repo/CONFIG,repo/spec['config'],repo/ds['train_file'],repo/ds['dev_file']]
    old=root/f"seed_{job['seed']}"
    if not job.get('dataset'):paths.append(old/job.get('source','full_alt')/'checkpoint.pt')
    if job.get('parity') or job.get('prefix_parity'):
        paths+=[old/'pcla'/'checkpoint.pt']+[old/'pcla'/f'predictions_round_{r:03d}.json'
                                            for r in range(1,read(repo/CONFIG)['training']['rounds']+1)]
    return sorted({str(p) for p in paths})


def validate(jobs):
    ids={j['id'] for j in jobs};require(len(ids)==len(jobs),'duplicate job IDs')
    for j in jobs:
        require(j['stage'] in STAGE_DEPENDENCIES and set(j['depends'])<=ids,'invalid dependencies')
        require(not j.get('donor') or j['donor'] in j['depends'],'donor must be a dependency')
    visiting,done=set(),set()
    lookup={j['id']:j for j in jobs}
    def visit(key):
        require(key not in visiting,'dependency cycle')
        if key in done:return
        visiting.add(key)
        j=lookup[key]
        deps=set(j['depends'])|{x['id'] for x in jobs if x['stage'] in STAGE_DEPENDENCIES[j['stage']]}
        for dep in deps:visit(dep)
        visiting.remove(key);done.add(key)
    for key in ids:visit(key)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();jobs=build();validate(jobs)
    atomic_json(a.out,{'batch':BATCH,'jobs':jobs,'count':len(jobs),'stage_dependencies':STAGE_DEPENDENCIES,
        'gate_type':'technical_and_integrity_only; no accuracy-threshold selection'})

if __name__=='__main__':main()
