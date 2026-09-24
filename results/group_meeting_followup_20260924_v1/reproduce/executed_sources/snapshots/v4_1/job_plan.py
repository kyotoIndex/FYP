"""Fixed V4 scientific cells; technical stages do not select favorable results."""
from __future__ import annotations
import argparse
from common import BATCH,SEEDS,atomic_json,require


def build():
    jobs=[]
    parity=[]
    for source in ('pcla','replay_random_rank'):
        name=f'parity_{source}_seed7';parity.append(name)
        jobs.append(dict(id=name,family='aggregation',kind='fixed',source=source,rule='fixed',
            seed=7,parity=True,stage='A',pool='4090',depends=[],priority=0))
    for seed in SEEDS:
        for anchor in (5,15):
            jobs.append(dict(id=f'fixed_state_s{seed}_a{anchor}',family='aggregation',kind='diagnostic',
                seed=seed,anchor=anchor,stage='A',pool='4090',depends=[],priority=1))
    for seed in SEEDS:
        stage='B' if seed in (7,13) else 'C'
        for source in ('pcla','replay_random_rank','random'):
            for rule in ('contributors','function_norm_matched'):
                jobs.append(dict(id=f'trace_{source}_{rule}_s{seed}',family='aggregation',kind='fixed',
                    source=source,rule=rule,seed=seed,stage=stage,pool='4090',depends=parity,priority=4))
        name=f'online_pcla_contributors_s{seed}'
        jobs.append(dict(id=name,family='aggregation',kind='online',source='pcla',rule='contributors',
            seed=seed,stage=stage,pool='4090',depends=parity,priority=3))
        jobs.append(dict(id=f'online_replay_contributors_s{seed}',family='aggregation',kind='online_replay',
            source='pcla',rule='contributors',seed=seed,stage=stage,pool='4090',depends=[name],donor=name,priority=3))
    modes=('central_pooled_both','central_serial_both','central_serial_alt','fed_both_mean','fed_both_exact')
    smokes=[]
    for mode in ('bridge_full_alt','central_pooled_both','fed_both_mean','fed_both_exact'):
        name=f'smoke_{mode}_s7';smokes.append(name)
        jobs.append(dict(id=name,family='lora_control',kind='control',mode=mode,seed=7,
            stop_after=5,stage='M0',pool='4090' if mode=='bridge_full_alt' else 'technical',depends=[],priority=2))
    for seed in SEEDS:
        for mode in modes:
            jobs.append(dict(id=f'{mode}_s{seed}',family='lora_control',kind='control',mode=mode,seed=seed,
                stage='M1' if seed in (7,13) else 'M2',pool='4090',depends=smokes,priority=4))
    require(len(jobs)==81 and len({j['id'] for j in jobs})==81,'job inventory differs')
    return jobs


# Attempt 2 restores the V4 scheduling rule "do not enforce a global barrier that leaves cards
# idle": each job keeps only its task-level technical dependencies (parity, donor, smokes).
# Stage labels stay for reporting; no stage waits for a whole earlier stage.
STAGE_DEPENDENCIES={'A':[],'M0':[],'B':[],'C':[],'M1':[],'M2':[]}


def validate(jobs):
    ids={j['id'] for j in jobs};require(len(ids)==len(jobs),'duplicate job IDs')
    for j in jobs:
        require(j['stage'] in STAGE_DEPENDENCIES and set(j['depends'])<=ids,'invalid dependencies')
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
