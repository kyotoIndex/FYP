"""V5c open-question cells: personalization readouts (P) and the upstream FedEx-LoRA reference (R); technical gates only."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import BATCH,SEEDS,atomic_json,require

PERSONAL_SOURCES={'old_pcla':('source_root','pcla'),'old_full_alt':('source_root','full_alt'),
                  'fed_both_mean':('v41_materialized','fed_both_mean'),
                  'fed_ffa_b':('s1_materialized','fed_ffa_b'),'central_iid_both':('s1_materialized','central_iid_both')}
PERSONAL_FINETUNE=('old_pcla','old_full_alt','fed_both_mean')
FEDEX_AGGREGATIONS=('normal','ffa','ours','ours_fixed')
FEDEX_PARTITIONS=('noniid','iid')


def personal_job(source,seed,**extra):
    ft=source in PERSONAL_FINETUNE
    return {**dict(id=f"P_{source}{'_ft' if ft else ''}_s{seed}",family='personal',kind='personal_eval',source=source,
                   finetune=ft,seed=seed,stage='S1',pool='4090',depends=[],priority=4),**extra}


def fedex_job(partition,agg,seed,**extra):
    return {**dict(id=f'R_{partition}_{agg}_s{seed}',family='fedex_ref',kind='fedex_ref',partition=partition,agg=agg,
                   seed=seed,stage='S1',pool='A100',depends=[],priority=4),**extra}


def build():
    jobs=[{**personal_job('old_pcla',7,stop_after=2),'id':'smoke_P_old_pcla_ft_s7','stage':'S0','priority':2}]
    jobs+=[{**fedex_job('noniid',agg,7,stop_after=5),'id':f'smoke_R_noniid_{agg}_s7','stage':'S0','priority':2}
           for agg in FEDEX_AGGREGATIONS]
    for seed in SEEDS:
        jobs+=[personal_job(source,seed) for source in PERSONAL_SOURCES]
        jobs+=[fedex_job(partition,agg,seed,depends=[f'smoke_R_noniid_{agg}_s7'])
               for partition in FEDEX_PARTITIONS for agg in FEDEX_AGGREGATIONS]
    require(len(jobs)==70 and len({j['id'] for j in jobs})==70,'job inventory differs')
    return jobs


STAGE_DEPENDENCIES={'S0':[],'S1':[]}


def source_policy(job):
    if job['kind']=='personal_eval':return 'pcla' if job['source']=='old_pcla' else 'full_alt'
    if job['family']=='aggregation':return job.get('source','full_alt')
    return 'pcla' if job['kind']=='diagnostic' else 'full_alt'


def personal_input(site,source,seed):
    key,name=PERSONAL_SOURCES[source]
    if key=='source_root':return Path(site[key])/f'seed_{seed}'/name/'checkpoint.pt'
    return Path(site[key])/f'{name}_s{seed}'/'run'/'checkpoint.pt'


def job_inputs(site,job):
    """Every checkpoint a job reads, with how init validates it; frozen by SHA256 at queue init."""
    s=job['seed'];policy=source_policy(job)
    out={str(Path(site['source_root'])/f'seed_{s}'/policy/'checkpoint.pt'):('original',policy,s)}
    if job['kind']=='personal_eval':
        out.setdefault(str(personal_input(site,job['source'],s)),
                       ('original' if job['source'].startswith('old_') else 'control',PERSONAL_SOURCES[job['source']][1],s))
    return out


def validate(jobs):
    ids={j['id'] for j in jobs};require(len(ids)==len(jobs),'duplicate job IDs')
    for j in jobs:
        require(j['stage'] in STAGE_DEPENDENCIES and set(j['depends'])<=ids,'invalid dependencies')
        if j['kind']=='personal_eval':require(j['source'] in PERSONAL_SOURCES,'unknown personal source')
        if j['kind']=='fedex_ref':require(j['agg'] in FEDEX_AGGREGATIONS and j['partition'] in FEDEX_PARTITIONS,'unknown FedEx reference cell')
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
