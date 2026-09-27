"""The queue's only job execution entry; no arbitrary shell commands in plans."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
from common import read,require,atomic_json
from context import prepare_real,synthetic_bundle,original_synthetic,original_environment
from job_plan import source_policy,personal_input,PERSONAL_SOURCES


def test_source(root,seed,policy):
    from common import bind_repo
    _,core,rt,_=bind_repo()
    root=Path(root)/f'seed_{seed}'
    path=root/policy/'checkpoint.pt'
    if path.exists():return rt.load_checkpoint(path),path
    b=synthetic_bundle(seed=seed)
    if policy=='replay_random_rank':
        _,donor=test_source(root.parent,seed,'pcla')
        rt.run_experiment(b.model,b.layout,b.clients,b.capacities,b.evaluation,b.cfg,seed,policy,
            path.parent,b.provenance,donor_path=donor,partition=b.partition)
        cp=rt.load_checkpoint(path)
    else:cp=original_synthetic(b,path.parent,policy)
    return cp,path


def test_control_source(site,source,seed):
    """Synthetic stand-in for a materialized, audited V4.1/S1 control run."""
    path=personal_input(site,source,seed)
    if source.startswith('old_') or path.exists():return path
    from run_lora_controls import run_bundle
    from audit_run import audit
    b=synthetic_bundle(seed=seed);b.source,b.source_path=test_source(site['source_root'],seed,'full_alt')
    run_bundle(b,path.parent,PERSONAL_SOURCES[source][1])
    atomic_json(path.parent.parent/'audit.json',audit(path.parent))
    return path


def execute(site,job,out,dependencies,resume=False,reason=None):
    test=bool(site.get('testing',False));root=site['source_root'];repo=site['repo'];threads=site.get('threads_per_gpu',4)
    old_policy=source_policy(job)
    role='tuning' if job['kind']=='diagnostic' else 'benchmark_dev'
    if test:
        original_environment('cpu',1)
        cp,path=test_source(root,job['seed'],old_policy)
        b=synthetic_bundle(seed=job['seed'],role=role);b.source,b.source_path=cp,path
    else:b=prepare_real(repo,root,job['seed'],old_policy,role,threads=threads)
    stop=2 if test and job.get('stop_after') else job.get('stop_after')
    if job['kind'] in ('fixed','online','online_replay'):
        from run_aggregation import run_bundle
        donor=Path(dependencies[job['donor']])/'run'/'run'/'checkpoint.pt' if job.get('donor') else None
        run_bundle(b,out,job['rule'],b.source,job['kind']!='fixed',donor,stop,resume,reason,job.get('parity',False))
    elif job['kind']=='diagnostic':
        from run_fixed_state import run_bundle
        if test:
            from common import tensor_load
            replay,_=test_source(root,job['seed'],'replay_random_rank')
            anchor=0;parent=tensor_load(b.source_path.parent/'probe_round_000.pt')['adapter']
        else:
            from context import load_original
            replay,_=load_original(root,job['seed'],'replay_random_rank',repo)
            anchor=job['anchor'];parent=None
        run_bundle(b,out,b.source,replay,anchor,parent,resume=resume,recovery_reason=reason)
    elif job['kind']=='control':
        from run_lora_controls import run_bundle
        run_bundle(b,out,job['mode'],stop,resume,reason)
    elif job['kind']=='personal_eval':
        from run_personal import run_bundle,local_tests
        path=test_control_source(site,job['source'],job['seed']) if test else personal_input(site,job['source'],job['seed'])
        run_bundle(b,out,job['source'],path,job['finetune'],local_tests(b,repo),stop,resume,reason)
    elif job['kind']=='fedex_ref':
        from fedex_ref import run_bundle
        run_bundle(b,out,job['agg'],job['partition'],repo,stop,resume,reason)
    else:raise ValueError('unknown job kind')
    from audit_run import audit
    report=audit(out,repo,require_complete=not bool(job.get('stop_after')))
    if job.get('parity') or job.get('mode')=='bridge_full_alt':
        require(read(Path(out)/'parity.json')['passed'],'technical parity did not pass')
    if job.get('stop_after'):
        expected=2 if test else job['stop_after']
        done=report['p2_completed'] if job['kind']=='personal_eval' else report['rows'][-1]['round']
        require(done==expected,'technical smoke horizon differs')
    atomic_json(Path(out).parent/'audit.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--job',required=True)
    p.add_argument('--out',required=True);p.add_argument('--dependencies',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--reason');a=p.parse_args()
    execute(read(a.site),read(a.job),a.out,read(a.dependencies),a.resume,a.reason)

if __name__=='__main__':main()
