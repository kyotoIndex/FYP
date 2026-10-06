#!/usr/bin/env python3
"""One immutable PCLA server-feedback trajectory on a single GPU."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_pcla_v7 import prepare,persist_failed_cost_timing
from pcla_repair.runtime import atomic_json
from pcla_research_v7.runtime import run_experiment
from pcla_research_v7.execution_timing import ExecutionTimingLedger,COST_LABEL_CATEGORIES,required_breakdown

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path);p.add_argument('--out',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args(); cfg=json.loads(a.config.read_text());job=cfg['job']
    if a.source_root and a.source_root.resolve()!=ROOT: raise ValueError('runner source-root differs')
    out=a.out or a.root/'results'/job['job_id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() != a.resume: raise ValueError('resume must match existing checkpoint')
    if a.stop_after and job['phase']!='TECH': raise ValueError('stop only for technical trajectories')
    lock=out/'.v8_run.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'job':os.environ.get('SLURM_JOB_ID'),'step':os.environ.get('SLURM_STEP_ID')}).encode());os.close(fd)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    timing=ExecutionTimingLedger(COST_LABEL_CATEGORIES,synchronize=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else None)
    try:
        model,clients,caps,evalset,prov=prepare(cfg,a.config,device,timing=timing)
        prov['v8_source_manifest_sha256']=sha(ROOT/'SOURCE_MANIFEST.json')
        prov['server_rule']=cfg['recipe']['server_rule']
        controller=None
        rule=cfg['recipe']['server_rule']
        if rule not in {'mean_r8','mean_r4'}:
            from pcla_research_v8.server_control import apply_server_rule
            from pcla_research_v7.data import TokenCache
            mf=json.loads(Path(cfg['manifest']).read_text())
            tokens=TokenCache(Path(cfg['token_cache']),Path(cfg['token_cache_metadata']))
            public=tokens.dataset(mf,'public_training','algorithm') if cfg['recipe'].get('public_training') else None
            recipe=cfg['recipe'];seed=job['seed']
            def controller(**kw):
                return apply_server_rule(**kw,public_dataset=public,recipe=recipe,seed=seed)
        runtime=dict(cfg['runtime'])
        if cfg['recipe'].get('head_lr')!=cfg['recipe'].get('client_lr'): runtime['head_lr']=cfg['recipe']['head_lr']
        rows=run_experiment(model,clients,evalset,caps,job,runtime,out,prov,timing=timing,v8_mode=True,server_controller=controller,stop_after=a.stop_after)
        timing.end_envelope();record=timing.snapshot()
        atomic_json(out/f'timing_attempt_{len(json.loads((out/"execution_attempts.json").read_text())):03d}.json',record)
        atomic_json(out/'run_identity.json',{'job':job,'provenance':prov})
        atomic_json(out/'result.json',{'job_id':job['job_id'],'recipe':cfg['recipe'],'rounds':len(rows),'budget_return_round':rows[-1]['budget_stop_round'] if rows else None,'required_cost_breakdown':required_breakdown(record),'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'config_sha256':sha(a.config),'state':'executed_pending_audit' if not a.stop_after else 'technical_paused','gpu':prov['hardware']})
        with (out/'metrics.jsonl').open('w') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        with (out/'controller.jsonl').open('w') as f:
            for row in rows:
                if row['server_controller'] is not None: f.write(json.dumps({'round':row['round'],**row['server_controller']})+'\n')
        print(json.dumps({'job':job['job_id'],'rounds':len(rows),'out':str(out)}),flush=True)
    except BaseException:
        trace=traceback.format_exc();(out/'exception.txt').write_text(trace)
        persist_failed_cost_timing(out,timing,trace.splitlines()[-1]);raise
    finally: lock.unlink(missing_ok=True)
if __name__=='__main__': main()
