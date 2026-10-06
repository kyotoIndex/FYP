#!/usr/bin/env python3
"""Audit completed V executions and atomically attach audit receipts to the queue."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from dispatch_pcla_v7 import lock,read,write,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ids',nargs='*');p.add_argument('--limit',type=int,default=24)
a=p.parse_args();auditor=Path(__file__).with_name('audit_pcla_v7_run.py');code_sha=sha(auditor)
with lock(a.root/'validation_queue'):
 items=[(p,read(p)) for p in sorted((a.root/'validation_queue/jobs').glob('*.json'))]
items=[(p,v) for p,v in items if v['state']=='executed_pending_audit' and v['job'].get('phase')=='V' and (not a.ids or v['job']['job_id'] in a.ids)][:a.limit]
for path,v in items:
 jid=v['job']['job_id'];out=a.root/'checks/validation_audits'/f'{jid}.{code_sha[:12]}.{time.time_ns()}.json'
 argv=[sys.executable,str(auditor),'--run',v['out'],'--config',v['config'],'--source',v['source'],'--out',str(out),'--phase','V']
 proc=subprocess.run(argv,capture_output=True,text=True)
 if not out.is_file():raise RuntimeError(f'{jid}: auditor failed to produce a receipt: {proc.stderr[-1000:]}')
 report=read(out)
 if sha(auditor)!=code_sha:raise RuntimeError('Auditor changed during batch; queue not updated for this result')
 good=proc.returncode==0 and report.get('ok') is True and report.get('errors')==[] and report.get('checks',{}).get('planned_job_id')==jid and Path(report.get('run_dir','')).resolve()==Path(v['out']).resolve()
 receipt={'report':str(out),'report_sha256':sha(out),'auditor_sha256':code_sha,'argv':argv,'exit_code':proc.returncode,'time':time.time(),'config_sha256':sha(v['config']),'source_manifest_sha256':sha(Path(v['source'])/'SOURCE_MANIFEST.json')}
 with lock(a.root/'validation_queue'):
  live=read(path)
  if live!=v:raise RuntimeError(f'{jid}: queue record changed during audit')
  live.update(state='completed' if good else 'audit_failed_needs_review',audit=receipt);write(path,live)
 print(json.dumps({'job_id':jid,'state':live['state'],'errors':report.get('errors',[])[:4]}),flush=True)
