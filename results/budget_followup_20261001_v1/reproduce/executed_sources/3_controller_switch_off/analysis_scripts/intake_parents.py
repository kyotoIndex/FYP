"""Read-only accepted V8 parent inventory; no model training or reconstruction."""
import argparse,hashlib,json,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def require(v,w):
 if not v:raise RuntimeError(w)
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-root',type=Path,required=True);a.add_argument('--matrix',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args()
 sys.path.insert(0,str(x.parent_root/'sources/core02'))
 import torch
 from pcla_research_v8.server_control import _hash_state as tensor_digest
 torch.set_num_threads(4)
 src=x.parent_root/'sources/core02/SOURCE_MANIFEST.json';sm=json.loads(src.read_text())
 for p,s in sm['files'].items():require(sha(src.parent/p)==s,'frozen parent source differs: '+p)
 rows=json.loads(x.matrix.read_text());records=[]
 for v in rows:
  jid=v['source_job_id'];qp=x.parent_root/'queues/validation01/jobs'/(jid+'.json');q=json.loads(qp.read_text());cp=Path(q['config']);cfg=json.loads(cp.read_text());out=Path(q['output'])
  require(q['state']=='completed' and q['return_code']==0,'parent not accepted '+jid)
  require(q['recipe_id']==v['source_recipe'] and q['seed']==int(v['seed']),'parent identity differs')
  require(sha(cp)==q['config_sha256']==v['parent_config_sha256'],'parent config differs')
  require(sha(src)==v['parent_source_manifest_sha256']==cfg['binding']['source_manifest_sha256'],'parent source differs')
  receipt=Path(q['audit_receipt']); require(sha(receipt)==q['audit_receipt_sha256'],'parent audit changed')
  ar=json.loads(receipt.read_text());require(ar.get('accepted') is True,'parent audit rejected')
  require(sha(Path(q['result_path']))==q['result_sha256'],'parent result changed')
  ckpath=out/'boundary_round_060.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=True)
  require(ck['format']=='pcla-v7-runtime-1' and ck['round']==60,'not full r60 checkpoint')
  require(tensor_digest(ck['adapter'])==v['parent_state_sha256'],'r60 model state differs')
  required={'cache','torch_rng','python_rng','cuda_rng','ema_effective','deployed_bytes','diagnostic_bytes','budget_stop_round','half_adapter','half_round','half_readout','metrics','identity'}
  require(required<=set(ck),'incomplete recovery state')
  require(ck['budget_stop_round'] is None and len(ck['metrics'])==60 and [r['round'] for r in ck['metrics']]==list(range(1,61)),'invalid r60 prefix')
  require(ck['torch_rng'].numel()>0 and ck['cuda_rng'] and all(t.numel()>0 for t in ck['cuda_rng']),'missing RNG')
  require({'clients','events'}<=set(ck['cache']) and len(ck['cache']['clients'])==50,'missing cache')
  metrics=out/'metrics.json';mr=json.loads(metrics.read_text());require(mr[59]['server_controller']['committed_state_hash']==v['parent_state_sha256'],'control boundary differs')
  require([r['round'] for r in mr[:80]]==list(range(1,81)),'missing on-policy control')
  identity=out/'run_identity.json';journal=out/'communication_journal.json';labels=out/'evaluation_labels.json'
  prefix=[e for e in json.loads(journal.read_text())['entries'] if e['round']<=60]
  require(len(prefix)==60 and all(e['status']=='committed' for e in prefix),'parent journal not complete')
  rec={**v,'seed':int(v['seed']),'parent_checkpoint_path':str(ckpath),'parent_checkpoint_file_sha256':sha(ckpath),'parent_checkpoint_identity':ck['identity'],'parent_config_path':str(cp),'control_metrics_path':str(metrics),'control_metrics_sha256':sha(metrics),'parent_run_identity_path':str(identity),'parent_run_identity_sha256':sha(identity),'parent_journal_path':str(journal),'parent_journal_sha256':sha(journal),'parent_labels_path':str(labels),'parent_labels_sha256':sha(labels),'parent_queue_path':str(qp),'parent_queue_sha256':sha(qp),'parent_audit_receipt_path':str(receipt),'parent_audit_receipt_sha256':sha(receipt),'data_identity_sha256':cfg['binding']['data_manifest_sha256'],'status':'parent_verified','recovery_fields':sorted(required),'cache_clients':len(ck['cache']['clients']),'cache_events':len(ck['cache']['events']),'prefix_deployed_bytes':ck['deployed_bytes'],'prefix_diagnostic_bytes':ck['diagnostic_bytes']}
  records.append(rec);print(json.dumps({'parent_verified':jid,'count':len(records)}),flush=True)
 require(len(records)==24 and len({r['job_id'] for r in records})==24,'matrix incomplete')
 x.out.parent.mkdir(parents=True,exist_ok=True)
 with x.out.open('x') as f:json.dump({'accepted':True,'matrix_sha256':sha(x.matrix),'parent_source_sha256':sha(src),'records':records},f,indent=2)
 print(json.dumps({'accepted':True,'records':24,'receipt_sha256':sha(x.out)}))
if __name__=='__main__':main()
