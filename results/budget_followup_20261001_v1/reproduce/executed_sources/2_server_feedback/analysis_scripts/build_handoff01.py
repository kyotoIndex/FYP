from pathlib import Path
import json,hashlib,zipfile,difflib,datetime
p=Path(__file__).resolve().parent;r=p.parents[1];h=r.parents[1]/'handoffs/PCLA_V8_Completed_20261001';manifest=json.loads((p/'PHASE_INDEPENDENT_AUDIT_2026.json').read_text());assert manifest['accepted'];files={}
def add(path,logical):
 if path.is_file():
  if logical in files:assert files[logical]==path,(logical,path,files[logical])
  files[logical]=path
for dataset in manifest['outputs']:
 for q in (p/dataset).iterdir():
  if q.is_file():add(q,'outputs/'+dataset+'/'+q.name)
for q in p.iterdir():
 if q.is_file() and q.suffix in ['.py','.md','.json']:add(q,'audit_records/phase_closeout01/'+q.name)
for name in ['core02','native02','native03','diagnostic_preflight02']:
 d=r/'sources'/name;j=json.loads((d/'SOURCE_MANIFEST.json').read_text());add(d/'SOURCE_MANIFEST.json','source_snapshots/'+name+'/SOURCE_MANIFEST.json')
 for rel in j['files']:add(d/rel,'source_snapshots/'+name+'/'+rel)
for q in (r/'checks/validation_selection01').iterdir():
 if q.is_file() and q.suffix in ['.json','.md','.py','.csv']:add(q,'selection/'+q.name)
for n in ['scope_authorization.json','CONTINUATION.md','EXECUTION.md']:add(r/n,'execution/'+n)
for q in h.iterdir():
 if q.is_file() and q.suffix in ['.md','.json'] and q.name not in ['ZIP_VERIFICATION.json','PACKAGE_PLAN.json']:add(q,q.name)
closure=json.loads((p/'originals_intake01/ORIGINALS_CLOSURE.json').read_text())
for x in closure['copies']:
 if not Path(x['relative_path']).name.startswith('timing_attempt_'):add(Path(x['local_path']),'originals/'+x['relative_path'])
for rel,local in closure['reused'].items():
 if not Path(rel).name.startswith('timing_attempt_'):add(Path(local),'originals/'+rel)
for n in ['ORIGINALS_CLOSURE.json','SELF_TRANSFER_STOP_2026.json','COPY_PROGRESS_2026.json']:add(p/'originals_intake01'/n,'audit_records/originals_intake01/'+n)
old=r/'checks/diagnostic_auditor02/cpu_audit61_batch12';oldrep=json.loads((old/'LOCAL_REMOTE_HASH_VERIFICATION.json').read_text())
for x in oldrep['artifacts']:add(Path(x['matching_local_path']),'originals/batch04/'+x['relative_path'])
for q in old.glob('*.json'):
 if q.stat().st_size<1048576:add(q,'audit_records/batch04/'+q.name)
# Minimal readable source diff from previous accepted core to frozen V8 core.
baseline=r.parent/'pcla-global-budget-20260928-v7/sources/validation02';new=r/'sources/core02'
if (baseline/'SOURCE_MANIFEST.json').exists():
 bm=json.loads((baseline/'SOURCE_MANIFEST.json').read_text());nm=json.loads((new/'SOURCE_MANIFEST.json').read_text());parts=[]
 for rel in sorted(set(bm['files'])|set(nm['files'])):
  a=baseline/rel;b=new/rel
  if a.is_file() and b.is_file() and a.read_bytes()==b.read_bytes():continue
  try:before=a.read_text().splitlines(True) if a.exists() else [];after=b.read_text().splitlines(True) if b.exists() else []
  except UnicodeError:continue
  parts.extend(difflib.unified_diff(before,after,fromfile='V7_validation02/'+rel,tofile='V8_core02/'+rel))
 diff=h/'V7_TO_V8_CORE_SOURCE.patch';diff.write_text(''.join(parts));add(diff,diff.name);add(baseline/'SOURCE_MANIFEST.json','baseline/V7_validation02_SOURCE_MANIFEST.json')
zip_path=h/'PCLA_V8_Completed_20261001.zip';assert not zip_path.exists();index={};seen={};total=0
with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for logical,actual in sorted(files.items()):
  data=actual.read_bytes();digest=hashlib.sha256(data).hexdigest();obj='objects/'+digest
  index[logical]={'object':obj,'sha256':digest,'bytes':len(data)};total+=len(data)
  if digest not in seen:z.writestr(obj,data);seen[digest]=len(data)
 z.writestr('INDEX.json',json.dumps({'schema':'pcla-content-addressed-evidence-package-1','logical_paths':index,'entrypoint':['README.md','STAGE_REPORT.md','PROMPT_TO_PRO.md'],'full_timing_events':'HPC exact paths in outputs/cost_source_audit02; no PT/model/data copies'},indent=2))
 z.writestr('reconstruct_package.py',"""import argparse,json,zipfile,hashlib\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('zip');p.add_argument('out',type=Path);p.add_argument('--prefix',default='');a=p.parse_args();assert not a.out.exists();a.out.mkdir()\nwith zipfile.ZipFile(a.zip) as z:\n for rel,v in json.loads(z.read('INDEX.json'))['logical_paths'].items():\n  if not rel.startswith(a.prefix):continue\n  p=a.out/rel;assert '..' not in Path(rel).parts and not Path(rel).is_absolute();data=z.read(v['object']);assert hashlib.sha256(data).hexdigest()==v['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)\n""")
plan={'logical_files':len(index),'unique_content_objects':len(seen),'uncompressed_logical_bytes':total,'uncompressed_unique_bytes':sum(seen.values()),'zip_bytes':zip_path.stat().st_size,'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_stage_audit_sha256':hashlib.sha256((p/'PHASE_INDEPENDENT_AUDIT_2026.json').read_bytes()).hexdigest()};(h/'PACKAGE_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print(plan)
