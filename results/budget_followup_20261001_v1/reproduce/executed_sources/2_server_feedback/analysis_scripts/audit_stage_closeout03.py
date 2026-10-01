"""Independent stage integrity audit. Does not call exporters, selection or training."""
from pathlib import Path
import json,csv,hashlib,collections,math,datetime
p=Path(__file__).resolve().parent;r=p.parents[1]
def read(q):return json.loads(q.read_text())
def sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
outputs={};counts={}
for name in ['development_face02','validation_base01','quality_controller01','diagnostic_chain01','validation_summary01','diagnostic_summary01','execution_cost02','operational_supplement01','cost_source_audit02']:
 d=p/name;m=read(d/'MANIFEST.json');files={}
 for n,h in m['files'].items():assert sha(d/n)==h,(name,n);files[n]=h
 outputs[name]={'manifest_sha256':sha(d/'MANIFEST.json'),'files':files}
 for q in d.glob('*.csv'):counts[name+'/'+q.name]=sum(1 for _ in csv.DictReader(q.open()))
assert counts['development_face02/endpoints.csv']==128 and counts['development_face02/recipe_summary.csv']==32
assert counts['validation_base01/endpoints.csv']==108 and counts['quality_controller01/quality_points.csv']==324
assert counts['diagnostic_chain01/rules.csv']==1080 and counts['diagnostic_chain01/clients.csv']==2160 and counts['diagnostic_chain01/four_corners.csv']==864
identities={}
for name,filename,n,seeds in [('development_face02','accepted_originals.jsonl',128,[503,509,521,541]),('validation_base01','accepted_originals.jsonl',108,[601,607,613,617,619,631,641,643,647,653,659,661]),('diagnostic_chain01','metadata_originals.jsonl',120,[101,109,127,149,163,181,199,223,241,263,281,307])]:
 ids=[];ss=collections.Counter();phases=collections.Counter();calls=steps=0
 for line in (p/name/filename).open():
  j=json.loads(line);i=j['identity'];q=j['queue_original'];a=j['audit_original'];assert q['state']=='completed' and (q.get('audit_receipt') or q.get('audit_receipt_path')) and q.get('audit_receipt_sha256');assert a.get('accepted',a.get('status')=='accepted')
  ids.append(i['job_id']);ss[i['seed']]+=1;phases[i.get('phase','training')]+=1
  if name=='diagnostic_chain01':calls+=j['collection_original']['local_calls'];steps+=j['collection_original']['optimizer_steps']
 assert len(ids)==len(set(ids))==n and set(ss)==set(seeds),(name,ss)
 assert len(set(ss.values()))==1,(name,ss)
 identities[name]={'jobs':n,'seed_counts':dict(ss),'phases':dict(phases)}
 if name=='diagnostic_chain01':assert calls==240 and steps==3840 and phases['BANK_READOUT']==96;identities[name].update(new_calls=calls,new_steps=steps)
source_hashes={'core02':'a6c570095a1b88a2ef65865fc1f041cf0ed17114072686cff89a946bf970dc4e','native02':'d9fc9e074d856106ecea45ef4c2849ab6a2c0eb793cbbeb33b9dc04a45d498ea','native03':'4f89964f1fc8da23e3685cceebe11c39eb1b2bc4a54bfa3cb04459fc672f3e05','diagnostic_preflight02':'b4f1f1f117213e87b435417cc83fca948f26c37f70d6eb1c1321f7a5b0c1655b'};source_counts={}
for name,h in source_hashes.items():
 d=r/'sources'/name;m=read(d/'SOURCE_MANIFEST.json');assert sha(d/'SOURCE_MANIFEST.json')==h
 for rel,digest in m['files'].items():assert sha(d/rel)==digest,(name,rel)
 source_counts[name]=len(m['files'])
reviews={}
for name in ['VALIDATION_SUMMARY_INDEPENDENT_1941.json','DIAGNOSTIC_SUMMARY_INDEPENDENT_1941.json','DEVELOPMENT_OUTPUT_INDEPENDENT_2026.json','EXECUTION_COST_OUTPUT02_INDEPENDENT_2026.json']:
 j=read(p/name);assert j.get('accepted',j.get('status')=='accepted');reviews[name]=sha(p/name)
source_audit=read(p/'cost_source_audit02/SOURCE_AUDIT.json');assert source_audit['accepted'] and source_audit['timing_source_records']==265 and source_audit['source_input_hashes_checked']==273
viewrows=[json.loads(x) for x in (p/'cost_source_audit02/timing_original_key_views.jsonl').open()];assert len(viewrows)==265 and all(isinstance(v['event_count'],int) for v in viewrows)
cs={v['source_path']:v for v in viewrows}
for row in csv.DictReader((p/'execution_cost02/timing_attempts.csv').open()):
 v=cs[row['source_path']];assert row['source_sha256']==v['source_sha256'];assert math.isclose(float(row['measured_total_seconds']),v['original_key_cost_view']['total_seconds'],abs_tol=1e-7)
closure=read(p/'originals_intake01/ORIGINALS_CLOSURE.json');assert not closure['unclosed_metadata']
for row in closure['copies']:assert sha(Path(row['local_path']))==row['sha256']
expected={x['source_path'].removeprefix('$HPC_HOME/pcla_realized_update_feedback_20260930_v8/'):x['sha256'] for x in read(p/'execution_cost02/OPERATIONAL_ORIGINAL_INDEX.json')};expected.update({x['source_path'].removeprefix('$HPC_HOME/pcla_realized_update_feedback_20260930_v8/'):x['sha256'] for x in read(p/'operational_supplement01/ORIGINAL_INDEX.json')});expected.update({k.removeprefix('$HPC_HOME/pcla_realized_update_feedback_20260930_v8/'):v for k,v in read(p/'execution_cost02/INPUT_HASHES.json').items()})
for rel,local in closure['reused'].items():assert sha(Path(local))==expected[rel],rel
for row in closure['formal_event_arrays_remote']:assert row['sha256']==cs['$HPC_HOME/pcla_realized_update_feedback_20260930_v8/'+row['relative_path']]['source_sha256']
old=read(r/'checks/diagnostic_auditor02/cpu_audit61_batch12/LOCAL_REMOTE_HASH_VERIFICATION.json');assert old['status']=='all_matched' and old['artifact_count']==282
for row in old['artifacts']:assert sha(Path(row['matching_local_path']))==row['remote_sha256']
selection=r/'checks/validation_selection01/SELECTION.json';assert sha(selection)=='4dec551ef96807c4cbefc98fdaf1f83d7942d43a7411d7e38de64102abeeb2fe'
report={'schema':'pcla-v8-independent-phase-audit-1','accepted':True,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'matrices':identities,'outputs':outputs,'csv_rows':counts,'frozen_source_manifests':source_hashes,'source_files_verified':source_counts,'selection_sha256':sha(selection),'independent_recomputations':reviews,'original_cost_inputs_independently_verified_on_HPC':273,'original_timing_records':265,'cost_event_arrays_remote_with_hash_and_compact_original_key_view':len(closure['formal_event_arrays_remote']),'original_operating_partial_cost_files_local':len(closure['copies']),'operating_originals_reused':len(closure['reused']),'old_batch04_originals_closed':282,'retained_limitations':['continuous GPU occupancy and monetary charges not recorded','original cache/RNG coverage unknown where not measured','launcher inclusive time overlaps diagnostic components; do not add','native failed initialization is part of failure envelope','time comparisons preserve actual physical UUID matching counts','full event sequences remain on HPC; no PT/backbone/data copies in handoff'],'scientific_action':'Frozen continue threshold not met; no MNLI/f5/new-seed branch activated. Submit complete evidence for next concrete scientific decision.'}
(p/'PHASE_INDEPENDENT_AUDIT_2026.json').write_text(json.dumps(report,indent=2)+'\n');print({'accepted':True,'DEV':128,'DIAG':120,'VAL':108,'output_files':sum(len(x['files']) for x in outputs.values()),'frozen_source_files':sum(source_counts.values()),'originals_copy_and_reuse':len(closure['copies'])+len(closure['reused'])})
