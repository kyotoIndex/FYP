"""Independent source-level check; retains original-key cost view, full event originals on HPC."""
from pathlib import Path
import csv,json,hashlib,math,argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;c=r/'checks/phase_closeout01';src=c/'execution_cost02'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
inputs=json.loads((src/'INPUT_HASHES.json').read_text());source_checks=[]
for s,h in inputs.items():
 q=Path(s); assert q.is_file() and sha(q)==h,s;source_checks.append({'source_path':s,'sha256':h,'bytes':q.stat().st_size})
rows=list(csv.DictReader((src/'timing_attempts.csv').open()));compact=[]
for row in rows:
 q=Path(row['source_path']);v=json.loads(q.read_text());assert sha(q)==row['source_sha256'];assert math.isclose(sum(v['category_seconds'].values()),float(row['measured_total_seconds']),rel_tol=1e-9,abs_tol=1e-7)
 for k,x in v['category_seconds'].items():assert math.isclose(x,float(row[k]),rel_tol=1e-9,abs_tol=1e-7),(q,k)
 categories={}
 for bucket,labels in v['bucket_label_seconds'].items():
  for label,x in labels.items():categories[v['label_categories'][label]]=categories.get(v['label_categories'][label],0)+x
 for k,x in v['category_seconds'].items():assert math.isclose(x,categories.get(k,0),abs_tol=1e-7,rel_tol=1e-9),(q,k)
 view={k:v[k] for k in ['schema','label_categories','bucket_label_seconds','bucket_category_seconds','label_seconds','category_seconds','total_seconds','envelope_bounds','partial','open_intervals','tail_unmeasured','capture_sample_error'] if k in v}
 compact.append({'source_path':str(q),'source_sha256':sha(q),'original_bytes':q.stat().st_size,'event_count':v['events'],'completed_spans_count':v['completed_spans'],'original_key_cost_view':view})
assert len(compact)==265
assert not a.out.exists();a.out.mkdir()
with (a.out/'timing_original_key_views.jsonl').open('w') as f:
 for v in compact:f.write(json.dumps(v,separators=(',',':'))+'\n')
rep={'accepted':True,'source_input_hashes_checked':len(source_checks),'timing_source_records':len(compact),'independent_original_category_and_label_sums_checked':True,'full_event_sequences':'Preserved at exact source_path on HPC; source SHA256, size and original-key cost fields supplied without downloading 1.94GB event arrays','sources':source_checks}
(a.out/'SOURCE_AUDIT.json').write_text(json.dumps(rep,indent=2)+'\n');(a.out/'MANIFEST.json').write_text(json.dumps({'schema':'pcla-v8-independent-original-cost-view-1','audit_code_sha256':sha(Path(__file__)),'files':{f.name:sha(f) for f in a.out.iterdir()}},indent=2)+'\n');print({'accepted':True,'original_hashes_checked':len(source_checks),'timing_records':265})
