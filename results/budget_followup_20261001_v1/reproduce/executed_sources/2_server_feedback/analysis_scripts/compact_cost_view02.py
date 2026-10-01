from pathlib import Path
import json,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();c=a.root/'checks/phase_closeout01';old=c/'cost_source_audit01';out=c/'cost_source_audit02';assert not out.exists();out.mkdir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
m=json.loads((old/'MANIFEST.json').read_text());assert all(sha(old/n)==h for n,h in m['files'].items())
count=0
with (out/'timing_original_key_views.jsonl').open('w') as f:
 for line in (old/'timing_original_key_views.jsonl').open():
  v=json.loads(line);v['event_count']=len(v['event_count']);v['completed_spans_count']=len(v['completed_spans_count']);f.write(json.dumps(v,separators=(',',':'))+'\n');count+=1
assert count==265
(out/'SOURCE_AUDIT.json').write_bytes((old/'SOURCE_AUDIT.json').read_bytes());(out/'MANIFEST.json').write_text(json.dumps({'schema':'pcla-v8-independent-original-cost-view-2','count':265,'source_audit01_manifest_sha256':sha(old/'MANIFEST.json'),'change':'event_count and completed_spans_count are scalar lengths; no original-key cost fields change; full timing event originals remain at audited HPC paths','code_sha256':sha(Path(__file__)),'files':{x.name:sha(x) for x in out.iterdir()}},indent=2)+'\n');print({'records':count,'bytes':(out/'timing_original_key_views.jsonl').stat().st_size})
