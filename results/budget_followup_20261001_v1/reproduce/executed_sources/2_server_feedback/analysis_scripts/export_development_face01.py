"""Export the 128 accepted DEV originals against the existing frozen selection."""
import argparse,csv,json,statistics
from pathlib import Path
import export_validation_base01 as b
SELECTION_SHA='4dec551ef96807c4cbefc98fdaf1f83d7942d43a7411d7e38de64102abeeb2fe'
SNAPSHOT_SHA='8f2fab18fee7e0459fa754d4d5c145b57e1d8bb27e23240a6bbcf57375485ede'
SEEDS={503,509,521,541}
WINNERS=['U002','U010','U014','U018','U020','U023','U027','U004','U006']
def require_matrix(rows):
    cells={(r['recipe_id'],r['seed']) for r in rows}
    b.require(len(rows)==128 and cells=={(f'U{x:03d}',s) for x in range(1,33) for s in SEEDS},'complete 32x4 development matrix')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sp=a.root/'checks/validation_selection02/SELECTION.json';sn=a.root/'checks/validation_selection02/ACCEPTED_DEV_ENDPOINT_SNAPSHOT.json';rv=a.root/'checks/validation_selection02/ROOT_SELECTION_REVIEW.json'
    b.require(b.digest(sp)==SELECTION_SHA and b.digest(sn)==SNAPSHOT_SHA,'original selection/snapshot pins')
    sel=b.read(sp);snap=b.read(sn);review=b.read(rv);b.require(review['accepted'] and review['selection_sha256']==SELECTION_SHA and review['snapshot_sha256']==SNAPSHOT_SHA,'independent original selection review')
    b.require([x['recipe_id'] for x in sel['winners']]==WINNERS,'nine original winners')
    require_matrix(sel['evidence_rows']);evidence={str(p):b.digest(p) for p in [sp,sn,rv]};records=[];endpoints=[];costs=[];curves=[]
    queuepaths={p.stem:p for p in (a.root/'queues/development01/jobs').glob('*.json')};b.require(len(queuepaths)==128 and set(queuepaths)==set(snap['jobs']),'development queue/snapshot identity')
    for row in sel['evidence_rows']:
        jid=f"DEV_{row['recipe_id']}_s{row['seed']}";qp=queuepaths[jid];q=b.read(qp);ss=snap['jobs'][jid]
        b.require(q['state']=='completed' and q['return_code']==0 and row['independent_accepted'] and row['exclusive_4090D_four_cpu'],'completed development')
        paths=[Path(q[k]) for k in ['config','result_path','audit_receipt']];c,v,audit=map(b.read,paths)
        for field,path in zip(['config_sha256','result_sha256','audit_receipt_sha256'],paths):b.require(b.digest(path)==q[field]==ss[field],'snapshot accepted binding '+jid);evidence[str(path)]=q[field]
        b.require(audit['accepted'] and not audit['issues'] and audit['result_sha256']==q['result_sha256'],'accepted audit')
        b.require(q['recipe_id']==row['recipe_id']==c['recipe']['recipe_id'] and q['seed']==row['seed']==c['job']['seed'] and c['recipe']==v['recipe'],'development recipe binding')
        b.require(b.digest(qp)==ss['queue_record_sha256'],'sealed development queue');evidence[str(qp)]=b.digest(qp)
        source=Path(c['source_manifest']);b.require(b.digest(source)==c['binding']['source_manifest_sha256']==v['source_manifest_sha256'],'source manifest');evidence[str(source)]=b.digest(source)
        for rel,h in c['binding']['source_file_hashes'].items():b.require(b.digest(source.parent/rel)==h,'source original');evidence[str(source.parent/rel)]=h
        out=Path(q['output']);mp=out/'metrics.json';b.require(b.digest(mp)==ss['artifacts']['metrics.json'],'sealed original metrics');evidence[str(mp)]=b.digest(mp);m=b.read(mp)
        window=b.quality_window(m,v,c['runtime']['byte_cap']);b.require(window['returned_round']==ss['budget_return_round'] and m[-1]['evaluation']['samples']==row['samples']==4096 and window['raw_B8_accuracy']==row['correct']/row['samples'],'original B8 exact count')
        b.require(v['required_cost_breakdown']==audit['checks']['original_cost_fields']['required_cost_breakdown'],'original required cost fields')
        original_seconds=b.required_time(v);ident={'job_id':jid,'recipe_id':row['recipe_id'],'seed':row['seed'],'method':c['recipe']['method'],'gpu_uuid':q['claim']['gpu_uuid'],'hardware_family':q['hardware_family'],'source_manifest_sha256':v['source_manifest_sha256'],'selected_for_VAL':row['recipe_id'] in WINNERS}
        endpoints.append({**ident,**window,'correct':row['correct'],'samples':row['samples'],'required_prefix_seconds_selection':row['required_prefix_seconds'],'original_required_seconds':original_seconds,'selection_timing_coverage':'exact_single_attempt' if row['required_prefix_seconds'] is not None else 'not_qualified_for_tie'})
        costs.append({**ident,**v['required_cost_breakdown'],'original_required_seconds':original_seconds,'selection_tie_seconds':row['required_prefix_seconds']})
        records.append({'identity':ident,'result_original':v,'audit_original':audit,'config_original':c,'queue_original':q,'selection_evidence_original':row,'raw_snapshot_original':ss})
        for x in m:curves.append({**ident,'round':x['round'],'deployed_bytes':x['deployed_bytes'],'raw_accuracy':x['evaluation']['accuracy'],'raw_NLL':x['evaluation']['loss'],'EMA_accuracy':x['ema_evaluation']['accuracy'],'EMA_NLL':x['ema_evaluation']['loss']})
    b.require(not a.out.exists(),'new development export directory');a.out.mkdir(parents=True)
    for name,rows in [('endpoints.csv',endpoints),('required_costs.csv',costs),('curves.csv',curves)]:b.write_csv(a.out/name,rows)
    means=[]
    for rid in sorted({r['recipe_id'] for r in endpoints}):
        rows=[r for r in endpoints if r['recipe_id']==rid];means.append({'recipe_id':rid,'seeds':4,'mean_raw_B8_accuracy':statistics.mean(r['raw_B8_accuracy'] for r in rows),'mean_raw_B8_NLL':statistics.mean(r['raw_B8_NLL'] for r in rows),'mean_original_required_seconds':statistics.mean(r['original_required_seconds'] for r in rows),'selected_for_VAL':rid in WINNERS})
    b.write_csv(a.out/'recipe_summary.csv',means)
    with (a.out/'accepted_originals.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    (a.out/'INPUT_HASHES.json').write_text(json.dumps(evidence,indent=2))
    manifest={'schema':'pcla-v8-development-face-export-1','jobs':128,'recipes':32,'seeds':sorted(SEEDS),'curve_rows':len(curves),'selection_sha256':SELECTION_SHA,'accepted_snapshot_sha256':SNAPSHOT_SHA,'selection_repeated':False,'missing_cost':'unqualified selection tie time remains null; original per-attempt cost retained separately','code_sha256':b.digest(__file__),'files':{p.name:b.digest(p) for p in a.out.iterdir()}}
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'jobs':128,'curve_rows':len(curves),'recipes':32,'selection_repeated':False}))
if __name__=='__main__':main()
