"""Read-only, fully accepted V8 VAL original-value base export (no PT/GPU/CI)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

INDEX_SHA = '40046d70bd6822d82c386817d0a186ba4c05a682fcaafb1bdb53725be04856ef'
GATE_SHA = '7e9da286a0943da67f1ec5b4ed1572b56667f37a4e7caa702f5108e5c25e2b63'
AUDITOR_SHA = '2d381c565cad1cb777c6a7f1444105e24ded01d3450e81adf446cba7f95e5a3b'
SEEDS = [601,607,613,617,619,631,641,643,647,653,659,661]
RECIPES = ['U002','U010','U014','U018','U020','U023','U027','U004','U006']
COST_FIELDS = ['task_initialization','state_and_optimizer_setup','client_training',
               'scoring_and_shared_information','coordinate_and_server_decision',
               'message_processing','budget_and_return_control']

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def quality_window(metrics, result, byte_cap):
    n = result['budget_return_round']
    require(n == result['rounds'] == len(metrics) and 100 <= n <= 200, 'budget round coverage')
    require([m['round'] for m in metrics] == list(range(1,n+1)), 'round sequence')
    deployed = [m['deployed_bytes'] for m in metrics]
    require(all(a <= b for a,b in zip(deployed,deployed[1:])) and deployed[-1] <= byte_cap, 'byte prefix')
    require(all(m['diagnostic_after_budget_bytes'] == 0 for m in metrics), 'after-budget bytes')
    for m in metrics:
        for role in ('evaluation','ema_evaluation'):
            e = m[role]
            require(math.isfinite(e['accuracy']) and 0 <= e['accuracy'] <= 1 and
                    math.isfinite(e['loss']) and e['loss'] >= 0 and e['samples'] > 0, 'quality numeric')
    require(len({m['evaluation']['samples'] for m in metrics}) == 1, 'readout denominator')
    half = [m for m in metrics if m['deployed_bytes'] <= byte_cap//2]
    require(half, 'half-budget coverage')
    r100 = metrics[99]
    tail20 = sorted(m['evaluation']['accuracy'] for m in metrics[-20:])
    # Original records retained; this explicit lower-tail convention is descriptive.
    return {'returned_round':n, 'half_round':half[-1]['round'],
            'deployed_bytes':deployed[-1], 'raw_samples':metrics[-1]['evaluation']['samples'],
            'raw_B8_accuracy':metrics[-1]['evaluation']['accuracy'],
            'raw_B8_NLL':metrics[-1]['evaluation']['loss'],
            'EMA_B8_accuracy':metrics[-1]['ema_evaluation']['accuracy'],
            'raw_half_accuracy':half[-1]['evaluation']['accuracy'],
            'raw_r100_accuracy':r100['evaluation']['accuracy'],
            'raw_r96_100_mean':sum(m['evaluation']['accuracy'] for m in metrics[95:100])/5,
            'raw_last5_mean':sum(m['evaluation']['accuracy'] for m in metrics[-5:])/5,
            'raw_last20_min':tail20[0],
            'raw_last20_lower5_mean':sum(tail20[:5])/5}

def required_time(result):
    values = result['required_cost_breakdown']
    require(set(values) == set(COST_FIELDS), 'required cost seven fields')
    require(all(math.isfinite(v) and v >= 0 for v in values.values()), 'cost numeric')
    return sum(values.values())

def matrix_status(root):
    root = Path(root)
    index_path = root/'checks/validation_configuration01/COMPILED_INDEX.json'
    require(digest(index_path) == INDEX_SHA, 'compiled index hash')
    index = read(index_path)
    expected = {f'VAL_S{i:02d}_s{s}' for i in range(1,10) for s in SEEDS}
    paths = {p.stem:p for p in (root/'queues/validation01/jobs').glob('*.json')}
    require(set(paths) == set(index['config_sha256']) == expected, 'complete 9x12 matrix')
    states = {}
    for p in paths.values():
        q = read(p); states[q['state']] = states.get(q['state'],0)+1
    return index,paths,states

def load_accepted(qpath, index):
    q = read(qpath); jid=q['job_id']; require(q['state']=='completed', 'not completed '+jid)
    cp=Path(q['config']); rp=Path(q['result_path']); ap=Path(q['audit_receipt'])
    config=read(cp); result=read(rp); audit=read(ap); check=audit['checks']
    require(digest(cp)==q['config_sha256']==index['config_sha256'][jid]==result['config_sha256']==check['config_sha256'], 'config binding '+jid)
    require(digest(rp)==q['result_sha256']==audit['result_sha256'], 'result binding '+jid)
    require(digest(ap)==q['audit_receipt_sha256'] and audit['accepted'] and not audit['issues'] and audit['auditor']['sha256']==AUDITOR_SHA, 'audit binding '+jid)
    require(digest(q['gate_receipt'])==q['gate_receipt_sha256']==GATE_SHA==check['validation_root_gate_sha256'], 'gate binding '+jid)
    require(config['job']['job_id']==result['job_id']==audit['job_id']==jid and config['recipe']==result['recipe'], 'job/recipe binding '+jid)
    require(q['seed']==config['job']['seed'] and q['slot']==config['job']['slot'] and q['recipe_id']==config['recipe']['recipe_id']==RECIPES[q['slot']-1], 'slot/seed identity '+jid)
    require(digest(config['source_manifest'])==config['binding']['source_manifest_sha256']==result['source_manifest_sha256']==check['source_manifest_sha256'], 'frozen source manifest '+jid)
    source_dir=Path(config['source_manifest']).parent
    for relative, expected_sha in config['binding']['source_file_hashes'].items():
        require(digest(source_dir/relative)==expected_sha, 'frozen source file '+relative)
    require(check['validation_config_index_sha256']==INDEX_SHA and
            config['binding']['selection_sha256']==check['validation_selection_sha256']==index['selection_sha256'], 'selection/index binding '+jid)
    bd=ap.parent; inventory_path=bd/'INVENTORY.json'; root_path=bd/'ROOT_ACCEPTANCE_EXECUTE.json'
    inventory=read(inventory_path); accepted=read(root_path)
    require(accepted['accepted'] and accepted['execute'] and digest(inventory_path)==accepted['inventory_sha256'], 'root acceptance inventory '+jid)
    require(jid in accepted['completed'], 'root completed membership '+jid)
    row=next(x for x in inventory['results'] if x['job_id']==jid)
    review=next(x for x in accepted['items'] if x['job_id']==jid)
    require(row['accepted'] and row['return_code']==0 and row['receipt_sha256']==review['audit_receipt_sha256']==q['audit_receipt_sha256'] and row['result_sha256']==review['result_sha256']==q['result_sha256'], 'root item binding '+jid)
    out=Path(q['output']); mp=out/'metrics.json'; metrics=read(mp)
    endpoint=quality_window(metrics,result,config['runtime']['byte_cap'])
    require(endpoint['half_round']==check['half_round'] and endpoint['returned_round']==check['budget_return_round'] and endpoint['deployed_bytes']==check['deployed_bytes'], 'audited window '+jid)
    require(result['required_cost_breakdown']==check['original_cost_fields']['required_cost_breakdown']==review['required_cost_original'], 'audited cost '+jid)
    require(sum(len(m['client_training']) for m in metrics)==10*len(metrics) and all(len(m['selected_clients'])==len(set(m['selected_clients']))==10 and all(t['optimizer_steps']==16 for t in m['client_training']) for m in metrics), 'client update counts '+jid)
    files={str(p):digest(p) for p in [qpath,cp,rp,ap,mp,inventory_path,root_path,Path(config['source_manifest']),Path(q['gate_receipt'])]}
    require(check['pretraining_failed_queue_attempts']==len(q['attempts'])-1 and check['pretraining_global_archive_count']==14, 'failure denominator '+jid)
    if check['pretraining_failed_queue_attempts']:
        archive=Path(check['pretraining_failure_archive'])
        for rel,sha in check['pretraining_failure_file_sha256'].items():
            p=archive/rel;require(digest(p)==sha, 'failure original '+jid);files[str(p)]=sha
    identity={'job_id':jid,'recipe_id':q['recipe_id'],'seed':q['seed'],'slot':q['slot'],
              'eval_role':config['job']['eval_role'],'model_revision':config['binding']['model_revision'],
              'hardware_family':q['hardware_family'],'gpu_uuid':q['claim']['gpu_uuid'],
              'source_manifest_sha256':result['source_manifest_sha256'],
              'config_sha256':q['config_sha256'],'queue_attempts':len(q['attempts'])}
    endpoint={**identity,**endpoint,'required_prefix_seconds':required_time(result)}
    return {'identity':identity,'endpoint':endpoint,'queue_original':q,'result_original':result,
            'config_original':config,'audit_original':audit,'metrics_original':metrics,
            'evidence_sha256_at_export':files,
            'missing':{'balanced_accuracy':'requires separate CPU logit-derived readout export',
                       'prediction_class_counts':'requires separate CPU logit-derived readout export',
                       'continuous_GPU_occupancy':'not sampled continuously'}}

def write_csv(path, rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();index,paths,states=matrix_status(args.root)
    if args.preflight_only:
        print(json.dumps({'states':states,'matrix_jobs':108,'ready':states=={'completed':108},'writes':0}));return
    require(states=={'completed':108}, 'full accepted matrix required before export')
    require(args.out is not None and not args.out.exists(), 'new output directory required')
    # All input checks precede output creation. No partial formal export on validation failure.
    records=[load_accepted(paths[jid],index) for jid in sorted(paths)]
    args.out.mkdir(parents=True)
    with (args.out/'accepted_originals.jsonl').open('x') as f:
        for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
    write_csv(args.out/'endpoints.csv',[r['endpoint'] for r in records])
    costs=[];curves=[]
    for r in records:
        costs.append({**r['identity'],**r['result_original']['required_cost_breakdown'],
                      'required_prefix_seconds':r['endpoint']['required_prefix_seconds']})
        for m in r['metrics_original']:
            curves.append({**r['identity'],'round':m['round'],'deployed_bytes':m['deployed_bytes'],
                           'raw_accuracy':m['evaluation']['accuracy'],'raw_NLL':m['evaluation']['loss'],
                           'raw_samples':m['evaluation']['samples'],
                           'EMA_accuracy':m['ema_evaluation']['accuracy'],'EMA_NLL':m['ema_evaluation']['loss']})
    write_csv(args.out/'required_costs.csv',costs);write_csv(args.out/'curves.csv',curves)
    manifest={'schema':'pcla-v8-validation-original-base-export-1','jobs':108,'seeds':SEEDS,
              'statistical_unit':'seed','source_index_sha256':INDEX_SHA,'exporter_sha256':digest(__file__),
              'fields_missing_reason':records[0]['missing'],
              'scope':'Original-value base export; logit-derived quality, full historical operating costs and final statistics are separate.',
              'files':{p.name:digest(p) for p in args.out.iterdir()},'curve_rows':len(curves)}
    (args.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'jobs':108,'curve_rows':len(curves),'out':str(args.out)}))

if __name__=='__main__':main()
