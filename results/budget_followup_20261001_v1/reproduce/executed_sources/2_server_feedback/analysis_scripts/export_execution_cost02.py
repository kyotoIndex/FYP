"""Measured execution cost and operational-original inventory; unknowns stay null."""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
import export_validation_base01 as b
TECH_DIRS=['core01','core02','controller_gpu01','controller_families_gpu02','controller_families_gpu03','native_gate02','budget_gate02','diagnostic_gpu02','diagnostic_preflight01','diagnostic_preflight02','worker_boundary01','native_tail_guard01','diagnostic_tail_followon01','diagnostic_tail_followon02','validation_pretraining_recovery01','validation_queue01','validation_queue02','controller_pipeline02']
EXCLUDE_NAMES={'metrics.json','communication_journal.json','evaluation_labels.json','timing_events.json','cache.events.json'}
def measured_timing(path):
    v=b.read(path);b.require(v['schema']=='pcla-exclusive-host-timing-2','timing schema')
    categories=v['category_seconds'];b.require(all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in categories.values()),'valid category cost')
    b.require(math.isclose(sum(categories.values()),v['total_seconds'],abs_tol=1e-7,rel_tol=1e-9),'non-overlapping measured categories')
    return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
    output_sets={};states={}
    for phase,directory,count in [('DEV','development01',128),('VAL','validation01',108),('DIAG','diagnostic01',120)]:
        jobs=[b.read(p) for p in (r/'queues'/directory/'jobs').glob('*.json')];b.require(len(jobs)==count and all(j['state']=='completed' for j in jobs),'accepted '+phase+' matrix');states[phase]=len(jobs)
        if phase!='DIAG':output_sets.update({Path(j['output']).resolve():(phase,j['job_id']) for j in jobs})
    paths=set()
    for base in [r/'results']+[r/'checks'/x for x in TECH_DIRS]:
        if not base.exists():continue
        paths.update(base.rglob('timing_attempt_*.json'));paths.update(base.rglob('timing_events_partial.json'))
    attempts=[];buckets=[];inputs={};failure_jids=set()
    for path in sorted(paths):
        v=measured_timing(path);inputs[str(path)]=b.digest(path);parent=path.parent.resolve();phase,jid=output_sets.get(parent,('TECH_OR_HISTORY',parent.name))
        if 'failure_pretraining01' in path.parts:phase='VAL_PRETRAINING_FAILURE';failure_jids.add(parent.name)
        elif v.get('partial'):phase='HISTORICAL_PARTIAL'
        rows={k:float(v['category_seconds'].get(k,0)) for k in ['necessary_algorithm','research_observation','post_budget_diagnostic','measurement_overhead','unclassified_gap','environment_wait']}
        attempts.append({'phase':phase,'job_or_case_id':jid,'source_path':str(path),'source_sha256':inputs[str(path)],'attempt_file':path.name,'partial':v.get('partial',False),'tail_unmeasured':v.get('tail_unmeasured'),'open_interval_count':len(v.get('open_intervals',[])),'measured_total_seconds':v['total_seconds'],**rows})
        for bucket,labels in v['bucket_label_seconds'].items():
            for label,seconds in labels.items():buckets.append({'phase':phase,'job_or_case_id':jid,'attempt_file':path.name,'source_path':str(path),'source_sha256':inputs[str(path)],'bucket':bucket,'label':label,'category':v['label_categories'][label],'seconds':seconds})
    b.require(len(failure_jids)==14,'14 archived failed validation startups')
    dp=r/'checks/phase_closeout01/diagnostic_chain01';dm=b.read(dp/'MANIFEST.json')
    for name,sha in dm['files'].items():b.require(b.digest(dp/name)==sha,'accepted diagnostic export');inputs[str(dp/name)]=sha
    diagnostic=[];support_cost=[]
    for line in (dp/'metadata_originals.jsonl').open():
        m=json.loads(line);ident=m['identity'];col=m['collection_original'];diagnostic.append({'job_id':ident['job_id'],'phase':ident['phase'],'seed':ident['seed'],'elapsed_seconds_inclusive':m['elapsed_seconds'],'new_local_calls':col['local_calls'],'optimizer_steps':col['optimizer_steps'],'collection_training_seconds':col.get('training_seconds',0 if ident['phase']=='BANK_READOUT' else None),'collection_reused':col.get('reused'),'supports':len(m['supports'])})
        for support,values in m['support_records'].items():
            tm=values['timers_original'];rule_times=tm['standalone_server_wall_seconds_by_rule'];b.require(math.isclose(sum(rule_times.values()),tm['standalone_server_wall_seconds_total'],abs_tol=1e-9),'standalone rule sum')
            for rule,seconds in rule_times.items():support_cost.append({'job_id':ident['job_id'],'phase':ident['phase'],'seed':ident['seed'],'support':support,'rule':rule,'standalone_server_seconds':seconds,'independent_readout_callback_seconds_support_total':tm['independent_readout_callback_seconds'],'readout_cost_repeated_for_each_rule':True})
    b.require(len(diagnostic)==120 and len(support_cost)==1080 and sum(d['new_local_calls'] for d in diagnostic)==240 and sum(d['optimizer_steps'] for d in diagnostic)==3840,'diagnostic cost matrix')
    # Inventory all operational originals, excluding model/data/quality arrays and copied audit bundles.
    originals=[];operation_results=[];roots=[r/'operations']+[x for x in (r/'checks').iterdir() if x.is_dir() and x.name not in ['__pycache__','data01','phase_closeout01']]
    roots.append(r/'checks')
    for directory in roots:
        if not directory.exists():continue
        for path in sorted(directory.glob('*') if directory==r/'checks' else directory.rglob('*')):
            if not path.is_file() or path.is_symlink() or any(x in path.parts for x in ['__pycache__','metadata','originals','.pytest_cache','sources','configs','jobs']):continue
            if path.name=='EXECUTION_COST_EXPORT02_2026.log':continue
            if path.suffix not in ['.json','.jsonl','.log','.txt','.md','.csv','.exit'] or path.name in EXCLUDE_NAMES or path.name.startswith(('logits','controller_round')):continue
            originals.append({'source_path':str(path),'sha256':b.digest(path),'size_bytes':path.stat().st_size,'copy_candidate':path.stat().st_size<=1048576})
            if path.suffix=='.json' and path.stat().st_size<=1048576:
                try:v=b.read(path)
                except (ValueError,UnicodeError):continue
                if isinstance(v,dict) and any(k in v for k in ['return_code','exit_code','failure','failed_operation']):
                    start=v.get('started_at',v.get('started'));end=v.get('ended_at',v.get('ended'));elapsed=end-start if isinstance(start,(int,float)) and isinstance(end,(int,float)) and end>=start else None
                    operation_results.append({'source_path':str(path),'source_sha256':b.digest(path),'return_code':v.get('return_code',v.get('exit_code')),'failure':v.get('failure',v.get('failed_operation')),'launcher_elapsed_seconds_inclusive_of_resource_wait':elapsed,'allocated_GPU_seconds':None,'measured_component_cost':None})
    # Native init component is separately recorded, not added to envelope totals.
    native=[]
    for path in (r/'results/failure_pretraining01').rglob('native_initialization_attempts.json'):
        inputs[str(path)]=b.digest(path)
        for value in b.read(path):native.append({'job_id':path.parent.name,'attempt':value['attempt'],'source_path':str(path),'source_sha256':inputs[str(path)],'native_initialization_wall_seconds_component':value['cost']['total_wall_seconds'],'included_in_failed_timing_total':True})
    b.require(not a.out.exists(),'new cost export');a.out.mkdir(parents=True)
    for name,rows in [('timing_attempts.csv',attempts),('timing_labels.csv',buckets),('diagnostic_object_costs.csv',diagnostic),('diagnostic_rule_costs.csv',support_cost),('operational_exit_records.csv',operation_results),('native_failed_initialization_components.csv',native)]:
        b.require(rows,'nonempty cost table '+name);b.write_csv(a.out/name,rows)
    (a.out/'INPUT_HASHES.json').write_text(json.dumps(inputs,indent=2));(a.out/'OPERATIONAL_ORIGINAL_INDEX.json').write_text(json.dumps(originals,indent=2))
    totals=defaultdict(lambda:defaultdict(float))
    for x in attempts:
        for key in ['measured_total_seconds','necessary_algorithm','research_observation','post_budget_diagnostic','measurement_overhead','unclassified_gap','environment_wait']:totals[x['phase']][key]+=x[key]
    manifest={'schema':'pcla-v8-execution-cost-export-2','accepted_matrices':states,'timing_attempt_records':len(attempts),'timing_label_records':len(buckets),'partial_timing_records':sum(x['partial'] for x in attempts),'failed_VAL_startups':14,'operational_originals':len(originals),'operational_exit_records':len(operation_results),'measured_timing_totals_by_phase':dict(totals),'interpretation':'Host seconds are measured intervals, not monetary charges or continuous GPU utilization. Inclusive diagnostic and launcher times overlap component times and are never summed with them. Readout support total is repeated as annotation in rule rows; count once per support. Native failed initialization is included in failed timing, not an extra addend. Unmeasured CPU checks, transfer retries, allocation waits and absent coverage remain unknown. TECH_OR_HISTORY is a provenance grouping, not an all-failed classification.','code_sha256':b.digest(__file__),'files':{p.name:b.digest(p) for p in a.out.iterdir()}}
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['timing_attempt_records','partial_timing_records','failed_VAL_startups','operational_originals','operational_exit_records']}))
if __name__=='__main__':main()
