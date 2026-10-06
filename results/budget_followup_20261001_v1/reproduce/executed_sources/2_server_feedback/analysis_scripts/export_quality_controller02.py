"""CPU-only original logit quality and controller export; never changes model/queue."""
import argparse
import json
import math
from pathlib import Path
import export_validation_base01 as base

BASE_SHA='e3c988f2042fe6956bbf8a4b60cbe81bc763d35650191f397d4f2e5f96bb8c7f'

def classification_stats(logits, labels):
    base.require(len(logits)==len(labels)>0, 'logit/label coverage')
    truth=[0,0];predicted=[0,0];correct=[0,0];loss=[]
    for values,label in zip(logits,labels):
        base.require(type(label) is int and label in (0,1) and len(values)==2 and
                     all(math.isfinite(v) for v in values), 'binary logit domain')
        # Match argmax first-index ties. Float32 stored logits, float64 loss sum.
        pred=0 if values[0]>=values[1] else 1
        truth[label]+=1;predicted[pred]+=1;correct[label]+=int(pred==label)
        top=max(values);loss.append(top+math.log(math.fsum(math.exp(v-top) for v in values))-values[label])
    return {'samples':len(labels),'accuracy':sum(correct)/len(labels),
            'NLL':math.fsum(loss)/len(labels),
            'balanced_accuracy':sum(correct[i]/truth[i] for i in (0,1))/2 if all(truth) else None,
            'balanced_accuracy_missing_reason':None if all(truth) else 'one truth class absent',
            'prediction_class_0':predicted[0],'prediction_class_1':predicted[1],
            'truth_class_0':truth[0],'truth_class_1':truth[1],
            'correct_class_0':correct[0],'correct_class_1':correct[1]}

def inspect_job(record):
    q=record['queue_original'];out=Path(q['output']);identity=record['identity']
    metrics=record['metrics_original'];endpoint=record['endpoint']
    lp=out/'evaluation_labels.json';labels=base.read(lp)
    evidence={**record['evidence_sha256_at_export'],str(lp):base.digest(lp)}
    points=[];controllers=[]
    for role,rnd in [('B8',endpoint['returned_round']),('half',endpoint['half_round']),('r100',100)]:
        p=out/f'logits_round_{rnd:03d}.json';stats=classification_stats(base.read(p),labels)
        raw=metrics[rnd-1]['evaluation']
        base.require(stats['samples']==raw['samples'] and stats['accuracy']==raw['accuracy'] and
                     abs(stats['NLL']-raw['loss'])<=1e-6, 'sealed raw readout differs '+q['job_id'])
        evidence[str(p)]=base.digest(p)
        points.append({**identity,'role':role,'round':rnd,**stats,'logits_sha256':evidence[str(p)],'labels_sha256':evidence[str(lp)]})
    for m in metrics:
        rnd=m['round'];p=out/f'controller_round_{rnd:03d}.json'
        if identity['recipe_id']=='U006':
            original=m['server_controller'];source=str(out/'metrics.json')
        else:
            base.require(p.is_file(), 'core controller file missing '+q['job_id'])
            original=base.read(p);source=str(p);evidence[source]=base.digest(p)
            base.require(original==m['server_controller'], 'inline/controller original differs')
        base.require(isinstance(original,dict) and original, 'controller coverage')
        controllers.append({**identity,'round':rnd,'original_source':source,'original':original})
    # Missing counter totals remain null; coverage column documents why.
    summary={**identity,'rounds':len(metrics)}
    for key in ('features_count','encoder_batches','head_evals','public_dataset_passes','public_dataset_passes_for_selection'):
        values=[c['original'][key] for c in controllers if key in c['original']]
        base.require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values), 'controller counter numeric')
        summary[key+'_recorded_rounds']=len(values)
        summary[key+'_total']=sum(values) if len(values)==len(controllers) else None
    summary['alpha_unique_count']=len({json.dumps(c['original'].get('alpha'),sort_keys=True) for c in controllers}) if all('alpha' in c['original'] for c in controllers) else None
    summary['beta_unique_count']=len({json.dumps(c['original'].get('beta'),sort_keys=True) for c in controllers}) if all('beta' in c['original'] for c in controllers) else None
    return {'quality_points':points,'controller_rounds':controllers,'controller_summary':summary,
            'evidence_sha256_at_export':evidence}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();base.require(base.digest(base.__file__)==BASE_SHA,'base exporter version')
    index,paths,states=base.matrix_status(args.root)
    base.require(states=={'completed':108},'full accepted matrix required before export')
    base.require(not args.out.exists(),'new output directory required')
    # Validate every original before making the formal output directory.
    records=[inspect_job(base.load_accepted(paths[j],index)) for j in sorted(paths)]
    args.out.mkdir(parents=True)
    points=[x for r in records for x in r['quality_points']]
    base.write_csv(args.out/'quality_points.csv',points)
    base.write_csv(args.out/'controller_summary.csv',[r['controller_summary'] for r in records])
    with (args.out/'controller_originals.jsonl').open('x') as f:
        for r in records:
            for c in r['controller_rounds']:f.write(json.dumps(c,allow_nan=False)+'\n')
    evidence={p:h for r in records for p,h in r['evidence_sha256_at_export'].items()}
    (args.out/'INPUT_HASHES.json').write_text(json.dumps(evidence,indent=2))
    manifest={'schema':'pcla-v8-quality-controller-export-1','jobs':108,'quality_points':len(points),
              'controller_rounds':sum(len(r['controller_rounds']) for r in records),
              'exporter_sha256':base.digest(__file__),'base_exporter_sha256':BASE_SHA,
              'precision':'stored float32 logits; Python float64 stable logsumexp and math.fsum; raw NLL check atol1e-6',
              'NA':'counter total null unless explicitly recorded on every round; native initialization and selection fields remain separate',
              'files':{p.name:base.digest(p) for p in args.out.iterdir()}}
    (args.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'jobs':108,'quality_points':len(points),'out':str(args.out)}))

if __name__=='__main__':main()
