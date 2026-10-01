#!/usr/bin/env python3
"""Seed-level paired estimates. Full planned denominator is required by default."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--seeds",default="7,13,21,42,87")
    p.add_argument("--policies",default="pcla,random,layer_only,replay_random_rank,random_layer,uniform_cyclic,head_only,full_alt")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args();seeds=list(map(int,a.seeds.split(',')));policies=a.policies.split(',')
    records={};missing=[];identities={};protocols=set()
    for seed in seeds:
        for policy in policies:
            run=a.root/f"seed_{seed}"/policy
            try:
                status=json.loads((run/'status.json').read_text())
                identity=json.loads((run/'identity.json').read_text())
                metrics=json.loads((run/'metrics.json').read_text())
                if identity.get('seed') != seed or identity.get('policy') != policy:
                    raise ValueError('directory seed/policy does not match recorded identity')
                identity_payload={k:v for k,v in identity.items() if k not in ('identity_hash','policy')}
                identity_digest=hashlib.sha256(json.dumps(identity_payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                if identity_digest != identity.get('identity_hash'):
                    raise ValueError('recorded identity hash is invalid')
                provenance=identity['provenance']
                common_protocol={'config':identity['config'],'provenance':{k:v for k,v in provenance.items()
                    if k not in ('partition_sha256','initial_loaded_model_sha256')}}
                protocol_digest=hashlib.sha256(json.dumps(common_protocol,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                rounds=identity['config']['rounds']
                if rounds<5:raise ValueError('last-five primary endpoint requires at least five rounds')
                if status['state']!='completed' or [x['round'] for x in metrics]!=list(range(1,rounds+1)):
                    raise ValueError('not a complete fixed-horizon run')
                if status.get('completed_round') != rounds:
                    raise ValueError('status round does not match fixed horizon')
                if len({x['samples'] for x in metrics})!=1 or any(x['samples']<=0 for x in metrics):
                    raise ValueError('evaluation set size changed or empty')
                if any(not math.isfinite(x['accuracy']) or not 0<=x['accuracy']<=1 for x in metrics):
                    raise ValueError('invalid accuracy')
                protocols.add(protocol_digest)
                identities.setdefault(seed,set()).add(identity['identity_hash'])
                records[(seed,policy)]={'last5_accuracy':float(np.mean([x['accuracy'] for x in metrics[-5:]])),
                      'final_accuracy':metrics[-1]['accuracy'],
                      'serialized_traffic_bytes':sum(x['serialized_traffic_bytes'] for x in metrics),
                      'probe_seconds':sum(x['probe_seconds'] for x in metrics)}
            except Exception as exc: missing.append({'seed':seed,'policy':policy,'reason':str(exc)})
    if any(len(v)>1 for v in identities.values()): raise ValueError('paired identities differ; do not pool these runs')
    if len(protocols)>1: raise ValueError('cross-seed source/config/data protocol differs; do not pool these runs')
    estimates=[]
    for policy in policies:
        if policy=='pcla': continue
        valid=[s for s in seeds if (s,'pcla') in records and (s,policy) in records]
        d=np.array([100*(records[(s,'pcla')]['last5_accuracy']-records[(s,policy)]['last5_accuracy']) for s in valid])
        if not len(d):continue
        half=float(t.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else None
        estimates.append({'comparison':f'pcla - {policy}','paired_seeds':valid,'planned_seeds':seeds,
                          'mean_last5_difference_pp':float(d.mean()),'per_seed_difference_pp':d.tolist(),
                          'descriptive_t_interval':None if half is None else [float(d.mean()-half),float(d.mean()+half)],
                          'role':'trajectory-conditioned intervention, not standalone policy' if policy=='replay_random_rank' else 'standalone policy comparison'})
    output={'complete_planned_matrix':not missing,'missing_or_failed':missing,'paired_estimates':estimates,
            'records':[{'seed':s,'policy':p,**v} for (s,p),v in records.items()],
            'inference_note':'seeds are independent units; rounds/clients/ranks are not extra independent replicates. No automatic acceptance, equivalence or superiority verdict.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    if missing and not a.allow_partial: raise SystemExit('Incomplete planned matrix: inventory written; no full-matrix conclusion authorized.')
if __name__=='__main__':main()
