"""Derive V5e EMA endpoints and adjacent-round SVD subspace overlap from verified replays."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'handoff_v5e'))
from common import read,require,tensor_load,bind_repo

SEEDS=(7,13,21,42,87)
KS=(1,4,8,16)


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def overlap(factors_a,factors_b,module,k,rank_a,rank_b):
    if min(rank_a,rank_b)<k:return None
    out=[]
    for suffix,columns in (('b',True),('a',False)):
        x=factors_a[module+'.lora_'+suffix].double()
        y=factors_b[module+'.lora_'+suffix].double()
        if not columns:x,y=x.T,y.T
        x,y=x[:,:k],y[:,:k]
        norm_x,norm_y=x.norm(dim=0),y.norm(dim=0)
        require(bool((norm_x>0).all() and (norm_y>0).all()),'zero singular vector within numerical rank')
        x,y=x/norm_x,y/norm_y
        ident=torch.eye(k,dtype=torch.float64)
        require(float((x.T@x-ident).abs().max())<2e-3 and
                float((y.T@y-ident).abs().max())<2e-3,'stored factors not orthogonal SVD vectors')
        value=float(((x.T@y)**2).sum()/k)
        require(-1e-8<=value<=1+2e-3,'subspace overlap outside [0,1]')
        out.append(min(1.,max(0.,value)))
    return out


def boundary_gap(module_record,k):
    """The top-k projector is not unique when the k/k+1 boundary is tied."""
    values=module_record['singular_values']
    if k>=len(values):return None
    require(values and values[0]>=0,'invalid singular values')
    return (values[k-1]-values[k])/values[0] if values[0]>0 else 0.


def effective_rank(module_record):
    values=module_record['singular_values']
    if not values or values[0]<=0:return 0
    threshold=max(module_record['tolerance'],1e-5*values[0])
    return min(module_record['numerical_rank'],sum(x>threshold for x in values))


def run(root,out):
    _,core,_,_=bind_repo()
    tensor_digest=core.tensor_digest
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    ema_rows=[];sub_rows=[];per_seed=[]
    for seed in SEEDS:
        e=root/f'L2_ema_s{seed}'/'run';s=root/f'V4_factors_s{seed}'/'run'
        for d in (e,s):
            require(read(d/'v5e_parity.json')['passed'],'V5e parity has not passed: '+str(d))
        rows=read(e/'v5e_ema_rows.json')
        require([x['round'] for x in rows]==list(range(1,101)),'EMA rounds missing')
        original=read(e/'run/metrics.json')
        for start,end in ((26,30),(96,100)):
            er=[x['accuracy'] for x in rows[start-1:end]]
            br=[x['accuracy'] for x in original[start-1:end]]
            ema_rows.append({'seed':seed,'round_start':start,'round_end':end,
                             'ema_mean_accuracy':statistics.mean(er),
                             'global_mean_accuracy':statistics.mean(br),
                             'ema_minus_global_pp':100*(statistics.mean(er)-statistics.mean(br))})
        previous=None
        for r in range(1,31):
            bank=tensor_load(s/'v5e_factors'/f'round_{r:03d}.pt')
            rec=read(s/'evidence'/f'{r:03d}'/'refactor.json')
            require(bank['round']==r and bank['post_lora_sha256']==tensor_digest(bank['post'])==
                    rec['post_lora_sha256'],'SVD factors differ from refactor record')
            current={x['module']:x for x in rec['modules']}
            if previous is not None:
                old_bank,old_modules=previous
                require(set(old_modules)==set(current),'SVD module inventory changed')
                for module in sorted(current):
                    ra=old_modules[module]['numerical_rank'];rb=current[module]['numerical_rank']
                    ea=effective_rank(old_modules[module]);eb=effective_rank(current[module])
                    for k in KS:
                        ga=boundary_gap(old_modules[module],k)
                        gb=boundary_gap(current[module],k)
                        if min(ea,eb)<k: reason='effective_rank_below_k'
                        elif (ga is not None and ga<=1e-5) or (gb is not None and gb<=1e-5):
                            reason='boundary_near_tie'
                        else:reason='valid'
                        pair=overlap(old_bank['post'],bank['post'],module,k,ra,rb) if reason=='valid' else None
                        sub_rows.append({'seed':seed,'round_before':r-1,'round_after':r,
                                         'module':module,'k':k,'rank_before':ra,'rank_after':rb,
                                         'effective_rank_before':ea,'effective_rank_after':eb,
                                         'boundary_gap_relative_before':ga,'boundary_gap_relative_after':gb,
                                         'validity':reason,
                                         'left_overlap':None if pair is None else pair[0],
                                         'right_overlap':None if pair is None else pair[1]})
            previous=bank,current
    write_csv(out/'ema_endpoints.csv',ema_rows,list(ema_rows[0]))
    write_csv(out/'adjacent_subspace.csv',sub_rows,list(sub_rows[0]))
    summary=[]
    for seed in SEEDS:
        for k in KS:
            rows=[x for x in sub_rows if x['seed']==seed and x['k']==k]
            present=[x for x in rows if x['left_overlap'] is not None]
            summary.append({'seed':seed,'k':k,'transitions_modules':len(rows),'available':len(present),
                            'effective_rank_below_k':sum(x['validity']=='effective_rank_below_k' for x in rows),
                            'boundary_near_tie':sum(x['validity']=='boundary_near_tie' for x in rows),
                            'mean_left_overlap':statistics.mean(x['left_overlap'] for x in present) if present else None,
                            'mean_right_overlap':statistics.mean(x['right_overlap'] for x in present) if present else None})
    write_csv(out/'subspace_summary_by_seed.csv',summary,list(summary[0]))
    (out/'status.json').write_text(json.dumps({'passed':True,'seeds':list(SEEDS),
      'ema_endpoint_rows':len(ema_rows),'subspace_rows':len(sub_rows),
      'definition':'top-k left/right singular subspace projector overlap: ||U_t^T U_(t+1)||_F^2 / k; effective rank uses max(refactor tolerance, 1e-5 * largest singular value); unavailable if either effective rank < k or relative k/k+1 gap <= 1e-5 in either round',
      'k':list(KS)},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--materialized',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.materialized,a.out)

if __name__=='__main__':main()
