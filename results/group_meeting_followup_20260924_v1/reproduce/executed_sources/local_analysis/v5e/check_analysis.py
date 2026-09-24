"""Independent consistency checks for derived V5e CSVs and archive parity summary."""
from __future__ import annotations
import csv,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
A=ROOT/'analysis'
rows=list(csv.DictReader((A/'adjacent_subspace.csv').open(newline='')))
summary=list(csv.DictReader((A/'subspace_summary_by_seed.csv').open(newline='')))
ema=list(csv.DictReader((A/'ema_endpoints.csv').open(newline='')))
parity=json.loads((A/'parity_summary.json').read_text())
status=json.loads((A/'status.json').read_text())
assert len(rows)==13920==5*29*24*4 and len(summary)==20 and len(ema)==10
assert len({(r['seed'],r['round_before'],r['module'],r['k']) for r in rows})==len(rows)
assert {int(r['seed']) for r in rows}=={7,13,21,42,87}
assert {int(r['k']) for r in rows}=={1,4,8,16}
for r in rows:
    assert int(r['round_after'])==int(r['round_before'])+1
    assert 1<=int(r['round_before'])<=29
    valid=r['validity']=='valid'
    assert valid==(r['left_overlap']!='' and r['right_overlap']!='')
    assert r['validity'] in {'valid','effective_rank_below_k','boundary_near_tie'}
    if valid:
        assert 0<=float(r['left_overlap'])<=1 and 0<=float(r['right_overlap'])<=1
    elif r['validity']=='effective_rank_below_k':
        assert min(int(r['effective_rank_before']),int(r['effective_rank_after']))<int(r['k'])
    else:
        assert min(int(r['effective_rank_before']),int(r['effective_rank_after']))>=int(r['k'])
        assert min(float(r['boundary_gap_relative_before']),float(r['boundary_gap_relative_after']))<=1e-5
for s in summary:
    group=[r for r in rows if r['seed']==s['seed'] and r['k']==s['k']]
    valid=[r for r in group if r['validity']=='valid']
    assert len(group)==696==int(s['transitions_modules'])
    assert len(valid)==int(s['available'])
    for key in ('effective_rank_below_k','boundary_near_tie'):
        assert sum(r['validity']==key for r in group)==int(s[key])
    assert math.isclose(statistics.mean(float(r['left_overlap']) for r in valid),float(s['mean_left_overlap']),abs_tol=1e-12)
    assert math.isclose(statistics.mean(float(r['right_overlap']) for r in valid),float(s['mean_right_overlap']),abs_tol=1e-12)
for r in ema:
    assert int(r['round_end'])-int(r['round_start'])==4
    assert math.isclose(100*(float(r['ema_mean_accuracy'])-float(r['global_mean_accuracy'])),float(r['ema_minus_global_pp']),abs_tol=1e-10)
assert status['passed'] and status['ema_endpoint_rows']==10 and status['subspace_rows']==13920
assert parity['passed'] and parity['formal_count']==10 and parity['smoke_count']==2
assert len(parity['rows'])==12 and all(r['state']=='completed' and r['parity_passed'] and all(r['checks'].values()) for r in parity['rows'])
assert sum(r['audit_complete'] for r in parity['rows'])==10
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
outputs={p.name:sha(p) for p in sorted(A.iterdir()) if p.is_file() and p.name not in ('verification.json','provenance.json')}
ema_windows={}
for start,end in ((26,30),(96,100)):
    group=[r for r in ema if int(r['round_start'])==start and int(r['round_end'])==end]
    assert len(group)==5
    differences=[float(r['ema_minus_global_pp']) for r in group]
    mean=statistics.mean(differences);sem=statistics.stdev(differences)/math.sqrt(5)
    ema_windows[f'{start}-{end}']={
        'global_mean_accuracy':statistics.mean(float(r['global_mean_accuracy']) for r in group),
        'ema_mean_accuracy':statistics.mean(float(r['ema_mean_accuracy']) for r in group),
        'paired_difference_pp':mean,
        'descriptive_95pct_t_interval_pp':[mean-2.7764451051977987*sem,mean+2.7764451051977987*sem],
        'positive_seeds':sum(x>0 for x in differences),
    }
subspace_by_k={}
for k in (1,4,8,16):
    group=[r for r in summary if int(r['k'])==k]
    subspace_by_k[str(k)]={'available':sum(int(r['available']) for r in group),
                          'total':sum(int(r['transitions_modules']) for r in group),
                          'seed_mean_left_overlap':statistics.mean(float(r['mean_left_overlap']) for r in group),
                          'seed_mean_right_overlap':statistics.mean(float(r['mean_right_overlap']) for r in group)}
result={'passed':True,'subspace_rows':len(rows),'valid_by_k':{str(k):sum(r['validity']=='valid' for r in rows if r['k']==str(k)) for k in (1,4,8,16)},
        'ema_endpoint_rows':len(ema),'ema_windows':ema_windows,'subspace_by_k':subspace_by_k,
        'all_formal_parity':True,'output_sha256':outputs}
(A/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
