"""Seed-level descriptive summaries of the complete hashed V8 diagnostic chain."""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path
from scipy.stats import t
import export_validation_base01 as b
SEEDS={101,109,127,149,163,181,199,223,241,263,281,307}
RULES={'ordinary_mean','scalar4','scalar17','block16','separable'}

def describe(v):
    n=len(v)
    if not n:return {'n_seeds':0,'mean':None,'t95_low':None,'t95_high':None,'positive_seeds':0}
    assert all(math.isfinite(x) for x in v)
    mean=statistics.mean(v);m=float(t.ppf(.975,n-1))*statistics.stdev(v)/math.sqrt(n) if n>1 else None
    return {'n_seeds':n,'mean':mean,'t95_low':mean-m if m is not None else None,'t95_high':mean+m if m is not None else None,'positive_seeds':sum(x>0 for x in v)}

def collect(rows,kind):
    expected=1080 if kind=='rules' else 216
    b.require(len(rows)==expected,'complete diagnostic rows')
    identities=set(); groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        seed=int(r['seed']);ph=r['phase'];parent=r['parent_recipe'] or 'rank8';support=r['support'];anchor=int(r['anchor_round'])
        key=(r['job_id'],support,r.get('rule'))
        b.require(key not in identities and seed in SEEDS,'duplicate/unknown diagnostic cell');identities.add(key)
        b.require(ph in ['BANK_READOUT','R8_TRANSFER'] and anchor in [10,60],'diagnostic phase/anchor')
        if kind=='rules':b.require(r['rule'] in RULES,'diagnostic rule')
        for scope in ['all_anchors',str(anchor)]:groups[(ph,parent,support,r.get('rule','chain'),scope)][seed].append(r)
    output=[];seedrows=[]
    for key,seed_cells in sorted(groups.items()):
        ph,parent,support,rule,scope=key
        b.require(set(seed_cells)==SEEDS,'12 seed group required')
        expected_n=(4 if ph=='BANK_READOUT' else 2)//(1 if scope=='all_anchors' else 2)
        b.require(all(len(x)==expected_n for x in seed_cells.values()),'parent/anchor/factor coverage')
        fields=(['independent_gain_vs_parent','independent_gain_fraction_parent','independent_NLL_benefit_vs_ordinary_mean','independent_NLL_benefit_vs_scalar17','standalone_measured_server_wall_seconds'] if kind=='rules' else ['weighted_individual_global_NLL_gain','aggregate_global_NLL_gain','aggregate_minus_sum_individual','public_interaction_I','independent_interaction_I'])
        metrics={f:{} for f in fields}
        if kind=='rules':metrics.update({f:{} for f in ['gain_fraction_ratio_of_within_seed_sums','public_independent_direction_agreement_fraction','alpha_zero_fraction','beta_zero_fraction','both_zero_fraction']})
        for seed,rs in sorted(seed_cells.items()):
            for field in fields:
                values=[float(r[field]) for r in rs if r[field]!='']
                metrics[field][seed]=statistics.mean(values) if len(values)==len(rs) else None
            if kind=='rules':
                denominator=sum(float(r['parent_independent_NLL']) for r in rs)
                metrics['gain_fraction_ratio_of_within_seed_sums'][seed]=sum(float(r['independent_gain_vs_parent']) for r in rs)/denominator if denominator!=0 else None
                metrics['public_independent_direction_agreement_fraction'][seed]=statistics.mean(r['public_independent_gain_direction_agrees']=='True' for r in rs)
                for field,fn in [('alpha_zero_fraction',lambda r:float(r['alpha'])==0),('beta_zero_fraction',lambda r:float(r['beta'])==0),('both_zero_fraction',lambda r:float(r['alpha'])==float(r['beta'])==0)]:metrics[field][seed]=statistics.mean(fn(r) for r in rs)
        ident=dict(zip(['phase','parent_recipe','support','rule','anchor_scope'],key))
        for field,vals in metrics.items():
            values=[x for x in vals.values() if x is not None]
            output.append({**ident,'metric':field,'objects_per_seed':expected_n,'missing_seeds':12-len(values),**describe(values)})
            seedrows.extend({**ident,'metric':field,'seed':s,'objects':expected_n,'value':v} for s,v in vals.items())
    return output,seedrows

def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic-export',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    m=b.read(a.diagnostic_export/'MANIFEST.json');b.require(m['objects']==120 and m['supports']==216,'complete diagnostic manifest')
    for name,h in m['files'].items():b.require(b.digest(a.diagnostic_export/name)==h,'diagnostic output hash '+name)
    tables=[]
    for kind,name in [('rules','rules.csv'),('chain','transfer_interactions.csv')]:
        with (a.diagnostic_export/name).open() as f:tables.append(collect(list(csv.DictReader(f)),kind))
    b.require(not a.out.exists(),'new diagnostic summary output')
    a.out.mkdir(parents=True)
    for name,rows in zip(['rule_summary.csv','rule_seed_values.csv','chain_summary.csv','chain_seed_values.csv'],[tables[0][0],tables[0][1],tables[1][0],tables[1][1]]):b.write_csv(a.out/name,rows)
    manifest={'schema':'pcla-v8-diagnostic-seed-summary-1','objects':120,'supports':216,'statistical_unit':'12 independent seeds; first equal mean over prescribed within-seed parent/factor/anchor views','gain_normalization':'mean per-object gain/parent_NLL and ratio of within-seed sums exported separately','interval':'descriptive Student t95; no new selection or hypothesis test','missing':'null retained; zero parent denominator not assigned a default','input_manifest_sha256':b.digest(a.diagnostic_export/'MANIFEST.json'),'code_sha256':b.digest(__file__),'files':{p.name:b.digest(p) for p in a.out.iterdir()}}
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'rule_summary_rows':len(tables[0][0]),'chain_summary_rows':len(tables[1][0])}))
if __name__=='__main__':main()
