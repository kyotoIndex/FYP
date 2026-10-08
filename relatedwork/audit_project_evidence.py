"""Recalculate report numbers from existing seed records; never runs training."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'relatedwork'
T4 = 2.7764451051977987  # two-sided 95% Student t, df=4

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def paired(values):
    assert len(values) == 5
    mean, sd = st.mean(values), st.stdev(values)
    margin = T4 * sd / math.sqrt(len(values))
    return dict(n=5, per_seed_difference_pp=values, mean_difference_pp=mean,
                sd_difference_pp=sd, ci95_pp=[mean-margin, mean+margin],
                positive_seed_count=sum(v > 0 for v in values))

base = ROOT / 'hetsa/results/hetsa_lora_confirm_extension_v1/analysis_v1'
hrows = read_csv(base / 'seed_method_summary.csv')
hdata = {m: {int(r['seed']): r for r in hrows if r['method'] == m}
         for m in sorted({r['method'] for r in hrows})}
seeds = [202, 303, 404, 505, 606]
assert all(sorted(rows) == seeds for rows in hdata.values())
hetsa = {'seeds': seeds, 'endpoint': 'mean client-local holdout accuracy over rounds 26-30',
         'method_means_percent': {}, 'comparisons': {}, 'cost': {},
         'cost_scope': 'time: simulated parallel FL completion from measured local elapsed plus resource/network coefficients; communication: fp32 parameter payload MiB, not serialized wire bytes'}
for method, rows in hdata.items():
    hetsa['method_means_percent'][method] = st.mean(float(rows[s]['local_accuracy_r26_30_mean'])*100 for s in seeds)
    hetsa['cost'][method] = dict(
        simulated_seconds=st.mean(float(rows[s]['cumulative_time_r30']) for s in seeds),
        parameter_payload_mib=st.mean(float(rows[s]['cumulative_comm_mb_r30']) for s in seeds))
    if method != 'hetsa_lora':
        hetsa['comparisons']['hetsa_lora-'+method] = paired([
            100*(float(hdata['hetsa_lora'][s]['local_accuracy_r26_30_mean']) - float(rows[s]['local_accuracy_r26_30_mean'])) for s in seeds])
        for column, label in [('cumulative_time_r30', 'simulated_time_reduction_percent'), ('cumulative_comm_mb_r30', 'payload_reduction_percent')]:
            hetsa['comparisons']['hetsa_lora-'+method][label] = 100*(1-st.mean(float(hdata['hetsa_lora'][s][column])/float(rows[s][column]) for s in seeds))
for published in read_csv(base / 'paired_statistics.csv'):
    calc = hetsa['comparisons'][published['comparison']]
    assert abs(calc['mean_difference_pp']-float(published['mean_difference_pp'])) < 1e-6
    assert abs(calc['ci95_pp'][0]-float(published['ci95_lower_pp'])) < 1e-6
    assert abs(calc['ci95_pp'][1]-float(published['ci95_upper_pp'])) < 1e-6

ppath = ROOT / 'relatedwork/project_context/pcla_v2_confirmatory/results/pcla_wl_ablation_v2/summary.json'
pjson = json.loads(ppath.read_text(encoding='utf-8'))
pseeds = [601, 631, 661, 691, 721]
pdata = {m: {r['seed']: r for r in pjson['records'] if r['policy'] == m}
         for m in sorted({r['policy'] for r in pjson['records']})}
assert len(pjson['records']) == 30 and all(sorted(rows) == pseeds for rows in pdata.values())
pcla = {'seeds': pseeds, 'endpoint': pjson['primary_endpoint'], 'methods': {},
        'comparisons': {}, 'decision': pjson['decision'],
        'scope': 'recomputed from seed records in saved summary; not independent reanalysis of all round logs'}
for method, rows in pdata.items():
    vals = [100*rows[s]['last5_accuracy'] for s in pseeds]
    pcla['methods'][method] = dict(mean_accuracy_percent=st.mean(vals), sample_sd_pp=st.stdev(vals))
    for col in ['total_train_seconds', 'total_probe_seconds', 'total_round_wall_seconds', 'total_serialized_traffic_bytes']:
        pcla['methods'][method]['mean_'+col] = st.mean(rows[s][col] for s in pseeds)
    if method != 'layer_only':
        pcla['comparisons']['layer_only - '+method] = paired([
            100*(pdata['layer_only'][s]['last5_accuracy']-rows[s]['last5_accuracy']) for s in pseeds])
for published in pjson['paired_comparisons']:
    calc = pcla['comparisons'][published['comparison']]
    assert abs(calc['mean_difference_pp']-published['mean_difference_pp']) < 1e-9
    assert abs(calc['ci95_pp'][0]-published['two_sided_95pct_t_interval_pp'][0]) < 1e-8
pcla['mean_paired_train_reduction_vs_pcla_percent'] = 100*(1-st.mean(pdata['layer_only'][s]['total_train_seconds']/pdata['pcla'][s]['total_train_seconds'] for s in pseeds))

paths = [base/'seed_method_summary.csv', base/'paired_statistics.csv', base/'cost_summary.csv', base/'round30_server_summary.csv', ppath,
    ROOT/'hetsa/src/fedhello_sst2.py', ROOT/'hetsa/docs/HETSA_LORA_V1.md', ROOT/'hetsa/docs/HETSA_LORA_CONFIRM_EXTENSION_V1.md',
    ROOT/'relatedwork/project_context/pcla_v2_confirmatory/docs/PCLA_WL_LAYER_STATE_V1.md', ROOT/'relatedwork/project_context/pcla_v2_confirmatory/docs/PCLA_WL_ABLATION_V2.md']
snapshot = dict(date='2026-09-30', checks='all assertions passed', t_critical_df4=T4,
    hetsa=hetsa, pcla=pcla, files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
(OUT/'project_evidence_snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Verified 25 HetSA seed records and 30 PCLA seed records; paired intervals match saved tables.')
print('PCLA-WL paired train reduction:', pcla['mean_paired_train_reduction_vs_pcla_percent'])
