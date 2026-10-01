#!/usr/bin/env python3
"""Recompute every number quoted in README.md from the published data files.

Run from the results directory:  python3 reproduce/verify_tables.py
The script prints the recomputed values and exits non-zero if any quoted
value (rounded as in README.md) disagrees with the data.
"""
import csv, gzip, json, math, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, 'data', '1_budget_selection')
P2 = os.path.join(ROOT, 'data', '2_server_feedback')
P3 = os.path.join(ROOT, 'data', '3_controller_switch_off')
T95_DF11 = 2.200985160082949
failures = []
report = {}


def rows(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check(name, value, quoted, digits):
    got = round(value, digits)
    report[name] = got
    if abs(got - quoted) > 10 ** (-digits) / 2 + 1e-12:
        failures.append(f'{name}: data {got} vs README {quoted}')


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    m = statistics.mean(d)
    h = T95_DF11 * statistics.stdev(d) / math.sqrt(len(d))
    return m, m - h, m + h, sum(x > 0 for x in d), sum(x < 0 for x in d), len(d)


# ---------------------------------------------------------------- part 1
v = rows(os.path.join(P1, 'validation', 'v_endpoints.csv'))
acc1 = {}
for r in v:
    acc1.setdefault(int(r['slot']), {})[int(r['seed'])] = 100 * float(r['b8_accuracy'])
assert len(acc1) == 16 and all(len(x) == 12 for x in acc1.values())
quoted1 = {1: 85.91, 2: 84.81, 3: 86.14, 4: 86.33, 5: 85.43, 6: 85.36, 7: 89.19, 8: 90.50,
           9: 87.35, 10: 88.25, 11: 87.19, 12: 84.97, 13: 88.77, 14: 84.80, 15: 83.98, 16: 85.52}
for slot, q in quoted1.items():
    check(f'p1_slot{slot}_b8_accuracy', statistics.mean(acc1[slot].values()), q, 2)
m, lo, hi, pos, neg, n = paired(acc1[3], acc1[16])
check('p1_score_minus_random_index', m, 0.61, 2)
check('p1_score_minus_random_index_low', lo, 0.35, 2)
check('p1_score_minus_random_index_high', hi, 0.88, 2)
check('p1_score_minus_random_index_positive', pos, 12, 0)
m, lo, hi, pos, neg, n = paired(acc1[3], acc1[8])
check('p1_rotated_score_minus_rank8', m, -4.36, 2)
check('p1_rotated_score_minus_rank8_negative', neg, 11, 0)
m, *_ = paired(acc1[4], acc1[3])
check('p1_rotated_cyclic_minus_score', m, 0.19, 2)

surf = rows(os.path.join(P1, 'development', 'representation_lr_surface.csv'))
best = {}
for r in surf:
    k = (r['method'], r['representation'])
    best[k] = max(best.get(k, 0), 100 * float(r['b8_raw_accuracy_mean']))
check('p1_dev_best_local_original', best[('local', 'N')], 86.28, 2)
check('p1_dev_best_local_svd', best[('local', 'S')], 83.81, 2)
check('p1_dev_best_local_rotated', best[('local', 'T')], 86.86, 2)
check('p1_dev_best_local_rescaled', best[('local', 'D')], 86.76, 2)
recipes = rows(os.path.join(P1, 'development', 'recipe_117.csv'))
check('p1_dev_recipe_count', len(recipes), 117, 0)
check('p1_dev_run_count', len(rows(os.path.join(P1, 'development', 'seed_endpoints_351.csv'))), 351, 0)

join = {int(r['slot']): r for r in rows(os.path.join(P1, 'costs', 'quality_cost_join.csv'))}
for slot, q in {1: 1042, 3: 907, 4: 832, 6: 909, 7: 2068, 8: 1047, 13: 1692}.items():
    check(f'p1_cost_seconds_slot{slot}', float(join[slot]['COST1_required_seconds']), q, 0)
check('p1_fedsb_budget_fraction', 100 * float(join[13]['V_mean_deployed_bytes']) / 2382653750, 27.9, 1)
check('p1_rank8_return_round', float(join[8]['V_mean_return_round']), 100, 0)
check('p1_rank16_return_round', float(join[9]['V_mean_return_round']), 50, 0)
check('p1_rank4_return_round', float(join[7]['V_mean_return_round']), 198, 0)

desc = {}
for d in jsonl(os.path.join(P1, 'diagnostics', 'selection_bank', 'cohort_across_seed_descriptive.jsonl')):
    desc[(d['family'], d['choice_or_comparison'], d['metric'])] = d['mean_of_seed_values']
check('p1_rank_corr_compressed_vs_full', desc[('client_rank', 'h32_full_four', 'mean_defined_client_value')], 0.77, 2)
check('p1_rank_corr_prediction_vs_16_steps', desc[('client_rank', 'h32_vs_16', 'mean_defined_client_value')], 0.22, 2)
check('p1_rank_corr_full_gradient_vs_16_steps', desc[('client_rank', 'full_vs_16', 'mean_defined_client_value')], 0.23, 2)
check('p1_four_step_minus_one_step', desc[('client_rank', 'actual4_minus_actual1', 'mean_defined_client_value')], 0.033, 3)
check('p1_bank_vs_local_parent_loss_pct', 100 * desc[('selection16', 'finite_bank_search', 'delta_over_parent_loss')], 24.82, 2)
check('p1_cyclic_vs_local_parent_loss_pct', 100 * desc[('selection16', 'cyclic', 'delta_over_parent_loss')], 15.99, 2)

inter, lora_gap = {}, {}
with gzip.open(os.path.join(P1, 'diagnostics', 'selection_bank', 'cohort_object_values.jsonl.gz'), 'rt', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d['choice'] == 'local' and d['prefix'] == 16:
            joint, head, lora = (d['aggregate_readout2048_benefit'], d['head_only_readout2048_benefit'],
                                 d['lora_only_readout2048_benefit'])
            inter.setdefault(d['seed'], []).append(joint - head - lora)
            lora_gap.setdefault(d['seed'], []).append(lora - joint)
iv = [statistics.mean(x) for x in inter.values()]
lv = [statistics.mean(x) for x in lora_gap.values()]
check('p1_head_lora_interaction', statistics.mean(iv), -0.101, 3)
check('p1_head_lora_interaction_negative_seeds', sum(x < 0 for x in iv), 12, 0)
check('p1_lora_only_minus_joint', statistics.mean(lv), 0.085, 3)
check('p1_lora_only_better_seeds', sum(x > 0 for x in lv), 9, 0)

# ---------------------------------------------------------------- part 2
e = rows(os.path.join(P2, 'validation', 'endpoints.csv'))
acc2, nll2, sec2 = {}, {}, {}
for r in e:
    acc2.setdefault(r['recipe_id'], {})[int(r['seed'])] = 100 * float(r['raw_B8_accuracy'])
    nll2.setdefault(r['recipe_id'], {})[int(r['seed'])] = float(r['raw_B8_NLL'])
    sec2.setdefault(r['recipe_id'], {})[int(r['seed'])] = float(r['required_prefix_seconds'])
assert len(acc2) == 9 and all(len(x) == 12 for x in acc2.values())
quoted2 = {  # recipe: (accuracy, NLL, seconds over seeds 601/607/613)
    'U002': (89.78, 0.334, 1048), 'U010': (89.84, 0.319, 1037), 'U014': (88.27, 0.289, 1495),
    'U018': (88.48, 0.284, 1516), 'U020': (88.72, 0.280, 1472), 'U023': (89.93, 0.279, 1065),
    'U027': (87.12, 0.369, 1086), 'U004': (89.95, 0.396, 2061), 'U006': (87.09, 0.377, 1699)}
for rid, (qa, qn, qs) in quoted2.items():
    check(f'p2_{rid}_accuracy', statistics.mean(acc2[rid].values()), qa, 2)
    check(f'p2_{rid}_nll', statistics.mean(nll2[rid].values()), qn, 3)
    check(f'p2_{rid}_seconds_cost_seeds', statistics.mean(sec2[rid][s] for s in (601, 607, 613)), qs, 0)
for name, a, b, q, qneg in [('block_minus_rank8', 'U018', 'U002', -1.30, 10),
                            ('separable_minus_rank8', 'U020', 'U002', -1.06, 9),
                            ('block_minus_headfit', 'U018', 'U023', -1.45, 10),
                            ('headfit_minus_rank8', 'U023', 'U002', 0.15, None),
                            ('separable_minus_scalar', 'U020', 'U014', 0.45, None)]:
    m, lo, hi, pos, neg, n = paired(acc2[a], acc2[b])
    check(f'p2_{name}', m, q, 2)
    if qneg is not None:
        check(f'p2_{name}_negative_seeds', neg, qneg, 0)
m, lo, hi, pos, neg, n = paired(nll2['U023'], nll2['U002'])
check('p2_headfit_minus_rank8_nll', m, -0.055, 3)
check('p2_headfit_minus_rank8_nll_lower_seeds', neg, 9, 0)

dev = rows(os.path.join(P2, 'development', 'recipe_summary.csv'))
check('p2_dev_recipe_count', len(dev), 32, 0)
check('p2_dev_best_fedexp', 100 * max(float(r['mean_raw_B8_accuracy']) for r in dev if r['recipe_id'] in ('U029', 'U030', 'U031', 'U032')), 84.70, 2)

rule = {}
for r in rows(os.path.join(P2, 'diagnostics', 'summary', 'rule_summary.csv')):
    if r['phase'] == 'R8_TRANSFER':
        rule[(r['rule'], r['anchor_scope'], r['metric'])] = float(r['mean']) if r['mean'] else float('nan')
check('p2_rank8_scalar4_vs_mean', rule[('scalar4', 'all_anchors', 'independent_NLL_benefit_vs_ordinary_mean')], 0.200, 3)
check('p2_rank8_scalar17_vs_mean', rule[('scalar17', 'all_anchors', 'independent_NLL_benefit_vs_ordinary_mean')], 0.202, 3)
check('p2_rank8_block_vs_mean', rule[('block16', 'all_anchors', 'independent_NLL_benefit_vs_ordinary_mean')], 0.204, 3)
check('p2_rank8_block_vs_scalar17', rule[('block16', 'all_anchors', 'independent_NLL_benefit_vs_scalar17')], 0.002, 3)
check('p2_rank8_block_vs_mean_round10', rule[('block16', '10', 'independent_NLL_benefit_vs_ordinary_mean')], 0.379, 3)
check('p2_rank8_block_vs_mean_round60', rule[('block16', '60', 'independent_NLL_benefit_vs_ordinary_mean')], 0.029, 3)
chain = {}
for r in rows(os.path.join(P2, 'diagnostics', 'summary', 'chain_summary.csv')):
    chain[(r['phase'], r['parent_recipe'], r['support'], r['anchor_scope'], r['metric'])] = r
ri = chain[('R8_TRANSFER', 'D066', 'rank8', 'all_anchors', 'independent_interaction_I')]
check('p2_rank8_interaction', float(ri['mean']), -0.118, 3)
check('p2_rank8_interaction_negative_seeds', 12 - int(ri['positive_seeds']), 12, 0)

costs2 = rows(os.path.join(P2, 'validation', 'required_costs.csv'))
def cost_mean(rid, col):
    return statistics.mean(float(x[col]) for x in costs2 if x['recipe_id'] == rid and int(x['seed']) in (601, 607, 613))
check('p2_block_extra_seconds', cost_mean('U018', 'required_prefix_seconds') - cost_mean('U002', 'required_prefix_seconds'), 470, -1)
check('p2_block_extra_client_seconds', cost_mean('U018', 'client_training') - cost_mean('U002', 'client_training'), 340, -1)
check('p2_block_extra_server_seconds', cost_mean('U018', 'coordinate_and_server_decision') - cost_mean('U002', 'coordinate_and_server_decision'), 120, -1)
check('p2_block_time_ratio', statistics.mean(sec2['U018'][s] / sec2['U002'][s] for s in (601, 607, 613)), 1.45, 2)

noop = {}
with gzip.open(os.path.join(P2, 'validation', 'controller', 'controller_originals.jsonl.gz'), 'rt', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if 61 <= d['round'] <= 100:
            o = d['original']
            noop.setdefault(d['recipe_id'], {}).setdefault(d['seed'], []).append(o['parent_hash'] == o.get('committed_state_hash', o.get('accepted_hash')))
for rid, q in (('U014', 87.29), ('U018', 83.54), ('U020', 80.00), ('U002', 0.00)):
    per_seed = [100 * sum(v) / len(v) for v in noop[rid].values()]
    check(f'p2_{rid}_late_no_update_pct', statistics.mean(per_seed), q, 2)

curve = {}
for r in rows(os.path.join(P2, 'validation', 'curves.csv.gz')):
    curve.setdefault((r['recipe_id'], int(r['round'])), []).append(100 * float(r['raw_accuracy']))
quoted_curve = {('U002', 10): 80.06, ('U002', 60): 87.40, ('U002', 80): 88.66, ('U002', 100): 89.78,
                ('U018', 10): 81.01, ('U018', 60): 87.87, ('U018', 80): 88.07, ('U018', 100): 88.33,
                ('U014', 60): 85.31, ('U014', 80): 87.93}
for (rid, rnd), q in quoted_curve.items():
    assert len(curve[(rid, rnd)]) == 12
    check(f'p2_{rid}_round{rnd}_accuracy', statistics.mean(curve[(rid, rnd)]), q, 2)

# ---------------------------------------------------------------- part 3
s = json.load(open(os.path.join(P3, 'summary.json'), encoding='utf-8'))
quoted3 = {('U014', '61'): (-10.77, 12), ('U014', '65'): (-6.79, 10), ('U014', '70'): (-3.40, 9),
           ('U014', '80'): (-1.64, 5), ('U018', '61'): (-9.35, 12), ('U018', '65'): (-6.96, 12),
           ('U018', '70'): (-2.81, 8), ('U018', '80'): (-1.98, 5)}
for (rid, rnd), (qa, qneg) in quoted3.items():
    a = s['recipes'][rid][rnd]['accuracy_difference_pp']
    check(f'p3_{rid}_r{rnd}_accuracy', a['mean'], qa, 2)
    check(f'p3_{rid}_r{rnd}_negative_seeds', a['negative'], qneg, 0)
for rid, q in (('U014', 0.086), ('U018', 0.108)):
    check(f'p3_{rid}_r80_nll', s['recipes'][rid]['80']['nll_difference']['mean'], q, 3)

fc = json.load(open(os.path.join(P3, 'costs', 'FORMAL_COSTS.json'), encoding='utf-8'))
check('p3_branch_count', len(fc), 24, 0)
check('p3_necessary_hours', sum(x['necessary_seconds'] for x in fc) / 3600, 1.4, 1)
check('p3_additional_gb', sum(x['additional_bytes'] for x in fc) / 1e9, 11.25, 2)
for slot, q in {1: 128, 2: 95, 3: 95, 4: 92, 5: 91, 6: 91, 7: 198, 8: 100, 9: 50, 10: 71, 11: 147,
                12: 147, 13: 200, 14: 66, 15: 86, 16: 95}.items():
    check(f'p1_slot{slot}_return_round', float(join[slot]['V_mean_return_round']), q, 0)

print(json.dumps(report, indent=1, ensure_ascii=False))
if failures:
    print('FAILED:\n  ' + '\n  '.join(failures))
    sys.exit(1)
print(f'PASS: {len(report)} quoted values match the data')
