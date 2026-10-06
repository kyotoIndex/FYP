#!/usr/bin/env python3
"""Read-only intake, followed by a new immutable descriptive paired analysis.

CLI: --inventory inventory.json --matrix matrix.json --output NEW_DIRECTORY
Inventory is a list (or {"records": [...]}) of the 24 bound records. Paths
are relative to inventory.json. Audit receipts require accepted=true, matching
job_id, and a `bindings` object matching BINDINGS below. Parent/source file
paths, when provided, are hash checked; semantic checkpoint verification is
owned by the runtime audit. `evaluation.loss` is the original NLL field.
Costs may use row `policy_switch_research_cost` with nonnegative byte/second
values; missing fields stay null, and only rounds 61--80 are aggregated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SEEDS = (601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661)
RECIPES = ('U014', 'U018')
ROUNDS = (61, 65, 70, 80)
T95 = 2.200985160
BINDINGS = ('parent_state_sha256', 'parent_checkpoint_file_sha256',
            'parent_config_sha256', 'new_source_manifest_sha256',
            'control_metrics_sha256', 'off_metrics_sha256')
COST_FIELDS = ('necessary_seconds', 'research_seconds', 'upload_bytes', 'download_bytes')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f'invalid {name}')
    return value


def evaluations(path, *, control=False):
    rows = read_json(path)
    require(isinstance(rows, list), f'metrics must be a round list: {path}')
    indexed = {}
    for row in rows:
        r = row.get('round')
        require(type(r) is int and r >= 1 and (control or r <= 80) and r not in indexed,
                f'invalid/duplicate round: {r}')
        ev = row.get('evaluation', {})
        a = number(ev.get('accuracy'), f'accuracy round {r}')
        n = number(ev.get('nll', ev.get('loss')), f'NLL round {r}')
        require(0 <= a <= 1 and n >= 0, f'out of range evaluation: {r}')
        indexed[r] = row
    last_round = max(indexed, default=0) if control else 80
    require(last_round >= 80 and set(indexed) == set(range(1, last_round + 1)),
            f'incomplete continuous metrics: {path}')
    return indexed


def evaluation(row):
    ev = row['evaluation']
    return {'accuracy': ev['accuracy'], 'nll': ev.get('nll', ev.get('loss'))}


def describe(values):
    require(len(values) == 12, 'paired summaries require exactly 12 seeds')
    mean = statistics.mean(values)
    half = T95 * statistics.stdev(values) / math.sqrt(12)
    return {'n_pairs': 12, 'mean': mean, 'ci95': [mean-half, mean+half],
            'positive': sum(x > 0 for x in values), 'negative': sum(x < 0 for x in values),
            'zero': sum(x == 0 for x in values), 'values': values}


def analyze(inventory_path, matrix_path, output_path):
    inventory_path, matrix_path, output_path = map(Path, (inventory_path, matrix_path, output_path))
    require(not output_path.exists(), 'output must be a new directory')
    inventory_sha, matrix_sha = sha256(inventory_path), sha256(matrix_path)
    raw = read_json(inventory_path)
    records = raw.get('records') if isinstance(raw, dict) else raw
    matrix = read_json(matrix_path)
    require(isinstance(records, list) and len(records) == 24, 'inventory must contain all 24 pairs')
    require(isinstance(matrix, list) and len(matrix) == 24, 'matrix must contain all 24 pairs')
    expected = {(r, s) for r in RECIPES for s in SEEDS}
    by_job = {}
    matrix_pairs = set()
    for m in matrix:
        key = (m['source_recipe'], int(m['seed']))
        require(key in expected and key not in matrix_pairs and m['job_id'] not in by_job, 'matrix identity mismatch')
        require(m['job_id'] == f'OFF_{key[0]}_s{key[1]}_r60_to80', 'matrix job identity mismatch')
        matrix_pairs.add(key)
        by_job[m['job_id']] = m
    require(matrix_pairs == expected, 'matrix fixed seed set mismatch')
    result, seen, tracked = [], set(), {inventory_path: inventory_sha, matrix_path: matrix_sha}
    for rec in records:
        job = rec['job_id']
        require(job in by_job, f'unknown job: {job}')
        m = by_job[job]
        key = (rec['parent_recipe'], int(rec['seed']))
        require(key == (m['source_recipe'], int(m['seed'])) and key not in seen, 'duplicate/mismatched pair identity')
        seen.add(key)
        for name in BINDINGS + ('audit_receipt_sha256',):
            require(isinstance(rec.get(name), str) and re.fullmatch('[0-9a-f]{64}', rec[name]), f'missing/invalid {name}')
        for name in ('parent_state_sha256', 'parent_config_sha256'):
            require(rec[name] == m[name], f'matrix binding mismatch: {name}')
        def bound_file(path_key, hash_key):
            p = Path(rec[path_key])
            p = p if p.is_absolute() else inventory_path.parent / p
            actual = sha256(p)
            require(actual == rec[hash_key], f'file hash mismatch: {path_key}')
            require(p not in tracked or tracked[p] == actual, 'inconsistent file binding')
            tracked[p] = actual
            return p
        control_path = bound_file('control_metrics_path', 'control_metrics_sha256')
        off_path = bound_file('off_metrics_path', 'off_metrics_sha256')
        receipt_path = bound_file('audit_receipt_path', 'audit_receipt_sha256')
        require(control_path.resolve() != off_path.resolve(), 'control and off must be different objects')
        for pk, hk in (('parent_checkpoint_path', 'parent_checkpoint_file_sha256'),
                       ('parent_config_path', 'parent_config_sha256'),
                       ('new_source_manifest_path', 'new_source_manifest_sha256')):
            if pk in rec:
                bound_file(pk, hk)
        receipt = read_json(receipt_path)
        require(receipt.get('accepted') is True and receipt.get('job_id') == job, 'not formally accepted')
        bindings = receipt.get('bindings', receipt)
        require(all(bindings.get(k) == rec[k] for k in BINDINGS), 'audit evidence binding mismatch')
        control, off = evaluations(control_path, control=True), evaluations(off_path)
        require(all(control[r] == off[r] for r in range(1, 61)), 'original prefix differs')
        baseline = evaluation(control[60])
        paths = {}
        for r in ROUNDS:
            on_ev, off_ev = evaluation(control[r]), evaluation(off[r])
            da, dn = off_ev['accuracy']-on_ev['accuracy'], off_ev['nll']-on_ev['nll']
            paths[str(r)] = {'on': on_ev, 'off': off_ev, 'accuracy_difference_fraction': da,
                             'accuracy_difference_pp': 100*da, 'nll_difference': dn,
                             'on_change_from_parent60': {k: on_ev[k]-baseline[k] for k in baseline},
                             'off_change_from_parent60': {k: off_ev[k]-baseline[k] for k in baseline}}
        cost = {}
        for field in COST_FIELDS:
            vals = [off[r].get('policy_switch_research_cost', {}).get(field) for r in range(61, 81)]
            for v in vals:
                if v is not None:
                    require(number(v, field) >= 0, f'negative cost: {field}')
            cost[field] = sum(vals) if all(v is not None for v in vals) else None
        result.append({'job_id': job, 'parent_recipe': key[0], 'seed': key[1],
                       'parent60': baseline, 'rounds': paths, 'bindings': {k: rec[k] for k in BINDINGS},
                       'audit_receipt_sha256': rec['audit_receipt_sha256'],
                       'paths': {k: rec[k] for k in ('control_metrics_path', 'off_metrics_path', 'audit_receipt_path')},
                       'additional_research_cost_rounds61_80': cost})
    require(seen == expected, 'incomplete fixed pair matrix')
    result.sort(key=lambda x: (x['parent_recipe'], x['seed']))
    summaries = {}
    for recipe in RECIPES:
        subset = [r for r in result if r['parent_recipe'] == recipe]
        summaries[recipe] = {str(r): {metric: describe([x['rounds'][str(r)][metric] for x in subset])
                             for metric in ('accuracy_difference_fraction', 'accuracy_difference_pp', 'nll_difference')}
                             for r in ROUNDS}
    summary = {'primary_endpoint': 80, 'secondary_rounds': [61, 65, 70], 'parent_baseline': 60,
               't95_df11': T95, 'pair_count': 24, 'independent_seed_count': 12,
               'analysis': 'post-hoc descriptive; each parent recipe analyzed separately; no automated claim verdict',
               'inventory_sha256': inventory_sha, 'matrix_sha256': matrix_sha, 'recipes': summaries}
    # Detect changes during intake before the first output write.
    require(all(sha256(p) == h for p, h in tracked.items()), 'input changed during analysis')
    output_path.mkdir(parents=True, exist_ok=False)
    (output_path/'paired_values.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in result))
    (output_path/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (output_path/'HASHES').write_text(''.join(f'{sha256(output_path/n)}  {n}\n' for n in ('paired_values.jsonl', 'summary.json')))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', required=True)
    p.add_argument('--matrix', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        analyze(args.inventory, args.matrix, args.output)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as e:
        p.exit(2, f'analysis rejected: {e}\n')


if __name__ == '__main__':
    main()
