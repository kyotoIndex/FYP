"""Descriptive, seed-paired summaries of a complete and hashed VAL base export."""
import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t
import export_validation_base01 as base

BASE_SHA = 'e3c988f2042fe6956bbf8a4b60cbe81bc763d35650191f397d4f2e5f96bb8c7f'
ACCURACY = ['raw_B8_accuracy', 'EMA_B8_accuracy', 'raw_half_accuracy',
            'raw_r100_accuracy', 'raw_r96_100_mean', 'raw_last5_mean',
            'raw_last20_min', 'raw_last20_lower5_mean']
METRICS = ACCURACY + ['raw_B8_NLL', 'required_prefix_seconds', 'returned_round']
PRIMARY_COST_SEEDS = [601, 607, 613]


def describe(values):
    base.require(len(values) >= 2 and all(math.isfinite(v) for v in values),
                 'finite independent seed values required')
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    margin = float(t.ppf(.975, len(values)-1)) * sd / math.sqrt(len(values))
    return {'n_seeds': len(values), 'mean': mean, 'sample_sd': sd,
            't95_low': mean-margin, 't95_high': mean+margin,
            'minimum': min(values), 'maximum': max(values),
            'positive_seeds': sum(v > 0 for v in values),
            'negative_seeds': sum(v < 0 for v in values),
            'zero_seeds': sum(v == 0 for v in values)}


def validate_rows(rows):
    expected = {(r, s) for r in base.RECIPES for s in base.SEEDS}
    cells = {}
    for original in rows:
        row = dict(original)
        recipe, seed = row['recipe_id'], int(row['seed'])
        key = recipe, seed
        base.require(key in expected and key not in cells, 'duplicate/unknown seed cell')
        slot = base.RECIPES.index(recipe)+1
        base.require(int(row['slot']) == slot and row['job_id'] == f'VAL_S{slot:02d}_s{seed}',
                     'recipe/slot/job identity')
        base.require(row['hardware_family'] == 'RTX4090D', 'authorized hardware family')
        base.require(row['eval_role'] == 'recipe_dev', 'evaluation role')
        for metric in METRICS:
            row[metric] = float(row[metric])
            base.require(math.isfinite(row[metric]), 'nonfinite endpoint')
        base.require(all(0 <= row[m] <= 1 for m in ACCURACY), 'accuracy domain')
        base.require(row['raw_B8_NLL'] >= 0 and row['required_prefix_seconds'] > 0,
                     'loss/time domain')
        base.require(100 <= row['returned_round'] <= 200, 'budget return domain')
        cells[key] = row
    base.require(set(cells) == expected, 'full 9x12 accepted export required')
    return cells


def summaries(rows):
    cells = validate_rows(rows)
    methods, differences, seed_pairs, times = [], [], [], []
    for recipe in base.RECIPES:
        for metric in METRICS:
            scale = 100 if metric in ACCURACY else 1
            unit = 'percentage_points' if metric in ACCURACY else (
                'seconds' if metric == 'required_prefix_seconds' else
                'rounds' if metric == 'returned_round' else 'NLL')
            values = [cells[recipe, s][metric]*scale for s in base.SEEDS]
            methods.append({'recipe_id': recipe, 'metric': metric, 'unit': unit,
                            **describe(values)})
    for left, right in itertools.combinations(base.RECIPES, 2):
        for metric in METRICS:
            scale = 100 if metric in ACCURACY else 1
            values = []
            for seed in base.SEEDS:
                a, b = cells[left, seed], cells[right, seed]
                delta = (a[metric]-b[metric])*scale
                values.append(delta)
                seed_pairs.append({'left': left, 'right': right, 'seed': seed,
                    'metric': metric, 'left_original': a[metric], 'right_original': b[metric],
                    'left_minus_right': delta, 'accuracy_scale': scale,
                    'left_returned_round': a['returned_round'],
                    'right_returned_round': b['returned_round'],
                    'left_required_seconds': a['required_prefix_seconds'],
                    'right_required_seconds': b['required_prefix_seconds'],
                    'same_physical_gpu': a['gpu_uuid'] == b['gpu_uuid']})
            differences.append({'left': left, 'right': right, 'metric': metric,
                                'direction': 'left-minus-right',
                                'accuracy_scale': scale, **describe(values)})
        for label, seeds in [('primary_cost_seed_block', PRIMARY_COST_SEEDS),
                             ('all_seeds_descriptive', base.SEEDS)]:
            ratios = [cells[left, s]['required_prefix_seconds'] /
                      cells[right, s]['required_prefix_seconds'] for s in seeds]
            times.append({'left': left, 'right': right, 'scope': label,
                          'ratio': 'left-seconds/right-seconds',
                          'same_physical_gpu_seeds': sum(cells[left, s]['gpu_uuid'] ==
                                                       cells[right, s]['gpu_uuid'] for s in seeds),
                          **describe(ratios)})
    return methods, differences, seed_pairs, times


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-export', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest_path = args.base_export/'MANIFEST.json'
    manifest = base.read(manifest_path)
    base.require(manifest['jobs'] == 108 and manifest['exporter_sha256'] == BASE_SHA
                 and manifest['source_index_sha256'] == base.INDEX_SHA,
                 'accepted base exporter identity')
    for name, sha in manifest['files'].items():
        base.require(base.digest(args.base_export/name) == sha, 'base output hash '+name)
    base.require('endpoints.csv' in manifest['files'], 'bound endpoints required')
    with (args.base_export/'endpoints.csv').open() as f:
        outputs = summaries(list(csv.DictReader(f)))
    base.require(not args.out.exists(), 'new summary output directory required')
    args.out.mkdir(parents=True)
    names = ['method_summary.csv', 'paired_summary.csv', 'seed_pairs.csv', 'time_ratios.csv']
    for name, rows in zip(names, outputs):
        base.write_csv(args.out/name, rows)
    report = {'schema': 'pcla-v8-validation-seed-summary-1', 'jobs': 108,
        'statistical_unit': 'independent seed', 'seeds': base.SEEDS,
        'interval': 'two-sided descriptive Student t95; df=n_seeds-1',
        'primary_quality': 'raw_B8_accuracy',
        'other_windows': 'separate descriptive outcomes; preserve native half=B8 when recorded',
        'cost_scope': 'seven necessary algorithm fields; excludes separate research/operating costs',
        'time_ratio_aggregation': 'mean of within-seed ratios, not ratio of means',
        'cost_hardware': 'exclusive 1GPU/4CPU; physical GPU equality preserved per pair',
        'decision': 'root scientific assessment uses frozen DECISIONS.md; no automatic job launch',
        'summary_code_sha256': base.digest(__file__),
        'base_manifest_sha256': base.digest(manifest_path),
        'files': {name: base.digest(args.out/name) for name in names}}
    (args.out/'MANIFEST.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'jobs': 108, 'paired_seed_rows': len(outputs[2]), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
