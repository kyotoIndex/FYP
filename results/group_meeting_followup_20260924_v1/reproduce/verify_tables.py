#!/usr/bin/env python3
"""Verify released PCLA endpoint tables and derived statistics; no training required."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 13, 21, 42, 87)
T95 = 2.7764451052


def records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=2e-11, rel_tol=0):
        raise ValueError(f'{label}: {actual!r} != {expected!r}')


def main() -> None:
    sources = json.loads((ROOT / 'reproduce/data_sources.json').read_text())
    for item in sources:
        path = ROOT / item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'File checksum mismatch: {item["file"]}')
    raw = records(ROOT / 'data/endpoints.csv')
    lookup: dict[tuple[str, ...], tuple[float, str]] = {}
    for row in raw:
        key = tuple(row[k] for k in ('cell', 'metric', 'scope', 'round_start', 'round_end', 'seed'))
        value = (float(row['value']), row['sample_identity_sha256'])
        if key in lookup and lookup[key] != value:
            raise ValueError(f'Conflicting repeated endpoint: {key}')
        lookup[key] = value
    comparisons = records(ROOT / 'data/comparisons.csv')
    for c in comparisons:
        differences = []
        for seed in SEEDS:
            left = lookup[(c['left'], c['metric'], c['scope'], c['round_start'], c['round_end'], str(seed))]
            right = lookup[(c['right'], c['right_metric'], c['scope'], c['right_round_start'], c['right_round_end'], str(seed))]
            if left[1] != right[1]:
                raise ValueError(f'Paired evaluation identity differs: {c["comparison"]}, seed {seed}')
            differences.append(left[0] - right[0])
        mean = st.mean(differences)
        radius = T95 * st.stdev(differences) / math.sqrt(5)
        close(mean, float(c['mean_difference']), c['comparison'])
        low, high = json.loads(c['ci95_descriptive'])
        close(mean - radius, low, c['comparison'] + ' lower interval')
        close(mean + radius, high, c['comparison'] + ' upper interval')
    secondary = records(ROOT / 'data/secondary_rounds.csv')
    per_seed = records(ROOT / 'data/secondary_per_seed.csv')
    for row in per_seed:
        series = sorted([r for r in secondary if r['cell'] == row['cell'] and r['seed'] == row['seed']], key=lambda r: int(r['round']))
        if [int(r['round']) for r in series] != list(range(1, 101)):
            raise ValueError(f'Incomplete 100-round series: {row["cell"]}/{row["seed"]}')
        x = [float(r['participant_positive_fraction']) for r in series]
        y = [float(r['predicted_positive_fraction']) for r in series]
        close(st.pstdev(x), float(row['participant_fraction_sd']), 'participant fraction SD')
        close(st.pstdev(y), float(row['predicted_fraction_sd']), 'predicted fraction SD')
        close(st.correlation(x, y), float(row['positive_fraction_pearson_r']), 'correlation')
    print(json.dumps({'files_verified': len(sources), 'endpoint_rows': len(raw), 'paired_comparisons': len(comparisons), 'secondary_round_rows': len(secondary), 'secondary_seed_series': len(per_seed), 'passed': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
