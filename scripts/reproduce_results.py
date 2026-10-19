#!/usr/bin/env python3
"""Recompute released tables from trajectories; no model, dataset or GPU required."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'federated_finetuning/experiments/scra_diagnostics_20261017'
sys.path.insert(0, str(EXP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'federated_finetuning/runs')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/results')
    args = parser.parse_args()
    from scra.batch import write_batch_report
    from scra.batch2 import write_batch2_report
    from scra.c2fed import write_c2fed_report
    import numpy as np

    # The original report writers write beside their inputs. Recompute in an isolated
    # temporary copy to keep released evidence untouched.
    with tempfile.TemporaryDirectory(prefix='amplora-tables-') as temporary:
        runs = Path(temporary) / 'runs'
        shutil.copytree(args.runs, runs, ignore=shutil.ignore_patterns(
            '*.pt', '*.pth', '*.safetensors', '__pycache__', 'source_snapshot'))
        for p in (runs / 'c2_fed_20261018').glob('*/fed.json'):
            blob = json.loads(p.read_text())
            plan = json.loads((p.parent / 'identity.json').read_text())['plan']
            expected = plan['c2fed']['rounds']
            if blob['eval'][-1]['round'] != expected:
                raise ValueError(f'{p.parent.name}: incomplete ({blob["eval"][-1]["round"]}/{expected})')
        write_batch_report(runs)
        write_batch2_report(runs)
        config = json.loads((EXP / 'plans_c2fed/main_amp_zf.json').read_text())
        write_c2fed_report(runs / 'c2_fed_20261018', config['c2fed_report'])
        documents = [('scra_batch_20261018', 'summary.json', 'SUMMARY.md'),
                     ('scra_batch2_20261018', 'summary.json', 'SUMMARY.md'),
                     ('c2_fed_20261018', 'report_c2_1.json', 'REPORT_C2_1.md')]
        for folder, filename, report in documents:
            expected = json.loads((args.runs / folder / filename).read_text())
            actual = json.loads((runs / folder / filename).read_text())
            if actual != expected:
                keys = sorted(k for k in expected.keys() | actual.keys() if actual.get(k) != expected.get(k))
                raise ValueError(f'{folder}/{filename}: released result differs in {keys}')
            out = args.output / folder
            out.mkdir(parents=True, exist_ok=True)
            for name in (filename, report):
                shutil.copy2(runs / folder / name, out / name)
        b1 = json.loads((runs / documents[0][0] / 'summary.json').read_text())['runs']
        c2 = json.loads((runs / documents[2][0] / 'report_c2_1.json').read_text())
        if c2['multiseed']['blocked']:
            raise ValueError(f'C2 comparisons blocked: {c2["multiseed"]["blocked"]}')
        rows = []
        for model, data, methods in [('RoBERTa-base', b1, ('r8', 'vref', 'amp', 'L2')),
                                     ('Qwen2.5-0.5B', c2['multiseed']['runs'],
                                      ('r8', 'vref_zf', 'amp_zf', 'vref', 'amp', 'fslora_paper'))]:
            for method in methods:
                vals = [data[f'{method}_s{s}']['core'] for s in (2040, 2041, 2042)]
                rows.append({'model': model, 'method': method, 'seeds': [2040, 2041, 2042],
                             'accuracy_per_seed': vals, 'mean': float(np.mean(vals)),
                             'sample_sd': float(np.std(vals, ddof=1))})
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'main_results.json').write_text(json.dumps(rows, indent=2) + '\n')
        lines = ['# Recomputed main results', '', 'Accuracy in percentage points; sample SD (ddof=1).', '',
                 '| Model | Method | Seed 2040 | Seed 2041 | Seed 2042 | Mean ± sample SD |',
                 '|---|---|---:|---:|---:|---:|']
        for row in rows:
            vals = ' | '.join(f'{v:.2f}' for v in row['accuracy_per_seed'])
            lines.append(f'| {row["model"]} | {row["method"]} | {vals} | {row["mean"]:.2f} ± {row["sample_sd"]:.2f} |')
        (args.output / 'MAIN_RESULTS.md').write_text('\n'.join(lines) + '\n')
        print('PASS: all three published result JSON files recomputed exactly; C2 comparisons complete.')
        print('\n'.join(lines[4:]))
        print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
