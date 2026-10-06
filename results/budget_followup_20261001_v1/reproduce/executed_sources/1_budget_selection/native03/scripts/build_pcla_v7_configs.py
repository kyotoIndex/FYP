#!/usr/bin/env python3
"""Expand the fixed D0 matrix without silently enabling unsupported methods."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcla_research_v7.configuration import compile_first_batch, load_common, expand_for_runner
from pcla_research_v7.data import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--technical', action='store_true')
    args = p.parse_args()
    handoff, data = args.root / 'handoff', args.root / 'data/pcla_v7'
    common = load_common(handoff / 'configs/common.yaml')
    compiled = compile_first_batch(handoff / 'configs/recipes.json', handoff / 'jobs/first_batch.jsonl')
    b8 = json.loads((data / 'b8_wire.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    result = {'compiled': [], 'unsupported': [], 'b8_bytes': b8['total_bytes']}
    for item in compiled:
        if args.technical and (item['job']['seed'] != 401 or item['job']['lr'] != .0003):
            continue
        try:
            expanded = expand_for_runner(item, common, manifest_path=data / 'train_roles.json',
                train_file=args.train, token_cache=data / 'train_tokens.npz', model_path=args.model,
                b8_bytes=b8['total_bytes'])
        except NotImplementedError as e:
            result['unsupported'].append({'job': item['job']['job_id'], 'reason': str(e)})
            continue
        if args.technical:
            job = expanded['job'] = dict(expanded['job'])
            job.update(seed=9001, phase='TECH', job_id='TECH_' + item['job']['recipe_id'] + '_s9001')
            expanded['technical_profile_only'] = True
        expanded.pop('expanded_sha256', None)
        expanded['expanded_sha256'] = digest(expanded)
        path = args.out / (expanded['job']['job_id'] + '.json')
        payload = json.dumps(expanded, sort_keys=True, indent=2) + '\n'
        if path.exists() and path.read_text() != payload:
            raise ValueError(f'Existing configuration differs: {path}; use a new config directory')
        path.write_text(payload)
        result['compiled'].append(expanded['job']['job_id'])
    (args.out / 'compilation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'compiled': len(result['compiled']), 'unsupported': len(result['unsupported']),
                      'b8_bytes': result['b8_bytes'], 'output': str(args.out)}))


if __name__ == '__main__':
    main()
