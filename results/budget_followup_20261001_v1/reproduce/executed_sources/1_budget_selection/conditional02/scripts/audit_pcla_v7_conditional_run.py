#!/usr/bin/env python3
"""Independent conditional V audit, retaining the complete base V evidence checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.conditional_v_queue import _identity, canonical_hash


def _conditional_identity(cfg: dict, errors: list[str]) -> None:
    try:
        slot, _ = _identity(cfg['job'], cfg)
        if slot not in (3, 4, 16):
            raise ValueError('not a conditional V slot')
        for key in ('recipe_id', 'method', 'representation', 'rank', 'lr', 'period', 'xi'):
            if cfg.get('recipe', {}).get(key) != cfg['job'].get(key):
                raise ValueError(f'recipe/job {key} differs')
    except Exception as exc:
        errors.append(f'conditional V slot/config identity differs: {exc}')


def audit(config_path: Path, run_dir: Path, source: Path) -> dict:
    """Full V auditor with only its fixed-slot identity callback replaced."""
    if ROOT.resolve() != source.resolve():
        return {'ok': False, 'errors': ['conditional auditor execution root differs from frozen source'],
                'checks': {}, 'run_dir': str(run_dir.resolve())}
    from scripts import audit_pcla_v7_run as base
    original = base.check_validation_identity
    try:
        base.check_validation_identity = _conditional_identity
        result = base.audit(config_path, run_dir, source, phase='V')
    finally:
        base.check_validation_identity = original
    errors = result.setdefault('errors', [])
    checks = result.setdefault('checks', {})
    try:
        cfg = json.loads(config_path.read_text())
        slot, seed = _identity(cfg['job'], cfg)
        out = run_dir.resolve()
        identity = json.loads((out / 'run_identity.json').read_text())
        provenance = identity['provenance']
        rows = json.loads((out / 'metrics.json').read_text())
        if slot == 3:
            from pcla_repair.core import lora_modules
            from pcla_research_v7.conditional_replay import freeze_donor_trace
            from pcla_research_v7.runtime import state
            from scripts.run_pcla_v7 import prepare
            import torch
            model, _, capacities, _, prepared = prepare(cfg, config_path,
                                                         torch.device('cpu'))
            names = list(lora_modules(model))
            initial = json.loads((out / 'initial_state_identity.json').read_text())
            from pcla_repair.core import tensor_digest
            if (initial.get('schema') != 'pcla-v7-R-local-initial-state-1' or
                    initial.get('job_id') != cfg['job']['job_id'] or
                    initial.get('config_sha256') != base.sha(config_path) or
                    initial.get('initial_state_sha256') != tensor_digest(state(model)) or
                    initial.get('source_manifest_sha256') != provenance.get('source_manifest_sha256') or
                    initial.get('data_manifest_sha256') != provenance.get('manifest_sha256') or
                    prepared.get('pretrained_model_sha256') != provenance.get('pretrained_model_sha256')):
                raise ValueError('donor initial/source/data identity differs')
            donor_ids = {'donor_config_file_sha256': base.sha(config_path),
                         'source_manifest_sha256': provenance['source_manifest_sha256'],
                         'data_manifest_sha256': provenance['manifest_sha256'],
                         'initial_state_sha256': initial['initial_state_sha256'],
                         'raw_metrics_file_sha256': base.sha(out / 'metrics.json')}
            book = freeze_donor_trace(donor_job=cfg['job'], rows=rows,
                                      module_names=names, capacities=capacities,
                                      identities=donor_ids)
            seal = json.loads((out / 'donor_quota_seal.json').read_text())
            if (seal.get('schema') != 'pcla-v7-R-local-donor-trace-seal-1' or
                    seal.get('job_id') != cfg['job']['job_id'] or
                    seal.get('completed_round') != len(rows) or
                    seal.get('identities') != donor_ids or
                    seal.get('donor_support_trace_sha256') != book.donor_support_trace_sha256 or
                    seal.get('donor_book_sha256') != book.sha256):
                raise ValueError('donor complete support trace seal differs')
            checks['donor_book_sha256'] = book.sha256
            checks['donor_available_through'] = len(book.rounds)
        elif slot == 16:
            from pcla_repair.core import lora_modules, tensor_digest
            from pcla_research_v7.conditional_replay import replay_support
            from pcla_research_v7.runtime import state
            from scripts.run_pcla_v7 import load_quota_replay_binding, prepare
            import torch
            model, _, capacities, _, prepared = prepare(cfg, config_path,
                                                         torch.device('cpu'))
            init_sha = tensor_digest(state(model))
            book, donor_ids, binding_sha = load_quota_replay_binding(
                expanded=cfg, config_path=config_path,
                binding_path=Path(cfg['quota_replay']['binding_path']),
                capacities=capacities, module_names=list(lora_modules(model)),
                initial_sha=init_sha, provenance=prepared)
            dispatch_path = out / 'dispatch_identity.json'
            if (provenance.get('quota_replay_binding_sha256') != binding_sha or
                    provenance.get('quota_replay_donor_book_sha256') != book.sha256 or
                    (dispatch_path.is_file() and
                     json.loads(dispatch_path.read_text()).get('binding_sha256') != binding_sha)):
                raise ValueError('replay run/dispatch binding differs')
            module_ranks = {name: module.rank for name, module in lora_modules(model).items()}
            for row in rows:
                round_id = row['round']
                expected = replay_support(book=book, replay_job=cfg['job'],
                    round_id=round_id, cohort=row['selected_clients'],
                    module_ranks=module_ranks, expected_identities=donor_ids)
                trace = row.get('support_trace', {})
                for cid in row['selected_clients']:
                    actual_modules = trace.get(str(cid), {}).get('modules', {})
                    for name, masks in expected[cid].items():
                        for factor in ('a', 'b'):
                            if actual_modules.get(name, {}).get(factor) != masks[factor].nonzero().flatten().tolist():
                                raise ValueError(f'replay mask differs at r{round_id} c{cid} {name} {factor}')
                meta = row.get('quota_replay', {})
                if (meta.get('donor_job_id') != book.donor_job_id or
                        meta.get('donor_book_sha256') != book.sha256 or
                        meta.get('replayed_round') != round_id or
                        meta.get('coordinate_path') != cfg['job']['representation']):
                    raise ValueError(f'replay round {round_id} donor identity differs')
            checks['donor_book_sha256'] = book.sha256
            checks['replay_masks_verified_rounds'] = len(rows)
        checks['conditional_slot'] = slot
        checks['conditional_seed'] = seed
    except Exception as exc:
        errors.append(f'conditional evidence unreadable or inconsistent: {type(exc).__name__}: {exc}')
    result['ok'] = not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.config, args.run, args.source)
    except Exception as exc:
        result = {'ok': False, 'errors': [f'conditional auditor failed: {type(exc).__name__}: {exc}'],
                  'checks': {}, 'run_dir': str(args.run.resolve())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
