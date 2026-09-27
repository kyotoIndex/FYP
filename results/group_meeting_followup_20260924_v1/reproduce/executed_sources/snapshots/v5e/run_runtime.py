"""Original runtime policies, called unmodified; this wrapper only binds identity, donor, layout and recovery."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
from contextlib import nullcontext
import time
import traceback

import torch
from common import SEEDS, require, read, atomic_json, atomic_torch, sha256, exclusive
from context import prepare_fresh, load_original, base_study, write_evaluation_identity
from run_aggregation import ONLINE_PARITY_FIELDS, numerical_rows, preserve_ahead

POLICIES = ('pcla', 'full_alt', 'replay_random_rank')


def identity_of(bundle, cfg):
    study = base_study(bundle)
    study.update({'family': 'runtime_policy', 'dataset': bundle.dataset, 'rounds': cfg['rounds']})
    prov = copy.deepcopy(bundle.provenance); prov['v5b'] = study
    return {'seed': bundle.seed, 'config': cfg, 'provenance': prov,
            'initial_adapter_sha256': study['initial_adapter_sha256']}


def prefix_parity(reference, run, cp):
    """The first h rounds of a longer run must equal the completed h-round original run."""
    from pcla_repair import runtime as rt
    old, old_path = reference; h = old['round']; run = Path(run)
    require(old['round'] == old['identity']['config']['rounds'] and cp['round'] >= h, 'prefix horizon unavailable')
    require(h == cp['round'] or h in cp['identity']['config']['probe_rounds'], 'no saved state at prefix horizon')
    state = cp if h == cp['round'] else rt.load_checkpoint(run/f'probe_round_{h:03d}.pt')
    require(state['round'] == h and state['identity_hash'] == cp['identity_hash'], 'prefix state identity differs')
    keys = {k for k in cp['trace'] if int(k.split('/')[0]) <= h}
    check = {
        'adapter': set(old['adapter']) == set(state['adapter']) and all(
            torch.equal(v, state['adapter'][k]) for k, v in old['adapter'].items()),
        'trace': set(old['trace']) == keys and all(torch.equal(v, cp['trace'][k]) for k, v in old['trace'].items()),
        'metrics': numerical_rows(old['metrics'], ONLINE_PARITY_FIELDS) ==
                   numerical_rows(cp['metrics'][:h], ONLINE_PARITY_FIELDS),
        'predictions': all(read(Path(old_path).parent/f'predictions_round_{r:03d}.json') ==
                           read(run/f'predictions_round_{r:03d}.json') for r in range(1, h+1))}
    return {'passed': all(check.values()), 'checks': check, 'horizon': h,
            'reference_checkpoint_sha256': sha256(old_path), 'reference_identity_hash': old['identity_hash'],
            'compared_metric_fields': list(ONLINE_PARITY_FIELDS),
            'excluded_from_equality': ['timing', 'memory high-water marks', 'host', 'process IDs']}


def run_bundle(bundle, out, policy, donor=None, reference=None, stop_after=None, resume=False,
               recovery_reason=None, ema_readout=None):
    from pcla_repair import runtime as rt
    require(policy in POLICIES, 'unsupported runtime policy')
    require((policy == 'replay_random_rank') == (donor is not None), 'replay needs a PCLA donor; other policies take none')
    require(ema_readout is None or (policy == 'pcla' and 0 < ema_readout < 1), 'EMA readout requires PCLA')
    out = Path(out); require(not out.exists() or resume, 'output exists; use explicit same-identity recovery')
    out.mkdir(parents=True, exist_ok=True); run = out/'run'
    cfg = copy.deepcopy(bundle.cfg)
    identity = identity_of(bundle, cfg); identifier = rt.fingerprint(identity)
    meta = {'kind': 'runtime_policy', 'policy': policy, 'identity': identity, 'identity_hash': identifier,
            'online_donor': None, 'prefix_reference': None}
    if donor is not None:
        d = rt.load_checkpoint(Path(donor))
        require(d['identity_hash'] == identifier and d['policy'] == 'pcla' and d['round'] == cfg['rounds'],
                'donor must be a same-identity completed PCLA run')
        meta['online_donor'] = {'sha256': sha256(donor), 'identity_hash': d['identity_hash']}
    if reference is not None:
        meta['prefix_reference'] = {'sha256': sha256(reference[1]), 'identity_hash': reference[0]['identity_hash']}
    with exclusive(out/'.run.lock', {'identity_hash': identifier}):
        if resume:
            require(recovery_reason, 'recovery reason is required')
            if donor is not None:
                meta['embedded_donor_sha256'] = sha256(out/'donor_reference.pt')
            require(read(out/'study.json') == meta, 'recovery study identity differs')
            saved = rt.load_checkpoint(run/'checkpoint.pt')
            require(saved['identity_hash'] == identifier and saved['policy'] == policy, 'recovery identity differs')
            preserve_ahead(out, saved['round'])
            atomic_json(out/'recovery'/f'{time.time_ns()}.json', {'reason': recovery_reason, 'round': saved['round']})
        else:
            if donor is not None:
                ref = {k: d[k] for k in ('policy', 'identity', 'identity_hash', 'round', 'trace')}
                ref['original_file_sha256'] = sha256(donor)
                atomic_torch(out/'donor_reference.pt', ref)
                meta['embedded_donor_sha256'] = sha256(out/'donor_reference.pt')
            atomic_json(out/'study.json', meta); write_evaluation_identity(out, bundle)
        try:
            if ema_readout is not None:
                from v5e_metrics import runtime_ema
            with (runtime_ema(bundle, out, ema_readout) if ema_readout is not None else nullcontext()):
                rt.run_experiment(bundle.model, bundle.layout, bundle.clients, bundle.capacities, bundle.evaluation,
                                  cfg, bundle.seed, policy, run, identity['provenance'], partition=bundle.partition,
                                  donor_path=Path(donor) if donor else None, stop_after=stop_after)
            cp = rt.load_checkpoint(run/'checkpoint.pt')
            require(cp['identity_hash'] == identifier, 'runtime identity differs from wrapper identity')
            complete = cp['round'] == cfg['rounds']
            if reference is not None and complete:
                result = prefix_parity(reference, run, cp)
                atomic_json(out/'prefix_parity.json', result); require(result['passed'], 'prefix parity failed')
            atomic_json(out/'status.json', {'state': 'completed' if complete else 'paused', 'completed_round': cp['round']})
            return cp
        except BaseException:
            (out/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
            atomic_json(out/'status.json', {'state': 'failed'}); raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True); p.add_argument('--dataset', choices=('sst2', 'ag_news'), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--policy', choices=POLICIES, required=True); p.add_argument('--rounds', type=int)
    p.add_argument('--donor'); p.add_argument('--prefix-source-root')
    p.add_argument('--out', required=True); p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true'); p.add_argument('--reason')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    b = prepare_fresh(a.repo, a.dataset, a.seed, threads=a.threads)
    if a.rounds: b.cfg['rounds'] = a.rounds
    reference = load_original(a.prefix_source_root, a.seed, 'pcla', a.repo) if a.prefix_source_root else None
    run_bundle(b, a.out, a.policy, a.donor, reference, a.stop_after, a.resume, a.reason)

if __name__ == '__main__':
    main()
