"""Thin, reversible runtime adapter. All local training stays in the original runtime."""
from __future__ import annotations
import argparse
import contextlib
import copy
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import torch
from common import (BATCH, SEEDS, ROOT, require, read, atomic_json, atomic_torch,
                    sha256, fingerprint, source_files, exclusive, tensor_load)
from context import (Bundle, prepare_real, prepare_fresh, synthetic_bundle, original_synthetic,
                     base_study, write_evaluation_identity)
from aggregation import AggregationSnapshot

DRIVER = 'v41_fixed_support'
SCIENTIFIC_FIELDS = ('round', 'accuracy', 'loss', 'samples', 'selected_clients')
ONLINE_PARITY_FIELDS = SCIENTIFIC_FIELDS + ('serialized_traffic_bytes',)


def check_trace(source, bundle):
    from pcla_repair import core
    require(source['round'] == bundle.cfg['rounds'], 'support donor incomplete')
    require(source['identity']['seed'] == bundle.seed, 'support seed differs')
    trace = {}; expected = set()
    for r in range(1, bundle.cfg['rounds']+1):
        selected = random.Random(core.keyed_seed(bundle.seed, 'participants', r)).sample(
            sorted(bundle.clients), bundle.cfg['selected_clients'])
        for cid in selected:
            key = f'{r}/{cid}'; expected.add(key)
            mask = source['trace'][key].detach().cpu().clone()
            core.check_support(mask, bundle.shape, bundle.capacities[cid]*bundle.cfg['credits_per_layer'])
            if source['policy'] == 'random':
                expected_mask = core.allocate('random', torch.zeros(bundle.shape), bundle.capacities[cid],
                    bundle.cfg['credits_per_layer'], core.keyed_seed(bundle.seed,'allocation','random',r,cid))
                require(torch.equal(mask, expected_mask), 'random support cannot be regenerated')
            trace[key] = mask
    require(set(source['trace']) == expected, 'support trace inventory differs')
    return trace


def numerical_rows(rows, fields=SCIENTIFIC_FIELDS):
    return [{k: row[k] for k in fields} for row in rows]


def parity_report(original, replay, original_dir, replay_dir, fields=SCIENTIFIC_FIELDS):
    from pcla_repair import runtime as rt
    require(len(replay['metrics']) == len(original['metrics']), 'parity horizon differs')
    n = len(original['metrics'])
    check = {
        'adapter': set(original['adapter']) == set(replay['adapter']) and all(
            torch.equal(v, replay['adapter'][k]) for k, v in original['adapter'].items()),
        'trace': set(original['trace']) == set(replay['trace']) and all(
            torch.equal(v, replay['trace'][k]) for k, v in original['trace'].items()),
        'scientific_metrics': numerical_rows(original['metrics'], fields) == numerical_rows(replay['metrics'], fields),
        'predictions': all(read(Path(original_dir)/f'predictions_round_{r:03d}.json') ==
                           read(Path(replay_dir)/f'predictions_round_{r:03d}.json') for r in range(1,n+1)),
    }
    return {'passed': all(check.values()), 'checks': check, 'compared_metric_fields': list(fields),
            'excluded_from_equality': ['timing', 'memory high-water marks', 'host', 'process IDs']}


def preserve_ahead(out, committed):
    out = Path(out); evidence = out/'evidence'
    quarantine = out/'recovery'/str(time.time_ns())
    paths = []
    if evidence.exists():
        paths += [p for p in evidence.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) > committed]
    run = out/'run'
    if run.exists():
        for p in run.iterdir():
            if '_round_' in p.stem:
                try:
                    r = int(p.stem.rsplit('_round_', 1)[1])
                except ValueError:
                    continue
                if r > committed: paths.append(p)
        m = run/'metrics.json'
        if m.exists() and len(read(m)) > committed: paths.append(m)
    for p in paths:
        dest = quarantine/p.relative_to(out); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), dest)
    if paths:
        atomic_json(quarantine/'receipt.json', {'committed_round': committed, 'preserved_paths': len(paths)})


@contextlib.contextmanager
def instrument(bundle, out, rule, trace, resume_checkpoint=None, refactor=None):
    from pcla_repair import core, runtime as rt
    from lora_matrix import refactor_module
    from run_lora_controls import projection_state, load_projections
    out = Path(out); run = out/'run'
    original = (rt.POLICIES, rt.allocate, rt.DeltaAccumulator, rt.evaluate, rt.atomic_checkpoint, rt.payload_size)
    count = resume_checkpoint['round'] if resume_checkpoint else 0
    counters = {'aggregation': count, 'evaluation': count}
    receipts = copy.deepcopy(resume_checkpoint.get('v41_evidence', {})) if resume_checkpoint else {}
    lookup = {}
    if trace is not None:
        for key, support in trace.items():
            r, cid = map(int, key.split('/'))
            token = core.keyed_seed(bundle.seed, 'allocation', DRIVER, r, cid)
            require(token not in lookup, 'allocation-key collision'); lookup[token] = support
    scalings = {n: float(m.scaling) for n, m in core.lora_modules(bundle.model).items()}
    anchors = set(bundle.cfg['probe_rounds']) | {1, bundle.cfg['rounds']}
    sizes = {}

    def lora(state):
        return {n: v for n, v in state.items() if '.lora_' in n}

    def refactor_round(r, state):
        before = projection_state(bundle.model); after = {}; new = dict(state); modules = []
        for name in sorted(scalings):
            after[name], new[name+'.lora_b'], new[name+'.lora_a'], info = refactor_module(
                before[name], state[name+'.lora_b'], state[name+'.lora_a'], scalings[name])
            modules.append({'module': name, **info})
        load_projections(bundle.model, after)
        record = {'round': r, 'method': refactor, 'modules': modules,
                  'base_before_sha256': core.tensor_digest(before), 'base_after_sha256': core.tensor_digest(after),
                  'pre_lora_sha256': core.tensor_digest(lora(state)), 'post_lora_sha256': core.tensor_digest(lora(new)),
                  'download_bytes': sizes[r][0], 'adapter_only_download_bytes': sizes[r][1]}
        atomic_json(out/'evidence'/f'{r:03d}'/'refactor.json', record)
        if r in anchors:
            atomic_torch(out/'evidence'/f'{r:03d}'/'refactor_bank.pt', {
                'round': r, 'scalings': scalings, 'pre': lora(state), 'post': lora(new),
                'base_before': before, 'base_after': after})
        return new

    def payload_size(payload):
        if set(payload) != {'round', 'adapter'}:
            return original[5](payload)
        r = payload['round']
        if r not in sizes:
            sizes[r] = (original[5]({**payload, 'server_projection_weights': projection_state(bundle.model)}),
                        original[5](payload))
        return sizes[r][0]

    class Acc(AggregationSnapshot):
        def __init__(self, base, total, mode='fixed'):
            counters['aggregation'] += 1
            self.round = counters['aggregation']
            self.requested_rule = mode
            require(mode == rule, 'runtime changed aggregation rule')
            super().__init__(base, total, scalings, 'b' if self.round % 2 else 'a')

        def finish(self):
            state, _ = self.state(rule)
            atomic_json(out/'evidence'/f'{self.round:03d}'/'aggregation.json', self.readout(rule))
            return refactor_round(self.round, state) if refactor else state

    def allocate(policy, scores, q, m, token, **kwargs):
        if policy != DRIVER:
            return original[1](policy, scores, q, m, token, **kwargs)
        require(token in lookup, 'unplanned support request')
        mask = lookup[token]; core.check_support(mask, tuple(scores.shape), q*m)
        return mask.clone()

    def evaluate(*args, **kwargs):
        kwargs['record_logits'] = True
        result = original[3](*args, **kwargs)
        counters['evaluation'] += 1; r = counters['evaluation']
        require(result['labels'] == bundle.labels, 'evaluation label ordering changed')
        atomic_json(out/'evidence'/f'{r:03d}'/'evaluation.json', result)
        # Keep runtime fields but remove lists it did not historically own.
        result.pop('labels'); result.pop('logits')
        return result

    def checkpoint(path, value):
        if Path(path).parent != run:
            return original[4](path, value)
        r = value['round']
        if r:
            files = [out/'evidence'/f'{r:03d}'/'evaluation.json',
                     out/'evidence'/f'{r:03d}'/'aggregation.json',
                     run/f'clients_round_{r:03d}.json', run/f'predictions_round_{r:03d}.json']
            if refactor:
                files += [out/'evidence'/f'{r:03d}'/'refactor.json'] + (
                    [out/'evidence'/f'{r:03d}'/'refactor_bank.pt'] if r in anchors else [])
            receipts[str(r)] = {p.relative_to(out).as_posix(): sha256(p) for p in files}
        value = {**value, 'v41_evidence': copy.deepcopy(receipts)}
        if refactor:
            value['v5b_base'] = projection_state(bundle.model)
        return original[4](path, value)

    rt.POLICIES = (*original[0], DRIVER); rt.allocate = allocate
    rt.DeltaAccumulator = Acc; rt.evaluate = evaluate; rt.atomic_checkpoint = checkpoint
    if refactor:
        rt.payload_size = payload_size
    try:
        yield
    finally:
        rt.POLICIES, rt.allocate, rt.DeltaAccumulator, rt.evaluate, rt.atomic_checkpoint, rt.payload_size = original


def run_bundle(bundle, out, rule='fixed', source=None, online=False, donor=None,
               stop_after=None, resume=False, recovery_reason=None, parity=False, refactor=None):
    from pcla_repair import core, runtime as rt
    from run_lora_controls import projection_state, load_projections
    require(rule in ('fixed','contributors','function_norm_matched'), 'invalid long-run rule')
    require(refactor in (None, 'svd'), 'unknown refactor')
    require(refactor is None or (online and donor is None), 'SVD refactor is defined for online PCLA only')
    out = Path(out); exists = out.exists()
    require(not exists or resume, 'output exists; use explicit same-identity recovery')
    out.mkdir(parents=True, exist_ok=True); run = out/'run'; run.mkdir(exist_ok=True)
    if not online:
        require(source is not None, 'fixed-support run needs source')
        trace = check_trace(source, bundle); policy = DRIVER
    else:
        trace = None; policy = 'replay_random_rank' if donor is not None else 'pcla'
    cfg = copy.deepcopy(bundle.cfg); cfg['aggregation'] = rule
    common = base_study(bundle)
    common.update({'family': 'online' if online else 'fixed_support', 'rule': rule,
                   'fixed_source_policy': None if online else source['policy']})
    if refactor:
        common.update({'refactor': refactor,
                       'initial_projections_sha256': core.tensor_digest(projection_state(bundle.model))})
    prov = copy.deepcopy(bundle.provenance); prov['v41'] = common
    identity = {'seed': bundle.seed, 'config': cfg, 'provenance': prov,
                'initial_adapter_sha256': core.tensor_digest(core.adapter_state(bundle.model))}
    identifier = rt.fingerprint(identity)
    donor_meta = None
    if donor is not None:
        d = rt.load_checkpoint(Path(donor))
        require(d['identity_hash'] == identifier and d['policy'] == 'pcla' and
                d['round'] == cfg['rounds'], 'donor must be same-identity completed online PCLA')
        donor_meta = {'sha256': sha256(donor), 'identity_hash': d['identity_hash'], 'path': str(Path(donor).resolve())}
    meta = {'identity': identity, 'identity_hash': identifier, 'policy': policy,
            'kind': 'aggregation', 'source_policy': None if online else source['policy'],
            'online_donor': donor_meta, 'reference_source':
                {'path': str(bundle.source_path), 'sha256': sha256(bundle.source_path)} if bundle.source_path else None}
    with exclusive(out/'.run.lock', {'identity_hash': identifier}):
        saved = None
        if resume:
            require(recovery_reason, 'recovery reason is required')
            saved = rt.load_checkpoint(run/'checkpoint.pt')
            require(saved['identity_hash'] == identifier and saved['policy'] == policy, 'recovery identity differs')
            for files in saved.get('v41_evidence', {}).values():
                for rel, h in files.items(): require(sha256(out/rel) == h, 'committed evidence changed')
            preserve_ahead(out, saved['round'])
            if refactor:
                expected = (read(out/'evidence'/f"{saved['round']:03d}"/'refactor.json')['base_after_sha256']
                            if saved['round'] else common['initial_projections_sha256'])
                require(core.tensor_digest(saved['v5b_base']) == expected, 'saved refactored base differs from evidence')
                load_projections(bundle.model, saved['v5b_base'])
            atomic_json(out/'recovery'/f'{time.time_ns()}.json', {'reason': recovery_reason, 'round': saved['round']})
        else:
            if source is not None:
                ref = {k: source[k] for k in ('policy','identity','identity_hash','round','trace')}
                ref['original_file_sha256'] = sha256(bundle.source_path) if bundle.source_path else None
                atomic_torch(out/'source_reference.pt', ref)
                meta['embedded_source_sha256'] = sha256(out/'source_reference.pt')
            if donor is not None:
                ref = {k: d[k] for k in ('policy','identity','identity_hash','round','trace')}
                ref['original_file_sha256'] = sha256(donor)
                atomic_torch(out/'donor_reference.pt', ref)
                meta['embedded_donor_sha256'] = sha256(out/'donor_reference.pt')
            atomic_json(out/'study.json', meta); write_evaluation_identity(out, bundle)
        try:
            with instrument(bundle, out, rule, trace, saved, refactor):
                rows = rt.run_experiment(bundle.model, bundle.layout, bundle.clients, bundle.capacities,
                    bundle.evaluation, cfg, bundle.seed, policy, run, prov,
                    partition=bundle.partition, donor_path=Path(donor) if donor else None, stop_after=stop_after)
            cp = rt.load_checkpoint(run/'checkpoint.pt')
            complete = cp['round'] == cfg['rounds']
            if parity and complete:
                require(bundle.source_path is not None and rule == 'fixed' and refactor is None, 'parity source missing')
                require(not online or (donor is None and source['policy'] == 'pcla'), 'online parity needs the old PCLA run')
                result = parity_report(source, cp, bundle.source_path.parent, run,
                                       ONLINE_PARITY_FIELDS if online else SCIENTIFIC_FIELDS)
                atomic_json(out/'parity.json', result); require(result['passed'], 'fixed replay parity failed')
            atomic_json(out/'status.json', {'state': 'completed' if complete else 'paused', 'completed_round': cp['round']})
            return cp
        except BaseException:
            (out/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
            atomic_json(out/'status.json', {'state': 'failed'}); raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True); p.add_argument('--source-root', required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--source', choices=('pcla','random','replay_random_rank'), default='pcla')
    p.add_argument('--rule', choices=('fixed','contributors','function_norm_matched'), required=True)
    p.add_argument('--online', action='store_true'); p.add_argument('--donor')
    p.add_argument('--out', required=True); p.add_argument('--parity', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--reason'); p.add_argument('--stop-after', type=int)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--dataset', choices=('sst2', 'ag_news')); p.add_argument('--refactor', choices=('svd',))
    args = p.parse_args()
    if args.dataset:
        b = prepare_fresh(args.repo, args.dataset, args.seed, threads=args.threads)
    else:
        b = prepare_real(args.repo, args.source_root, args.seed, 'pcla' if args.online else args.source, threads=args.threads)
    run_bundle(b, args.out, args.rule, b.source, args.online, args.donor,
               args.stop_after, args.resume, args.reason, args.parity, args.refactor)

if __name__ == '__main__':
    main()
