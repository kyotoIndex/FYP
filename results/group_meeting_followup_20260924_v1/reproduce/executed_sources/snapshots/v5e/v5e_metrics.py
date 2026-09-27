"""Read-only EMA readout and parity checks for the V5e deterministic replays."""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from common import atomic_json, atomic_torch, read, require, sha256, tensor_load


@contextlib.contextmanager
def runtime_ema(bundle, out, beta=0.8):
    """Observe each post-aggregation state without changing the training trajectory."""
    from pcla_repair import core, runtime as rt
    out = Path(out)
    directory = out / 'v5e_ema'
    directory.mkdir(exist_ok=True)
    original = rt.evaluate
    checkpoint = out / 'run' / 'checkpoint.pt'
    committed = rt.load_checkpoint(checkpoint)['round'] if checkpoint.exists() else 0
    ema = None
    if committed:
        prior = tensor_load(directory / f'round_{committed:03d}.pt')
        require(prior['round'] == committed, 'EMA recovery horizon differs')
        ema = prior['ema']
    count = committed

    def evaluate(model, layout, dataset, batch_size, device, *, record_logits=False):
        nonlocal ema, count
        factual = original(model, layout, dataset, batch_size, device,
                           record_logits=record_logits)
        count += 1
        current = core.adapter_state(model)
        previous = ema
        ema = ({n: v.double() for n, v in current.items()} if previous is None else
               {n: beta * previous[n] + (1 - beta) * v.double() for n, v in current.items()})
        before_hash = core.tensor_digest(current)
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        try:
            core.load_adapter(model, {n: v.to(current[n].dtype) for n, v in ema.items()})
            observed = original(model, layout, dataset, batch_size, device, record_logits=True)
        finally:
            core.load_adapter(model, current)
        require(core.tensor_digest(core.adapter_state(model)) == before_hash,
                'EMA evaluation changed global training parameters')
        require(torch.equal(torch.get_rng_state(), rng), 'EMA evaluation changed CPU RNG')
        if cuda_rng is not None:
            require(torch.equal(torch.cuda.get_rng_state(device), cuda_rng),
                    'EMA evaluation changed CUDA RNG')
        require(observed['labels'] == bundle.labels, 'EMA labels/order differ')
        atomic_torch(directory / f'round_{count:03d}.pt',
                     {'round': count, 'beta': beta, 'global': current, 'ema': ema,
                      'global_sha256': before_hash, 'previous_ema_sha256':
                       None if previous is None else core.tensor_digest(previous),
                      'ema_sha256': core.tensor_digest(ema)})
        atomic_json(directory / f'evaluation_{count:03d}.json', observed)
        return factual

    rt.evaluate = evaluate
    try:
        yield
    finally:
        rt.evaluate = original


def parity(original_job, replay, family):
    """Compare scientific trajectory fields with a completed, read-only V5b job."""
    from pcla_repair import core, runtime as rt
    original_job, replay = Path(original_job), Path(replay)
    original_run = original_job / 'run' / 'run'
    new_run = replay / 'run'
    old = rt.load_checkpoint(original_run / 'checkpoint.pt')
    new = rt.load_checkpoint(new_run / 'checkpoint.pt')
    n = new['round']
    require(0 < n <= old['round'], 'parity horizon differs')
    old_state = old if n == old['round'] else rt.load_checkpoint(original_run / f'probe_round_{n:03d}.pt')
    checks = {}
    fields = ('round', 'accuracy', 'loss', 'samples', 'selected_clients',
              'serialized_traffic_bytes')
    checks['per_round_metrics'] = all(
        all(a.get(k) == b.get(k) for k in fields) for a, b in zip(old['metrics'][:n], new['metrics']))
    checks['predictions'] = all(
        read(original_run / f'predictions_round_{r:03d}.json') ==
        read(new_run / f'predictions_round_{r:03d}.json') for r in range(1, n + 1))
    checks['clients_and_support'] = all(
        _client_science(read(original_run / f'clients_round_{r:03d}.json')) ==
        _client_science(read(new_run / f'clients_round_{r:03d}.json')) for r in range(1, n + 1))
    checks['trace'] = (set(old_state['trace']) == set(new['trace']) and all(
        torch.equal(v, new['trace'][k]) for k, v in old_state['trace'].items()))
    checks['final_adapter'] = (set(old_state['adapter']) == set(new['adapter']) and all(
        torch.equal(v, new['adapter'][k]) for k, v in old_state['adapter'].items()))
    if family == 'svd':
        checks['final_base'] = (set(old_state['v5b_base']) == set(new['v5b_base']) and all(
            torch.equal(v, new['v5b_base'][k]) for k, v in old_state['v5b_base'].items()))
        checks['refactor_records'] = all(
            read(original_job / 'run' / 'evidence' / f'{r:03d}' / 'refactor.json') ==
            read(replay / 'evidence' / f'{r:03d}' / 'refactor.json')
            for r in range(1, n + 1))
    result = {'passed': all(checks.values()), 'checks': checks, 'family': family,
              'rounds': n, 'seed': old['identity']['seed'],
              'reference_checkpoint_sha256': sha256(original_run / 'checkpoint.pt'),
              'reference_archive_sha256': read(original_job / 'EXTRACTION_RECEIPT.json')['archive_sha256'],
              'final_adapter_sha256': core.tensor_digest(new['adapter']),
              'final_base_sha256': core.tensor_digest(new['v5b_base']) if family == 'svd' else None}
    return result


def _client_science(clients):
    return [{k: c[k] for k in ('client', 'support', 'factor', 'sample_weight', 'credits',
                               'logical_update_parameters', 'optimizer_tensor_parameters')}
            for c in clients]


def audit_ema(out, beta=0.8):
    from pcla_repair import core, runtime as rt
    out = Path(out)
    cp = rt.load_checkpoint(out / 'run' / 'checkpoint.pt')
    expected = None
    rows = []
    for r in range(1, cp['round'] + 1):
        bank = tensor_load(out / 'v5e_ema' / f'round_{r:03d}.pt')
        evaluation = read(out / 'v5e_ema' / f'evaluation_{r:03d}.json')
        require(bank['round'] == r and bank['beta'] == beta, 'EMA round/beta differs')
        global_state = bank['global']
        require(core.tensor_digest(global_state) == bank['global_sha256'], 'EMA global hash differs')
        previous = expected
        expected = ({n: v.double() for n, v in global_state.items()} if previous is None else
                    {n: beta * previous[n] + (1 - beta) * v.double() for n, v in global_state.items()})
        require(bank['previous_ema_sha256'] ==
                (None if previous is None else core.tensor_digest(previous)), 'EMA chain broken')
        require(all(torch.equal(expected[n], bank['ema'][n]) for n in expected), 'EMA state differs')
        require(core.tensor_digest(expected) == bank['ema_sha256'], 'EMA hash differs')
        require(evaluation['labels'] and len(evaluation['predictions']) == cp['metrics'][r-1]['samples'],
                'EMA evaluation sample count differs')
        accuracy = sum(p == y for p, y in zip(evaluation['predictions'], evaluation['labels'])) / len(evaluation['labels'])
        require(accuracy == evaluation['accuracy'], 'EMA accuracy differs from predictions')
        rows.append({'round': r, 'accuracy': accuracy, 'loss': evaluation['loss'],
                     'global_sha256': bank['global_sha256'], 'ema_sha256': bank['ema_sha256']})
    require(core.tensor_digest(tensor_load(out / 'v5e_ema' / f'round_{cp["round"]:03d}.pt')['global']) ==
            core.tensor_digest(cp['adapter']), 'final EMA source differs from final global adapter')
    return rows
