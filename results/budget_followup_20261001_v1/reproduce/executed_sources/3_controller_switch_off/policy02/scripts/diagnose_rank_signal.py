#!/usr/bin/env python3
"""Train-only, checkpoint-conditioned rank-score diagnostics; never reads dev data."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from pcla_repair.core import (
    BudgetLoRALinear, adapter_state, component_scores, configure, keyed_seed,
    load_adapter, lora_modules, tensor_digest,
)
from pcla_repair.runtime import (
    atomic_json, batches, fingerprint, load_checkpoint, local_train, seed_train, to_device,
)

SCORES = ('legacy', 'normalized_projection', 'gate_shared_qv', 'gradient_norm')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_ids(source_ids, seed, client):
    # Identical at every checkpoint and for both trained factors.
    ordered = sorted(source_ids, key=lambda x: (keyed_seed('rank-signal-v1', seed, client, x), x))
    assert len(ordered) == len(set(ordered)) == 256
    groups = {'score_0': ordered[:16], 'score_1': ordered[16:32],
              'score_2': ordered[32:48], 'update': ordered[48:64],
              'heldout': ordered[64:128]}
    assert len(set(x for group in groups.values() for x in group)) == 128
    return groups


def score_batch(model, layout, dataset, factor, device):
    rank = next(iter(lora_modules(model).values())).rank
    shape = (max(layout.values()) + 1, rank)
    configure(model, layout, torch.ones(shape, dtype=torch.bool), factor, head=False)
    model.eval()
    model.zero_grad(set_to_none=True)
    batch = to_device(next(iter(batches(dataset, len(dataset), 0, False))), device)
    output = model(**batch)
    if not torch.isfinite(output.loss):
        raise FloatingPointError('nonfinite score loss')
    output.loss.backward()
    scores = {kind: component_scores(model, layout, factor, kind)
              for kind in ('legacy', 'normalized_projection', 'gradient_norm')}
    signed_gate = torch.zeros(shape, dtype=torch.float64)
    for name, module in lora_modules(model).items():
        p = getattr(module, 'lora_' + factor)
        signed_gate[layout[name]] += (p.grad.detach().double() * p.detach().double()).sum(
            dim=0 if factor == 'b' else 1).cpu()
    # One shared gate scales both Q/V contributions: sum signed derivatives BEFORE abs.
    scores['gate_shared_qv'] = signed_gate.abs()
    if any(not torch.isfinite(s).all() for s in scores.values()):
        raise FloatingPointError('nonfinite score')
    logits = output.logits.detach().cpu().double()
    model.zero_grad(set_to_none=True)
    return scores, logits


@torch.no_grad()
def losses(model, layout, dataset, device):
    rank = next(iter(lora_modules(model).values())).rank
    configure(model, layout, torch.ones(max(layout.values()) + 1, rank, dtype=torch.bool),
              'none', head=False)
    model.eval()
    values = []
    for batch in batches(dataset, 64, 0, False):
        batch = to_device(batch, device)
        logits = model(**batch).logits.double()
        values.extend(F.cross_entropy(logits, batch['labels'], reduction='none').cpu().tolist())
    if not values or not all(torch.isfinite(torch.tensor(values))):
        raise FloatingPointError('invalid heldout losses')
    return values


def scale_adapter(model, layout):
    # Powers of two minimize rounding; scales differ by position to test ranking invariance.
    scales = {}
    with torch.no_grad():
        for name, module in lora_modules(model).items():
            vals = [2.0 ** (-2, -1, 1, 2)[(layout[name] + j) % 4] for j in range(module.rank)]
            c = module.lora_a.new_tensor(vals)
            module.lora_a.mul_(c[:, None])
            module.lora_b.div_(c[None, :])
            scales[name] = vals
    return scales


def prepare_train_only(repo, checkpoint, partition, clients, device):
    from pcla_repair.bridge import checked_legacy
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    identity = checkpoint['identity']
    prov = identity['provenance']
    for name, expected in prov['code_files'].items():
        if sha(repo / name) != expected:
            raise ValueError(f'execution code differs: {name}')
    if fingerprint(partition) != prov['partition_sha256']:
        raise ValueError('partition identity differs')
    old = prov['base_config']
    legacy = checked_legacy(repo)
    train_path = repo / old['dataset']['train_file']
    if sha(train_path) != prov['train_file_sha256']:
        raise ValueError('training input differs')
    # Do not call bridge.prepare: it also opens the benchmark dev file.
    rows = legacy.read_text_rows(train_path, old['dataset'])
    seed_train(keyed_seed(identity['seed'], 'model_initialization'))
    reference = old['dataset']['model']
    if (repo / reference).is_dir():
        reference = str(repo / reference)
    model = AutoModelForSequenceClassification.from_pretrained(
        reference, num_labels=old['dataset']['num_labels'], local_files_only=True)
    if tensor_digest(dict(model.state_dict())) != prov['initial_loaded_model_sha256']:
        raise ValueError('pretrained initialization differs')
    tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=True)
    for layer in legacy.encoder_layers(model):
        for projection in ('query', 'value'):
            base = getattr(layer.attention.self, projection)
            setattr(layer.attention.self, projection, BudgetLoRALinear(
                base, old['lora']['rank'], old['lora']['alpha'], old['lora']['dropout']))
    layer_by_id = {id(getattr(layer.attention.self, p)):i
                   for i, layer in enumerate(legacy.encoder_layers(model)) for p in ('query', 'value')}
    layout = {name:layer_by_id[id(module)] for name,module in model.named_modules() if id(module) in layer_by_id}
    load_adapter(model, checkpoint['adapter'])
    datasets, splits = {}, {}
    for client in clients:
        groups = split_ids(partition['clients'][str(client)], identity['seed'], client)
        splits[str(client)] = groups
        datasets[client] = {}
        for name, ids in groups.items():
            assert all(x.startswith('train:') for x in ids)
            selected = [rows[int(x.split(':')[1])] for x in ids]
            datasets[client][name] = legacy.TextClassificationDataset(selected, tokenizer, old['dataset']['max_length'])
    return model.to(device), layout, datasets, splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    assert a.seed in protocol['seeds'] and a.round in protocol['checkpoint_rounds']
    assert protocol['layer_indices'] == [0,5,10] and protocol['factors'] == ['a','b']
    if a.out.exists():
        raise FileExistsError('Preserve existing diagnostic results; choose a new output path')
    a.out.mkdir(parents=True)
    atomic_json(a.out/'status.json', {'state':'running'})
    started = time.time()
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(a.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        source = a.source_root / f'seed_{a.seed}/pcla'
        cp_path = source / f'probe_round_{a.round:03d}.pt'
        checkpoint = load_checkpoint(cp_path)
        assert checkpoint['round'] == a.round and checkpoint['policy'] == 'pcla'
        assert checkpoint['identity']['seed'] == a.seed
        assert checkpoint['identity']['config']['score'] == 'normalized_projection'
        partition = json.loads((source/'partition_ids.json').read_text())
        clients = [min(int(cid) for cid, cap in partition['capacities'].items() if cap == q)
                   for q in (6,8,12)]
        model, layout, datasets, splits = prepare_train_only(REPO, checkpoint, partition, clients, device)
        baseline = adapter_state(model)
        cfg = copy.deepcopy(checkpoint['identity']['config'])
        cfg.update(train_head=False, local_epochs=1, batch_size=16)
        frozen_digest = tensor_digest({n:p for n,p in model.named_parameters() if n not in baseline})
        manifest = {'seed':a.seed, 'round':a.round, 'source_checkpoint_sha256':sha(cp_path),
                    'source_identity_hash':checkpoint['identity_hash'], 'protocol_sha256':sha(a.protocol),
                    'diagnostic_script_sha256':sha(__file__), 'training_sources':checkpoint['identity']['provenance']['code_files'],
                    'clients':clients, 'capacities':[partition['capacities'][str(c)] for c in clients],
                    'splits':splits, 'heldout_is_client_training_data':True,
                    'benchmark_dev_read':False, 'torch':str(torch.__version__), 'cuda':torch.version.cuda,
                    'device_name':torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}
        atomic_json(a.out/'identity.json', manifest)
        total_steps = 0
        for client in clients:
            data = datasets[client]
            for factor in ('a','b'):
                load_adapter(model, baseline)
                baseline_losses = losses(model, layout, data['heldout'], device)
                assert baseline_losses == losses(model, layout, data['heldout'], device)
                original_scores, scaled_scores, scaling_checks = [], [], []
                for batch_id in range(3):
                    scores, original_logits = score_batch(model, layout, data[f'score_{batch_id}'], factor, device)
                    original_scores.append(scores)
                    scale_adapter(model, layout)
                    scaled, scaled_logits = score_batch(model, layout, data[f'score_{batch_id}'], factor, device)
                    scaled_scores.append(scaled)
                    error = float((original_logits-scaled_logits).abs().max())
                    if error > 3e-5:
                        raise ValueError(f'reciprocal scaling changed forward logits: {error}')
                    scaling_checks.append({'batch':batch_id,'max_logit_absolute_difference':error})
                    load_adapter(model, baseline)
                for layer in protocol['layer_indices']:
                    candidates = []
                    for rank in range(16):
                        load_adapter(model, baseline)
                        support = torch.zeros(12,16,dtype=torch.bool)
                        support[layer,rank] = True
                        diagnostic = {}
                        masks, _, _ = local_train(model, layout, data['update'], support, factor,
                            keyed_seed('rank-signal-update-v1',a.seed,a.round,client,factor),cfg,device,diagnostic=diagnostic)
                        assert diagnostic['optimizer_steps'] == 1 and diagnostic['training_examples_seen'] == 16
                        after = adapter_state(model)
                        for name, value in after.items():
                            mask = masks.get(name, torch.zeros_like(value,dtype=torch.bool)).cpu()
                            if not torch.equal(value[~mask],baseline[name][~mask]):
                                raise ValueError(f'frozen parameters changed: {name}')
                        after_losses = losses(model, layout, data['heldout'], device)
                        candidates.append({'rank':rank,'after_losses':after_losses,
                            'heldout_gain':sum(x-y for x,y in zip(baseline_losses,after_losses))/64})
                        total_steps += 1
                    result = {'seed':a.seed,'round':a.round,'client':client,
                        'capacity':partition['capacities'][str(client)],'layer_index':layer,'factor':factor,
                        'baseline_losses':baseline_losses,'candidates':candidates,
                        'scores':[{kind:values[layer].tolist() for kind,values in s.items()} for s in original_scores],
                        'scaled_scores':[{kind:values[layer].tolist() for kind,values in s.items()} for s in scaled_scores],
                        'scaling_checks':scaling_checks,
                        'random_scores':[[keyed_seed('rank-signal-random-v1',a.seed,a.round,client,factor,layer,b,j)/2**63
                                          for j in range(16)] for b in range(3)]}
                    atomic_json(a.out/f'client_{client}_layer_{layer}_{factor}.json',result)
                    print(json.dumps({'seed':a.seed,'round':a.round,'client':client,'factor':factor,'layer':layer,'steps':total_steps}),flush=True)
                load_adapter(model, baseline)
        assert total_steps == 288
        assert tensor_digest({n:p for n,p in model.named_parameters() if n not in baseline}) == frozen_digest
        assert tensor_digest(adapter_state(model)) == tensor_digest(baseline)
        atomic_json(a.out/'status.json',{'state':'completed','optimizer_steps':total_steps,'units':18,
                    'elapsed_seconds':time.time()-started,'frozen_parameters_verified':True,
                    'zero_update_loss_exact':True,'restored_checkpoint_exact':True})
    except BaseException:
        import traceback
        (a.out/'exception.txt').write_text(traceback.format_exc())
        atomic_json(a.out/'status.json',{'state':'failed','elapsed_seconds':time.time()-started})
        raise


if __name__ == '__main__':
    main()
