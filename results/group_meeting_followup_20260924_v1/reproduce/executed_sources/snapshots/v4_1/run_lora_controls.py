"""Centralized controls and matched simultaneous-factor aggregation mechanisms."""
from __future__ import annotations
import argparse
import copy
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch
from torch.utils.data import ConcatDataset, Subset
from common import (SEEDS, require, read, atomic_json, atomic_torch, tensor_load,
                    fingerprint, sha256, exclusive, source_files)
from context import Bundle, prepare_real, base_study, write_evaluation_identity
from lora_matrix import exact_server_correction

MODES = ('central_pooled_both', 'central_serial_both', 'central_serial_alt',
         'fed_both_mean', 'fed_both_exact', 'bridge_full_alt')


def projection_state(model):
    from pcla_repair import core
    return {n: m.base.weight.detach().cpu().clone() for n, m in core.lora_modules(model).items()}


def load_projections(model, values):
    from pcla_repair import core
    modules = core.lora_modules(model)
    require(set(modules) == set(values), 'projection inventory changed')
    with torch.no_grad():
        for n, m in modules.items():
            v = values[n]
            require(v.shape == m.base.weight.shape and v.dtype == m.base.weight.dtype
                    and bool(torch.isfinite(v).all()), 'projection shape/dtype invalid')
            m.base.weight.copy_(v)


def frozen_other_digest(model):
    from pcla_repair import core
    projections = {n+'.base.weight' for n in core.lora_modules(model)}
    return core.tensor_digest({n: p for n, p in model.named_parameters()
        if '.lora_' not in n and not n.startswith('classifier.') and n not in projections})


def client_sequence(bundle, r):
    from pcla_repair.core import keyed_seed
    return random.Random(keyed_seed(bundle.seed, 'participants', r)).sample(
        sorted(bundle.clients), bundle.cfg['selected_clients'])


def expected_work(bundle, selected):
    c = bundle.cfg
    steps = sum(math.ceil(len(bundle.clients[i])/c['batch_size']) for i in selected)*c['local_epochs']
    examples = sum(len(bundle.clients[i]) for i in selected)*c['local_epochs']
    return steps, examples


def pooled_step(bundle, selected, r, optimizer):
    from pcla_repair import core, runtime as rt
    from pcla_repair.optimization_dev import make_optimizer
    cfg, model = bundle.cfg, bundle.model
    full = torch.ones(bundle.shape, dtype=torch.bool)
    core.configure(model, bundle.layout, full, 'both', forward_mode=cfg['forward_mode'], head=cfg['train_head'])
    if optimizer is None:
        optimizer = make_optimizer(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    ds = ConcatDataset([bundle.clients[c] for c in selected])
    rt.seed_train(core.keyed_seed(bundle.seed, 'central_train', r))
    model.train(); count = steps = 0; orders = []
    for epoch in range(cfg['local_epochs']):
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(
            core.keyed_seed(bundle.seed, 'central_order', r, epoch))).tolist()
        orders.append(order)
        for batch in rt.batches(Subset(ds, order), cfg['batch_size'], 0, shuffle=False):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**rt.to_device(batch, bundle.device)).loss
            require(bool(torch.isfinite(loss)), 'nonfinite pooled loss')
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params),
                    'nonfinite pooled gradient')
            if cfg.get('clip_grad_norm') is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg['clip_grad_norm'])
            optimizer.step(); count += batch['labels'].numel(); steps += 1
    model.zero_grad(set_to_none=True)
    return optimizer, {'steps': steps, 'examples': count,
                       'pooled_order_sha256': fingerprint(orders),
                       'optimizer_tensor_parameters': sum(p.numel() for p in params)}


def train_round(bundle, mode, r, optimizer=None, save_bank=False):
    from pcla_repair import core, runtime as rt
    cfg, model = bundle.cfg, bundle.model
    selected = client_sequence(bundle, r)
    full = torch.ones(bundle.shape, dtype=torch.bool)
    before = core.adapter_state(model)
    base = projection_state(model)
    before_projection_digest = core.tensor_digest(base)
    mods = core.lora_modules(model)
    info = {'round': r, 'mode': mode, 'selected_clients': selected,
            'base_before_sha256': before_projection_digest,
            'initial_round_adapter_sha256': core.tensor_digest(before),
            'sample_multiset_sha256': fingerprint([s for c in selected for s in bundle.partition['clients'][str(c)]]),
            'download_bytes': 0, 'upload_bytes': 0, 'client_details': []}
    rt.synchronize(bundle.device); started = time.perf_counter()
    bank, alternative = None, None
    if mode == 'central_pooled_both':
        optimizer, extra = pooled_step(bundle, selected, r, optimizer); info.update(extra)
    else:
        fed = mode in ('fed_both_mean', 'fed_both_exact', 'bridge_full_alt')
        factor = ('b' if r % 2 else 'a') if mode in ('central_serial_alt','bridge_full_alt') else 'both'
        total = sum(len(bundle.clients[c]) for c in selected)
        accumulator = core.DeltaAccumulator(before, total, 'fixed') if fed else None
        targets = {n: torch.zeros_like(m.base.weight, dtype=torch.float64, device='cpu') for n,m in mods.items()} if fed else None
        bank = {'parent_adapter': before, 'parent_projections': base, 'clients': [],
                'scalings': {n: float(m.scaling) for n,m in mods.items()}, 'factor': factor} if fed and save_bank else None
        info.update({'steps': 0, 'examples': 0, 'factor': factor})
        if fed:
            down = {'round': r, 'adapter': before}
            if mode == 'fed_both_exact': down['server_projection_weights'] = base
            per_download = rt.payload_size(down)
            info['download_bytes'] = per_download*len(selected)
        for cid in selected:
            if fed: core.load_adapter(model, before)
            local_start = core.adapter_state(model)
            masks, seconds, allocated = rt.local_train(model, bundle.layout, bundle.clients[cid], full,
                factor, core.keyed_seed(bundle.seed,'train',r,cid), cfg, bundle.device)
            local = core.adapter_state(model)
            if factor in ('a','b'):
                partner = 'a' if factor == 'b' else 'b'
                for n in mods:
                    require(torch.equal(local[n+'.lora_'+partner], local_start[n+'.lora_'+partner]),
                            'common frozen factor changed')
            steps = math.ceil(len(bundle.clients[cid])/cfg['batch_size'])*cfg['local_epochs']
            samples = len(bundle.clients[cid])*cfg['local_epochs']
            info['steps'] += steps; info['examples'] += samples
            info['client_details'].append({'client': cid, 'sample_weight': len(bundle.clients[cid]),
                'steps_derived': steps, 'examples_derived': samples, 'train_seconds': seconds,
                'optimizer_tensor_parameters': allocated, 'factor': factor})
            if fed:
                accumulator.add(local,masks,len(bundle.clients[cid]))
                for n,m in mods.items():
                    targets[n] += len(bundle.clients[cid])*float(m.scaling)*(local[n+'.lora_b'].double()@local[n+'.lora_a'].double())
                info['upload_bytes'] += rt.payload_size(rt.upload_payload(before,local,bundle.layout,full,masks,r,cid,factor))
                if bank is not None:
                    bank['clients'].append({'client': cid,'weight': len(bundle.clients[cid]),'adapter': local})
        require(core.tensor_digest(projection_state(model)) == before_projection_digest,
                'local training modified frozen projection weights')
        if fed:
            mean = accumulator.finish()
            corrected, algebra = {}, []
            for n,m in mods.items():
                target = targets[n]/total
                corrected[n], record = exact_server_correction(base[n],mean[n+'.lora_b'],mean[n+'.lora_a'],float(m.scaling),target)
                algebra.append({'module': n, **record})
            if mode == 'bridge_full_alt':
                # Common partner implies exactness up to averaging/cast arithmetic.
                require(all(x['residual_norm'] < 1e-4*max(1., x['target_norm']) for x in algebra),
                        'alternating shared-factor algebra does not match')
            if mode == 'fed_both_exact':
                load_projections(model, corrected)
                if save_bank: alternative = {'adapter': mean, 'base': base, 'treatment': 'mean_same_updates'}
            elif mode == 'fed_both_mean' and save_bank:
                alternative = {'adapter': mean, 'base': corrected, 'treatment': 'exact_same_updates'}
            core.load_adapter(model, mean)
            info['algebra'] = algebra
        else:
            info['algebra'] = []
    require((info['steps'],info['examples']) == expected_work(bundle,selected), 'sample/step budget differs')
    if mode != 'fed_both_exact':
        require(core.tensor_digest(projection_state(model)) == before_projection_digest, 'base unexpectedly changed')
    rt.synchronize(bundle.device)
    info['training_and_server_seconds'] = time.perf_counter()-started
    info['steps_measurement'] = 'observed_batches' if mode == 'central_pooled_both' else 'derived_from_fixed_full_local_epoch'
    return optimizer, info, bank, alternative


def run_bundle(bundle, out, mode, stop_after=None, resume=False, recovery_reason=None):
    from pcla_repair import core, runtime as rt
    from pcla_repair.optimization_dev import make_optimizer
    require(mode in MODES, 'unknown LoRA control')
    out = Path(out); require(not out.exists() or resume, 'use new output directory')
    out.mkdir(parents=True, exist_ok=True)
    original_base = projection_state(bundle.model)
    frozen_hash = frozen_other_digest(bundle.model)
    identity = {**base_study(bundle), 'kind': 'lora_control', 'mode': mode,
                'config': bundle.cfg, 'provenance': bundle.provenance,
                'initial_projections_sha256': core.tensor_digest(original_base),
                'other_frozen_sha256': frozen_hash}
    identifier = fingerprint(identity)
    checkpoint = out/'checkpoint.pt'
    optimizer, metrics, start = None, [], 1
    with exclusive(out/'.run.lock', {'identity_hash': identifier}):
        if resume:
            require(recovery_reason and checkpoint.exists(), 'resume requires checkpoint and reason')
            cp = tensor_load(checkpoint)
            require(cp['identity_hash'] == identifier, 'resume identity differs')
            for row in cp['metrics']:
                for rel,h in row['evidence'].items(): require(sha256(out/rel) == h, 'committed record changed')
            # Save uncommitted round evidence, never silently erase it.
            for p in (out/'rounds').glob('*'):
                if p.is_dir() and p.name.isdigit() and int(p.name) > cp['round']:
                    dest=out/'recovery'/str(time.time_ns())/p.name; dest.parent.mkdir(parents=True,exist_ok=True)
                    p.rename(dest)
            core.load_adapter(bundle.model,cp['adapter']); load_projections(bundle.model,cp['projections'])
            metrics, start = cp['metrics'], cp['round']+1
            if cp['optimizer'] is not None:
                core.configure(bundle.model,bundle.layout,torch.ones(bundle.shape,dtype=torch.bool),'both',head=True)
                optimizer=make_optimizer(bundle.model,bundle.cfg); optimizer.load_state_dict(cp['optimizer'])
        else:
            atomic_json(out/'study.json', {'kind':'lora_control','identity':identity,'identity_hash':identifier})
            write_evaluation_identity(out,bundle)
        attempts_path=out/'execution_attempts.json'
        attempts=read(attempts_path) if attempts_path.exists() else []
        attempts.append({'pid':os.getpid(),'host':os.uname().nodename,'start_round':start,
                         'reason':recovery_reason,'time':time.time()})
        atomic_json(attempts_path,attempts)
        def state(r):
            return {'format':'pcla-control-v4.1','identity':identity,'identity_hash':identifier,
                    'round':r,'adapter':core.adapter_state(bundle.model),'projections':projection_state(bundle.model),
                    'optimizer':optimizer.state_dict() if optimizer else None,'metrics':copy.deepcopy(metrics)}
        try:
            if start==1:
                baseline=rt.evaluate(bundle.model,bundle.layout,bundle.evaluation,bundle.cfg['eval_batch_size'],bundle.device,record_logits=True)
                atomic_json(out/'evaluation_initial.json',baseline)
                atomic_torch(checkpoint,state(0))
            anchors=set(bundle.cfg['probe_rounds'])|{1,bundle.cfg['rounds']}
            for r in range(start,bundle.cfg['rounds']+1):
                if bundle.device.type=='cuda': torch.cuda.reset_peak_memory_stats(bundle.device)
                optimizer,record,bank,alternative=train_round(bundle,mode,r,optimizer,save_bank=r in anchors)
                record['evaluation']=rt.evaluate(bundle.model,bundle.layout,bundle.evaluation,
                    bundle.cfg['eval_batch_size'],bundle.device,record_logits=True)
                require(record['evaluation']['labels']==bundle.labels,'evaluation identity changed')
                record['cuda_peak_allocated']=torch.cuda.max_memory_allocated(bundle.device) if bundle.device.type=='cuda' else 0
                files=[]; directory=out/'rounds'/f'{r:03d}'; directory.mkdir(parents=True,exist_ok=False)
                if alternative is not None:
                    factual_adapter=core.adapter_state(bundle.model); factual_base=projection_state(bundle.model)
                    load_projections(bundle.model,alternative['base']); core.load_adapter(bundle.model,alternative['adapter'])
                    counter=rt.evaluate(bundle.model,bundle.layout,bundle.evaluation,bundle.cfg['eval_batch_size'],bundle.device,record_logits=True)
                    record['same_update_alternative']={'treatment':alternative['treatment'],'evaluation':counter}
                    load_projections(bundle.model,factual_base);core.load_adapter(bundle.model,factual_adapter)
                p=directory/'record.json';atomic_json(p,record);files.append(p)
                if bank is not None:
                    p=directory/'local_bank.pt';atomic_torch(p,bank);files.append(p)
                metrics.append({'round':r,'accuracy':record['evaluation']['accuracy'],
                    'gpu_nll':record['evaluation']['loss'],
                    'evidence':{p.relative_to(out).as_posix():sha256(p) for p in files}})
                atomic_torch(checkpoint,state(r))
                if r in anchors: atomic_torch(out/f'probe_round_{r:03d}.pt',state(r))
                if stop_after is not None and r>=stop_after and r<bundle.cfg['rounds']: break
            require(frozen_other_digest(bundle.model)==frozen_hash,'other frozen weights changed')
            complete=metrics[-1]['round']==bundle.cfg['rounds']
            atomic_json(out/'status.json',{'state':'completed' if complete else 'paused','completed_round':metrics[-1]['round']})
            if mode=='bridge_full_alt' and bundle.source_path is not None:
                last=metrics[-1]['round']
                ref=rt.load_checkpoint(bundle.source_path.parent/f'probe_round_{last:03d}.pt')
                checks={'adapter':all(torch.equal(v,core.adapter_state(bundle.model)[n]) for n,v in ref['adapter'].items()),
                        'quality':all(x['accuracy']==y['accuracy'] and x['gpu_nll']==y['loss']
                                      for x,y in zip(metrics,ref['metrics'])),
                        'predictions':all(read(out/'rounds'/f'{r:03d}'/'record.json')['evaluation']['predictions']==
                            read(bundle.source_path.parent/f'predictions_round_{r:03d}.json') for r in range(1,last+1))}
                atomic_json(out/'parity.json',{'passed':all(checks.values()),'checks':checks})
                require(all(checks.values()),'full_alt bridge parity failed')
            return tensor_load(checkpoint)
        except BaseException:
            (out/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
            atomic_json(out/'status.json',{'state':'failed'});raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True);p.add_argument('--source-root',required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--out',required=True);p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--reason');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();bundle=prepare_real(a.repo,a.source_root,a.seed,'full_alt',threads=a.threads)
    run_bundle(bundle,a.out,a.mode,a.stop_after,a.resume,a.reason)

if __name__=='__main__':main()
