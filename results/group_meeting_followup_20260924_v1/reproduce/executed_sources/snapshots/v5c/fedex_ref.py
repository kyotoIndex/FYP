"""Upstream FedEx-LoRA (commit 2fa2e4a) aggregation and client loop on our partition, participants and evaluation.

Function bodies are compiled from the pinned upstream files; every edit is listed in ADAPTATIONS and recorded per run.
"""
from __future__ import annotations
import ast
import copy
import difflib
import hashlib
import math
import os
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from common import ROOT, require, read, atomic_json, atomic_torch, tensor_load, fingerprint, sha256, exclusive
from context import base_study, write_evaluation_identity

UPSTREAM = ROOT/'fedex_lora_upstream'
COMMIT = '2fa2e4a243f93c829e2b601e459e45aa748e4149'
UPSTREAM_SHA256 = {'fed_agg.py': '674c17d3fc4c9f41836b538227a3630433a0b674414d87c2466cd80714323840',
                   'models.py': 'fece12e32b1c4bdffab978a5c40fcf66bff992d5cca790f71c9d3d04aa1c15ad',
                   'train_eval.py': '15bb75638bf32f3b68d5df70a28c43719574f53172d116a7c941e32c8181fe07',
                   'fed_train_glue.py': '541da5b94dcadd3304f239e33e57e1cc29659c2d048bd9d55b9fd0604e77aa12'}
FORMAT = 'pcla-fedex-ref-v5c'
PREFIX = 'base_model.model.'
WARMUP_RATIO = 0.06
AGGREGATIONS = {'normal': 'aggregate_models_normal', 'ffa': 'aggregate_models_ffa',
                'ours': 'aggregate_models_ours', 'ours_fixed': 'aggregate_models_ours'}
PARTITIONS = ('noniid', 'iid')
ADAPTATIONS = (
    {'where': 'fed_train_glue.py:106-111', 'change': 'aggregation receives the trained state dicts of the participating clients',
     'reason': 'upstream passes the model objects client_models while fed_agg indexes client_models[i][k] (fed_agg.py:26,31,45,50,69,77,88,91); '
               'the state dicts it collects at fed_train_glue.py:97-104 are otherwise unused'},
    {'where': 'fed_train_glue.py:97-104 / train_eval.py:75', 'change': 'client state dicts are cloned and restricted to the keys the aggregators read (lora/classifier)',
     'reason': 'one client model object is reused, and train_client returns model.state_dict(), which aliases live parameters'},
    {'where': 'fed_train_glue.py:94-104', 'change': 'each round only the 10 sampled clients train and are aggregated',
     'reason': 'our participation sequence random.Random(keyed_seed(seed,"participants",r)).sample(sorted(clients),10); upstream trains all clients'},
    {'where': 'train_eval.py:67', 'change': 'wandb.log statement removed from train_client (AST edit); tqdm bound to identity',
     'reason': 'wandb is absent on the HPC; logging and progress display have no numerical effect'},
    {'where': 'fed_agg.py:1-17, train_eval.py:1-42, models.py:1-23', 'change': 'module-level imports not executed; functions compiled with only the names they use',
     'reason': 'datasets/sklearn/opacus/nltk/pycocoevalcap/transformers.AdamW are unused by these functions and partly unavailable'},
    {'where': 'models.py:28-30,48-50', 'change': 'RobertaForSequenceClassification.from_pretrained replaced by local bert-base-uncased (BertForSequenceClassification)',
     'reason': 'same backbone, tokenization and evaluation set as our runs; LoraConfig body unchanged (r=16, alpha=16, dropout=0.1, query/value, rslora off)'},
    {'where': 'fed_train_glue.py:78-81', 'change': 'initial LoRA A (B is zero in both) and classifier head copied from the same-seed bundle initial adapter',
     'reason': 'identical starting point to our runs; backbone verified byte-equal to the bundle backbone'},
    {'where': 'data_utils.py:95-102, fed_train_glue.py:58-60', 'change': 'bundle client datasets and benchmark dev; DataLoader(batch_size=16, shuffle=True) with generator '
     'keyed_seed(keyed_seed(seed,"train",r,c),"data",0) and torch/python RNG seeded by keyed_seed(seed,"train",r,c) before each client',
     'reason': 'same samples and batch order as our local_train for that client-round; resume reproduces the run exactly'},
    {'where': 'train_eval.py:104-156', 'change': 'fp32 evaluation that stores logits (batch 16, no shuffle)',
     'reason': 'upstream evaluation only logs metrics to wandb; logits are needed for independent audit'},
    {'where': 'peft/tuners/lora/torchao.py (environment, not upstream)', 'change': 'peft torchao availability check forced to False before building models',
     'reason': 'peft 0.19.1 raises ImportError on the HPC torchao 0.9.0 (<0.16) while dispatching LoRA layers; torchao only serves quantized layers, unused here'},
    {'where': 'fed_agg.py:112-114', 'change': 'ours_fixed only: torch.transpose(residue * scaling_factor, 1, 0) -> residue * scaling_factor',
     'reason': 'for nn.Linear, lora_B @ lora_A already has the shape and orientation of base_layer.weight'},
)


def _is_call(node, owner, attr):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr and
            isinstance(node.func.value, ast.Name) and node.func.value.id == owner)


class DropTranspose(ast.NodeTransformer):
    def __init__(self): self.edits = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if _is_call(node, 'torch', 'transpose'):
            require(len(node.args) == 3 and [getattr(a, 'value', None) for a in node.args[1:]] == [1, 0], 'unexpected upstream transpose')
            self.edits += 1
            return node.args[0]
        return node


class DropWandb(ast.NodeTransformer):
    def __init__(self): self.edits = 0

    def visit_Expr(self, node):
        if _is_call(node.value, 'wandb', 'log'):
            self.edits += 1
            return None
        return node


def upstream_file(name):
    require((UPSTREAM/'UPSTREAM_COMMIT.txt').read_text().strip() == COMMIT, 'upstream commit differs')
    require(sha256(UPSTREAM/name) == UPSTREAM_SHA256[name], 'upstream file differs from pinned commit: '+name)
    return UPSTREAM/name


def compile_upstream(file, name, namespace, edit=None):
    path = upstream_file(file); source = path.read_text(encoding='utf-8')
    found = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'upstream function missing: '+name)
    node = copy.deepcopy(found[0]); original = ast.unparse(found[0])
    if edit is not None:
        node = edit.visit(node); require(edit.edits == 1, 'upstream edit did not apply exactly once')
    executed = ast.unparse(node)
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return scope[node.name], {'file': file, 'function': name, 'lines': [found[0].lineno, found[0].end_lineno],
        'upstream_source_sha256': digest(ast.get_source_segment(source, found[0])),
        'executed_sha256': digest(executed), 'edits': 0 if edit is None else edit.edits,
        'diff': list(difflib.unified_diff(original.splitlines(), executed.splitlines(), lineterm='', n=0))}


def disable_torchao():
    import peft.import_utils as iu
    import peft.tuners.lora.torchao as lt
    iu.is_torchao_available = lt.is_torchao_available = lambda: False


def upstream_functions(agg, loader=None):
    """Aggregation, client loop and model factory for one aggregation type; loader=None yields records only."""
    require(agg in AGGREGATIONS, 'unknown aggregation')
    for f in UPSTREAM_SHA256: upstream_file(f)
    agg_ns, train_ns, model_ns = {'torch': torch, 'np': np}, {}, {}
    if loader is not None:
        import peft
        disable_torchao()
        from transformers import get_linear_schedule_with_warmup
        from torch.cuda.amp import GradScaler, autocast
        train_ns = {'torch': torch, 'get_linear_schedule_with_warmup': get_linear_schedule_with_warmup,
                    'GradScaler': GradScaler, 'autocast': autocast, 'tqdm': lambda iterable: iterable}
        model_ns = {'RobertaForSequenceClassification': loader, 'LoraConfig': peft.LoraConfig,
                    'TaskType': peft.TaskType, 'get_peft_model': peft.get_peft_model}
    functions, records = {}, {}
    functions['aggregate'], records['aggregate'] = compile_upstream('fed_agg.py', AGGREGATIONS[agg], agg_ns,
        DropTranspose() if agg == 'ours_fixed' else None)
    functions['train_client'], records['train_client'] = compile_upstream('train_eval.py', 'train_client', train_ns, DropWandb())
    functions['create_model'], records['create_model'] = compile_upstream(
        'models.py', 'create_peft_FFA_model' if agg == 'ffa' else 'create_peft_model', model_ns)
    return functions, records


class Pretrained:
    """Stands in for RobertaForSequenceClassification inside the upstream create_peft_* bodies."""
    def __init__(self, build): self.build = build

    def from_pretrained(self, name, num_labels): return self.build(name, num_labels)


class SyntheticBase(nn.Module):
    """TinyModel topology with plain nn.Linear projections, wrapped by peft exactly as BERT query/value are."""
    def __init__(self, tiny):
        super().__init__()
        from transformers import PretrainedConfig
        self.config = PretrainedConfig()
        self.blocks = nn.ModuleList([nn.ModuleDict({p: copy.deepcopy(b[p].base) for p in ('query', 'value')}) for b in tiny.blocks])
        self.classifier = copy.deepcopy(tiny.classifier)

    def forward(self, x=None, labels=None, **_):
        from transformers.modeling_outputs import SequenceClassifierOutput
        for block in self.blocks:
            x = torch.tanh(.5*block['query'](x)+.5*block['value'](x))+0.1*x
        logits = self.classifier(x)
        return SequenceClassifierOutput(loss=nn.functional.cross_entropy(logits, labels), logits=logits)


def base_builder(bundle):
    if bundle.synthetic:
        return lambda name, num_labels: SyntheticBase(bundle.model)
    from transformers import AutoModelForSequenceClassification
    ds = bundle.provenance['base_config']['dataset']

    def build(name, num_labels):
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels,
                                                                   local_files_only=ds.get('local_files_only', True))
        require(type(model).__name__ == 'BertForSequenceClassification', 'reference backbone must be BertForSequenceClassification')
        return model
    return build


def model_reference(bundle, repo):
    if bundle.synthetic: return 'synthetic-tiny'
    ds = bundle.provenance['base_config']['dataset']; p = Path(repo)/ds['model']
    return str(p) if p.is_dir() else ds['model']


def peft_keys(bundle):
    from pcla_repair import core
    keys = {}
    for n in core.lora_modules(bundle.model):
        keys[n+'.lora_a'] = (PREFIX+n+'.lora_A.default.weight',)
        keys[n+'.lora_b'] = (PREFIX+n+'.lora_B.default.weight',)
    for n, _ in bundle.model.classifier.named_parameters():
        keys['classifier.'+n] = tuple(f'{PREFIX}classifier.{w}.{n}' for w in ('modules_to_save.default', 'original_module'))
    return keys


def as_ours(state, keys):
    return {k: state[p[0]] for k, p in keys.items()}


def tracked(model, device='cpu'):
    return {k: v.detach().to(device, copy=True) for k, v in model.state_dict().items()
            if 'lora_' in k or 'classifier' in k or '.base_layer.' in k}


def adapter_part(state):
    return {k: v for k, v in state.items() if '.base_layer.' not in k}


def base_part(state):
    return {k: v for k, v in state.items() if '.base_layer.' in k}


def backbone_ours(model):
    return {k.replace('.base.', '.'): v for k, v in model.state_dict().items()
            if '.lora_' not in k and not k.startswith('classifier.') and not k.endswith('.update_mask')}


def backbone_peft(model, lora_bases=True):
    return {k[len(PREFIX):].replace('.base_layer.', '.'): v for k, v in model.state_dict().items()
            if 'lora_' not in k and not k.startswith(PREFIX+'classifier.') and (lora_bases or '.base_layer.' not in k)}


def module_names(model):
    return [n for n, m in model.named_modules() if hasattr(m, 'lora_A') and hasattr(m, 'lora_B')]


def trainable(model):
    groups = {'lora_A': 0, 'lora_B': 0, 'classifier': 0, 'other': 0}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups['lora_A' if 'lora_A' in n else 'lora_B' if 'lora_B' in n else 'classifier' if 'classifier' in n else 'other'] += p.numel()
    return groups


def install_initial(model, bundle, keys):
    from pcla_repair import core
    initial = core.adapter_state(bundle.model)
    require(set(initial) == set(keys), 'adapter/peft key inventory differs')
    values = {p: initial[k] for k, ps in keys.items() for p in ps}
    require(set(values) <= set(model.state_dict()), 'peft key mapping failed')
    require(not model.load_state_dict(values, strict=False).unexpected_keys, 'peft key mapping failed')


def iid_assignment(partition, seed):
    from pcla_repair.core import keyed_seed
    order = sorted(map(int, partition['clients'])); sizes = {len(partition['clients'][str(c)]) for c in order}
    require(len(sizes) == 1, 'IID reference requires equal client sizes')
    n = sizes.pop(); pool = [(c, i) for c in order for i in range(n)]
    random.Random(keyed_seed(seed, 'iid_partition')).shuffle(pool)
    return {str(c): pool[j*n:(j+1)*n] for j, c in enumerate(order)}


def client_data(bundle, partition):
    order = sorted(bundle.clients)
    if partition == 'noniid':
        return {c: bundle.clients[c] for c in order}, {str(c): list(bundle.partition['clients'][str(c)]) for c in order}
    assignment = iid_assignment(bundle.partition, bundle.seed); offsets, total = {}, 0
    for c in order:
        offsets[c] = total; total += len(bundle.clients[c])
    pool = ConcatDataset([bundle.clients[c] for c in order])
    return ({c: Subset(pool, [offsets[o]+i for o, i in assignment[str(c)]]) for c in order},
            {str(c): [bundle.partition['clients'][str(o)][i] for o, i in assignment[str(c)]] for c in order})


class Counted:
    """Transparent loader wrapper recording the batches the upstream loop consumes."""
    def __init__(self, loader): self.loader, self.steps, self.examples, self.labels = loader, 0, 0, {}

    def __len__(self): return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            y = batch['labels'].tolist(); self.steps += 1; self.examples += len(y)
            for v in y: self.labels[str(v)] = self.labels.get(str(v), 0)+1
            yield batch


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    from pcla_repair.optimization_dev import classification_readout
    model.eval(); logits, labels, loss_sum = [], [], 0.
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        b = {k: v.to(device) for k, v in batch.items()}; out = model(**b)
        require(bool(torch.isfinite(out.logits).all() and torch.isfinite(out.loss)), 'nonfinite evaluation')
        logits.append(out.logits.detach().float().cpu()); labels += b['labels'].cpu().tolist()
        loss_sum += float(out.loss)*b['labels'].numel()
    z = torch.cat(logits); pred = z.argmax(-1).tolist(); n = len(labels)
    return {'accuracy': sum(a == b for a, b in zip(labels, pred))/n, 'loss': loss_sum/n, 'samples': n, 'predictions': pred,
            **classification_readout(z, torch.tensor(labels)), 'logits': z.tolist(), 'labels': labels}


def product_means(states, modules, scaling):
    return {n: scaling*sum(s[n+'.lora_B.default.weight'].double()@s[n+'.lora_A.default.weight'].double()
                           for s in states)/len(states) for n in modules}


def exactness(parent, post, target, modules, scaling, initial):
    err = ideal = shift = 0.
    for n in modules:
        a, b, w = n+'.lora_A.default.weight', n+'.lora_B.default.weight', n+'.base_layer.weight'
        prev = parent[w].double()+scaling*parent[b].double()@parent[a].double()
        want = parent[w].double()+target[n]
        got = post[w].double()+scaling*post[b].double()@post[a].double()
        err += float((got-want).square().sum()); ideal += float((want-prev).square().sum())
        shift += float((post[w].double()-initial[w].to(post[w].device).double()).square().sum())
    return {'effective_error_fro': math.sqrt(err), 'ideal_update_fro': math.sqrt(ideal),
            'relative_error': math.sqrt(err/ideal) if ideal else None, 'base_shift_from_initial_fro': math.sqrt(shift),
            'definition': 'achieved W+sBA versus parent W + mean_k s B_k A_k, summed over LoRA modules'}


def norms(state):
    groups = {'lora_A': 0., 'lora_B': 0., 'classifier': 0.}
    for k, v in adapter_part(state).items():
        groups['lora_A' if 'lora_A' in k else 'lora_B' if 'lora_B' in k else 'classifier'] += float(v.double().square().sum())
    return {k: math.sqrt(v) for k, v in groups.items()}


def run_bundle(bundle, out, agg, partition='noniid', repo=None, stop_after=None, resume=False, recovery_reason=None):
    from pcla_repair import core, runtime as rt
    from run_lora_controls import client_sequence
    require(agg in AGGREGATIONS and partition in PARTITIONS, 'unknown FedEx reference cell')
    out = Path(out); require(not out.exists() or resume, 'use new output directory')
    out.mkdir(parents=True, exist_ok=True)
    cfg, device = bundle.cfg, bundle.device; study = base_study(bundle)
    mods = core.lora_modules(bundle.model); first = next(iter(mods.values())); alpha = first.scaling*first.rank
    args = SimpleNamespace(model=model_reference(bundle, repo), lora_r=first.rank,
                           lora_alpha=int(alpha) if float(alpha).is_integer() else alpha, lora_dropout=first.dropout.p,
                           rslora=False, lr=cfg['learning_rate'], local_epochs=cfg['local_epochs'],
                           warmup_ratio=WARMUP_RATIO, batch_size=cfg['batch_size'], agg_type=agg)
    if not bundle.synthetic:
        require((args.lora_r, args.lora_alpha, args.lora_dropout, args.batch_size, args.local_epochs, args.lr) ==
                (16, 16, .1, 16, 1, .001), 'real reference recipe differs from the protocol')
    functions, records = upstream_functions(agg, Pretrained(base_builder(bundle)))
    num_labels = 2 if bundle.synthetic else bundle.provenance['base_config']['dataset']['num_labels']
    global_model = functions['create_model'](num_labels, args).to(device)
    client_model = functions['create_model'](num_labels, args).to(device)
    keys = peft_keys(bundle); install_initial(global_model, bundle, keys)
    require(core.tensor_digest(backbone_peft(global_model)) == core.tensor_digest(backbone_ours(bundle.model)),
            'peft backbone differs from the bundle backbone')
    require(core.tensor_digest(as_ours(global_model.state_dict(), keys)) == study['initial_adapter_sha256'],
            'initial adapter mapping differs')
    modules = module_names(global_model)
    require(sorted(m[len(PREFIX):] for m in modules) == sorted(mods), 'peft LoRA modules differ from bundle LoRA modules')
    groups = trainable(client_model)
    require(groups['other'] == 0 and groups['lora_B'] > 0 and groups['classifier'] > 0 and (groups['lora_A'] == 0) == (agg == 'ffa'),
            'upstream trainable parameter set differs')
    initial = tracked(global_model); initial_device = base_part(tracked(global_model, device))
    scaling = args.lora_alpha/args.lora_r
    datasets, client_ids = client_data(bundle, partition)
    banks = sorted({1, cfg['rounds']})
    identity = {**study, 'kind': 'fedex_ref', 'agg': agg, 'partition': partition, 'config': cfg, 'provenance': bundle.provenance,
                'bundle_source_path': str(bundle.source_path) if bundle.source_path else None,
                'upstream': {'commit': COMMIT, 'files': {f: UPSTREAM_SHA256[f] for f in sorted(UPSTREAM_SHA256)},
                             'functions': records, 'adaptations': list(ADAPTATIONS)},
                'args': vars(args), 'scaling': scaling, 'trainable_parameters': groups,
                'client_partition_sha256': fingerprint(client_ids), 'initial_state_sha256': core.tensor_digest(initial),
                'backbone_sha256': core.tensor_digest(backbone_peft(global_model)),
                'other_frozen_sha256': core.tensor_digest(backbone_peft(global_model, False)),
                'bank_rounds': banks, 'evaluation_batch_size': args.batch_size,
                'order_seed': 'keyed_seed(keyed_seed(seed,"train",r,client),"data",0)'}
    identifier = fingerprint(identity); checkpoint = out/'checkpoint.pt'; metrics, start = [], 1
    with exclusive(out/'.run.lock', {'identity_hash': identifier}):
        if resume:
            require(recovery_reason and checkpoint.exists(), 'resume requires checkpoint and reason')
            cp = tensor_load(checkpoint); require(cp['identity_hash'] == identifier, 'resume identity differs')
            for row in cp['metrics']:
                for rel, h in row['evidence'].items(): require(sha256(out/rel) == h, 'committed record changed')
            for p in (out/'rounds').glob('*'):
                if p.is_dir() and p.name.isdigit() and int(p.name) > cp['round']:
                    dest = out/'recovery'/str(time.time_ns())/p.name; dest.parent.mkdir(parents=True, exist_ok=True); p.rename(dest)
            require(set(cp['state']) == set(initial), 'checkpoint state inventory differs')
            require(not global_model.load_state_dict(cp['state'], strict=False).unexpected_keys, 'checkpoint state differs')
            metrics, start = cp['metrics'], cp['round']+1
        else:
            atomic_json(out/'study.json', {'kind': 'fedex_ref', 'identity': identity, 'identity_hash': identifier})
            write_evaluation_identity(out, bundle)
            atomic_json(out/'client_partition.json', client_ids)
        attempts_path = out/'execution_attempts.json'
        attempts = read(attempts_path) if attempts_path.exists() else []
        attempts.append({'pid': os.getpid(), 'host': os.uname().nodename, 'start_round': start,
                         'reason': recovery_reason, 'time': time.time()})
        atomic_json(attempts_path, attempts)

        def state(r):
            return {'format': FORMAT, 'identity': identity, 'identity_hash': identifier, 'round': r,
                    'state': tracked(global_model), 'metrics': copy.deepcopy(metrics)}
        try:
            if start == 1:
                atomic_json(out/'evaluation_initial.json', evaluate(global_model, bundle.evaluation, args.batch_size, device))
                atomic_torch(checkpoint, state(0))
            for r in range(start, cfg['rounds']+1):
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                selected = client_sequence(bundle, r); parent = tracked(global_model, device)
                states, details = [], []
                rt.synchronize(device); started = time.perf_counter()
                for c in selected:
                    s = core.keyed_seed(bundle.seed, 'train', r, c); rt.seed_train(s)
                    client_model.load_state_dict(global_model.state_dict())
                    loader = Counted(DataLoader(datasets[c], batch_size=args.batch_size, shuffle=True,
                                                generator=torch.Generator().manual_seed(core.keyed_seed(s, 'data', 0))))
                    t0 = time.perf_counter()
                    trained = functions['train_client'](client_model, loader, args)
                    rt.synchronize(device)
                    states.append({k: v.detach().clone() for k, v in trained.items() if 'lora' in k or 'classifier' in k})
                    details.append({'client': c, 'steps': loader.steps, 'examples': loader.examples, 'label_counts': loader.labels,
                                    'sample_ids_sha256': fingerprint(client_ids[str(c)]), 'train_seconds': time.perf_counter()-t0})
                bank = {'clients': [{'client': c, 'state': {k: v.cpu() for k, v in s.items()}} for c, s in zip(selected, states)]} if r in banks else None
                target = product_means(states, modules, scaling)
                returned = functions['aggregate'](global_model, states, args) if agg.startswith('ours') else functions['aggregate'](global_model, states)
                require(returned is global_model, 'upstream aggregation replaced the global model object')
                post = tracked(global_model, device); rt.synchronize(device)
                summary = {'function_executed_sha256': records['aggregate']['executed_sha256'],
                           'parent_state_sha256': core.tensor_digest(parent), 'post_state_sha256': core.tensor_digest(post),
                           'parent_adapter_sha256': core.tensor_digest(adapter_part(parent)),
                           'post_adapter_sha256': core.tensor_digest(adapter_part(post)),
                           'parent_base_sha256': core.tensor_digest(base_part(parent)), 'post_base_sha256': core.tensor_digest(base_part(post)),
                           'post_norms': norms(post), 'exactness': exactness(parent, post, target, modules, scaling, initial_device)}
                seconds = time.perf_counter()-started
                record = {'round': r, 'agg': agg, 'partition': partition, 'selected_clients': selected, 'client_details': details,
                          'aggregation': summary, 'training_and_server_seconds': seconds}
                record['evaluation'] = evaluate(global_model, bundle.evaluation, args.batch_size, device)
                require(record['evaluation']['labels'] == bundle.labels, 'evaluation identity changed')
                record['cuda_peak_allocated'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                directory = out/'rounds'/f'{r:03d}'; directory.mkdir(parents=True, exist_ok=False)
                files = [directory/'record.json']; atomic_json(files[0], record)
                if bank is not None:
                    bank.update({'parent': {k: v.cpu() for k, v in adapter_part(parent).items()},
                                 'post': {k: v.cpu() for k, v in adapter_part(post).items()},
                                 'base_delta': {n: (post[n+'.base_layer.weight']-parent[n+'.base_layer.weight']).cpu()
                                                for n in modules} if agg.startswith('ours') else {},
                                 'base_parent_maxabs': {n: float(parent[n+'.base_layer.weight'].abs().max()) for n in modules},
                                 'scaling': scaling, 'modules': modules})
                    files.append(directory/'bank.pt'); atomic_torch(files[-1], bank)
                metrics.append({'round': r, 'accuracy': record['evaluation']['accuracy'], 'gpu_nll': record['evaluation']['loss'],
                                'evidence': {p.relative_to(out).as_posix(): sha256(p) for p in files}})
                atomic_torch(checkpoint, state(r))
                del parent, post, target, states
                if stop_after is not None and r >= stop_after and r < cfg['rounds']: break
            require(core.tensor_digest(backbone_peft(global_model, False)) == identity['other_frozen_sha256'], 'frozen backbone changed')
            complete = metrics[-1]['round'] == cfg['rounds']
            atomic_json(out/'status.json', {'state': 'completed' if complete else 'paused', 'completed_round': metrics[-1]['round']})
            return tensor_load(checkpoint)
        except BaseException:
            (out/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
            atomic_json(out/'status.json', {'state': 'failed'}); raise
