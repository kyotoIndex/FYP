"""Strict bridge to the archived, source-locked PCLA training contract."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import torch
from common import (ROOT, CONFIG, BATCH, DATASETS, ORIGINAL_TORCH, ORIGINAL_CUDA, ORIGINAL_TRANSFORMERS,
                    bind_repo, verify_repo, require, read, sha256, fingerprint, source_files,
                    atomic_json, git_blob)


@dataclass
class Bundle:
    model: object
    layout: dict
    clients: dict
    capacities: dict
    evaluation: object
    cfg: dict
    provenance: dict
    partition: dict
    labels: list
    seed: int
    source: dict | None = None
    source_path: Path | None = None
    synthetic: bool = False
    dataset: str | None = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def shape(self):
        from pcla_repair.core import lora_modules
        return max(self.layout.values())+1, next(iter(lora_modules(self.model).values())).rank

    @property
    def classes(self):
        return int(self.provenance.get('base_config', {}).get('dataset', {}).get('num_labels', 2))


def numerical_environment():
    return {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
            'python': platform.python_version(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'threads': torch.get_num_threads()}


def original_environment(device='cuda', threads=4):
    """Replicate the *actual original CLI*, not just its runtime module defaults."""
    require(threads > 0, 'invalid thread count')
    torch.set_num_threads(threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    d = torch.device(device)
    if d.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA required; refusing silent CPU fallback')
        require(torch.cuda.device_count() == 1, 'Slurm step must expose exactly one GPU')
    return d


def visible_gpu(expected_uuid=None):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one visible CUDA device required')
    x = torch.empty(1, device='cuda'); torch.cuda.synchronize()
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True)
    uuids = {parts[1].strip() for line in output.splitlines()
             if len(parts := line.split(',')) == 2 and parts[0].strip() == str(os.getpid())}
    require(len(uuids) == 1, 'could not verify current process GPU UUID')
    uuid = next(iter(uuids)); del x
    if expected_uuid is not None:
        require(uuid == expected_uuid, 'assigned GPU UUID differs from visible device')
    return {'uuid': uuid, 'name': torch.cuda.get_device_name(0),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'memory_bytes': torch.cuda.get_device_properties(0).total_memory}


def load_original(root, seed, policy, repo=None):
    _, core, rt, _ = bind_repo(repo)
    require(policy in ('pcla', 'replay_random_rank', 'random', 'full_alt'), 'unknown old policy')
    path = Path(root)/f'seed_{seed}'/policy/'checkpoint.pt'
    cp = rt.load_checkpoint(path)
    require(cp['policy'] == policy and cp['round'] == cp['identity']['config']['rounds'],
            'old source incomplete')
    require(cp['identity']['seed'] == seed and cp['identity_hash'] == rt.fingerprint(cp['identity']),
            'old identity differs')
    require(cp['identity']['config']['aggregation'] == 'fixed', 'old source is not fixed aggregation')
    if repo is not None:
        for rel, digest in cp['identity']['provenance']['code_files'].items():
            require(sha256(Path(repo)/rel) == digest, 'old execution source differs: ' + rel)
    return cp, path


def labels_from_tsv(repo, partition, provenance, role):
    """Independent label resolution; does not need transformers or legacy tokenizer."""
    ds = provenance['base_config']['dataset']
    require(ds.get('format', 'sst2_tsv') in ('sst2_tsv', 'named_tsv'), 'unsupported label source format')
    column = ds.get('label_column', 'label')
    mapping = {str(k): int(v) for k, v in ds.get('label_mapping', {}).items()}
    registered = {}
    for prefix, key in [('train', 'train_file'), ('dev', 'dev_file')]:
        p = Path(ds[key]); p = p if p.is_absolute() else Path(repo)/p
        require(sha256(p) == provenance[prefix+'_file_sha256'], 'dataset file hash changed')
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        require(rows and column in rows[0], 'dataset labels missing')
        for i, row in enumerate(rows):
            registered[f'{prefix}:{i}'] = mapping[row[column]] if mapping else int(row[column])
    return [registered[s] for s in partition[role]]


def validate_partition(partition):
    groups = list(partition['clients'].values()) + [partition['proxy'], partition['tuning'], partition['benchmark_dev']]
    seen = set()
    for group in groups:
        require(len(group) == len(set(group)) and not seen.intersection(group), 'source IDs overlap')
        seen.update(group)
    require(set(partition['clients']) == set(partition['capacities']), 'capacity/client identity differs')


def prepare_real(repo, source_root, seed, policy, role='benchmark_dev', device='cuda', threads=4):
    repo = verify_repo(repo)
    _, core, rt, bridge = bind_repo(repo)
    device = original_environment(device, threads)
    require(device.type == 'cuda', 'real BERT scientific runs require an assigned CUDA GPU')
    visible_gpu(os.environ.get('PCLA_EXPECTED_GPU_UUID'))
    cp, path = load_original(source_root, seed, policy, repo)
    plan = read(repo/CONFIG)
    require(plan['training'] == cp['identity']['config'], 'base config differs from old controls')
    require(role in ('tuning', 'benchmark_dev'), 'invalid evaluation role')
    plan['evaluation_split'] = role
    model, layout, clients, caps, ev, prov, part = bridge.prepare(repo, plan, seed, device)
    old = cp['identity']['provenance']
    for key in ('legacy_blob', 'base_config', 'train_file_sha256', 'dev_file_sha256',
                'partition_sha256', 'initial_loaded_model_sha256'):
        require(prov[key] == old[key], 'preparation invariant differs: '+key)
    require(str(torch.__version__) == old['torch'] and torch.version.cuda == old['cuda'],
            'use original PCLA PyTorch/CUDA build')
    require(importlib.metadata.version('transformers') == '4.57.1', 'transformers differs from original evidence')
    require(core.tensor_digest(core.adapter_state(model)) == cp['identity']['initial_adapter_sha256'],
            'adapter initialization differs')
    validate_partition(part)
    labels = labels_from_tsv(repo, part, prov, role)
    from pcla_repair.optimization_dev import dataset_labels
    require(labels == dataset_labels(ev).tolist(), 'prepared labels differ from source TSV')
    prov.update({'code_files': old['code_files'], 'torch': str(torch.__version__), 'cuda': torch.version.cuda})
    return Bundle(model, layout, clients, caps, ev, copy.deepcopy(plan['training']),
                  prov, part, labels, seed, cp, path, False)


def code_files(repo):
    """The original CLI's execution-source record (scripts/run_pcla_repaired.py)."""
    repo = Path(repo)
    files = {str(p.relative_to(repo)): sha256(p) for p in sorted((repo/'pcla_repair').glob('*.py'))}
    files['scripts/run_pcla_repaired.py'] = sha256(repo/'scripts/run_pcla_repaired.py')
    return files


def fresh_plan(repo, dataset, role='benchmark_dev'):
    """v3 plan with only base_config/base_config_blob redirected; SST-2 yields the v3 plan itself."""
    require(dataset in DATASETS, 'unknown dataset: ' + str(dataset))
    require(role in ('tuning', 'benchmark_dev'), 'invalid evaluation role')
    spec = DATASETS[dataset]; plan = read(Path(repo)/CONFIG)
    require(git_blob(Path(repo)/spec['config']) == spec['blob'], 'dataset configuration differs from pinned blob')
    plan.update(base_config=spec['config'], base_config_blob=spec['blob'], evaluation_split=role)
    return plan


def build_fresh(repo, dataset, seed, role='benchmark_dev', device='cpu'):
    """Checkpoint-free preparation through the original bridge; environment checks live in prepare_fresh."""
    from pcla_repair import core, bridge
    from pcla_repair.optimization_dev import dataset_labels
    repo = Path(repo).resolve(); spec = DATASETS[dataset]
    plan = fresh_plan(repo, dataset, role)
    model, layout, clients, caps, ev, prov, part = bridge.prepare(repo, plan, seed, torch.device(device))
    require(prov['train_file_sha256'] == spec['train_sha256'] and prov['dev_file_sha256'] == spec['dev_sha256'],
            'dataset files differ from pinned hashes')
    validate_partition(part)
    flat = [x for c in sorted(clients) for x in part['clients'][str(c)]]
    resolved = labels_from_tsv(repo, {'all': part[role] + flat}, prov, 'all')
    labels = resolved[:len(part[role])]
    require(labels == dataset_labels(ev).tolist(), 'prepared labels differ from source TSV')
    require(resolved[len(labels):] == [y for c in sorted(clients) for y in dataset_labels(clients[c]).tolist()],
            'client labels differ from source TSV')
    prov.update({'code_files': code_files(repo), 'torch': str(torch.__version__), 'cuda': torch.version.cuda})
    prov['fresh'] = {'dataset': dataset, 'plan': plan, 'base_config_blob': spec['blob'],
                     'train_file_sha256': prov['train_file_sha256'], 'dev_file_sha256': prov['dev_file_sha256'],
                     'partition_sha256': prov['partition_sha256'],
                     'initial_adapter_sha256': core.tensor_digest(core.adapter_state(model))}
    return Bundle(model, layout, clients, caps, ev, copy.deepcopy(plan['training']), prov, part, labels, seed,
                  dataset=dataset)


def prepare_fresh(repo, dataset, seed, role='benchmark_dev', device='cuda', threads=4):
    repo = verify_repo(repo)
    bind_repo(repo)
    device = original_environment(device, threads)
    require(device.type == 'cuda', 'real BERT scientific runs require an assigned CUDA GPU')
    visible_gpu(os.environ.get('PCLA_EXPECTED_GPU_UUID'))
    require(str(torch.__version__) == ORIGINAL_TORCH and torch.version.cuda == ORIGINAL_CUDA,
            'use original PCLA PyTorch/CUDA build')
    require(importlib.metadata.version('transformers') == ORIGINAL_TRANSFORMERS, 'transformers differs from original evidence')
    return build_fresh(repo, dataset, seed, role, device)


def synthetic_bundle(seed=7, init_seed=123, rounds=4, role='benchmark_dev'):
    _, core, rt, _ = bind_repo()
    from pcla_repair.testing import fixture, tiny_config
    model, layout, clients, caps, evaluation = fixture(init_seed)
    cfg = tiny_config(); cfg['rounds'] = rounds
    cfg['probe_rounds'] = sorted({0, min(2, rounds), rounds})
    ids = {str(c): [f'synthetic_train:{c}:{i}' for i in range(len(ds))] for c, ds in clients.items()}
    part = {'clients': ids, 'capacities': {str(c): v for c, v in caps.items()},
            'proxy': [], 'tuning': [f'synthetic_tune:{i}' for i in range(len(evaluation))],
            'benchmark_dev': [f'synthetic_dev:{i}' for i in range(len(evaluation))]}
    prov = {'synthetic': True, 'evaluation_split': role, 'partition_sha256': rt.fingerprint(part)}
    labels = [int(evaluation[i]['labels']) for i in range(len(evaluation))]
    return Bundle(model, layout, clients, caps, evaluation, cfg, prov, part, labels, seed, synthetic=True)


def original_synthetic(bundle, path, policy='pcla'):
    from pcla_repair import runtime as rt
    rt.run_experiment(bundle.model, bundle.layout, bundle.clients, bundle.capacities,
        bundle.evaluation, bundle.cfg, bundle.seed, policy, Path(path), bundle.provenance,
        partition=bundle.partition)
    return rt.load_checkpoint(Path(path)/'checkpoint.pt')


def base_study(bundle):
    from pcla_repair import core
    return {'batch': BATCH, 'seed': bundle.seed, 'synthetic': bundle.synthetic,
            'source_files': source_files(), 'numerical_environment': numerical_environment(),
            'initial_adapter_sha256': core.tensor_digest(core.adapter_state(bundle.model)),
            'source_original_identity': bundle.source['identity_hash'] if bundle.source else None,
            'source_original_checkpoint_sha256': sha256(bundle.source_path) if bundle.source_path else None,
            'partition_sha256': fingerprint(bundle.partition)}


def write_evaluation_identity(out, bundle):
    role = bundle.provenance['evaluation_split']
    atomic_json(Path(out)/'evaluation_identity.json', {
        'role': role, 'source_ids': bundle.partition[role], 'labels': bundle.labels,
        'partition_sha256': fingerprint(bundle.partition),
        'data_provenance': {k: v for k, v in bundle.provenance.items()
                            if k in ('base_config', 'train_file_sha256', 'dev_file_sha256')},
        'synthetic': bundle.synthetic})
    atomic_json(Path(out)/'partition_ids.json', bundle.partition)
