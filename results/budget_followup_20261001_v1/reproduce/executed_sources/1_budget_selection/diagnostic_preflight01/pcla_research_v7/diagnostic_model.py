"""Real-model and frozen-data bridge for fixed-parent diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from pcla_repair.core import tensor_digest
from .data import TokenCache, role_ids
from .diagnostic_configuration import verify_anchor, sha
from .diagnostic_readout import DiagnosticView
from .runtime import evaluate, load_state, state


def source_rows(ids):
    """Keep train source row identity, not an index within a role subset."""
    rows = []
    for value in ids:
        if not isinstance(value, str) or not value.startswith('train:'):
            raise ValueError('diagnostics require train source identities')
        suffix = value.removeprefix('train:')
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError('invalid train source row')
        rows.append(int(suffix))
    if len(rows) != len(set(rows)):
        raise ValueError('duplicate source rows')
    return tuple(rows)


def diagnostic_views(tokens, manifest, seed, participants):
    """Bind each reader to its existing allowed role and full source SHA."""
    def view(name, role, actor, cid=None):
        ids = role_ids(manifest, role, actor, seed, cid)
        return DiagnosticView(name, tokens.dataset(manifest, role, actor, seed, cid),
                              manifest['source_sha256'], source_rows(ids))
    local = {cid: view('local64', 'client_local_readout', 'local_diagnostic', cid)
             for cid in participants}
    selection = view('selection1024', 'cohort_select', 'finite_bank_search')
    readout = view('readout2048', 'cohort_readout', 'cohort_report')
    # All views share the source identity, so cross-role overlap cannot be
    # hidden by a distinct role label.
    seen = set()
    for item in [selection, readout, *local.values()]:
        ids = set(item.row_ids)
        if ids & seen:
            raise ValueError('diagnostic roles overlap')
        seen.update(ids)
    return local, selection, readout


class ModelLoss:
    """Evaluate registered diagnostic views, restore parent and RNG after each call.

    Repeated bank combinations reuse an exact tensor/data identity cache. This
    saves duplicate readouts without selecting candidates using new information.
    """
    def __init__(self, model, views, *, batch_size=64):
        self.model = model
        self.batch_size = batch_size
        self.views = {id(v.dataset): v for v in views}
        if len(self.views) != len(views):
            raise ValueError('duplicate diagnostic dataset object')
        self.cache = {}
        self.evaluations = []

    def __call__(self, parameters, dataset):
        view = self.views.get(id(dataset))
        if view is None or view.dataset is not dataset:
            raise PermissionError('unregistered diagnostic data role')
        key = (view.source_id, view.row_ids, tensor_digest(parameters))
        if key in self.cache:
            return self.cache[key]
        parent = state(self.model)
        modes = {name: m.training for name, m in self.model.named_modules()}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        started = time.perf_counter()
        try:
            load_state(self.model, parameters)
            value = evaluate(self.model, dataset, batch_size=self.batch_size)['loss']
        finally:
            load_state(self.model, parent)
            for name, module in self.model.named_modules():
                module.training = modes[name]
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
        self.cache[key] = float(value)
        self.evaluations.append({'role': view.role, 'rows': len(dataset),
                                 'state_sha256': key[2], 'loss': float(value),
                                 'seconds': time.perf_counter() - started})
        return float(value)


def prepare_context(config, *, device, model_prepare=None):
    """Verify the saved V anchor and recreate its base model using the real loader."""
    if config.get('schema') != 'pcla-v7-diagnostic-config-1':
        raise ValueError('invalid diagnostic config')
    parent = config['parent']
    parent_config_path = Path(parent['config'])
    expanded = json.loads(parent_config_path.read_text())
    manifest_path = Path(expanded['manifest'])
    if not manifest_path.is_absolute():
        manifest_path = parent_config_path.parent / manifest_path
    if sha(manifest_path) != config['frozen_data_manifest_sha256']:
        raise ValueError('frozen diagnostic manifest differs')
    manifest = json.loads(manifest_path.read_text())
    ids = sorted(map(int, manifest['partitions'][str(config['seed'])]['client_train']))
    anchor = verify_anchor(Path(parent['anchor']), seed=config['seed'],
                           anchor=config['anchor_round'], client_ids=ids,
                           parent_config=parent_config_path,
                           parent_config_sha256=parent['config_sha256'],
                           source_manifest_sha256=parent['source_manifest_sha256'])
    if anchor['state'] != 'anchor_verified' or anchor['participants'] != config['participants']:
        raise ValueError('parent anchor not ready or participants differ')
    if model_prepare is None:
        from scripts.run_pcla_v7 import prepare as model_prepare
    model, clients, capacities, _unused_development_view, provenance = model_prepare(
        expanded, parent_config_path, torch.device(device))
    original = json.loads((Path(parent['output']) / 'run_identity.json').read_text())['provenance']
    for key in ('manifest_sha256', 'split_sha256', 'source_sha256', 'token_cache_sha256',
                'token_cache_identity_sha256', 'pretrained_model_sha256', 'b8_wire_manifest_sha256'):
        if provenance[key] != original[key]:
            raise ValueError(f'parent model/data provenance differs: {key}')
    payload = torch.load(parent['anchor'], map_location='cpu', weights_only=False)
    if sha(Path(parent['anchor'])) != anchor['anchor_sha256']:
        raise ValueError('anchor changed during loading')
    load_state(model, payload['adapter'])
    cache_path = Path(expanded['token_cache'])
    if not cache_path.is_absolute():
        cache_path = parent_config_path.parent / cache_path
    tokens = TokenCache(cache_path)
    local, selection, readout = diagnostic_views(tokens, manifest, config['seed'], config['participants'])
    public = tokens.dataset(manifest, 'public_training', 'algorithm')
    if len(public) != 512:
        raise ValueError('public direction requires exactly the frozen 512 rows')
    return dict(model=model, parent_state=payload['adapter'], anchor=anchor,
                anchor_metrics=payload['metrics'], parent_cache=payload['cache'],
                clients={cid: clients[cid] for cid in config['participants']},
                capacities={cid: capacities[cid] for cid in config['participants']},
                visits={int(cid): count for cid, count in anchor['visits_from_anchor_metrics'].items()},
                public=public, local64=local, selection=selection, readout=readout,
                provenance=provenance)
