"""Persist the cohort bank before search, and lock search before readout.

This is a diagnostic stage, not a whole-object completion receipt: exact versus
sketched gradient diagnostics must also be integrated before formal dispatch.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from pcla_repair.runtime import atomic_checkpoint, atomic_json
from .diagnostic_bank import collect_with_runtime, support_digest
from .diagnostic_readout import finite_bank_selection, measure_bank_readouts
from .diagnostic_model import ModelLoss
from .runtime import aggregate


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_bank_readout(*, config, context, supports, jq_lock, output,
                     config_sha256, source_sha256, collect=collect_with_runtime):
    """Resume a saved bank without repeating its 40 local training calls."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clients = config['participants']
    identity = dict(object_id=config['job_id'], anchor_sha256=context['anchor']['anchor_sha256'],
                    config_sha256=config_sha256, source_sha256=source_sha256,
                    jq_lock_sha256=jq_lock.sha256,
                    support_sha256={str(cid): {name: support_digest(mask, factor=config['factor'])
                                               for name, mask in supports[cid].items()} for cid in clients})
    identity_path = output / 'bank_stage_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('existing bank stage has a different frozen identity')
    else:
        if any(output.iterdir()):
            raise ValueError('unidentified output directory is not empty')
        atomic_json(identity_path, identity)
        atomic_json(output / 'jq_pretraining_lock.json', asdict(jq_lock))
    bank_path = output / 'bank.pt'
    bank_receipt = output / 'bank_receipt.json'
    stage_path = output / 'bank_readout_stage.json'
    if stage_path.exists():
        receipt = json.loads(stage_path.read_text())
        if (receipt.get('identity') != identity or receipt.get('status') != 'bank_readout_complete'
                or receipt['bank_sha256'] != file_sha(bank_path)
                or receipt['result_sha256'] != file_sha(receipt['result_path'])):
            raise ValueError('completed bank stage no longer matches its raw evidence')
        return receipt
    if bank_path.exists():
        if not bank_receipt.exists() or json.loads(bank_receipt.read_text())['sha256'] != file_sha(bank_path):
            raise ValueError('saved bank lacks its matching receipt; inspect before recovery')
        bank = torch.load(bank_path, map_location='cpu', weights_only=False)
        if bank['object_id'] != identity['object_id'] or bank['anchor_sha256'] != identity['anchor_sha256'] or bank['jq_lock_sha256'] != jq_lock.sha256:
            raise ValueError('bank payload identity differs')
    else:
        started = time.perf_counter()
        bank = collect(object_id=config['job_id'],
                       anchor_sha256=context['anchor']['anchor_sha256'],
                       anchor_identity=context['anchor']['checkpoint_identity'],
                       factor=config['factor'], participants=clients,
                       visits=context['visits'], anchor_metrics=context['anchor_metrics'],
                       parent_state=context['parent_state'], datasets=context['clients'],
                       supports=supports, jq_lock=jq_lock, model=context['model'],
                       cache=context['parent_cache'])
        atomic_checkpoint(bank_path, bank)
        atomic_json(bank_receipt, dict(sha256=file_sha(bank_path), seconds=time.perf_counter()-started,
                                      local_calls=40, optimizer_steps=640))
    views = [context['selection'], context['readout'], *context['local64'].values()]
    loss = ModelLoss(context['model'], views)
    selection_path = output / 'selection_lock.pt'
    selection_receipt = output / 'selection_lock.json'
    if selection_path.exists():
        receipt = json.loads(selection_receipt.read_text())
        if receipt['file_sha256'] != file_sha(selection_path):
            raise ValueError('selection lock changed')
        selection = torch.load(selection_path, map_location='cpu', weights_only=False)
    else:
        selection = finite_bank_selection(bank=bank, parent=context['parent_state'],
                                          selection=context['selection'], loss=loss, aggregate=aggregate)
        atomic_checkpoint(selection_path, selection)
        atomic_json(selection_receipt, dict(**asdict(selection), file_sha256=file_sha(selection_path)))
    # Even after recovery, the readout module validates this lock against the
    # bank and checks the row identities for overlap before evaluating.
    result = measure_bank_readouts(bank=bank, parent=context['parent_state'],
                                  local64=context['local64'], readout=context['readout'],
                                  selection_lock=selection, jq_lock=jq_lock, loss=loss, aggregate=aggregate)
    stamp = time.time_ns()
    atomic_json(output / f'evaluations_{stamp}.json', loss.evaluations)
    result_path = output / f'bank_readout_{stamp}.json'
    atomic_json(result_path, result)
    receipt = dict(schema='pcla-v7-bank-readout-stage-1', status='bank_readout_complete',
                   diagnostic_object_complete=False, identity=identity,
                   bank_sha256=file_sha(bank_path), result_path=str(result_path),
                   result_sha256=file_sha(result_path))
    atomic_json(output / 'bank_readout_stage.json', receipt)
    return receipt
