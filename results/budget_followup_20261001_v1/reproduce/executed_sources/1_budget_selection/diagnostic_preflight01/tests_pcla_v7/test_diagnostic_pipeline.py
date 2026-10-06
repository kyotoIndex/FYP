from dataclasses import dataclass
from pathlib import Path
import pytest
import torch

from pcla_research_v7.diagnostic_bank import make_jq_lock, support_digest, SUPPORTS
import pcla_research_v7.diagnostic_pipeline as pipeline


@dataclass
class Lock:
    sha256: str


def test_stage_locks_before_readout_and_recovery_never_retrains(tmp_path, monkeypatch):
    ids = list(range(10)); order = []
    mask = {'module': {'a': torch.tensor([True]), 'b': torch.tensor([False])}}
    supports = {cid: {name: mask for name in SUPPORTS} for cid in ids}
    jq = make_jq_lock(object_id='obj', participants=ids,
                     choice_by_client={cid: 'local' for cid in ids},
                     support_sha256_by_client={cid: support_digest(mask, factor='a') for cid in ids},
                     h32_sha256='c'*64, locked_at_utc='2026-09-28T00:00:00Z', information_access=['h32'])
    config = {'participants': ids, 'job_id': 'obj', 'factor': 'a'}
    ctx = dict(anchor={'anchor_sha256': 'a'*64, 'checkpoint_identity': 'anchor'},
               visits={}, anchor_metrics=[], parent_state={}, clients={}, model=None,
               parent_cache={}, selection='selection', readout='readout', local64={})
    def collect(**kw):
        assert (tmp_path/'jq_pretraining_lock.json').is_file()
        order.append('train')
        return {'object_id': 'obj', 'anchor_sha256': 'a'*64, 'jq_lock_sha256': jq.sha256}
    def select(**kw):
        assert (tmp_path/'bank_receipt.json').is_file()
        order.append('select'); return Lock('s'*64)
    def readout(**kw):
        assert (tmp_path/'selection_lock.json').is_file()
        order.append('readout'); return {'result': 1}
    class Loss:
        def __init__(self, *a): self.evaluations = []
    monkeypatch.setattr(pipeline, 'ModelLoss', Loss)
    monkeypatch.setattr(pipeline, 'finite_bank_selection', select)
    monkeypatch.setattr(pipeline, 'measure_bank_readouts', readout)
    args = dict(config=config, context=ctx, supports=supports, jq_lock=jq, output=tmp_path,
                config_sha256='f'*64, source_sha256='b'*64, collect=collect)
    receipt = pipeline.run_bank_readout(**args)
    assert order == ['train', 'select', 'readout'] and receipt['diagnostic_object_complete'] is False
    assert pipeline.run_bank_readout(**args) == receipt and len(order) == 3
    with pytest.raises(ValueError, match='identity'):
        pipeline.run_bank_readout(**{**args, 'source_sha256': 'd'*64})
    Path(receipt['result_path']).write_text('{}')
    with pytest.raises(ValueError, match='evidence'): pipeline.run_bank_readout(**args)
