import random
import torch
import pytest
from pcla_repair.testing import fixture, TinyDataset
from pcla_repair.core import lora_modules
from pcla_research_v7.runtime import state
from pcla_research_v7.cohort_runtime import prepare_cohort
from pcla_research_v7.execution_timing import ExecutionTimingLedger, COST_LABEL_CATEGORIES
from pcla_research_v7.timing_hooks import timed, timing_context, verification


@pytest.mark.parametrize('method,representation', [('JL','N'),('JQ','N'),('local','T'),('cyclic','T'),('local','D')])
def test_cohort_timing_preserves_parameters_masks_packets_and_rng(method, representation):
    def execute(timing=None):
        model, _, _, _, _ = fixture(23)
        with torch.no_grad():
            for m in lora_modules(model).values(): m.lora_b.normal_()
        clients = {0:TinyDataset(21,64),1:TinyDataset(22,64)}
        plan = prepare_cohort(model,state(model),clients,[0,1],{0:2,1:2},{0:1,1:0},
            round_id=2,seed=23,factor='a',method=method,representation=representation,
            period=1,learning_rate=.001,batch_size=16,timing=timing)
        return plan,torch.get_rng_state(),random.getstate()
    plain, trng, prng = execute()
    ledger=ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    measured, trng2, prng2=execute(ledger)
    assert torch.equal(trng,trng2) and prng==prng2
    assert all(torch.equal(plain.parent[k],measured.parent[k]) for k in plain.parent)
    for cid in plain.supports:
        for name in plain.supports[cid]:
            for factor in ('a','b'):
                assert torch.equal(plain.supports[cid][name][factor],measured.supports[cid][name][factor])
    assert plain.control_packets==measured.control_packets
    assert plain.cache_after_controls==measured.cache_after_controls
    assert plain.decision==measured.decision
    record=ledger.snapshot()
    for label in ('state_and_optimizer_setup','scoring_and_shared_information','coordinate_and_server_decision','technical_logits_check'):
        assert record['completed_spans'][label]>0
    if representation=='T' or method in ('JL','JQ'):
        assert record['completed_spans']['message_processing']>0


def test_failed_timed_call_restores_context_and_closes_intervals():
    @timed('message_processing')
    def helper(): pass
    @timing_context('coordinate_and_server_decision')
    def fail(*,timing=None):
        helper()
        raise ValueError('probe failed')
    ledger=ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    with pytest.raises(ValueError):fail(timing=ledger)
    before=ledger.snapshot()
    helper()
    assert ledger.snapshot()==before


def test_audit_repeat_excludes_nested_algorithm_hooks_but_restores_them():
    @timed('budget_and_return_control')
    def bound(value): return value + 1
    @timing_context('coordinate_and_server_decision')
    def execute(*, timing=None):
        assert verification(bound, 4) == 5
        return bound(4)
    ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    assert execute(timing=ledger) == 5
    record = ledger.snapshot()
    assert record['completed_spans']['technical_equivalence_check'] == 1
    assert record['completed_spans']['budget_and_return_control'] == 1
