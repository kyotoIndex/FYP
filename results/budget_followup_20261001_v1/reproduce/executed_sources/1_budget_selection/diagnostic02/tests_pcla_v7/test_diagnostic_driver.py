from dataclasses import dataclass
import json
import pytest
from scripts.run_pcla_v7_diagnostic import run_stage, validate_config


@dataclass
class Lock:
    sha256: str


def config():
    return dict(schema='pcla-v7-diagnostic-config-1', line='cohort', seed=101,
                anchor_round=10, factor='a', participants=list(range(10)), job_id='obj',
                calls={'clients': 10, 'local_calls': 40, 'optimizer_steps': 640},
                training=dict(batch_size=16, local_examples=256, local_optimizer_steps=16,
                              learning_rate=.001, head_learning_rate=.001, optimizer='adamw',
                              betas=[.9, .999], eps=1e-8, weight_decay=.01, prefix_steps=[1, 4, 16],
                              paired_rng_key_scope=['diagnostic_object_id', 'client_id']))


def test_staged_driver_restores_supports_without_new_probe_or_bank(tmp_path):
    calls = []
    ctx = dict(model=None, parent_state={}, anchor={'anchor_sha256': 'a'*64, 'checkpoint_identity': 'parent'},
               clients={}, public=None, visits={}, capacities={})
    def prepare(*a, **kw): return ctx
    def construct(**kw):
        calls.append('probe')
        return dict(anchor_sha256='a'*64, supports={}, jq_lock=Lock('jq'), public_linear_rule_id='public-rule')
    def pipeline(**kw):
        calls.append('bank'); return {'diagnostic_object_complete': False}
    args = dict(config=config(), out=tmp_path, config_sha='c'*64, source_sha='s'*64,
                device='cpu', prepare=prepare, construct=construct, pipeline=pipeline)
    first = run_stage(**args, stage='supports')
    assert first['state'] == 'supports_complete' and calls == ['probe']
    second = run_stage(**args, stage='bank-readout')
    assert second['diagnostic_object_complete'] is False and calls == ['probe', 'bank']
    with pytest.raises(ValueError, match='identity'):
        run_stage(**{**args, 'source_sha': 'z'*64}, stage='supports')
    (tmp_path/'supports.pt').write_bytes(b'corruption')
    with pytest.raises(ValueError, match='hash'):
        run_stage(**args, stage='supports')


def test_wrong_training_factor_or_diagnostic_line_rejected():
    for key, value in [('factor', 'both'), ('line', 'direction'), ('seed', 401)]:
        with pytest.raises(ValueError): validate_config({**config(), key: value})
    cfg = config(); cfg['training']['learning_rate'] = .0003
    with pytest.raises(ValueError, match='training'): validate_config(cfg)


def test_direction_stage_recovery_reuses_actual_bank_and_rejects_line_mix(tmp_path):
    cfg = config()
    cfg.update(line='direction', calls=dict(clients=10, local_calls=120, optimizer_steps=1920))
    cfg['training'].update(learning_rate=.0003, head_learning_rate=.0003)
    ctx = dict(anchor={'anchor_sha256': 'a'*64, 'checkpoint_identity': 'parent'})
    calls = []
    def collect(config, context):
        calls.append('direction')
        return dict(local_calls=120, optimizer_steps=1920, readouts={'N': {'loss': .5}})
    args = dict(config=cfg, out=tmp_path, config_sha='c'*64, source_sha='s'*64,
                device='cpu', prepare=lambda *a, **kw: ctx, direction_collect=collect)
    with pytest.raises(ValueError, match='line'):
        run_stage(**args, stage='bank-readout')
    result = run_stage(**args, stage='direction')
    assert result['state'] == 'direction_executed_unreviewed'
    assert result['diagnostic_object_complete'] is False
    assert run_stage(**args, stage='direction') == result
    assert calls == ['direction']
    (tmp_path/'direction_bank.pt').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='hash'):
        run_stage(**args, stage='direction')
