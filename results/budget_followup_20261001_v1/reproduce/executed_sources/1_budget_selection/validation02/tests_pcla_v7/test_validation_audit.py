"""Formal V remains independently audited; phase alone never makes evidence valid."""
import json
import pytest

from scripts.audit_pcla_v7_run import audit, canonical_hash, sha, check_validation_identity
from tests_pcla_v7.test_audit import evidence, put


def validation_evidence(root, seed=101):
    config, run, source = evidence(root, seed=seed)
    cfg = json.loads(config.read_text())
    job = cfg['job']
    job.update(phase='V', job_id=f'V_S07_D063_s{seed}', recipe_id='D063',
               representation='N', lr=.001, period=0, xi=0., hardware_block=f'V_s{seed}',
               output_dir=f'runs/pcla-global-budget-20260928-v7/V/V_S07_D063_s{seed}')
    cfg['recipe'] = {key: job[key] for key in
                     ('recipe_id', 'method', 'representation', 'rank', 'lr', 'period', 'xi')}
    cfg['expanded_sha256'] = canonical_hash({k: v for k, v in cfg.items() if k != 'expanded_sha256'})
    put(config, cfg)
    identity = json.loads((run / 'run_identity.json').read_text())
    identity['job'] = job
    identity['provenance']['config_sha256'] = sha(config)
    identity['provenance']['hardware'] = {'name': 'NVIDIA A100-PCIE-40GB' if seed in (281,307) else 'NVIDIA GeForce RTX 4090 D'}
    put(run / 'run_identity.json', identity)
    return config, run, source


@pytest.mark.parametrize('seed', [101, 281])
def test_v_requires_explicit_phase_and_matching_hardware(tmp_path, seed):
    config, run, source = validation_evidence(tmp_path, seed)
    assert not audit(config, run, source)['ok']
    result = audit(config, run, source, phase='V')
    assert result['ok'], result['errors']
    assert result['checks']['phase'] == 'V'
    identity = json.loads((run/'run_identity.json').read_text())
    identity['provenance']['hardware']['name'] = 'wrong-family'
    put(run/'run_identity.json', identity)
    assert 'V seed hardware family differs from frozen assignment' in audit(config, run, source, phase='V')['errors']


@pytest.mark.parametrize('field,value', [('seed',401),('recipe_id','D066'),
    ('job_id','V_S03_D035_s101'),('output_dir','runs/pcla-global-budget-20260928-v7/D0/D063_s401'),
    ('hardware_block','D0_s101_lr2')])
def test_wrong_validation_identity_rejected(tmp_path, field, value):
    config, _, _ = validation_evidence(tmp_path)
    cfg = json.loads(config.read_text()); cfg['job'][field] = value
    errors = []; check_validation_identity(cfg, errors)
    assert errors


def test_d0_cannot_be_relabelled_by_audit_flag(tmp_path):
    config, run, source = evidence(tmp_path)
    assert not audit(config, run, source, phase='V')['ok']
