import copy
import pytest
from export_validation_base01 import quality_window, required_time, COST_FIELDS

def sample(n=150):
    metrics=[{'round':r,'deployed_bytes':r*10,'diagnostic_after_budget_bytes':0,
              'evaluation':{'accuracy':r/200,'loss':2-r/200,'samples':4096},
              'ema_evaluation':{'accuracy':0.99,'loss':0.01,'samples':4096}}
             for r in range(1,n+1)]
    result={'rounds':n,'budget_return_round':n,'required_cost_breakdown':{k:i+1 for i,k in enumerate(COST_FIELDS)}}
    return metrics,result

def test_windows_preserve_raw_and_budget_units():
    m,r=sample();s=quality_window(m,r,2000)
    assert s['half_round']==100
    assert s['raw_B8_accuracy']==0.75 and s['EMA_B8_accuracy']==0.99
    assert s['raw_r100_accuracy']==0.5
    assert s['raw_r96_100_mean']==pytest.approx(0.49)
    assert s['raw_last5_mean']==pytest.approx(0.74)
    assert s['raw_last20_min']==0.655
    assert s['raw_last20_lower5_mean']==pytest.approx(0.665)

def test_native_cap_before_byte_budget_has_full_prefix_half():
    m,r=sample(200);s=quality_window(m,r,5000)
    assert s['half_round']==200 and s['returned_round']==200

@pytest.mark.parametrize('kind',['duplicate','missing','overspend','afterbudget','nan','denominator'])
def test_reject_incomplete_or_changed_scientific_window(kind):
    m,r=sample()
    if kind=='duplicate':m[50]['round']=50
    if kind=='missing':m.pop()
    if kind=='overspend':m[-1]['deployed_bytes']=2001
    if kind=='afterbudget':m[-1]['diagnostic_after_budget_bytes']=1
    if kind=='nan':m[-1]['evaluation']['accuracy']=float('nan')
    if kind=='denominator':m[-1]['evaluation']['samples']=2048
    with pytest.raises(ValueError):quality_window(m,r,2000)

def test_cost_seven_fields_exclude_research_wall():
    _,r=sample();r['research_evaluation_seconds']=9999
    assert required_time(r)==28
    changed=copy.deepcopy(r);changed['required_cost_breakdown']['research_wall']=9999
    with pytest.raises(ValueError):required_time(changed)
    r['required_cost_breakdown']['client_training']=float('nan')
    with pytest.raises(ValueError):required_time(r)

def test_full_matrix_export_waits_for_acceptance_and_creates_no_output(tmp_path, monkeypatch):
    import json
    import sys
    import shutil
    from pathlib import Path
    import export_validation_base01 as e
    origin=Path(__file__).parents[1]/'validation_configuration01/actual/COMPILED_INDEX.json'
    dst=tmp_path/'checks/validation_configuration01/COMPILED_INDEX.json'
    dst.parent.mkdir(parents=True);shutil.copyfile(origin,dst)
    index=json.loads(dst.read_text());jobs=tmp_path/'queues/validation01/jobs';jobs.mkdir(parents=True)
    for jid in index['config_sha256']:(jobs/(jid+'.json')).write_text(json.dumps({'state':'completed'}))
    (jobs/'VAL_S01_s601.json').write_text(json.dumps({'state':'executed_pending_audit'}))
    out=tmp_path/'formal'
    monkeypatch.setattr(sys,'argv',['export','--root',str(tmp_path),'--out',str(out)])
    with pytest.raises(ValueError,match='full accepted matrix'):e.main()
    assert not out.exists()
    (jobs/'VAL_S01_s601.json').unlink()
    with pytest.raises(ValueError,match='9x12 matrix'):e.matrix_status(tmp_path)
