"""Failure-path tests for accidental mixing/mislabeling of completed run metadata."""
import copy,hashlib,json,subprocess,sys
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/summarize_pcla_repaired.py'


def put_run(root,seed,*,rate=.001,recorded_seed=None):
    path=root/f'seed_{seed}'/'pcla';path.mkdir(parents=True)
    ident={'seed':seed if recorded_seed is None else recorded_seed,'config':{'rounds':5,'learning_rate':rate},
           'provenance':{'code_files':{'core.py':'same-source'},'train_file_sha256':'same-data',
                         'partition_sha256':f'partition-{seed}','initial_loaded_model_sha256':f'init-{seed}'},
           'initial_adapter_sha256':f'adapter-{seed}'}
    digest=hashlib.sha256(json.dumps(ident,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    (path/'identity.json').write_text(json.dumps({**ident,'identity_hash':digest,'policy':'pcla'}))
    (path/'status.json').write_text(json.dumps({'state':'completed','completed_round':5}))
    (path/'metrics.json').write_text(json.dumps([{'round':r,'samples':8,'accuracy':.5,'serialized_traffic_bytes':100,'probe_seconds':0} for r in range(1,6)]))


def run(root):
    return subprocess.run([sys.executable,str(SCRIPT),'--root',str(root),'--seeds','7,13','--policies','pcla','--out',str(root/'summary.json')],capture_output=True,text=True)


def test_seeds_with_different_training_protocol_cannot_be_pooled(tmp_path):
    put_run(tmp_path,7);put_run(tmp_path,13,rate=.002)
    result=run(tmp_path)
    assert result.returncode!=0 and 'cross-seed' in result.stderr


def test_seed_specific_partitions_and_initializations_are_allowed(tmp_path):
    put_run(tmp_path,7);put_run(tmp_path,13)
    result=run(tmp_path)
    assert result.returncode==0, result.stderr
    assert json.loads((tmp_path/'summary.json').read_text())['complete_planned_matrix']


def test_mislabeled_seed_is_not_reported_complete(tmp_path):
    put_run(tmp_path,7);put_run(tmp_path,13,recorded_seed=7)
    result=run(tmp_path)
    assert result.returncode!=0
    output=json.loads((tmp_path/'summary.json').read_text())
    assert not output['complete_planned_matrix']
    assert 'directory seed/policy' in output['missing_or_failed'][0]['reason']
