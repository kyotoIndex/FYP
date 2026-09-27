import json
import pytest
import torch
from pcla_repair.runtime import run_experiment,load_checkpoint,upload_payload,payload_size
from pcla_repair.testing import fixture,tiny_config
from pcla_repair.core import configure,adapter_state,keyed_seed

torch.set_num_threads(1)

def run(path,policy="pcla",stop=None,donor=None,cfg=None):
    model,layout,clients,capacities,evaluation=fixture()
    return run_experiment(model,layout,clients,capacities,evaluation,cfg or tiny_config(),7,policy,path,
                          {"synthetic":True},stop_after=stop,donor_path=donor)


def test_resume_matches_uninterrupted_weights_and_allocations(tmp_path):
    run(tmp_path/"whole")
    run(tmp_path/"resume",stop=2)
    run(tmp_path/"resume")
    a=load_checkpoint(tmp_path/"whole/checkpoint.pt")
    b=load_checkpoint(tmp_path/"resume/checkpoint.pt")
    assert a["round"]==b["round"]==4
    for key in a["adapter"]: assert torch.equal(a["adapter"][key],b["adapter"][key]),key
    for key in a["trace"]: assert torch.equal(a["trace"][key],b["trace"][key])
    assert [m["accuracy"] for m in a["metrics"]]==[m["accuracy"] for m in b["metrics"]]


def test_replay_matches_donor_layers_counts_not_model_path(tmp_path):
    run(tmp_path/"donor")
    run(tmp_path/"replay",policy="replay_random_rank",donor=tmp_path/"donor/checkpoint.pt")
    a=load_checkpoint(tmp_path/"donor/checkpoint.pt")
    b=load_checkpoint(tmp_path/"replay/checkpoint.pt")
    assert any(not torch.equal(a["trace"][k],b["trace"][k]) for k in a["trace"])
    for k in a["trace"]: assert torch.equal(a["trace"][k].sum(1),b["trace"][k].sum(1))


def test_incomplete_donor_refused(tmp_path):
    run(tmp_path/"donor",stop=2)
    with pytest.raises(ValueError,match="complete PCLA"):
        run(tmp_path/"replay",policy="replay_random_rank",donor=tmp_path/"donor/checkpoint.pt")


def test_config_change_refuses_resume(tmp_path):
    run(tmp_path/"x",stop=2)
    config=tiny_config();config["score"]="legacy"
    with pytest.raises(ValueError,match="resume identity"):
        run(tmp_path/"x",cfg=config)


def test_log_json_is_not_a_model_checkpoint(tmp_path):
    path=tmp_path/"checkpoint.json";path.write_text('{"metrics":[]}')
    with pytest.raises(ValueError,match="logs"):
        load_checkpoint(path)


@pytest.mark.parametrize("policy",["layer_only","random","cyclic","deep_layer","random_layer","uniform_cyclic","flat_topk","full_alt","full_both","head_only"])
def test_each_policy_runs_offline(tmp_path,policy):
    run(tmp_path/policy,policy=policy)
    status=json.loads((tmp_path/policy/"status.json").read_text())
    assert status["state"]=="completed"
    assert status["completed_round"]==4


def test_payload_includes_head_ids_and_serialization_overhead():
    model,layout,*_=fixture()
    base=adapter_state(model)
    mask=torch.zeros(4,4,dtype=torch.bool);mask[0,0]=True
    masks=configure(model,layout,mask,"b")
    data=upload_payload(base,base,layout,mask,masks,1,0,"b")
    assert any(k.startswith("classifier.") for k in data["delta"])
    assert torch.equal(data["rank_ids"]["0"],torch.tensor([0]))
    raw=sum(v.numel()*v.element_size() for v in data["delta"].values())
    assert payload_size(data)>raw


def test_counterfactual_diagnostic_uses_disjoint_batches_and_restores_model():
    from pcla_repair.diagnostics import diagnose,summarize_within_layer
    model,layout,clients,_,_=fixture()
    initial=adapter_state(model)
    rows,split=diagnose(model,layout,clients[0],tiny_config(),7,"b",[0])
    assert len(rows)==4
    assert len(set(split["score_indices"]+split["update_indices"]+split["heldout_indices"]))==24
    for n,value in adapter_state(model).items(): assert torch.equal(value,initial[n])
    summary=summarize_within_layer(rows)
    assert len(summary)==5
    assert next(x for x in summary if x["score"]=="gate")["constant_score"]
    assert all(x["top1_regret"]>=0 for x in summary)
