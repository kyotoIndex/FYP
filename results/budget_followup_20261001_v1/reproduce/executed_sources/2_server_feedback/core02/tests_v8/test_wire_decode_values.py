import torch
from pcla_research_v7.runtime import RawTensorWire

def test_real_array_decode_matches_fp32_factor_and_head_deltas():
    parent={"m.lora_a":torch.zeros(2,3),"m.lora_b":torch.zeros(3,2),"classifier.weight":torch.zeros(2,3),"classifier.bias":torch.zeros(2)}
    child={k:torch.arange(v.numel(),dtype=torch.float32).reshape(v.shape)/7 for k,v in parent.items()}
    supports={3:{"m":{"a":torch.ones(2),"b":torch.ones(2)}}}
    wire=RawTensorWire();cache={"clients":{},"events":[]}
    preview=wire.preview(round_id=1,cohort=[3],parent=parent,supports=supports,cache=cache)
    result=wire.commit(preview,cache,parent,[child],[3],supports)
    assert len(result["events"])==2
    assert result["clients"]["3"]["version"]==0
    assert sum(e["bytes"] for e in result["events"])==preview["bytes"]
