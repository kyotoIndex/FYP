import torch
from pcla_repair.testing import TinyModel,TinyDataset
from pcla_repair.core import tensor_digest
from pcla_research_v7.runtime import state,load_state,local_train,full_support

def test_default_group_matches_explicit_equal_head_lr():
    torch.manual_seed(901);m=TinyModel();initial=state(m);ds=TinyDataset(7,32)
    default=local_train(m,ds,full_support(m),seed=41,lr=.001)
    a=state(m);load_state(m,initial)
    equal=local_train(m,ds,full_support(m),seed=41,lr=.001,head_lr=.001)
    assert all(torch.equal(a[k],state(m)[k]) for k in a)
    assert len(default['optimizer_groups'])==1 and len(equal['optimizer_groups'])==2

def test_lower_head_lr_uses_actual_groups():
    torch.manual_seed(902);m=TinyModel();initial=state(m);ds=TinyDataset(8,32)
    row=local_train(m,ds,full_support(m),seed=42,lr=.001,head_lr=.0001)
    assert [x['lr'] for x in row['optimizer_groups']]==[.001,.0001]
    assert row['optimizer_steps']==2
    assert any(not torch.equal(initial[k],state(m)[k]) for k in initial if k.startswith('classifier.'))
