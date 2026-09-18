import copy
import pytest
import torch
from torch import nn
from pcla_repair.core import *
from pcla_repair.runtime import local_train
from pcla_repair.testing import fixture,tiny_config

torch.set_num_threads(1)

@pytest.mark.parametrize("factor",["a","b"])
def test_active_frozen_forward_and_upstream_gradient(factor):
    torch.manual_seed(2)
    mod=BudgetLoRALinear(nn.Linear(6,5,dtype=torch.float64),4,4,0)
    mod.lora_b.data.normal_(); mod.factor=factor
    x=torch.randn(3,6,dtype=torch.float64,requires_grad=True)
    mod.update_mask[:]=torch.tensor([1,0,1,0])
    y=mod(x)
    expected=mod.base(x)+nn.functional.linear(nn.functional.linear(x,mod.lora_a),mod.lora_b)
    torch.testing.assert_close(y,expected,rtol=1e-12,atol=1e-12)
    g=torch.autograd.grad(y.sum(),x,retain_graph=True)[0]
    expected_g=torch.autograd.grad(expected.sum(),x,retain_graph=True)[0]
    torch.testing.assert_close(g,expected_g,rtol=1e-12,atol=1e-12)
    y.square().sum().backward()
    grad=getattr(mod,"lora_"+factor).grad
    inactive=grad[1::2,:] if factor=="a" else grad[:,1::2]
    assert torch.count_nonzero(inactive)==0


def test_mask_out_is_a_different_function():
    torch.manual_seed(3)
    mod=BudgetLoRALinear(nn.Linear(3,3),4,4,0)
    mod.lora_b.data.normal_();mod.update_mask[:]=0
    x=torch.randn(4,3)
    full=mod(x)
    mod.forward_mode="mask_out"
    truncated=mod(x)
    torch.testing.assert_close(truncated,mod.base(x))
    assert not torch.allclose(full,truncated)

@pytest.mark.parametrize("factor",["a","b"])
def test_adamw_does_not_mutate_unselected_slices(factor):
    model,layout,clients,_,_=fixture()
    for m in lora_modules(model).values(): m.lora_b.data.normal_()
    original=adapter_state(model)
    support=torch.zeros(4,4,dtype=torch.bool);support[0,[0,2]]=True
    masks,_,_=local_train(model,layout,clients[0],support,factor,9,tiny_config(),torch.device("cpu"))
    final=adapter_state(model)
    for n,old in original.items():
        mask=masks.get(n,torch.zeros_like(old,dtype=torch.bool)).cpu()
        assert torch.equal(final[n][~mask],old[~mask]),n
    assert any(not torch.equal(final[n],old) for n,old in original.items())


def test_zero_gradient_without_guard_still_decays():
    p=nn.Parameter(torch.tensor([1.,2.]))
    opt=torch.optim.AdamW([p],lr=.1,weight_decay=.1)
    p.grad=torch.tensor([1.,0.]);opt.step()
    assert p[1]!=2.  # Regression witness for the legacy freeze assumption.


def test_guard_clears_existing_momentum():
    model,layout,*_=fixture()
    m=next(iter(lora_modules(model).values()))
    m.lora_b.data.fill_(1)
    p=m.lora_b
    optimizer=torch.optim.AdamW([p],lr=.1,weight_decay=.1)
    p.grad=torch.ones_like(p);optimizer.step()
    original=p.detach().clone()
    mask=torch.zeros_like(p,dtype=torch.bool);mask[:,0]=True
    name=next(iter(lora_modules(model)))+".lora_b"
    guard=FrozenSliceGuard(model,{name:mask})
    p.grad=torch.zeros_like(p);p.grad[:,0]=1;optimizer.step();guard.restore(optimizer)
    assert torch.equal(p[:,1:],original[:,1:])
    assert torch.count_nonzero(optimizer.state[p]["exp_avg"][:,1:])==0


@pytest.mark.parametrize("factor",["a","b"])
@pytest.mark.parametrize("kind",["normalized_projection","gate"])
def test_scores_invariant_under_reciprocal_scaling(factor,kind):
    model,layout,clients,_,_=fixture()
    model.double()
    for m in lora_modules(model).values(): m.lora_b.data.normal_()
    c=torch.tensor([.1,.7,3.,10.],dtype=torch.float64)
    def score(net):
        configure(net,layout,torch.ones(4,4,dtype=torch.bool),factor,head=False)
        net.eval();net.zero_grad(set_to_none=True)
        batch={"x":clients[0].x[:8].double(),"labels":clients[0].y[:8]}
        out=net(**batch);out.loss.backward()
        return out.logits.detach(),component_scores(net,layout,factor,kind)
    logits,s1=score(model)
    transformed=copy.deepcopy(model)
    for m in lora_modules(transformed).values():
        m.lora_a.data*=c[:,None];m.lora_b.data/=c[None,:]
    logits2,s2=score(transformed)
    torch.testing.assert_close(logits,logits2,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(s1,s2,rtol=1e-9,atol=1e-10)


def test_zero_B_gate_is_blind_but_B_projection_is_not():
    model,layout,clients,_,_=fixture()
    configure(model,layout,torch.ones(4,4,dtype=torch.bool),"b",head=False)
    model.eval();model(x=clients[0].x[:8],labels=clients[0].y[:8]).loss.backward()
    assert torch.count_nonzero(component_scores(model,layout,"b","gate"))==0
    assert component_scores(model,layout,"b","normalized_projection").max()>0


def test_same_function_does_not_imply_same_sgd_function_update():
    a=torch.tensor([1.,2.]);b=torch.tensor([.3,.5])
    def one_step(a,b):
        b=b.clone().requires_grad_()
        loss=(b[:,None]*a[None,:]).square().sum()
        grad=torch.autograd.grad(loss,b)[0]
        return (b-.01*grad)[:,None]*a[None,:]
    assert not torch.allclose(one_step(a,b),one_step(a*3,b/3))

@pytest.mark.parametrize("policy",["pcla","random","layer_only","deep_layer","flat_topk","cyclic","random_layer","uniform_cyclic"])
@pytest.mark.parametrize("q",[6,8,12])
def test_exact_primary_budgets(policy,q):
    scores=torch.arange(192,dtype=torch.float64).reshape(12,16)
    mask=allocate(policy,scores,q,8,42)
    assert int(mask.sum())==q*8
    if policy in ("layer_only","deep_layer"):
        assert set(mask.sum(1).tolist())<={0,16}


def test_layer_only_refuses_72_credit_false_match():
    with pytest.raises(ValueError,match="exactly"):
        allocate("layer_only",torch.ones(12,16),9,8,7)


def test_random_rank_trace_preserves_every_layer_count():
    scores=torch.randn(12,16)
    donor=allocate("pcla",scores,8,8,1)
    shuffled=allocate("replay_random_rank",scores,8,8,9,donor=donor)
    assert torch.equal(donor.sum(1),shuffled.sum(1))
    assert not torch.equal(donor,shuffled)


def test_fixed_denominator_fedavg_and_conditional_difference():
    base={"p":torch.tensor([1.,2.])}
    fixed=DeltaAccumulator(base,4,"fixed")
    conditional=DeltaAccumulator(base,4,"contributors")
    for acc in (fixed,conditional):
        acc.add({"p":torch.tensor([3.,2.])},{"p":torch.tensor([True,False])},1)
        acc.add(base,{"p":torch.tensor([False,False])},3)
    torch.testing.assert_close(fixed.finish()["p"],torch.tensor([1.5,2.]))
    torch.testing.assert_close(conditional.finish()["p"],torch.tensor([3.,2.]))
    # Ordinary parameter averaging with unchanged absent clients is exactly the
    # fixed-denominator delta update; it is not a separate debiasing theorem.
    torch.testing.assert_close(fixed.finish()["p"],(.25*torch.tensor([3.,2.])+.75*base["p"]))


def test_full_selection_matches_weighted_parameter_average():
    acc=DeltaAccumulator({"p":torch.tensor([1.,2.])},5)
    acc.add({"p":torch.tensor([2.,4.])},{"p":torch.ones(2,dtype=torch.bool)},2)
    acc.add({"p":torch.tensor([4.,8.])},{"p":torch.ones(2,dtype=torch.bool)},3)
    torch.testing.assert_close(acc.finish()["p"],torch.tensor([3.2,6.4]))


def test_missing_client_is_not_silently_accepted():
    acc=DeltaAccumulator({"p":torch.ones(2)},3)
    acc.add({"p":torch.ones(2)},{"p":torch.ones(2,dtype=torch.bool)},2)
    with pytest.raises(RuntimeError,match="missing"):
        acc.finish()


def test_changed_frozen_slice_rejected():
    acc=DeltaAccumulator({"p":torch.ones(2)},1)
    with pytest.raises(RuntimeError,match="frozen"):
        acc.add({"p":torch.tensor([2.,2.])},{"p":torch.tensor([True,False])},1)


def test_inactive_entire_tensor_cannot_change():
    acc=DeltaAccumulator({"p":torch.ones(2),"frozen":torch.ones(2)},1)
    with pytest.raises(RuntimeError,match="inactive"):
        acc.add({"p":torch.ones(2),"frozen":torch.zeros(2)},{"p":torch.ones(2,dtype=torch.bool)},1)
