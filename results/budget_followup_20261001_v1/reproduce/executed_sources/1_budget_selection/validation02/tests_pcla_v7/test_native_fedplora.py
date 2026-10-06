import pytest
import torch

from pcla_research_v7.native_fedplora import (FedPLoRALocalLinear,
    aggregate_contributors, folded_local_weight, select_indices)


def test_select_fold_preserves_full_parent_and_trains_only_selected_pairs():
    torch.manual_seed(4)
    base = torch.nn.Linear(6, 5)
    a = torch.randn(4, 6)
    b = torch.randn(5, 4)
    ids = select_indices(4, 2, protocol="pcla-global-budget-20260928-v7",
                         seed=401, round_id=3, client_id=1, module="query")
    assert len(ids) == 2
    assert ids == select_indices(4, 2, protocol="pcla-global-budget-20260928-v7",
                                 seed=401, round_id=3, client_id=1, module="query")
    local = FedPLoRALocalLinear(base, a, b, ids, dropout=0)
    x = torch.randn(3, 6, requires_grad=True)
    assert torch.allclose(local(x), base(x) + x @ (b @ a).T, atol=1e-6)
    assert torch.allclose(local.folded_weight, folded_local_weight(base.weight, a, b, ids))
    local(x).sum().backward()
    assert local.a.grad is not None and local.b.grad is not None and x.grad is not None
    assert base.weight.grad is None
    next_a, next_b = local.expand_into(a, b)
    unselected = [j for j in range(4) if j not in ids]
    assert torch.equal(next_a[unselected], a[unselected])
    assert torch.equal(next_b[:, unselected], b[:, unselected])


def test_contributor_mean_and_missing_pair_retention():
    parent = {"q.lora_a": torch.arange(8, dtype=torch.float32).reshape(2, 4),
              "q.lora_b": torch.ones(3, 2), "classifier.bias": torch.zeros(2)}
    children = {cid: {k: v.clone() for k, v in parent.items()} for cid in (0, 1, 2)}
    masks = {0: {"q": (0,)}, 1: {"q": (0,)}, 2: {"q": (1,)}}
    children[0]["q.lora_a"][0] += 2
    children[1]["q.lora_a"][0] += 4
    children[2]["q.lora_a"][1] += 6
    children[0]["q.lora_b"][:, 0] += 2
    children[1]["q.lora_b"][:, 0] += 4
    children[2]["q.lora_b"][:, 1] += 6
    children[0]["classifier.bias"] += 3
    result, audit = aggregate_contributors(parent, children, masks, {0: 1, 1: 1, 2: 2})
    assert torch.allclose(result["q.lora_a"][0], parent["q.lora_a"][0] + 3)
    assert torch.allclose(result["q.lora_a"][1], parent["q.lora_a"][1] + 6)
    assert torch.allclose(result["classifier.bias"], torch.full((2,), .75))
    assert audit["contributors_per_projection_rank"]["q"] == [2, 1]
    # The paper divides each pair by its contributors, not their data weight.
    unbalanced, _ = aggregate_contributors(parent, children, masks,
                                            {0: 1, 1: 100, 2: 2})
    assert torch.allclose(unbalanced["q.lora_a"][0], parent["q.lora_a"][0] + 3)
    children[2]["q.lora_a"][0] += 1
    with pytest.raises(ValueError, match="unselected"):
        aggregate_contributors(parent, children, masks, {0: 1, 1: 1, 2: 2})


def test_select_fold_effective_matrix_and_selected_only_dropout():
    torch.manual_seed(21)
    base = torch.nn.Linear(6, 5, bias=False)
    a = torch.randn(4, 6)
    b = torch.randn(5, 4)
    ids = (0, 2)
    local = FedPLoRALocalLinear(base, a, b, ids, scaling=.7, dropout=.5)
    x = torch.randn(3, 6)
    local.train()
    torch.manual_seed(99)
    got = local(x)
    torch.manual_seed(99)
    expected = torch.nn.functional.linear(x, base.weight + .7 * (b[:, [1, 3]] @ a[[1, 3], :]))
    expected += .7 * torch.nn.functional.linear(
        torch.nn.functional.linear(torch.nn.functional.dropout(x, p=.5, training=True), a[[0, 2], :]),
        b[:, [0, 2]])
    assert torch.allclose(got, expected, atol=1e-6)
    local.eval()
    assert torch.allclose(local(x), base(x) + .7 * (x @ (b @ a).T), atol=1e-6)
