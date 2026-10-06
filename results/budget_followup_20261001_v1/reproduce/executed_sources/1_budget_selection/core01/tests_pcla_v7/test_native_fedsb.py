import pytest
import torch
from types import SimpleNamespace

from pcla_research_v7.native_fedsb import (FedSBLinear, aggregate_r_head,
                                           effective_init_lr, initialize_pair,
                                           signed_task_step, estimate_signed_steps)


def test_author_orientation_and_frozen_basis_training():
    torch.manual_seed(23)
    task_step = torch.randn(5, 7)
    a, b, r, audit = initialize_pair(task_step, 3)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        u, s, v = torch.svd_lowrank(task_step.T, q=3, niter=10)
    assert torch.allclose(a, u.T)
    assert torch.allclose(b, v)
    assert torch.allclose(r, torch.diag(s))
    base = torch.nn.Linear(7, 5, bias=False)
    layer = FedSBLinear(base, a, b, r, dropout=0)
    x = torch.randn(4, 7, requires_grad=True)
    expected = base(x) + (x @ a.T @ r.T @ b.T)
    assert torch.allclose(layer(x), expected, atol=1e-6)
    layer(x).sum().backward()
    assert layer.r.grad is not None and layer.a.grad is None and layer.b.grad is None
    assert x.grad is not None
    assert audit["rank"] == 3


def test_signed_step_and_latent_aggregation():
    assert effective_init_lr(.001, 16) == .001 / 16
    assert torch.equal(signed_task_step(torch.tensor([-2., 0., 3.]), .1),
                       torch.tensor([.1, 0., -.1]))
    parent = {"q.a": torch.randn(3, 7), "q.b": torch.randn(5, 3),
              "q.r": torch.eye(3), "classifier.bias": torch.zeros(2)}
    first = {k: v.clone() for k, v in parent.items()}
    second = {k: v.clone() for k, v in parent.items()}
    first["q.r"] += 2
    second["q.r"] -= 1
    first["classifier.bias"] += 3
    result = aggregate_r_head(parent, [first, second], [1, 3])
    assert torch.allclose(result["q.r"], parent["q.r"]-.25)
    assert torch.allclose(result["classifier.bias"], parent["classifier.bias"]+.75)
    assert torch.equal(result["q.a"], parent["q.a"])
    first["q.a"] += 1
    with pytest.raises(ValueError, match="frozen basis"):
        aggregate_r_head(parent, [first, second], [1, 3])


def test_public_role_signed_step_does_not_mutate_model():
    class Data(torch.utils.data.Dataset):
        role = "public_training"
        actor = "algorithm"
        def __len__(self): return 512
        def __getitem__(self, i):
            return {"x": torch.tensor([float(i % 7), 1.]),
                    "labels": torch.tensor(i % 2)}
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.Module()
            self.attention.self = torch.nn.Module()
            self.attention.self.query = torch.nn.Linear(2, 2, bias=False)
        def forward(self, x, labels):
            return SimpleNamespace(loss=torch.nn.functional.cross_entropy(
                self.attention.self.query(x), labels))
    model = Tiny()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    name = "attention.self.query.weight"
    result = estimate_signed_steps(model, Data(), [name], .001,
                                   device=torch.device("cpu"), batch_size=32)
    assert result[name].shape == (2, 2)
    assert all(abs(abs(x) - .001/16) < 1e-9 or x == 0
               for x in result[name].flatten().tolist())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    bad = Data(); bad.role = "recipe_dev"
    with pytest.raises(ValueError, match="public_training512"):
        estimate_signed_steps(model, bad, [name], .001, device=torch.device("cpu"))
