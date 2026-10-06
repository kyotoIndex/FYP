"""Synthetic numerical checks; these outputs are never research evidence."""
import random

import pytest
import torch

from pcla_repair.testing import TinyDataset, TinyModel
from pcla_research_v7.runtime import load_state, state
from pcla_research_v8.server_control import apply_server_rule, interpolate


def setup():
    torch.manual_seed(31)
    model = TinyModel()
    parent = state(model)
    children = []
    for i in range(3):
        children.append({k: v + .02 * (i+1) * torch.randn_like(v) for k, v in parent.items()})
    proposal = {k: sum(child[k] for child in children) / 3 for k in parent}
    return model, parent, proposal, children, TinyDataset(82, 32)


def call(rule, **extra):
    model, parent, proposal, children, public = setup()
    recipe = dict(server_rule=rule, **extra)
    result, record = apply_server_rule(model, parent, proposal, children, [1,1,1],
                                       public if rule in ("scalar4", "scalar17", "block16", "separable", "headfit", "fedlaw_peft") else None,
                                       recipe, 13, 2)
    return model, parent, proposal, children, public, result, record


def test_exact_endpoints_and_fixed_blocks():
    model, parent, proposal, children, _, result, rec = call("fixed_head", alpha=1, beta=.25)
    assert all(torch.equal(result[k], proposal[k]) for k in parent if not k.startswith("classifier."))
    assert all(torch.equal(interpolate(parent, proposal, 0, 0)[k], parent[k]) for k in parent)
    assert all(torch.equal(interpolate(parent, proposal, 1, 1)[k], proposal[k]) for k in parent)
    assert all(torch.equal(state(model)[k], parent[k]) for k in parent)
    assert rec["alpha"] == 1 and rec["beta"] == .25


@pytest.mark.parametrize("rule,passes,heads", [("scalar4",4,4),("scalar17",17,17),
                                                 ("block16",4,16),("separable",4,16)])
def test_public_grid_exact_logits_and_state(rule, passes, heads):
    model, parent, proposal, children, public = setup()
    model.train()
    before_torch = torch.random.get_rng_state().clone()
    before_py = random.getstate()
    accepted, rec = apply_server_rule(model, parent, proposal, children, [1,1,1], public,
                                      {"server_rule":rule}, 7, 3)
    assert rec["public_dataset_passes"] == passes
    assert rec["encoder_batches"] == passes  # 32 examples, batch64
    assert rec["features_count"] == passes * 32
    assert rec["head_evals"] == heads
    assert model.training and torch.equal(before_torch, torch.random.get_rng_state())
    assert random.getstate() == before_py
    assert all(torch.equal(state(model)[k], parent[k]) for k in parent)
    selected = next(r for r in rec["grid_public_NLL"] if r["alpha"] == rec["alpha"] and r["beta"] == rec["beta"])
    assert selected["loss"] == rec["chosen_public_NLL"]
    # Independent complete model forward for each grid candidate.
    model.eval()
    with torch.no_grad():
        for row in rec["grid_public_NLL"]:
            load_state(model, interpolate(parent, proposal, row["alpha"], row["beta"]))
            batch = next(iter(torch.utils.data.DataLoader(public, batch_size=64)))
            direct = model(**batch).logits.float()
            loss = torch.nn.functional.cross_entropy(direct.double(), batch["labels"], reduction="mean")
            assert abs(loss.item() - row["loss"]) < 1e-6


def test_headfit_changes_head_only_and_is_reproducible():
    a = call("headfit", server_head_lr=.001, server_head_steps=32, server_batch_size=16)
    b = call("headfit", server_head_lr=.001, server_head_steps=32, server_batch_size=16)
    result, rec = a[-2:]
    assert rec["encoder_batches"] == 1
    assert rec["headfit_steps"] == 32
    assert all(torch.equal(result[k], a[2][k]) for k in result if not k.startswith("classifier."))
    assert any(not torch.equal(result[k], a[2][k]) for k in ("classifier.weight", "classifier.bias"))
    assert all(torch.equal(result[k], b[-2][k]) for k in result)


def test_fedexp_unbounded_and_fedlaw_simplex():
    model, parent, proposal, children, public = setup()
    # Opposing clients make the mean delta small, exposing eta > 1.
    children = [{k: parent[k] + sign * .1 for k in parent} for sign in (1.,-1.,.1)]
    proposal = {k: sum(c[k] for c in children)/3 for k in parent}
    chosen, rec = apply_server_rule(model, parent, proposal, children, [1,1,1], None,
                                    {"server_rule":"fedexp_peft","server_epsilon":.001}, 1, 1)
    assert rec["eta"] > 1
    assert torch.allclose(chosen["classifier.bias"], parent["classifier.bias"] + rec["eta"]*(proposal["classifier.bias"]-parent["classifier.bias"]), atol=1e-5)
    result, law = apply_server_rule(model, parent, proposal, children, [1,1,1], public,
                                    {"server_rule":"fedlaw_peft","server_lr":.005,"server_batch_size":64}, 1, 1)
    assert law["gamma"] > 0
    assert law["server_gradient_batches"] == 1
    assert all(x > 0 for x in law["lambda_weights"])
    assert sum(law["lambda_weights"]) == pytest.approx(1, abs=1e-6)
    assert all(torch.isfinite(v).all() for v in result.values())


def test_invalid_proposal_rejected_even_for_noop():
    model, parent, proposal, children, _ = setup()
    proposal["classifier.bias"][0] = float("nan")
    with pytest.raises(ValueError):
        apply_server_rule(model, parent, proposal, children, [1,1,1], None,
                          {"server_rule":"fixed_scalar","alpha":0,"beta":0}, 1, 1)
