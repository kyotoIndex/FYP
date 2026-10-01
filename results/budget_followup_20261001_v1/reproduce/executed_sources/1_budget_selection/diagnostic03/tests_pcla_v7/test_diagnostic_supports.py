"""CPU checks for shared h32 masks, pretraining JQ lock and public role gate."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pcla_repair.core import BudgetLoRALinear, tensor_digest
from pcla_research_v7.cohort_runtime import _cyclic_offsets, probe_full_gradients, probe_seed
from pcla_research_v7.coordination import ClientPayload, coordinate_masks, projection
from pcla_research_v7.diagnostic_bank import SUPPORTS, support_digest
from pcla_research_v7.diagnostic_supports import (
    PUBLIC_RULE_ID, PublicSupportDecision, construct_diagnostic_supports,
    public512_mean_eval_gradient, public512_mean_eval_h32_JL_v1,
)
from pcla_research_v7.runtime import state
from pcla_research_v7.selectors import module_order, topk_mask


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([BudgetLoRALinear(nn.Linear(2, 2), 2, 2., 0.)
                                     for _ in range(24)])
        self.classifier = nn.Linear(2, 2)
        with torch.no_grad():
            for block in self.blocks:
                block.lora_b.fill_(.1)

    def forward(self, input_ids, labels):
        value = input_ids.float()
        for block in self.blocks:
            value = value + .01 * block(value)
        logits = self.classifier(value)
        return SimpleNamespace(logits=logits, loss=F.cross_entropy(logits, labels))


class Rows(torch.utils.data.Dataset):
    def __init__(self, count, role, actor="algorithm", shift=0.):
        self.role, self.actor = role, actor
        self.values = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2) / 100 + shift
        self.labels = torch.arange(count) % 2
        self.visited = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.visited.append(index)
        return {"input_ids": self.values[index], "labels": self.labels[index]}


def setup(factor="a"):
    torch.manual_seed(9)
    model = TinyModel()
    clients = tuple(range(10))
    probes = {cid: Rows(256, "client_train", shift=cid / 100) for cid in clients}
    public = Rows(512, "public_training")
    counts = {"public": 0, "rule": 0}

    def public_gradient(current_model, dataset, current_factor, *, batch_size):
        counts["public"] += 1
        assert dataset is public and dataset.role == "public_training" and len(dataset) == 512
        assert batch_size == 16 and current_factor == factor
        random.random(); np.random.rand(); torch.rand(1)
        return {name: torch.full_like(getattr(module, "lora_" + factor), .2)
                for name, module in ((f"blocks.{i}", block)
                                     for i, block in enumerate(current_model.blocks))}, 512

    def public_rule(*, public_gradients, payloads, factors, factor, budgets,
                    weights, seed, round_id):
        counts["rule"] += 1
        assert len(payloads) == 10 and set(public_gradients) == set(factors)
        assert weights == {cid: .1 for cid in clients}
        assert seed == 101 and round_id == 11
        # Deliberate test-only rule; production must provide a frozen rule ID.
        scores = {name: torch.arange(pair[0].shape[1], dtype=torch.float64)
                  for name, pair in factors.items()}
        masks = {cid: topk_mask(scores, budgets[cid]) for cid in clients}
        return PublicSupportDecision("test-only-explicit-rule",
                                     tensor_digest(public_gradients), masks)

    args = dict(model=model, parent_state=state(model), object_id="Cohort_s101_r10_a",
                anchor_identity="V_S10_s101-r10", anchor_sha256="a" * 64,
                seed=101, round_id=11, factor=factor, participants=clients,
                clients=probes, public=public,
                visits={cid: cid for cid in clients}, capacities={cid: 3 for cid in clients},
                locked_at_utc="2026-09-28T01:02:03Z",
                public_gradient=public_gradient, public_linear_rule=public_rule)
    return args, counts


def test_one_h32_path_feeds_identical_bank_masks_and_pretraining_jq_lock():
    args, counts = setup()
    model = args["model"]
    before_state = tensor_digest(state(model))
    before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    seen_seeds = {}
    def keyed_probe(model, dataset, factor, *, seed, batch_size, probe_batches):
        cid = next(cid for cid, rows in args["clients"].items() if rows is dataset)
        seen_seeds[cid] = seed
        assert seed == probe_seed(101, 11, cid, factor)
        return probe_full_gradients(model, dataset, factor, seed=seed,
                                    batch_size=batch_size, probe_batches=probe_batches)
    args["client_gradient"] = keyed_probe
    result = construct_diagnostic_supports(**args)
    assert counts == {"public": 1, "rule": 1}
    assert result["h32_sha256"] == result["jq_lock"].h32_sha256
    assert result["public_rows"] == 512 and result["client_probe_rows"] == 32
    assert len(seen_seeds) == 10
    assert set(result["supports"]) == set(args["participants"])
    factors = {name: (module.lora_b.detach().cpu(), module.lora_a.detach().cpu())
               for name, module in ((f"blocks.{i}", block)
                                    for i, block in enumerate(model.blocks))}
    names = module_order(factors)
    offsets = {cid: _cyclic_offsets(101, cid, "a", names, 2) for cid in args["participants"]}
    practical = coordinate_masks(result["h32_payloads"], {cid: .1 for cid in args["participants"]},
              args["capacities"], factors, "a", args["visits"], "JQ", xi=1.,
              offsets_by_client=offsets, expected_parent_version=10)
    for cid in args["participants"]:
        assert set(result["supports"][cid]) == set(SUPPORTS)
        for name, mask in (("local", practical.local_masks[cid]),
                           ("cyclic", practical.cyclic_masks[cid]),
                           ("JL_h32", practical.jl_masks[cid])):
            support = result["supports"][cid][name]
            assert all(torch.equal(support[module]["a"], mask[module]) for module in names)
            assert all(not bool(support[module]["b"].any()) for module in names)
            assert sum(int(support[module]["a"].sum()) for module in names) == 3
        chosen = result["jq_lock"].choice_by_client[cid]
        assert result["jq_lock"].support_sha256_by_client[cid] == support_digest(
            result["supports"][cid][chosen], factor="a")
    assert result["jq_lock"].information_access == (
        "parent_state", "client_train_probe32", "local_h32", "JQ_xi1_coordination")
    assert tensor_digest(state(model)) == before_state and model.training
    assert random.getstate() == before_rng[0]
    assert np.array_equal(np.random.get_state()[1], before_rng[1][1])
    assert torch.equal(torch.get_rng_state(), before_rng[2])


def test_public_role_512_and_ambiguous_rule_are_hard_gates_before_probes():
    args, counts = setup("b")
    args["public"] = Rows(2048, "cohort_readout")
    with pytest.raises(ValueError, match="public direction"):
        construct_diagnostic_supports(**args)
    assert counts == {"public": 0, "rule": 0}
    args["public"] = Rows(512, "public_training", actor="cohort_report")
    with pytest.raises(ValueError, match="public direction"):
        construct_diagnostic_supports(**args)
    args["public"] = Rows(512, "public_training")
    args["public_linear_rule"] = None
    with pytest.raises(ValueError, match="must be frozen"):
        construct_diagnostic_supports(**args)
    assert counts == {"public": 0, "rule": 0}


def test_bad_probe_rows_or_public_gradient_restores_model_and_rng():
    args, counts = setup("b")
    model = args["model"]
    before = tensor_digest(state(model))
    rng = torch.get_rng_state().clone()
    def wrong_public(current_model, dataset, factor, *, batch_size):
        torch.rand(3)
        return {f"blocks.{i}": torch.ones_like(block.lora_b)
                for i, block in enumerate(current_model.blocks)}, 32
    args["public_gradient"] = wrong_public
    with pytest.raises(ValueError, match="all 512 rows"):
        construct_diagnostic_supports(**args)
    assert tensor_digest(state(model)) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert counts["rule"] == 0
    args["clients"][0] = Rows(31, "client_train")
    with pytest.raises(ValueError, match="256 train rows"):
        construct_diagnostic_supports(**args)


def test_b_factor_supports_and_public_budget_are_separate():
    args, _ = setup("b")
    def synthetic_full_gradient(model, dataset, factor, *, seed, batch_size, probe_batches):
        assert factor == "b" and batch_size == 16 and probe_batches == 2
        return {f"blocks.{i}": torch.full_like(block.lora_b, .01 * (i + 1))
                for i, block in enumerate(model.blocks)}, 32
    args["client_gradient"] = synthetic_full_gradient
    result = construct_diagnostic_supports(**args)
    for cid in args["participants"]:
        for support in SUPPORTS:
            masks = result["supports"][cid][support]
            assert sum(int(mask["b"].sum()) for mask in masks.values()) == 3
            assert all(not bool(mask["a"].any()) for mask in masks.values())
    assert result["public_linear_rule_id"] == "test-only-explicit-rule"


@pytest.mark.parametrize("factor", ["a", "b"])
def test_default_public512_gradient_source_order_and_global_rng_restoration(factor):
    args, _ = setup(factor)
    public = args["public"]
    args.pop("public_gradient")
    args.pop("public_linear_rule")
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    result = construct_diagnostic_supports(**args)
    assert public.visited == list(range(512))
    assert result["public_linear_rule_id"] == PUBLIC_RULE_ID
    assert result["public_rows"] == 512 and len(result["public_full_gradient"]) == 24
    assert all(gradient.shape == getattr(args["model"].blocks[int(name.split(".")[1])],
                                          "lora_" + factor).shape
               for name, gradient in result["public_full_gradient"].items())
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_public_h32_jl_score_matches_independent_component_inner_product():
    names = ("layer.0.query", "layer.0.value")
    factors = {name: (torch.eye(2), torch.eye(2)) for name in names}
    public = {names[0]: torch.tensor([[1., 2.], [3., 4.]]),
              names[1]: torch.tensor([[2., 1.], [0., 3.]])}
    payloads = []
    for cid in range(10):
        updates = {name: torch.stack((
            torch.linspace(-1., 1., 32) * (cid + 1) * (index + 1),
            torch.linspace(1., -1., 32) * (cid + 2)), dim=1)
            for index, name in enumerate(names)}
        payloads.append(ClientPayload(cid, {}, updates, {}, 10))
    weights = {cid: .1 for cid in range(10)}
    budgets = {cid: 2 for cid in range(10)}
    decision = public512_mean_eval_h32_JL_v1(
        public_gradients=public, payloads=payloads, factors=factors,
        factor="a", budgets=budgets, weights=weights, seed=101, round_id=11)
    assert decision.rule_id == PUBLIC_RULE_ID
    for item in payloads:
        scores = {}
        for name in names:
            signs = projection(101, 11, name, "a", 2, h=32)
            # Explicit element-wise h32 projection and the JL linear dot.
            c = torch.tensor([[sum(float(public[name][j, width]) * float(signs[h, width])
                                   for width in range(2)) for j in range(2)]
                              for h in range(32)], dtype=torch.float32)
            scores[name] = torch.tensor([
                -.1 * sum(float(c[h, j].double() * item.update_sketches[name][h, j].double())
                           for h in range(32)) for j in range(2)])
        expected = topk_mask(scores, 2)
        assert all(torch.equal(decision.masks[item.client_id][name], expected[name])
                   for name in names)
