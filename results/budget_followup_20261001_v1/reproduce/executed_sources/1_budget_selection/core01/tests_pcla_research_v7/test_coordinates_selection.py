import math

import pytest
import torch

from pcla_research_v7.coordinates import (balance_d, map_pair, refresh_due,
                                          rotate_gradient, rotation_from_sketch,
                                          rotate_t, split_s)
from pcla_research_v7.selectors import (both_half_masks, cyclic_mask,
                                        energy_scores, equalized_quotas,
                                        layer_mask, normalized_scores,
                                        tied_local_mask, topk_mask)


def test_all_maps_preserve_effective_product_and_never_modify_inputs():
    torch.manual_seed(4)
    b, a = torch.randn(7, 3), torch.randn(3, 5)
    b0, a0 = b.clone(), a.clone()
    b[:, 0] *= 100
    a[0] /= 100
    b0, a0 = b.clone(), a.clone()
    for kind in ("N", "D", "S"):
        result = map_pair(kind, b, a)
        assert result.b.dtype == result.a.dtype == torch.float32
        torch.testing.assert_close(result.b.double() @ result.a.double(),
                                   b.double() @ a.double(), rtol=1e-5, atol=1e-6)
    t = rotate_t(b, a, torch.randn(8, 3))
    torch.testing.assert_close(t.b.double() @ t.a.double(), b.double() @ a.double(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b, b0)
    torch.testing.assert_close(a, a0)


def test_rank_deficient_s_retains_trainable_null_a_row():
    a = torch.diag(torch.tensor([1., 3., 4.]))
    b = torch.diag(torch.tensor([2., 0., 0.]))
    result = split_s(b, a)
    assert result.numerical_rank == 1
    assert (result.b[:, 1:].norm(dim=0) == 0).all()
    assert (result.a[1:].norm(dim=1) > 0).all()
    torch.testing.assert_close(result.a[1:].norm(dim=1).sort().values,
                               torch.tensor([3., 4.]))
    torch.testing.assert_close(result.b @ result.a, b @ a)


def test_s_initial_zero_b_keeps_full_a_row_space_and_singular_spectrum():
    torch.manual_seed(19)
    a = torch.randn(4, 7)
    b = torch.zeros(6, 4)
    result = split_s(b, a)
    assert result.numerical_rank == 0
    assert torch.count_nonzero(result.b) == 0
    assert torch.linalg.matrix_rank(result.a.double()) == 4
    torch.testing.assert_close(torch.linalg.svdvals(result.a.double()),
                               torch.linalg.svdvals(a.double()), atol=1e-6, rtol=1e-6)
    assert result.absolute_error == 0


def test_d_zero_direction_preserves_partner():
    a = torch.eye(3)
    b = torch.diag(torch.tensor([2., 0., 0.]))
    result = balance_d(b, a)
    assert result.a[1:].norm() > 0
    torch.testing.assert_close(result.b @ result.a, b @ a)


def test_refresh_phase_and_rotation_chain_rule():
    assert [r for r in range(1, 13) if refresh_due(r, 5)] == [2, 7, 12]
    b, a = torch.randn(5, 3), torch.randn(3, 4)
    o = rotation_from_sketch(torch.randn(8, 3))
    gb, ga = torch.randn_like(b), torch.randn_like(a)
    torch.testing.assert_close(rotate_gradient(gb, o, "b").double(), gb.double() @ o, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rotate_gradient(ga, o, "a").double(), o.T @ ga.double(), rtol=1e-6, atol=1e-6)


def test_fresh_normalized_scores_are_independent_qv_positions():
    factors = {
        "q": (torch.tensor([[1., 0.], [0., 1.]]), torch.eye(2)),
        "v": (torch.tensor([[1., 0.], [0., 1.]]), torch.eye(2)),
    }
    gradients = {"q": torch.tensor([[10., 0.], [0., 1.]]),
                 "v": torch.tensor([[1., 0.], [0., 9.]])}
    scores = normalized_scores(factors, gradients, "b")
    mask = topk_mask(scores, 2)
    assert mask["q"].tolist() == [True, False]
    assert mask["v"].tolist() == [False, True]
    assert scores["q"].tolist() == [10., 1.]


def test_zero_partner_all_zero_marks_scores_unavailable():
    factors = {"q": (torch.zeros(3, 2), torch.zeros(2, 4))}
    gradients = {"q": torch.ones(3, 2)}
    assert normalized_scores(factors, gradients, "b")["q"].tolist() == [0., 0.]


def test_score_ties_follow_numeric_network_layer_order():
    mask = topk_mask({"encoder.layer.10.query": torch.tensor([1.]),
                      "encoder.layer.2.query": torch.tensor([1.])}, 1)
    assert mask["encoder.layer.2.query"].item()
    assert not mask["encoder.layer.10.query"].item()


def test_cyclic_module_quotas_and_visits_cover_rank():
    masks = [cyclic_mask(["q", "v"], 4, 2, visit, offsets={"q": 0, "v": 1}) for visit in range(4)]
    assert all(sum(int(x.sum()) for x in mask.values()) == 2 for mask in masks)
    assert torch.stack([x["q"] for x in masks]).any(0).all()
    assert torch.stack([x["v"] for x in masks]).any(0).all()
    assert masks[0]["q"].nonzero().item() != masks[0]["v"].nonzero().item()


def test_layer_tied_energy_bothhalf_equal_quota():
    scores = {"q0": torch.tensor([10., 1.]), "v0": torch.tensor([1., 1.]),
              "q1": torch.tensor([2., 2.]), "v1": torch.tensor([2., 2.])}
    pairs = {0: ("q0", "v0"), 1: ("q1", "v1")}
    assert layer_mask(scores, 4, pairs)["q0"].all()
    tied = tied_local_mask(scores, 2, pairs)
    assert tied["q0"].tolist() == tied["v0"].tolist() == [True, False]
    both = both_half_masks(scores, scores, 4, policy="local")
    assert sum(int(x.sum()) for factor in both.values() for x in factor.values()) == 4
    quotas = {1: 96, 2: 128, 3: 192}
    equal = equalized_quotas(quotas, 2)
    assert sum(equal.values()) == sum(quotas.values()) and max(equal.values()) - min(equal.values()) <= 1
