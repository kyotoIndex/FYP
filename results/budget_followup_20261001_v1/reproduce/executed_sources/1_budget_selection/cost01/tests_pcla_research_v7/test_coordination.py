import torch
import pytest

from pcla_research_v7.coordination import (ClientPayload, aggregate_gradient_sketches,
                                            client_payload, coordinate_masks,
                                            coordination_due, projection, sketch,
                                            virtual_adamw_step)


def test_projection_keying_and_sketch_orientation():
    p = projection(7, 1, "q", "b", 5, h=32)
    assert p.shape == (32, 5)
    assert (p.abs() == 1 / 32**.5).all()
    torch.testing.assert_close(p, projection(7, 1, "q", "b", 5))
    assert not torch.equal(p, projection(7, 2, "q", "b", 5))
    g_b = torch.randn(5, 3)
    g_a = torch.randn(3, 5)
    torch.testing.assert_close(sketch(g_b, p, "b").double(), (p @ g_b.double()).float().double())
    torch.testing.assert_close(sketch(g_a, p, "a").double(), ((g_a.double() @ p.T).T).float().double())


def test_virtual_step_matches_fresh_adamw_selected_tensor():
    parameter = torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]], dtype=torch.float64))
    gradient = torch.tensor([[1., 0.], [-3., 4.]], dtype=torch.float64)
    before = parameter.detach().clone()
    opt = torch.optim.AdamW([parameter], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    parameter.grad = gradient.clone()
    opt.step()
    torch.testing.assert_close(parameter.detach() - before,
                               virtual_adamw_step(before, gradient, .001).double(), atol=1e-10, rtol=1e-6)


def test_payload_requires_actual_gradient_and_parent_version():
    factors = {"q": (torch.ones(3, 2), torch.ones(2, 4))}
    p = client_payload(0, 8, factors, {"q": torch.ones(3, 2)}, "b", .001, 10, 1)
    assert p.gradient_sketches["q"].shape == (32, 2)
    assert p.update_sketches["q"].dtype == torch.float32
    with pytest.raises(ValueError):
        client_payload(0, 8, factors, {"q": torch.ones(3, 3)}, "b", .001, 10, 1)
    assert [r for r in range(1, 12) if coordination_due(r, 5)] == [1, 6, 11]


def _cohort():
    # Two clients, two rank positions, one shared module. Client 0's local
    # preference points against the cohort descent, while JL can use Cbar.
    payloads = [
        ClientPayload(0, {"q": torch.tensor([[1., 0.]])},
                      {"q": torch.tensor([[1., -1.]])},
                      {"q": torch.tensor([10., 1.])}, 3),
        ClientPayload(1, {"q": torch.tensor([[1., 0.]])},
                      {"q": torch.tensor([[1., -1.]])},
                      {"q": torch.tensor([1., 10.])}, 3),
    ]
    factors = {"q": (torch.eye(2), torch.eye(2))}
    return payloads, factors


def test_jl_uses_shared_direction_and_jq_is_finite_menu_ascent():
    payloads, factors = _cohort()
    weights, budgets, visits = {0: .5, 1: .5}, {0: 1, 1: 1}, {0: 0, 1: 0}
    cbar = aggregate_gradient_sketches(payloads, weights)
    assert cbar["q"].tolist() == [[1., 0.]]
    jl = coordinate_masks(payloads, weights, budgets, factors, "b", visits, "JL")
    assert jl.masks[0]["q"].tolist() == [False, True]
    assert jl.masks[1]["q"].tolist() == [False, True]
    jq = coordinate_masks(payloads, weights, budgets, factors, "b", visits, "JQ", xi=1.)
    assert jq.objective_final >= jq.objective_initial
    assert set(jq.choices.values()) <= {"local", "cyclic", "JL"}
    assert all(int(jq.masks[cid]["q"].sum()) == 1 for cid in budgets)


def test_stale_parent_rejected_before_joint_selection():
    payloads, factors = _cohort()
    bad = list(payloads)
    bad[1] = ClientPayload(1, bad[1].gradient_sketches, bad[1].update_sketches,
                           bad[1].local_scores, 2)
    with pytest.raises(ValueError):
        coordinate_masks(bad, {0: .5, 1: .5}, {0: 1, 1: 1}, factors, "b", {0: 0, 1: 0}, "JQ")
    with pytest.raises(ValueError, match="stale"):
        coordinate_masks(payloads, {0: .5, 1: .5}, {0: 1, 1: 1}, factors,
                         "b", {0: 0, 1: 0}, "JQ", expected_parent_version=4)


def test_jq_reduces_redundant_effective_update_in_finite_menu():
    payloads = [ClientPayload(cid, {"q": torch.tensor([[-1., -1.]])},
                              {"q": torch.tensor([[2., 1.]])},
                              {"q": torch.tensor([10., 1.])}, 3)
                for cid in (0, 1)]
    factors = {"q": (torch.eye(2), torch.eye(2))}
    kwargs = (payloads, {0: .5, 1: .5}, {0: 1, 1: 1}, factors, "b", {0: 0, 1: 1})
    jl = coordinate_masks(*kwargs, "JL")
    assert jl.masks[0]["q"].tolist() == jl.masks[1]["q"].tolist() == [True, False]
    jq = coordinate_masks(*kwargs, "JQ", xi=1.)
    assert jq.objective_final > jq.objective_initial
    assert jq.masks[0]["q"].tolist() != jq.masks[1]["q"].tolist()


def test_jq_multimodule_weighted_phi_matches_independent_full_matrix_formula():
    factors = {
        "q": (torch.tensor([[1., 0.], [0., 2.]]), torch.tensor([[2., 0.], [0., 1.]])),
        "v": (torch.tensor([[1., 1.], [0., 2.]]), torch.tensor([[1., 1.], [2., 0.]])),
    }
    payloads = [
        ClientPayload(0,
                      {"q": torch.tensor([[1., 2.], [0., 1.]]),
                       "v": torch.tensor([[0., 2.], [1., 0.]])},
                      {"q": torch.tensor([[-2., -1.], [0., -1.]]),
                       "v": torch.tensor([[1., -3.], [2., 0.]])},
                      {"q": torch.tensor([2., 1.]), "v": torch.tensor([1., 2.])}, 4),
        ClientPayload(1,
                      {"q": torch.tensor([[0., 1.], [1., 2.]]),
                       "v": torch.tensor([[2., 0.], [0., 1.]])},
                      {"q": torch.tensor([[-1., 1.], [-2., -1.]]),
                       "v": torch.tensor([[-2., 1.], [0., -1.]])},
                      {"q": torch.tensor([1., 2.]), "v": torch.tensor([2., 1.])}, 4),
    ]
    weights = {0: .25, 1: .75}
    result = coordinate_masks(payloads, weights, {0: 2, 1: 2}, factors,
                              "b", {0: 0, 1: 1}, "JQ", xi=.25,
                              expected_parent_version=4)

    def terms(masks):
        linear = quadratic = 0.
        for name, (_, a) in factors.items():
            cbar = sum(weights[p.client_id] * p.gradient_sketches[name].double()
                       for p in payloads)
            tsum = sum(weights[p.client_id] * p.update_sketches[name].double()
                       * masks[p.client_id][name][None, :] for p in payloads)
            linear -= float((cbar * tsum).sum())
            quadratic += float((tsum @ a.double()).square().sum())
        return linear, quadratic

    ref_l, ref_q = terms(result.cyclic_masks)
    expected_lambda = .25 * max(0., ref_l) / ref_q if ref_q else 0.
    assert abs(result.lambda_value - expected_lambda) < 1e-12
    initial_l, initial_q = terms(result.jl_masks)
    final_l, final_q = terms(result.masks)
    assert abs(result.objective_initial - (initial_l - .5 * expected_lambda * initial_q)) < 1e-12
    assert abs(result.objective_final - (final_l - .5 * expected_lambda * final_q)) < 1e-12
