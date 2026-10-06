import torch
from torch import nn
import random
import json
import pytest

from pcla_repair.core import BudgetLoRALinear, keyed_seed, lora_modules
from pcla_repair.runtime import batches
from pcla_repair.testing import TinyDataset, fixture
from pcla_research_v7.cohort_runtime import (_cyclic_offsets, assert_preflight_covers_actual,
                                              committed_control_cache,
                                              fixed_coordinate_verification_dataset, probe_seed,
                                              preflight_control_bytes,
                                              preflight_round_upper_bound,
                                              prepare_cohort, probe_full_gradients)
from pcla_research_v7.coordinates import rotate_gradient, rotate_with_matrix
from pcla_research_v7.coordination import ClientPayload, coordinate_masks
from pcla_research_v7.selectors import module_order
from pcla_research_v7.runtime import (RawTensorWire, aggregate, load_state,
                                       local_train, run_experiment, select_support,
                                       state, _load_checkpoint)
from pcla_research_v7.transport import decode_packet


def _direct_full_probe(model, dataset, factor, seed, batch_size):
    for p in model.parameters():
        p.requires_grad_(False)
    for module in lora_modules(model).values():
        getattr(module, "lora_" + factor).requires_grad_(True)
        module.factor = factor
    model.eval()
    out = {name: torch.zeros_like(getattr(module, "lora_" + factor), dtype=torch.float64)
           for name, module in lora_modules(model).items()}
    for i, batch in enumerate(batches(dataset, batch_size, seed)):
        if i == 2:
            break
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        for name, module in lora_modules(model).items():
            out[name] += getattr(module, "lora_" + factor).grad.detach().double()
    for module in lora_modules(model).values():
        module.factor = "none"
    return {name: value / 2 for name, value in out.items()}


def test_real_two_batch_probe_matches_full_vector_autograd_and_preserves_rng():
    model, _, clients, _, _ = fixture(31)
    seed = keyed_seed(31, "probe")
    before = torch.get_rng_state().clone()
    measured, examples = probe_full_gradients(model, clients[0], "b", seed=seed, batch_size=8)
    assert examples == 16
    assert torch.equal(before, torch.get_rng_state())
    reference = _direct_full_probe(model, clients[0], "b", seed, 8)
    for name in reference:
        torch.testing.assert_close(measured[name], reference[name], atol=1e-8, rtol=1e-6)


def test_coordinate_check_dataset_is_fixed_across_cohorts():
    fixed = TinyDataset(90, 32)
    others = {2: TinyDataset(91, 32), 10: TinyDataset(92, 32)}
    clients = {10: others[10], 0: fixed, 2: others[2]}
    assert fixed_coordinate_verification_dataset(clients) is fixed
    assert fixed_coordinate_verification_dataset(dict(reversed(list(clients.items())))) is fixed


def test_failed_probe_reports_already_encoded_control_packets():
    model, _, _, _, _ = fixture(42)
    parent = state(model)
    clients = {0: TinyDataset(100, 8), 1: TinyDataset(101, 32)}
    observed = []
    with pytest.raises(ValueError, match="two full probe batches"):
        prepare_cohort(model, parent, clients, [0, 1], {0: 2, 1: 2},
                       {0: 0, 1: 0}, round_id=1, seed=42, factor="b",
                       method="JQ", learning_rate=.001, batch_size=8,
                       on_control_packet=observed.append)
    assert [packet.kind for packet in observed] == ["coord_parent", "coord_parent"]
    assert all(packet.nbytes == len(packet.raw) for packet in observed)


def test_jq_bridge_packets_masks_and_cached_training_parent():
    model, _, clients, _, _ = fixture(17)
    parent = state(model)
    cohort = [0, 1]
    fixed_bytes = preflight_control_bytes(model, parent, cohort, round_id=1,
                                          method="JQ", representation="N", period=1,
                                          factor="b")
    observed = []
    plan = prepare_cohort(model, parent, clients, cohort, {0: 2, 1: 2}, {0: 0, 1: 0},
                          round_id=1, seed=17, factor="b", method="JQ",
                          representation="N", period=1, learning_rate=.001,
                          batch_size=8, weights={0: .5, 1: .5},
                          on_control_packet=observed.append)
    assert observed == list(plan.control_packets)
    assert plan.phase == "JQ_coordination"
    assert plan.probe_examples == {0: 16, 1: 16}
    assert [p.kind for p in plan.control_packets] == ["coord_parent", "coord_parent",
                                                     "coord_probe", "coord_probe",
                                                     "coord_support", "coord_support"]
    assert plan.control_bytes == sum(len(p.raw) for p in plan.control_packets)
    assert plan.control_bytes == fixed_bytes
    assert all(sum(int(x["b"].sum()) for x in plan.supports[cid].values()) == 2 for cid in cohort)
    wire = RawTensorWire()
    preview = wire.preview(round_id=1, cohort=cohort, parent=plan.parent,
                           supports=plan.supports, cache=plan.cache_after_controls)
    upper = preflight_round_upper_bound(model, parent, cohort, {0: 2, 1: 2},
                                        round_id=1, method="JQ", representation="N",
                                        period=1, factor="b")
    assert upper >= plan.control_bytes + preview["bytes"]
    assert_preflight_covers_actual(upper, plan.control_bytes + preview["bytes"], round_id=1)
    full_preview = RawTensorWire(force_full_download=True).preview(
        round_id=1, cohort=cohort, parent=plan.parent, supports=plan.supports,
        cache=plan.cache_after_controls)
    full_upper = preflight_round_upper_bound(model, parent, cohort, {0: 2, 1: 2},
        round_id=1, method="JQ", representation="N", period=1, factor="b",
        force_full_download=True)
    assert full_upper >= plan.control_bytes + full_preview["bytes"]
    assert full_upper > upper
    with pytest.raises(AssertionError, match="exceeds preflight"):
        assert_preflight_covers_actual(upper, upper + 1, round_id=1)
    for cid in cohort:
        _, tensors = decode_packet(preview["downloads"][cid])
        assert tensors == {}  # The actual coord_parent packet already sent these tensors.
    children = []
    for cid in cohort:
        load_state(model, plan.parent)
        local_train(model, clients[cid], plan.supports[cid], seed=cid + 100,
                    lr=.001, batch_size=8)
        children.append(state(model))
    cache = wire.commit(preview, committed_control_cache(plan, diagnostic=False),
                        plan.parent, children, cohort, plan.supports)
    assert sum(event["bytes"] for event in cache["events"]) == plan.control_bytes + preview["bytes"]
    assert len({event["event_id"] for event in cache["events"]}) == len(cache["events"])
    aggregate(plan.parent, children, [len(clients[cid]) for cid in cohort])


def test_t_bridge_reconstructs_same_float32_adapter_from_float64_rotation():
    model, _, clients, _, _ = fixture(23)
    clients[0], clients[1] = TinyDataset(100, 32), TinyDataset(101, 32)
    with torch.no_grad():
        for module in lora_modules(model).values():
            module.lora_b.normal_(std=.02)
    parent = state(model)
    fixed_bytes = preflight_control_bytes(model, parent, [0, 1], round_id=2,
                                          method="local", representation="T", period=5,
                                          factor="a")
    w0 = {name: module.base.weight.detach().clone() for name, module in lora_modules(model).items()}
    plan = prepare_cohort(model, parent, clients, [0, 1], {0: 2, 1: 2},
                          {0: 0, 1: 0}, round_id=2, seed=23, factor="a",
                          method="local", representation="T", period=5,
                          learning_rate=.001, batch_size=8, weights={0: .5, 1: .5})
    assert plan.phase == "T_refresh" and plan.representation_error_max <= 1e-5
    assert plan.control_bytes == fixed_bytes
    assert [p.kind for p in plan.control_packets] == ["t_d_parent", "t_d_parent",
                                                     "t_grad_sketch", "t_grad_sketch",
                                                     "t_rotation", "t_rotation"]
    upper = preflight_round_upper_bound(model, parent, [0, 1], {0: 2, 1: 2},
                                        round_id=2, method="local", representation="T",
                                        period=5, factor="a")
    preview = RawTensorWire().preview(round_id=2, cohort=[0, 1], parent=plan.parent,
                                      supports=plan.supports, cache=plan.cache_after_controls)
    assert upper >= plan.control_bytes + preview["bytes"]
    full_preview = RawTensorWire(force_full_download=True).preview(
        round_id=2, cohort=[0, 1], parent=plan.parent, supports=plan.supports,
        cache=plan.cache_after_controls)
    full_upper = preflight_round_upper_bound(model, parent, [0, 1], {0: 2, 1: 2},
        round_id=2, method="local", representation="T", period=5, factor="a",
        force_full_download=True)
    assert full_upper >= plan.control_bytes + full_preview["bytes"]
    assert full_upper > upper
    _, d_parent = decode_packet(plan.control_packets[0].raw)
    header, rotations = decode_packet(plan.control_packets[-1].raw)
    assert header["metadata"]["to_digest"]
    for name in lora_modules(model):
        reconstructed = rotate_with_matrix(torch.from_numpy(d_parent[name + ".lora_b"]),
                                            torch.from_numpy(d_parent[name + ".lora_a"]),
                                            torch.from_numpy(rotations[name]))
        assert torch.equal(reconstructed.b, plan.parent[name + ".lora_b"])
        assert torch.equal(reconstructed.a, plan.parent[name + ".lora_a"])
        assert torch.equal(w0[name], lora_modules(model)[name].base.weight)
    # Directly probing both coordinate systems on the same examples checks
    # the retained full-gradient chain rule used before virtual AdamW.
    load_state(model, {name: torch.from_numpy(value) for name, value in d_parent.items()})
    before_g, _ = probe_full_gradients(model, clients[0], "a", seed=77, batch_size=8)
    load_state(model, plan.parent)
    after_g, _ = probe_full_gradients(model, clients[0], "a", seed=77, batch_size=8)
    for name in before_g:
        predicted = rotate_gradient(before_g[name], torch.from_numpy(rotations[name]), "a")
        torch.testing.assert_close(predicted.double(), after_g[name], atol=1e-6, rtol=1e-4)


def test_remaining_d0_selectors_are_executable_and_spend_exact_units():
    model, _, clients, _, _ = fixture(29)
    with torch.no_grad():
        for module in lora_modules(model).values():
            module.lora_b.normal_(std=.02)
    parent = state(model)
    cases = [("tied_local", {0: 2, 1: 2}), ("layer", {0: 8, 1: 8}),
             ("energy", {0: 2, 1: 4}), ("equal_quota_cyclic", {0: 2, 1: 4})]
    for method, budgets in cases:
        plan = prepare_cohort(model, parent, clients, [0, 1], budgets,
                              {0: 0, 1: 0}, round_id=1, seed=29,
                              factor="b", method=method, representation="N",
                              learning_rate=.001, batch_size=8)
        used = {cid: sum(int(x["b"].sum()) for x in plan.supports[cid].values())
                for cid in (0, 1)}
        assert sum(used.values()) == sum(budgets.values())
        if method == "equal_quota_cyclic":
            assert used == {0: 3, 1: 3}
        else:
            assert used == budgets


def test_independent_local_and_j_probe_use_same_keyed_training_rows():
    model, _, clients, _, _ = fixture(41)
    with torch.no_grad():
        for module in lora_modules(model).values():
            module.lora_b.normal_(std=.02)
    parent = state(model)
    run_seed, cid = 41, 0
    plan_local = prepare_cohort(model, parent, clients, [cid], {cid: 3}, {cid: 0},
                                round_id=2, seed=run_seed, factor="a",
                                method="JL", representation="N", period=5,
                                learning_rate=.001, batch_size=8)
    # Round 2 is noncoordinated and therefore uses exactly the fresh local rule.
    local_round2 = select_support(model, clients[cid], "local", "a", 3,
                                  seed=keyed_seed(run_seed, "selection", cid), visit=0,
                                  batch_size=8, probe_seeds={"a": probe_seed(run_seed, 2, cid, "a")})
    assert all(torch.equal(local_round2[name]["a"], plan_local.supports[cid][name]["a"])
               for name in local_round2)


def test_cyclic_candidate_matches_standalone_with_layer_10_remainder():
    model, _, clients, _, _ = fixture(43)
    for _ in range(8):
        model.blocks.append(nn.ModuleDict({
            target: BudgetLoRALinear(nn.Linear(6, 6), 4, 4., .1)
            for target in ("query", "value")}))
    parent = state(model)
    run_seed, cid = 43, 0
    standalone = select_support(model, clients[cid], "cyclic", "b", 29,
                                seed=keyed_seed(run_seed, "selection", cid), visit=0)
    plan = prepare_cohort(model, parent, clients, [cid], {cid: 29}, {cid: 0},
                          round_id=1, seed=run_seed, factor="b", method="cyclic",
                          representation="N", learning_rate=.001)
    assert all(torch.equal(standalone[name]["b"], plan.supports[cid][name]["b"])
               for name in standalone)
    names = module_order(lora_modules(model))
    factors = {name: (module.lora_b.detach().cpu(), module.lora_a.detach().cpu())
               for name, module in lora_modules(model).items()}
    payload = ClientPayload(cid, {name: torch.zeros(1, 4) for name in names},
                            {name: torch.zeros(1, 4) for name in names},
                            {name: torch.zeros(4) for name in names}, 0)
    result = coordinate_masks([payload], {cid: 1.}, {cid: 29}, factors, "b",
                              {cid: 0}, "JQ", offsets_by_client={cid:
                              _cyclic_offsets(run_seed, cid, "b", names, 4)},
                              expected_parent_version=0)
    assert all(torch.equal(standalone[name]["b"], result.cyclic_masks[cid][name])
               for name in standalone)


@pytest.mark.parametrize("method,representation", [("JQ", "N"), ("local", "T")])
def test_coordinated_preflight_denial_sends_no_round_two_control(tmp_path, method, representation):
    seed = 53
    job = {"method": method, "representation": representation, "rank": 4,
           "seed": seed, "lr": .001, "period": 1, "min_rounds": 1,
           "max_rounds": 2, "xi": 1.}
    common = {"clients_per_round": 2, "batch_size": 8, "probe_batch_size": 8,
              "probe_batches": 2, "local_epochs": 1, "eval_batch_size": 16}

    def inputs():
        model, _, _, capacities, _ = fixture(seed)
        clients = {cid: TinyDataset(100 + cid, 32) for cid in capacities}
        return model, clients, capacities, TinyDataset(200, 32)

    model, clients, capacities, evaluation = inputs()
    ref = tmp_path / "reference"
    run_experiment(model, clients, evaluation, capacities, job,
                   {**common, "byte_cap": 10**8}, ref,
                   {"evaluation_role": "synthetic"}, stop_after=1)
    first = _load_checkpoint(ref / "latest.pt")
    assert first["round"] == 1
    cohort2 = random.Random(keyed_seed(seed, "participants", 2)).sample(sorted(clients), 2)
    upper2 = preflight_round_upper_bound(model, first["adapter"], cohort2,
                                          {cid: capacities[cid] for cid in cohort2},
                                          round_id=2, method=method,
                                          representation=representation, period=1,
                                          factor="a", cache=first["cache"])
    cap = first["deployed_bytes"] + upper2 - 1
    model2, clients2, capacities2, evaluation2 = inputs()
    denied = tmp_path / "denied"
    run_experiment(model2, clients2, evaluation2, capacities2, job,
                   {**common, "byte_cap": cap}, denied,
                   {"evaluation_role": "synthetic"})
    ck = _load_checkpoint(denied / "latest.pt")
    assert ck["round"] == ck["budget_stop_round"] == 1
    assert ck["deployed_bytes"] == first["deployed_bytes"]
    assert ck["diagnostic_bytes"] == 0
    assert all(event["round"] == 1 for event in ck["cache"]["events"])
    assert all(torch.equal(ck["budget_adapter"][name], first["adapter"][name])
               for name in first["adapter"])


def test_failed_j_broadcast_spend_blocks_retry_and_returns_initial_parent(tmp_path, monkeypatch):
    import pcla_research_v7.cohort_runtime as bridge

    seed = 67
    job = {"method": "JQ", "representation": "N", "rank": 4,
           "seed": seed, "lr": .001, "period": 1, "min_rounds": 1,
           "max_rounds": 2, "xi": 1.}
    config = {"clients_per_round": 2, "batch_size": 8, "probe_batch_size": 8,
              "probe_batches": 2, "local_epochs": 1, "eval_batch_size": 16}

    def inputs():
        model, _, _, capacities, _ = fixture(seed)
        clients = {cid: TinyDataset(100 + cid, 32) for cid in capacities}
        return model, clients, capacities, TinyDataset(200, 32)

    model, clients, capacities, evaluation = inputs()
    initial = state(model)
    cohort = random.Random(keyed_seed(seed, "participants", 1)).sample(sorted(clients), 2)
    cap = preflight_round_upper_bound(model, initial, cohort,
        {cid: capacities[cid] for cid in cohort}, round_id=1, method="JQ",
        representation="N", period=1, factor="b")
    real_probe = bridge.probe_full_gradients
    def fail_probe(*args, **kwargs):
        raise RuntimeError("injected failed J probe")
    monkeypatch.setattr(bridge, "probe_full_gradients", fail_probe)
    with pytest.raises(RuntimeError, match="injected failed J probe"):
        run_experiment(model, clients, evaluation, capacities, job,
                       {**config, "byte_cap": cap}, tmp_path,
                       {"evaluation_role": "synthetic"})
    failed = json.loads((tmp_path / "communication_journal.json").read_text())["entries"][0]
    sunk = sum(event["bytes"] for event in failed["sent_events"])
    assert sunk > 0

    monkeypatch.setattr(bridge, "probe_full_gradients", real_probe)
    model2, clients2, capacities2, evaluation2 = inputs()
    run_experiment(model2, clients2, evaluation2, capacities2, job,
                   {**config, "byte_cap": cap}, tmp_path,
                   {"evaluation_role": "synthetic"})
    ck = _load_checkpoint(tmp_path / "latest.pt")
    budget = torch.load(tmp_path / "budget_return.pt", weights_only=True)
    assert ck["budget_stop_round"] == budget["round"] == 0
    assert ck["deployed_bytes"] == budget["deployed_bytes"] == sunk
    assert ck["diagnostic_bytes"] > 0
    assert all(torch.equal(budget["adapter"][name], initial[name]) for name in initial)
    events = ck["cache"]["events"]
    assert len({event["event_id"] for event in events}) == len(events)
    assert sum(event["bytes"] for event in events if event["bucket"] == "deployed_prefix") == sunk
