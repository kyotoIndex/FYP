from pathlib import Path

import pytest
import torch

from pcla_repair.core import lora_modules
from pcla_repair.testing import fixture, TinyDataset
from pcla_research_v7.runtime import (
    attach_slices, release_slices, full_support, local_train, run_experiment,
    select_support, state, _load_checkpoint, RawTensorWire,
    initial_ema, advance_ema, evaluate_ema,
)
from pcla_research_v7.transport import bert_adapter_specs, bert_head_specs, b8_wire_manifest
from pcla_research_v7.cohort_runtime import prepare_cohort
from pcla_repair.core import keyed_seed


def _job(method="full_both"):
    return {"method": method, "representation": "N", "rank": 4,
            "seed": 123, "lr": .01, "min_rounds": 2, "max_rounds": 4}


def _config(cap=10**8):
    return {"byte_cap": cap, "clients_per_round": 2, "batch_size": 8,
            "eval_batch_size": 16, "local_epochs": 1}


def _same_state(a, b):
    return all(torch.equal(a[n], b[n]) for n in a)


def test_compact_forward_and_gradients_match_dense_mask():
    torch.manual_seed(4)
    model, _, _, _, _ = fixture(4)
    module = lora_modules(model)["blocks.0.query"]
    module.dropout.p = 0.
    with torch.no_grad():
        module.lora_b.normal_()
    original_w0 = module.base.weight.detach().clone()
    support = {name: {"a": torch.zeros(m.rank, dtype=torch.bool),
                      "b": torch.zeros(m.rank, dtype=torch.bool)}
               for name, m in lora_modules(model).items()}
    support["blocks.0.query"]["a"][0] = True
    support["blocks.0.query"]["b"][2] = True
    x = torch.randn(3, 6, requires_grad=True)
    dense_x = x.detach().clone().requires_grad_()
    a0, b0 = module.lora_a.detach().clone(), module.lora_b.detach().clone()
    dense_a = a0.clone().requires_grad_()
    dense_b = b0.clone().requires_grad_()
    dense_y = module.base(dense_x) + torch.nn.functional.linear(
        torch.nn.functional.linear(dense_x, dense_a), dense_b)
    dense_y.square().sum().backward()
    attach_slices(model, support, train_head=False)
    compact_y = module(x)
    compact_y.square().sum().backward()
    assert torch.allclose(compact_y, dense_y, atol=1e-6)
    assert torch.allclose(x.grad, dense_x.grad, atol=1e-6)
    assert torch.allclose(module.active_a.grad[0], dense_a.grad[0], atol=1e-6)
    assert torch.allclose(module.active_b.grad[0], dense_b.grad[:, 2], atol=1e-6)
    opt = torch.optim.AdamW([module.active_a, module.active_b], lr=.01,
                            weight_decay=.01, foreach=False)
    opt.step()
    dense_opt = torch.optim.AdamW([dense_a, dense_b], lr=.01,
                                  weight_decay=.01, foreach=False)
    dense_a.grad[1:] = 0
    dense_b.grad[:, [0, 1, 3]] = 0
    dense_opt.step()
    release_slices(model)
    assert torch.allclose(module.lora_a[0], dense_a[0], atol=1e-6)
    assert torch.allclose(module.lora_b[:, 2], dense_b[:, 2], atol=1e-6)
    assert torch.equal(module.lora_a[1:], a0[1:])
    assert torch.equal(module.lora_b[:, [0, 1, 3]], b0[:, [0, 1, 3]])
    assert torch.equal(module.base.weight, original_w0)


def test_independent_qv_and_fresh_optimizer():
    model, _, clients, _, _ = fixture(5)
    supports = full_support(model, "b")
    supports["blocks.0.query"]["b"][:] = False
    supports["blocks.0.value"]["b"][:] = False
    supports["blocks.0.query"]["b"][0] = True
    supports["blocks.0.value"]["b"][1] = True
    before = state(model)
    base = {n: m.base.weight.detach().clone() for n, m in lora_modules(model).items()}
    result = local_train(model, clients[0], supports, seed=16, lr=.01, batch_size=8)
    after = state(model)
    assert result["optimizer_steps"] == 3
    assert result["optimizer_tensor_parameters"] < sum(x.numel() for x in before.values())
    assert torch.equal(after["blocks.0.query.lora_b"][:, 1:], before["blocks.0.query.lora_b"][:, 1:])
    assert torch.equal(after["blocks.0.value.lora_b"][:, [0, 2, 3]], before["blocks.0.value.lora_b"][:, [0, 2, 3]])
    assert not torch.equal(after["blocks.0.query.lora_b"][:, 0], before["blocks.0.query.lora_b"][:, 0])
    assert all(torch.equal(m.base.weight, base[n]) for n, m in lora_modules(model).items())
    # Repeating from the same parent and keyed data/dropout must reproduce the
    # update: no first/second moment survives an earlier local call.
    from pcla_research_v7.runtime import load_state
    load_state(model, before)
    local_train(model, clients[0], supports, seed=16, lr=.01, batch_size=8)
    assert _same_state(state(model), after)


class FixedWire:
    def preview(self, **kwargs):
        return {"bytes": 100}

    def commit(self, preview, cache, parent, children, cohort, supports, *, diagnostic):
        return {"events": cache.get("events", []) + [(tuple(cohort), diagnostic)]}


def test_budget_endpoint_and_diagnostic_continuation(tmp_path):
    model, _, clients, capacities, evaluation = fixture(7)
    run_experiment(model, clients, evaluation, capacities, _job(), _config(250),
                   tmp_path, {"evaluation_role": "synthetic"}, FixedWire())
    ck = _load_checkpoint(tmp_path / "latest.pt")
    assert ck["round"] == 2
    assert ck["budget_stop_round"] == 2
    assert ck["deployed_bytes"] == 200
    assert ck["diagnostic_bytes"] == 0
    half = torch.load(tmp_path / "half_budget_return.pt", map_location="cpu", weights_only=True)
    assert half["round"] == 1 and half["deployed_bytes"] == 100
    assert half["readout"]["samples"] == 32
    assert ck["metrics"][-1]["ema_evaluation"]["samples"] == 32
    # A cap crossed before the mandatory round-100 analogue continues in the
    # diagnostic bucket, while the return object stays at the prior model.
    model2, _, clients2, capacities2, evaluation2 = fixture(7)
    job = _job(); job["min_rounds"] = 3
    other = tmp_path / "diagnostic"
    run_experiment(model2, clients2, evaluation2, capacities2, job, _config(250),
                   other, {"evaluation_role": "synthetic"}, FixedWire())
    diag = _load_checkpoint(other / "latest.pt")
    assert diag["round"] == 3
    assert diag["budget_stop_round"] == 2
    assert diag["deployed_bytes"] == 200
    assert diag["diagnostic_bytes"] == 100
    assert _same_state(diag["budget_adapter"], ck["adapter"])
    assert (other / "logits_round_003.json").exists()


def test_resume_matches_continuous_and_benchmark_role_is_rejected(tmp_path):
    model, _, clients, capacities, evaluation = fixture(9)
    with pytest.raises(ValueError, match="train-only"):
        run_experiment(model, clients, evaluation, capacities, _job(), _config(),
                       tmp_path / "forbidden", {"evaluation_role": "benchmark_dev"}, FixedWire())
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    run_experiment(model, clients, evaluation, capacities, _job(), _config(),
                   full_dir, {"evaluation_role": "synthetic"}, FixedWire())
    first = _load_checkpoint(full_dir / "latest.pt")
    model2, _, clients2, capacities2, evaluation2 = fixture(9)
    run_experiment(model2, clients2, evaluation2, capacities2, _job(), _config(),
                   resumed_dir, {"evaluation_role": "synthetic"}, FixedWire(), stop_after=2)
    model3, _, clients3, capacities3, evaluation3 = fixture(9)
    run_experiment(model3, clients3, evaluation3, capacities3, _job(), _config(),
                   resumed_dir, {"evaluation_role": "synthetic"}, FixedWire())
    second = _load_checkpoint(resumed_dir / "latest.pt")
    assert _same_state(first["adapter"], second["adapter"])
    assert first["deployed_bytes"] == second["deployed_bytes"]
    for a, b in zip(first["metrics"], second["metrics"]):
        for field in ("round", "selected_clients", "deployed_bytes",
                      "diagnostic_after_budget_bytes", "budget_stop_round", "evaluation"):
            assert a[field] == b[field]
        for ca, cb in zip(a["client_training"], b["client_training"]):
            for field in ("client", "optimizer_steps", "optimizer_tensor_parameters"):
                assert ca[field] == cb[field]


def test_selector_exact_support_and_unsupported_mode(tmp_path):
    model, _, clients, capacities, evaluation = fixture(11)
    local = select_support(model, clients[0], "local", "b", 3, seed=4, visit=0,
                           batch_size=8, probe_batches=2)
    assert sum(int(v["b"].sum()) for v in local.values()) == 3
    assert sum(int(v["a"].sum()) for v in local.values()) == 0
    both = select_support(model, clients[0], "cyclic_both_half", "b", 4,
                          seed=4, visit=0)
    assert sum(int(v["a"].sum()) + int(v["b"].sum()) for v in both.values()) == 4
    with pytest.raises(NotImplementedError):
        run_experiment(model, clients, evaluation, capacities,
                       {**_job(), "representation": "T"}, _config(),
                       tmp_path, {"evaluation_role": "synthetic"}, FixedWire())


def test_rank8_full_both_packet_matches_frozen_b8_schema():
    specs = {**bert_adapter_specs(8), **bert_head_specs()}
    parent = {n: torch.zeros(s.shape) for n, s in specs.items()}
    modules = sorted({n.rsplit(".", 1)[0] for n in specs if n.endswith("lora_a")})
    support = {n: {"a": torch.ones(8, dtype=torch.bool),
                   "b": torch.ones(8, dtype=torch.bool)} for n in modules}
    wire = RawTensorWire(force_full_download=True)
    plan = wire.preview(round_id=1, cohort=[0], parent=parent,
                        supports={0: support}, cache={})
    reference = b8_wire_manifest(physically_encode=False)
    # First round has clients 0..9, all one-digit IDs and equal packet sizes.
    assert plan["bytes"] * 10 == reference["rounds"][0]["total_bytes"]
    last = wire.preview(round_id=100, cohort=[40], parent=parent,
                        supports={40: support}, cache={"clients": {"40": {"digests": {}}}})
    assert last["bytes"] * 10 == reference["rounds"][99]["total_bytes"]


def test_effective_matrix_ema_is_dense_and_restores_base():
    model, _, _, _, evaluation = fixture(17)
    first = state(model)
    ema = initial_ema(model, first)
    module = lora_modules(model)["blocks.0.query"]
    w0 = module.base.weight.detach().clone()
    changed = {n: x.clone() for n, x in first.items()}
    changed["blocks.0.query.lora_b"][:, 0] = 1.
    changed["blocks.0.query.lora_a"][0] = 2.
    ema = advance_ema(model, ema, changed)
    expected = .2 * (changed["blocks.0.query.lora_b"].double() @
                     changed["blocks.0.query.lora_a"].double())
    assert torch.allclose(ema["blocks.0.query"], expected)
    assert evaluate_ema(model, ema, evaluation)["samples"] == 32
    assert torch.equal(module.base.weight, w0)


@pytest.mark.parametrize("method,representation", [("local", "T"), ("JL", "N"), ("JQ", "N")])
def test_real_cohort_control_and_wire_ledger(tmp_path, method, representation):
    model, _, clients, capacities, evaluation = fixture(19)
    clients = {cid: TinyDataset(100 + cid, 32) for cid in clients}
    job = {**_job(method), "representation": representation,
           "period": 1, "max_rounds": 2, "xi": 1.}
    rows = run_experiment(model, clients, evaluation, capacities, job,
                          {**_config(), "probe_batch_size": 8, "probe_batches": 2},
                          tmp_path / method, {"evaluation_role": "synthetic"})
    ck = _load_checkpoint(tmp_path / method / "latest.pt")
    events = ck["cache"]["events"]
    assert len({e["event_id"] for e in events}) == len(events)
    assert sum(e["bytes"] for e in events if e["bucket"] == "deployed_prefix") == ck["deployed_bytes"]
    assert rows[-1]["coordinate_refresh"]["phase"] in {"T_refresh", "JL_coordination", "JQ_coordination"}


def test_resume_same_hardware_family_new_uuid_and_reject_other_family(tmp_path, monkeypatch):
    model, _, clients, capacities, evaluation = fixture(23)
    first = {"evaluation_role": "synthetic", "hardware":
             {"name": "GPU family A", "total_memory": 24_000,
              "uuid": "GPU-1", "cuda_visible_devices": "0"}}
    monkeypatch.setenv("PCLA_GPU_SHARING", "exclusive")
    run_experiment(model, clients, evaluation, capacities, _job(), _config(),
                   tmp_path, first, FixedWire(), stop_after=2)
    model2, _, clients2, capacities2, evaluation2 = fixture(23)
    other_uuid = {**first, "hardware": {**first["hardware"],
                                      "uuid": "GPU-2", "cuda_visible_devices": "3"}}
    monkeypatch.setenv("PCLA_GPU_SHARING", "shared2")
    run_experiment(model2, clients2, evaluation2, capacities2, _job(), _config(),
                   tmp_path, other_uuid, FixedWire())
    attempts = __import__("json").loads((tmp_path / "execution_attempts.json").read_text())
    assert [x["hardware"]["uuid"] for x in attempts] == ["GPU-1", "GPU-2"]
    assert [x["gpu_sharing"] for x in attempts] == ["exclusive", "shared2"]
    model3, _, clients3, capacities3, evaluation3 = fixture(23)
    other_family = {**first, "hardware": {**first["hardware"], "name": "GPU family B"}}
    with pytest.raises(ValueError, match="resume identity changed"):
        run_experiment(model3, clients3, evaluation3, capacities3, _job(), _config(),
                       tmp_path, other_family, FixedWire())


@pytest.mark.parametrize("method", ["local", "cyclic"])
def test_simple_selector_matches_cohort_bridge(method):
    model, _, clients, capacities, _ = fixture(31)
    cid, round_id, seed = 0, 3, 31
    own = select_support(model, clients[cid], method, "b", capacities[cid],
        seed=keyed_seed(seed, "selection", cid), visit=1, batch_size=8,
        probe_seeds={f: keyed_seed(seed, "v7_probe", round_id, cid, f)
                     for f in ("a", "b")})
    plan = prepare_cohort(model, state(model), clients, [cid],
        {cid: capacities[cid]}, {cid: 1}, round_id=round_id, seed=seed,
        factor="b", method=method, representation="N", period=1,
        learning_rate=.01, batch_size=8)
    assert all(torch.equal(own[name][factor], plan.supports[cid][name][factor])
               for name in own for factor in ("a", "b"))


@pytest.mark.parametrize("method,representation", [
    ("tied_local", "N"), ("layer", "N"), ("layer", "S"),
    ("energy", "S"), ("equal_quota_cyclic", "N")])
def test_simple_control_methods_record_exact_support(tmp_path, method, representation):
    model, _, clients, capacities, evaluation = fixture(37)
    if method == "layer":
        capacities = {cid: 8 for cid in capacities}
    job = {**_job(method), "representation": representation,
           "period": 1 if representation == "S" else 0, "max_rounds": 2}
    rows = run_experiment(model, clients, evaluation, capacities, job,
                          {**_config(), "probe_batch_size": 8},
                          tmp_path / method, {"evaluation_role": "synthetic"})
    for row in rows:
        for record in row["support_trace"].values():
            actual = sum(len(ids) for module in record["modules"].values()
                         for ids in module.values())
            assert actual == record["selected_positions"]


@pytest.mark.parametrize("representation", ["D", "S"])
def test_coordinate_refresh_uses_train_rows_and_preserves_logits(tmp_path, representation):
    model, _, clients, capacities, evaluation = fixture(13)
    job = {**_job("local"), "representation": representation,
           "period": 1, "max_rounds": 2}
    rows = run_experiment(model, clients, evaluation, capacities, job,
                          _config(), tmp_path / representation,
                          {"evaluation_role": "synthetic"}, FixedWire())
    assert rows[0]["coordinate_refresh"] is None
    assert rows[1]["coordinate_refresh"]["logits_max_absolute_error"] < 1e-4
