import json
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pcla_repair.core import BudgetLoRALinear
from pcla_research_v7.native_fedsb import FedSBLinear
from pcla_research_v7.native_runtime import (NativeWire, local_train_native, native_state,
                                              run_native, evaluate_ema_native, initial_ema_native,
                                              initialize_fedsb)


class Data(torch.utils.data.Dataset):
    def __init__(self, n=8): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return {"input_ids": torch.tensor([float(i % 3), float((i + 1) % 4), 1.]),
                "labels": torch.tensor(i % 2)}


class Tiny(nn.Module):
    def __init__(self, method):
        super().__init__()
        base = nn.Linear(3, 3)
        if method == "fedsb":
            self.q = FedSBLinear(base, torch.randn(2, 3), torch.randn(3, 2),
                                 torch.eye(2), dropout=0.)
        else:
            self.q = BudgetLoRALinear(base, 2, 2., 0.)
            with torch.no_grad(): self.q.lora_b.zero_()
        self.classifier = nn.Linear(3, 2)
    def forward(self, input_ids, labels):
        logits = self.classifier(torch.tanh(self.q(input_ids)))
        return type("Output", (), {"logits": logits, "loss": F.cross_entropy(logits, labels)})()


def setup(method):
    torch.manual_seed(50)
    model = Tiny(method)
    clients = {i: Data() for i in range(4)}
    capacities = {0: 96, 1: 128, 2: 192, 3: 96}
    job = {"method": method, "seed": 401, "rank": 2, "period": 1,
           "min_rounds": 1, "max_rounds": 2}
    recipe = {"method": method, "required_adapter": method,
              "rank": 2, "native_ranks": [1, 1, 2], "period": 1,
              "lr": .01, "head_lr": .01, "optimizer": "sgd" if method == "fedsvd" else "adamw",
              "adapter_weight_decay": 0., "adapter_clip_norm": 2. if method == "fedsvd" else None}
    runtime = {"byte_cap": 10000000, "clients_per_round": 2,
               "batch_size": 4, "eval_batch_size": 4}
    return model, clients, capacities, job, recipe, runtime


@pytest.mark.parametrize("method", ["fedsb", "fedsvd", "fedplora"])
def test_native_wire_real_packet_and_full_resume(method, tmp_path):
    model, clients, capacities, job, recipe, runtime = setup(method)
    parent = native_state(model, method)
    selected = {0: {"q": (0,)}}
    wire = NativeWire()
    preview = wire.preview(method, 1, [0], parent, selected, {})
    child = {k: v.clone() for k, v in parent.items()}
    for name in child:
        if name.startswith("classifier."): child[name] += .1
    committed = wire.commit(method, preview, {}, parent, {0: child}, selected)
    assert sum(x["bytes"] for x in committed["events"]) == preview["bytes"]
    assert len(committed["events"]) == 2
    provenance = {"evaluation_role": "synthetic", "hardware": {"name": "cpu"},
                  "native_initialization": {"public_rows": 512,
                                            "cost": {"total_wall_seconds": 1.}}}
    first = tmp_path / "continuous"
    rows = run_native(model, clients, Data(), capacities, job, recipe, runtime,
                      first, provenance)
    assert len(rows) == 2
    assert rows[-1]["deployed_bytes"] <= runtime["byte_cap"]
    saved = torch.load(first / "latest.pt", weights_only=True)
    model2, clients2, capacities2, job2, recipe2, runtime2 = setup(method)
    second = tmp_path / "resume"
    run_native(model2, clients2, Data(), capacities2, job2, recipe2, runtime2,
               second, provenance, stop_after=1)
    model3, clients3, capacities3, job3, recipe3, runtime3 = setup(method)
    provenance["native_initialization"]["cost"]["total_wall_seconds"] = 999.
    resumed = run_native(model3, clients3, Data(), capacities3, job3, recipe3, runtime3,
                         second, provenance)
    saved2 = torch.load(second / "latest.pt", weights_only=True)
    assert len(resumed) == 2
    assert saved["identity"] == saved2["identity"]
    assert saved["deployed_bytes"] == saved2["deployed_bytes"]
    for name in saved["adapter"]:
        assert torch.equal(saved["adapter"][name], saved2["adapter"][name])
    assert [x["evaluation"] for x in rows] == [x["evaluation"] for x in resumed]


@pytest.mark.parametrize("method", ["fedsb", "fedsvd", "fedplora"])
def test_native_ema_restores_base_and_head(method):
    model, *_ = setup(method)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema = initial_ema_native(model, method, native_state(model, method))
    assert torch.count_nonzero(ema["q"]) == 0
    assert torch.equal(ema["classifier.weight"], original["classifier.weight"].double())
    output = evaluate_ema_native(model, method, ema, Data(), 4)
    assert output["samples"] == 8
    assert all(torch.equal(p, original[n]) for n, p in model.named_parameters())


def test_fedsb_rank32_first_download_and_r_only_upload():
    torch.manual_seed(41)
    parent = {"bert.encoder.layer.0.attention.self.query.a": torch.randn(32, 64),
              "bert.encoder.layer.0.attention.self.query.b": torch.randn(64, 32),
              "bert.encoder.layer.0.attention.self.query.r": torch.eye(32),
              "classifier.weight": torch.randn(2, 64),
              "classifier.bias": torch.zeros(2)}
    module = "bert.encoder.layer.0.attention.self.query"
    selections = {4: {module: tuple(range(32))}}
    wire = NativeWire()
    first = wire.preview("fedsb", 1, [4], parent, selections, {})
    child = {k: v.clone() for k, v in parent.items()}
    child[module + ".r"][0, 0] += 1
    after = wire.commit("fedsb", first, {}, parent, {4: child}, selections)
    assert set(first["uploads"][4]["specs"]) == {module + ".r", "classifier.weight", "classifier.bias"}
    second = wire.preview("fedsb", 2, [4], parent, selections, after)
    assert second["bytes"] < first["bytes"]


def test_fedsvd_refresh_continues_in_budget_diagnostic_with_fixed_probe(monkeypatch, tmp_path):
    from pcla_research_v7 import native_runtime as runtime
    original = runtime.refresh_state
    probe_sizes = []
    def checked(state, round_id, period, probe=None, **kwargs):
        probe_sizes.append(probe(state).shape[0])
        return original(state, round_id, period, probe=probe, **kwargs)
    monkeypatch.setattr(runtime, "refresh_state", checked)
    model, clients, capacities, job, recipe, config = setup("fedsvd")
    clients = {cid: Data(40) for cid in clients}
    job.update(min_rounds=2, max_rounds=3)
    config["byte_cap"] = 2400  # round 1 fits; round 2 enters diagnostic bucket.
    rows = run_native(model, clients, Data(), capacities, job, recipe, config,
                      tmp_path, {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}})
    assert len(rows) == 2 and rows[1]["coordinate_refresh"]["refreshed"]
    assert probe_sizes == [32, 32] and 0 not in rows[1]["selected_clients"]
    assert rows[1]["deployed_bytes"] == rows[0]["deployed_bytes"]
    assert rows[1]["diagnostic_after_budget_bytes"] > 0
    assert rows[1]["budget_stop_round"] == 1


def test_failed_message_attempt_is_charged_separately_and_resume_exact(monkeypatch, tmp_path):
    from pcla_research_v7 import native_runtime as runtime
    model, clients, capacities, job, recipe, config = setup("fedplora")
    provenance = {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}}
    original = runtime.local_train_native
    called = 0
    def fail_second(*args, **kwargs):
        nonlocal called
        called += 1
        if called == 2:
            raise RuntimeError("injected client failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, "local_train_native", fail_second)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected client failure"):
        run_native(model, clients, Data(), capacities, job, recipe, config, out, provenance)
    ck = torch.load(out / "latest.pt", weights_only=True)
    assert ck["round"] == 0 and ck["deployed_bytes"] == 0
    journal = json.loads((out / "wire_attempt_journal.json").read_text())
    failed_bytes = sum(event["bytes"] for event in journal["events"])
    assert [event["direction"] for event in journal["events"]].count("download") == 2
    assert [event["direction"] for event in journal["events"]].count("upload") == 1
    monkeypatch.setattr(runtime, "local_train_native", original)
    resumed_model, resumed_clients, resumed_capacities, resumed_job, resumed_recipe, resumed_cfg = setup("fedplora")
    resumed = run_native(resumed_model, resumed_clients, Data(), resumed_capacities,
                         resumed_job, resumed_recipe, resumed_cfg, out, provenance)
    status = json.loads((out / "status.json").read_text())
    assert status["failed_attempt_wire_bytes"] == failed_bytes
    assert status["actual_attempted_wire_bytes"] == (
        status["deployed_bytes"] + status["diagnostic_after_budget_bytes"] + failed_bytes)
    fresh_model, fresh_clients, fresh_capacities, fresh_job, fresh_recipe, fresh_cfg = setup("fedplora")
    clean = run_native(fresh_model, fresh_clients, Data(), fresh_capacities,
                       fresh_job, fresh_recipe, fresh_cfg, tmp_path / "clean", provenance)
    assert [r["evaluation"] for r in resumed] == [r["evaluation"] for r in clean]
    resumed_ck = torch.load(out / "latest.pt", weights_only=True)
    clean_ck = torch.load(tmp_path / "clean/latest.pt", weights_only=True)
    assert all(torch.equal(resumed_ck["adapter"][n], clean_ck["adapter"][n])
               for n in clean_ck["adapter"])


def test_checkpoint_write_failure_preserves_latest(monkeypatch, tmp_path):
    from pcla_research_v7 import native_runtime as runtime
    runtime._checkpoint(tmp_path, {"format": "old"})
    def fail(*args, **kwargs): raise OSError("injected write failure")
    monkeypatch.setattr(runtime, "atomic_checkpoint", fail)
    with pytest.raises(OSError, match="injected"):
        runtime._checkpoint(tmp_path, {"format": "new"})
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["format"] == "old"
    assert torch.load(tmp_path / "previous.pt", weights_only=True)["format"] == "old"


def test_resume_rejects_source_and_committed_journal_changes(tmp_path):
    model, clients, capacities, job, recipe, config = setup("fedsb")
    provenance = {"evaluation_role": "synthetic", "hardware": {"name": "cpu"},
                  "native_sources": {"native_runtime.py": "source-a"}}
    run_native(model, clients, Data(), capacities, job, recipe, config,
               tmp_path, provenance, stop_after=1)
    fresh, clients2, capacities2, job2, recipe2, config2 = setup("fedsb")
    changed = {**provenance, "native_sources": {"native_runtime.py": "source-b"}}
    with pytest.raises(ValueError, match="resume identity"):
        run_native(fresh, clients2, Data(), capacities2, job2, recipe2, config2,
                   tmp_path, changed)
    journal_path = tmp_path / "wire_attempt_journal.json"
    journal = json.loads(journal_path.read_text())
    journal["events"][0]["bytes"] += 1
    journal_path.write_text(json.dumps(journal))
    fresh, clients2, capacities2, job2, recipe2, config2 = setup("fedsb")
    with pytest.raises(ValueError, match="committed journal prefix"):
        run_native(fresh, clients2, Data(), capacities2, job2, recipe2, config2,
                   tmp_path, provenance)


def test_fedsb_initializer_records_public_gradient_svd_and_memory():
    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(3, 3)
            self.value = nn.Linear(3, 3)
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.self = Pair()
    class ToyBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = nn.Module()
            self.bert.encoder = nn.Module()
            self.bert.encoder.layer = nn.ModuleList([Layer() for _ in range(12)])
            self.classifier = nn.Linear(3, 2)
        def forward(self, input_ids, labels):
            x = input_ids
            for layer in self.bert.encoder.layer:
                x = torch.tanh(layer.attention.self.query(x) + layer.attention.self.value(x))
            logits = self.classifier(x)
            return type("Output", (), {"logits": logits,
                                         "loss": F.cross_entropy(logits, labels)})()
    class Public(Data):
        role = "public_training"
        actor = "algorithm"
    torch.manual_seed(12)
    model = ToyBert()
    audit = initialize_fedsb(model, Public(512), 2, .0003)
    assert audit["public_rows"] == 512 and len(audit["projections"]) == 24
    assert audit["effective_lr"] == .0003 / 16
    assert audit["cost"]["public_gradient_seconds"] > 0
    assert audit["cost"]["svd_and_install_seconds"] > 0
    assert audit["cost"]["cpu_peak_rss_after_kb"] >= audit["cost"]["cpu_peak_rss_before_kb"]
