from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pcla_repair.core import BudgetLoRALinear
from pcla_research_v7.native_fedsb import FedSBLinear
from pcla_research_v7.native_runtime import (NativeWire, local_train_native, native_state,
                                              run_native, evaluate_ema_native, initial_ema_native)


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
    provenance = {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}}
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
