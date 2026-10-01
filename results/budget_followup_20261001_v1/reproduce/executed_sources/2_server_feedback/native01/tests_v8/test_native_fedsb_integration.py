"""Native Fed-SB contract and R-only wire checks without BERT downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pcla_repair.testing import TinyDataset
from pcla_research_v7.native_fedsb import FedSBLinear, aggregate_r_head
from pcla_research_v7.native_runtime import NativeWire, local_train_native, native_state
from pcla_research_v8.native_fedsb_runtime import native_contract, _save_mean_records
from pcla_research_v7.execution_timing import ExecutionTimingLedger
from pcla_research_v7.native_runtime import NATIVE_TIMING_LABELS
from pcla_research_v8 import native_fedsb_runtime as bridge


CONFIG = (Path(__file__).resolve().parents[3] / "checks/configuration01/configs/DEV_U005_s503.json")


def test_u005_u006_native_contract_and_reject_full_both():
    cfg = json.loads(CONFIG.read_text())
    job, recipe = native_contract(cfg)
    assert job["protocol"] == "pcla-global-budget-20260928-v7"
    assert job["v8_protocol"] == cfg["job"]["protocol"]
    assert recipe["required_adapter"] == "fedsb"
    assert recipe["optimizer"] == "adamw" and recipe["factor_schedule"] == "R_only"
    assert recipe["rank"] == 32 and recipe["lr"] == cfg["recipe"]["client_lr"]
    cfg["recipe"]["factor_schedule"] = "both"
    with pytest.raises(ValueError):
        native_contract(cfg)


class SmallFedSB(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Linear(6, 6)
        self.proj = FedSBLinear(base, torch.randn(2, 6), torch.randn(6, 2),
                                torch.eye(2), scaling=1., dropout=.1)
        self.classifier = nn.Linear(6, 2)

    def forward(self, x, labels):
        logits = self.classifier(torch.tanh(self.proj(x)))
        return SimpleNamespace(logits=logits, loss=nn.functional.cross_entropy(logits, labels))


def test_actual_native_client_updates_only_r_head_and_wire_roundtrip():
    torch.manual_seed(12)
    model = SmallFedSB()
    parent = native_state(model, "fedsb")
    recipe = {"lr": .001, "head_lr": .001, "optimizer": "adamw",
              "adapter_weight_decay": .01, "adapter_clip_norm": None}
    selected = {"proj": (0, 1)}
    child, detail = local_train_native(model, "fedsb", TinyDataset(7, 16), parent,
                                       selected, recipe, seed=13, batch_size=8)
    assert detail["optimizer_steps"] == 2
    assert torch.equal(child["proj.a"], parent["proj.a"])
    assert torch.equal(child["proj.b"], parent["proj.b"])
    assert not torch.equal(child["proj.r"], parent["proj.r"])
    proposal = aggregate_r_head(parent, [child], [16])
    assert all(torch.equal(proposal[k], child[k]) for k in child)
    wire = NativeWire()
    preview = wire.preview("fedsb", 1, [0], parent, {0: selected}, {})
    assert preview["bytes"] > 0
    assert wire.validate_upload("fedsb", preview, 0, parent, child, selected) > 0
    committed = wire.commit("fedsb", preview, {}, parent, {0: child}, {0: selected})
    assert len(committed["events"]) == 2
    assert committed["events"][1]["direction"] == "upload"


def test_mean_controller_record_survives_checkpoint_resume(tmp_path):
    rows = [{"round": 1, "budget_stop_round": None, "evaluation": {"accuracy": .5}}]
    adapter = {"proj.r": torch.eye(2), "classifier.bias": torch.zeros(2)}
    torch.save({"format": "pcla-v7-native-1", "round": 1, "adapter": adapter,
                "metrics": rows}, tmp_path / "latest.pt")
    _save_mean_records(tmp_path, rows, None)
    saved = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=True)
    assert saved["round"] == 1
    assert torch.equal(saved["adapter"]["proj.r"], adapter["proj.r"])
    assert saved["metrics"][0]["server_controller"]["rule"] == "fedsb_r32"
    resumed_rows = saved["metrics"] + [{"round": 2, "budget_stop_round": None}]
    saved["round"] = 2
    saved["metrics"] = resumed_rows
    torch.save(saved, tmp_path / "latest.pt")
    _save_mean_records(tmp_path, resumed_rows, None)
    final = json.loads((tmp_path / "native_controller.json").read_text())
    assert [r["round"] for r in final] == [1, 2]
    assert all(r["alpha"] == r["beta"] == 1 for r in final)


def test_v8_bridge_passes_native_identity_and_persists_resume_metrics(tmp_path, monkeypatch):
    cfg = json.loads(CONFIG.read_text())
    manifest = tmp_path / "roles.json"
    manifest.write_text(json.dumps({"roles": {"public_training": []}}))
    cfg["manifest"] = str(manifest)
    cfg["token_cache"] = str(tmp_path / "tokens.npz")
    cfg["token_cache_metadata"] = str(tmp_path / "tokens.json")
    seen = {}

    def fake_prepare(expanded, _config_path, _device, *, timing):
        assert expanded["job"]["rank"] == 16
        timing.begin_envelope()
        return SmallFedSB(), {}, {}, [], {"evaluation_role": "synthetic", "hardware": {"name": "cpu"}}

    class Public:
        role, actor = "public_training", "algorithm"
        def __len__(self):
            return 512

    monkeypatch.setattr(bridge, "prepare", fake_prepare)
    monkeypatch.setattr(bridge, "TokenCache", lambda *args: SimpleNamespace(dataset=lambda *args: Public()))
    monkeypatch.setattr(bridge, "_install_fedsb", lambda *args: {
        "public_rows": 512, "full_weight_gradient_targets": 24,
        "effective_lr": .0003/16, "initialization_seed": 42,
        "svd_dtype": "float32", "projections": {}, "cost": {"total_wall_seconds": 0.1}})

    def fake_run_native(model, clients, evaluation, capacities, job, recipe, runtime,
                        out, provenance, *, stop_after, timing):
        assert timing.active_labels == ("envelope",)
        assert job["protocol"] == "pcla-global-budget-20260928-v7"
        assert provenance["v8_job"]["protocol"] == cfg["job"]["protocol"]
        assert recipe["required_adapter"] == "fedsb"
        seen["called"] = True
        rows = [{"round": 1, "budget_stop_round": None}]
        torch.save({"format": "pcla-v7-native-1", "metrics": rows,
                    "adapter": native_state(model, "fedsb")}, out / "latest.pt")
        return rows

    monkeypatch.setattr(bridge, "run_native", fake_run_native)
    out = tmp_path / "out"; out.mkdir()
    (out / ".run.lock").write_text("test")
    timing = ExecutionTimingLedger(NATIVE_TIMING_LABELS)
    rows, provenance = bridge.execute_native_fedsb(cfg, tmp_path / "config.json", out,
        torch.device("cpu"), timing=timing)
    timing.end_envelope()
    assert seen["called"] and rows[0]["server_controller"]["alpha"] == 1
    assert provenance["native_initial_state_hash"]
    assert json.loads((out / "run_identity.json").read_text())["job"]["recipe_id"] == "U005"
