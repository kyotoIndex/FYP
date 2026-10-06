import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pcla_repair.core import BudgetLoRALinear
from pcla_research_v7.diagnostic_configuration import participants
from pcla_research_v7.runtime import aggregate, state
from pcla_research_v8.diagnostic_inputs import _bank_updates, _role_rows, collect_rank8


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parents[1]
INDEX = ROOT / "checks/diagnostic_intake01/INPUT_INDEX.json"
OLD_ROLES = ROOT.parents[1] / "runs/pcla-global-budget-20260928-v7/data_manifest/train_roles.json"


def _tiny_state(value):
    return {"query.lora_a": torch.full((2, 3), value, dtype=torch.float32),
            "query.lora_b": torch.full((3, 2), value, dtype=torch.float32),
            "classifier.weight": torch.full((2, 3), value, dtype=torch.float32),
            "classifier.bias": torch.full((2,), value, dtype=torch.float32)}


def test_real_old_seed_schedule_and_saved_torch_bank_schema(tmp_path):
    index = json.loads(INDEX.read_text())
    manifest = json.loads(OLD_ROLES.read_text())
    record = next(x for x in index["cohort_bank_objects"] if x["object_id"] == "Cohort_s101_full_alt_r10_a")
    ids = sorted(int(cid) for cid in manifest["partitions"]["101"]["client_train"])
    assert participants(101, 11, ids) == record["participants"]
    assert len(ids) == 50
    selected = record["participants"]
    parent = _tiny_state(0.0)
    children = [_tiny_state(float(i + 1)) for i in range(10)]
    proposal = aggregate(parent, children, [1] * 10)
    bank = {"schema": "pcla-v7-cohort-bank-1", "object_id": record["object_id"],
            "factor": "a", "participants": selected,
            "anchor_sha256": record["parent_checkpoint_sha256_expected"],
            "aggregate_states": {s: {16: proposal} for s in ("local", "cyclic")},
            "prefix_states": {s: {cid: {16: children[i]} for i, cid in enumerate(selected)}
                              for s in ("local", "cyclic")}}
    path = tmp_path / "bank.pt"
    torch.save(bank, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    config = {"source": {"object_id": record["object_id"], "factor": "a",
                         "participants": selected,
                         "parent_checkpoint_sha256_historical_expected": record["parent_checkpoint_sha256_expected"]}}
    updates = _bank_updates(loaded, parent, config)
    assert [x["support"] for x in updates] == ["local", "cyclic"]
    assert all(len(x["children"]) == 10 and x["weights"] == [0.1] * 10 and
               x["children_are_raw_unweighted"] for x in updates)
    loaded["aggregate_states"]["local"][16]["classifier.bias"][0] += 1
    with pytest.raises(ValueError, match="saved proposal differs"):
        _bank_updates(loaded, parent, config)


def test_roles_from_real_manifest_are_separate():
    manifest = json.loads(OLD_ROLES.read_text())
    public = set(_role_rows(manifest["roles"]["public_training"]))
    readout = set(_role_rows(manifest["roles"]["cohort_readout"]))
    local = set(_role_rows(manifest["partitions"]["101"]["client_local_readout"]["47"]))
    assert (len(public), len(readout), len(local)) == (512, 2048, 64)
    assert not (public & readout or public & local or readout & local)
    with pytest.raises(ValueError, match="invalid train source"):
        _role_rows(["dev:1"])


class TinyRank8(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList(
            BudgetLoRALinear(nn.Linear(3, 3), 8, 8, 0.1) for _ in range(24))
        self.classifier = nn.Linear(3, 2)

    def forward(self, input_ids, labels):
        x = input_ids.float()
        for module in self.projections:
            x = module(x)
        logits = self.classifier(x)
        return SimpleNamespace(logits=logits, loss=nn.functional.cross_entropy(logits, labels))


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 256

    def __getitem__(self, index):
        return {"input_ids": torch.tensor([float(index % 3), 0.5, -0.25]),
                "labels": torch.tensor(index % 2)}


def test_rank8_real_torch_local_train_closure_and_restore():
    model = TinyRank8()
    parent = state(model)
    selected = list(range(10))
    context = {"model": model, "parent_state": parent, "participants": selected,
               "aggregate_updates": [],
               "client_train_by_client": {cid: TinyDataset() for cid in selected}}
    config = {"job": {"phase": "R8_TRANSFER", "new_client_training_calls": 10,
                      "new_client_steps": 160, "seed": 101, "anchor_round": 10},
              "model": {"rank": 8},
              "source": {"training": {"lr": 0.001, "head_lr": 0.001,
                                      "weight_decay": 0.01}}}
    result = collect_rank8(context, config)
    assert result["local_calls"] == 10 and result["optimizer_steps"] == 160
    assert len(result["client_details"]) == 10
    assert all(len(x["actual_child_state_sha256"]) == 64 for x in result["client_details"])
    assert result["aggregate_updates"][0]["weights"] == [0.1] * 10
    assert all(torch.equal(value, state(model)[name]) for name, value in parent.items())
