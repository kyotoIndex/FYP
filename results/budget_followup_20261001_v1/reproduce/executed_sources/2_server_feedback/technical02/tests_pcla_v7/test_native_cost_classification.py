"""CPU parity and attribution checks for discarded setup and native audits."""
import hashlib
import json
import random
import sys
from types import ModuleType

import torch

from pcla_research_v7.data import digest
from pcla_research_v7.execution_timing import ExecutionTimingLedger, COST_LABEL_CATEGORIES
from pcla_research_v7.native_fedsb import initialize_pair
from pcla_research_v7.native_fedsvd import refresh_state
from scripts import run_pcla_v7


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self = torch.nn.Module()
        self.self.query = torch.nn.Linear(4, 4)
        self.self.value = torch.nn.Linear(4, 4)


class TinyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Module()
        self.bert.encoder = torch.nn.Module()
        self.bert.encoder.layer = torch.nn.ModuleList(
            [torch.nn.Module() for _ in range(12)])
        for layer in self.bert.encoder.layer:
            layer.attention = TinyAttention()
        self.classifier = torch.nn.Linear(4, 2)


def _prepare_inputs(tmp_path, monkeypatch):
    train = tmp_path / "SST-2" / "train.tsv"
    train.parent.mkdir()
    train.write_text("frozen train stub")
    model_dir = tmp_path / "bert-base-uncased"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                                    "partitions": {"401": {}}, "split_sha256": "split"}))
    b8 = tmp_path / "b8.json"
    b8.write_text(json.dumps({"total_bytes": 123,
                              "wire_manifest_sha256": digest({"total_bytes": 123})}))
    source = tmp_path / "SOURCE_MANIFEST.json"
    source.write_text(json.dumps({"files": {}}))
    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "test"
    fake_transformers.AutoModelForSequenceClassification = type(
        "AutoModelForSequenceClassification", (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: TinyBert())})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    class FakeTokenCache:
        def __init__(self, path):
            self.meta = {"identity": {"max_length": 64, "tokenizer_files": {}},
                         "cache_sha256": "cache", "identity_sha256": "cache_identity"}
        def dataset(self, _manifest, role, *_args):
            return range(256 if role == "client_train" else 4096)
    monkeypatch.setattr(run_pcla_v7, "TokenCache", FakeTokenCache)
    monkeypatch.setattr(run_pcla_v7, "b8_wire_manifest", lambda **_kwargs: {"total_bytes": 123})
    expanded = {"job": {"seed": 401, "rank": 4, "method": "fedsb"},
                "runtime": {"byte_cap": 123},
                "model": {"repository": str(model_dir), "max_length": 64},
                "manifest": str(manifest), "b8_wire": str(b8),
                "source_manifest": str(source), "train_file": str(train),
                "token_cache": str(tmp_path / "cache")}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(expanded))
    return expanded, config_path


def test_fedsb_discarded_common_lora_excluded_without_parameter_or_rng_change(tmp_path, monkeypatch):
    expanded, config_path = _prepare_inputs(tmp_path, monkeypatch)
    torch.manual_seed(82)
    random.seed(82)
    plain_model, *_ = run_pcla_v7.prepare(expanded, config_path, torch.device("cpu"))
    plain_state = {name: value.clone() for name, value in plain_model.state_dict().items()}
    plain_torch_rng, plain_python_rng = torch.get_rng_state().clone(), random.getstate()

    torch.manual_seed(82)
    random.seed(82)
    timing = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    timed_model, *_ = run_pcla_v7.prepare(expanded, config_path, torch.device("cpu"), timing=timing)
    timing.end_envelope()
    assert set(timed_model.state_dict()) == set(plain_state)
    assert all(torch.equal(value, plain_state[name]) for name, value in timed_model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), plain_torch_rng)
    assert random.getstate() == plain_python_rng
    snapshot = timing.snapshot()
    assert snapshot["label_seconds"]["technical_equivalence_check"] > 0
    assert snapshot["label_seconds"]["task_initialization"] > 0
    assert timing.coverage("envelope")["passes"]


def test_native_factor_audits_excluded_but_outputs_and_rng_unchanged():
    torch.manual_seed(36)
    step = torch.randn(8, 8)
    before = torch.get_rng_state().clone()
    plain_a, plain_b, plain_r, plain_audit = initialize_pair(step, 4)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    timing = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    timing.begin_envelope()
    with timing.span("task_initialization"):
        a, b, r, audit = initialize_pair(step, 4, timing=timing)
    timing.end_envelope()
    assert all(torch.equal(x, y) for x, y in zip((a, b, r), (plain_a, plain_b, plain_r)))
    assert audit == plain_audit
    assert torch.equal(torch.get_rng_state(), after)
    assert timing.snapshot()["label_seconds"]["technical_equivalence_check"] > 0

    state = {"query.lora_a": torch.randn(4, 8),
             "query.lora_b": torch.randn(8, 4),
             "classifier.weight": torch.randn(2, 8),
             "classifier.bias": torch.randn(2)}
    before = torch.get_rng_state().clone()
    plain_state, plain_record = refresh_state(state, 2, 5)
    timing = ExecutionTimingLedger(COST_LABEL_CATEGORIES)
    timing.begin_envelope()
    with timing.span("coordinate_and_server_decision"):
        timed_state, timed_record = refresh_state(state, 2, 5, timing=timing)
    timing.end_envelope()
    assert all(torch.equal(timed_state[key], plain_state[key]) for key in state)
    assert timed_record == plain_record
    assert torch.equal(torch.get_rng_state(), before)
    assert timing.snapshot()["label_seconds"]["technical_equivalence_check"] > 0
