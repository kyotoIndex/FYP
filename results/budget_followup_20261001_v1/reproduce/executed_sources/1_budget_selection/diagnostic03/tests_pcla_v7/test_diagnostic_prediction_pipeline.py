"""Real four-bank prediction integrated with staged persistence; CPU only."""
from dataclasses import dataclass

import pytest
import torch

import pcla_research_v7.diagnostic_pipeline as pipeline
import pcla_research_v7.diagnostic_prediction as prediction
from pcla_research_v7.diagnostic_readout import DiagnosticView
from tests_pcla_v7.test_diagnostic_prediction import Rows, bank_from, fixture


@dataclass(frozen=True)
class Selection:
    sha256: str


def test_real_prediction_is_sealed_before_training_and_survives_resume(tmp_path, monkeypatch):
    inputs = fixture("a")
    participants = inputs["participants"]
    readout = DiagnosticView("readout2048", Rows(2048), "source", tuple(range(2048)))
    selection = DiagnosticView("selection1024", Rows(1024), "source", tuple(range(2048, 3072)))
    local = {cid: DiagnosticView("local64", Rows(64), "source",
                                tuple(range(3072 + 64 * cid, 3136 + 64 * cid)))
             for cid in participants}
    context = {"anchor": {"anchor_sha256": inputs["anchor_sha256"],
                          "checkpoint_identity": "parent-checkpoint"},
               "parent_state": inputs["parent"], "model": object(),
               "selection": selection, "readout": readout, "local64": local,
               "visits": {}, "anchor_metrics": [], "clients": {}, "parent_cache": {}}
    config = {"job_id": inputs["object_id"], "participants": participants,
              "factor": "a", "anchor_round": 10, "seed": 101,
              "training": {"learning_rate": .001}}
    bundle = {"full_probe_gradients": inputs["full_probe_gradients"],
              "h32_payloads": inputs["h32_payloads"], "h32_sha256": inputs["h32_sha256"],
              "budgets": inputs["budgets"]}
    order = []
    actual_predict = prediction.compare_fixed_bank_predictions
    actual_full_readout = prediction.readout_locked_full_choice

    def predict(**kwargs):
        assert not (tmp_path / "bank.pt").exists()
        assert not (tmp_path / "selection_lock.pt").exists()
        order.append("predict")
        return actual_predict(**kwargs)

    def collect(**kwargs):
        assert (tmp_path / "prediction_lock.pt").is_file()
        assert (tmp_path / "prediction_lock.json").is_file()
        order.append("train")
        return bank_from(inputs)

    def select(**kwargs):
        assert (tmp_path / "bank_receipt.json").is_file()
        assert (tmp_path / "prediction_lock.pt").is_file()
        order.append("selection")
        return Selection("s" * 64)

    def measure(**kwargs):
        assert (tmp_path / "selection_lock.json").is_file()
        assert (tmp_path / "prediction_lock.pt").is_file()
        order.append("readout")
        return {"schema": "test-bank-readout"}

    class Loss:
        def __init__(self, model, views):
            assert views[0] is selection and views[1] is readout
            self.evaluations = []

        def __call__(self, values, dataset):
            assert dataset is readout.dataset
            assert (tmp_path / "prediction_lock.pt").is_file()
            assert (tmp_path / "selection_lock.pt").is_file()
            return sum(float(value.square().sum()) for value in values.values()) / len(dataset)

    def full_choice(**kwargs):
        order.append("full-choice-readout")
        return actual_full_readout(**kwargs)

    monkeypatch.setattr(prediction, "compare_fixed_bank_predictions", predict)
    monkeypatch.setattr(prediction, "readout_locked_full_choice", full_choice)
    monkeypatch.setattr(pipeline, "finite_bank_selection", select)
    monkeypatch.setattr(pipeline, "measure_bank_readouts", measure)
    monkeypatch.setattr(pipeline, "ModelLoss", Loss)
    kwargs = dict(config=config, context=context, supports=inputs["supports"],
                  jq_lock=inputs["jq_lock"], output=tmp_path,
                  config_sha256="c" * 64, source_sha256="d" * 64,
                  support_bundle=bundle, collect=collect)
    receipt = pipeline.run_bank_readout(**kwargs)
    assert order == ["predict", "train", "selection", "readout", "full-choice-readout"]
    assert receipt["status"] == "bank_readout_complete"
    assert receipt["diagnostic_object_complete"] is False
    assert (tmp_path / "prediction_lock.pt").is_file()
    assert (tmp_path / "bank.pt").is_file()
    assert (tmp_path / "selection_lock.pt").is_file()
    assert pipeline.run_bank_readout(**kwargs) == receipt
    assert len(order) == 5
    (tmp_path / "prediction_lock.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="prediction lock changed"):
        pipeline.run_bank_readout(**kwargs)
