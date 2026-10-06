"""Method-specific and fail-closed checks for the independent v8 auditor."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

AUDIT_DIR = Path(__file__).resolve().parents[3] / "checks/result_auditor02"
sys.path.insert(0, str(AUDIT_DIR))
spec = importlib.util.spec_from_file_location("audit_pcla_v8_extended", AUDIT_DIR / "audit_pcla_v8_extended.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def core_case(tmp_path, rule):
    state = {f"bert.encoder.layer.{i}.attention.self.{target}.lora_{factor}":
             torch.zeros((8, 768) if factor == "a" else (768, 8))
             for i in range(12) for target in ("query", "value") for factor in ("a", "b")}
    state.update({"classifier.weight": torch.zeros(2, 768), "classifier.bias": torch.zeros(2)})
    torch.save({"format": "pcla-v7-runtime-1", "adapter": state}, tmp_path / "boundary_round_000.pt")
    record = {"rule": rule, "recipe_id": "U021", "seed": 503, "round_id": 1,
              "public_role": "train_public512" if rule != "fedexp_peft" else None,
              "parent_hash": auditor.base.state_hash(state),
              "proposal_hash": auditor.base.state_hash(state),
              "committed_state_hash": auditor.base.state_hash(state),
              "configuration_sha256": "a" * 64,
              "head_key_manifest": ["classifier.weight", "classifier.bias"],
              "factor_key_manifest": [k for k in state if k.endswith((".lora_a", ".lora_b"))],
              "grid_public_NLL": [], "features_count": 0, "encoder_batches": 0,
              "head_evals": 0, "public_dataset_passes": 0, "server_gradient_batches": 0}
    recipe = {"server_rule": rule, "config_sha256": "a" * 64,
              "public_training": rule != "fedexp_peft"}
    return state, record, {"recipe": recipe, "job": {"recipe_id": "U021", "seed": 503}}


def check_case(tmp_path, state, record, cfg):
    (tmp_path / "controller_round_001.json").write_text(json.dumps(record))
    (tmp_path / "controller.jsonl").write_text(json.dumps({"round": 1, **record}) + "\n")
    checks = {}
    auditor.core_controller(tmp_path, cfg, [{"round": 1, "server_controller": record}],
                            {"adapter": state}, checks)
    return checks


def test_headfit_work_and_public_role(tmp_path):
    state, rec, cfg = core_case(tmp_path, "headfit")
    cfg["recipe"].update(server_head_optimizer="AdamW", server_head_mode="eval feature cache, fresh optimizer, head grads only",
                         server_head_steps=32, server_batch_size=16)
    rec.update(alpha=1., beta=None, headfit_steps=32, features_count=512, encoder_batches=8,
               public_dataset_passes=1, head_evals=34, public_ordered_content_sha256="b" * 64,
               mean_update_public_NLL=.7, chosen_public_NLL=.6)
    assert check_case(tmp_path, state, rec, cfg)["controller_rounds"] == 1
    rec["head_evals"] = 33
    with pytest.raises(ValueError, match="headfit work"):
        check_case(tmp_path, state, rec, cfg)


def test_fedlaw_simplex_and_gradient_count(tmp_path):
    state, rec, cfg = core_case(tmp_path, "fedlaw_peft")
    cfg["recipe"].update(server_optimizer="Adam", server_betas=[.5, .999], server_epochs=1,
                         server_batch_size=64, server_parameterization="exp gamma + softmax lambda")
    rec.update(alpha=None, beta=None, features_count=512, encoder_batches=8,
               server_gradient_batches=8, head_evals=8, public_dataset_passes=1,
               gamma=.9, lambda_weights=[.1] * 10, public_ordered_content_sha256="b" * 64)
    check_case(tmp_path, state, rec, cfg)
    rec["lambda_weights"][0] = -.1
    with pytest.raises(ValueError, match="gamma/simplex"):
        check_case(tmp_path, state, rec, cfg)


def test_fedexp_eta_and_epsilon(tmp_path):
    state, rec, cfg = core_case(tmp_path, "fedexp_peft")
    cfg["recipe"]["server_epsilon"] = .001
    rec.update(alpha=1.2, beta=1.2, eta=1.2)
    assert check_case(tmp_path, state, rec, cfg)["fedexp_norm_evidence"].startswith("frozen-source-only")
    rec.update(alpha=.99, beta=.99, eta=.99)
    with pytest.raises(ValueError, match="FedExP eta"):
        check_case(tmp_path, state, rec, cfg)
    rec.update(alpha=1., beta=1., eta=1.)
    cfg["recipe"]["server_epsilon"] = 0
    with pytest.raises(ValueError, match="FedExP eta"):
        check_case(tmp_path, state, rec, cfg)


def test_native_state_rank32_and_frozen_base_absence():
    state = {f"bert.encoder.layer.{i}.attention.self.{target}.{suffix}":
             torch.zeros(shape) for i in range(12) for target in ("query", "value")
             for suffix, shape in (("a", (32, 768)), ("b", (768, 32)), ("r", (32, 32)))}
    state.update({"classifier.weight": torch.zeros(2, 768), "classifier.bias": torch.zeros(2)})
    auditor.native_state(state, "valid")
    state["bert.encoder.layer.0.attention.self.query.r"] = torch.zeros(31, 31)
    with pytest.raises(ValueError, match="native.*shape"):
        auditor.native_state(state, "bad")


def test_incomplete_native_rejected_and_base_pinned(tmp_path):
    auditor.verify_base()
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"job": {"job_id": "DEV_U005_s503"},
                               "recipe": {"server_rule": "fedsb_r32"}}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.json").write_text(json.dumps({"state": "technical_paused"}))
    receipt = tmp_path / "receipt.json"
    assert auditor.main(["--config", str(cfg), "--out", str(out),
                         "--source-root", str(tmp_path), "--receipt", str(receipt)]) == 1
    assert json.loads(receipt.read_text())["accepted"] is False
