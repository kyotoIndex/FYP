import json
from pathlib import Path

import pytest
import torch

from scripts import audit_pcla_v8 as auditor


def test_failure_emits_rejected_receipt_even_without_result(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"job": {"job_id": "DEV_U001_s503"}}))
    receipt = tmp_path / "audit.json"
    code = auditor.main(["--config", str(cfg), "--out", str(tmp_path / "missing"),
                         "--source-root", str(tmp_path), "--receipt", str(receipt)])
    report = json.loads(receipt.read_text())
    assert code == 1 and report["accepted"] is False
    assert report["job_id"] == "DEV_U001_s503" and report["result_sha256"] is None
    assert report["issues"]


def test_round_ledger_rejects_deployed_byte_overrun(tmp_path):
    rows = []
    for round_id in range(1, 101):
        rows.append({"round": round_id, "selected_clients": list(range(10)),
                     "client_training": [{"client": c, "optimizer_steps": 16,
                                          "optimizer_tensor_parameters": 3,
                                          "optimizer_groups": [{"lr": 0.001, "parameter_count": 3}]}
                                         for c in range(10)],
                     "deployed_bytes": round_id * (auditor.B8 // 100 + 1),
                     "diagnostic_after_budget_bytes": 0, "budget_stop_round": 100})
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "metrics.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="byte cap"):
        auditor.check_rounds(tmp_path, {"job": {"lr": 0.001, "head_lr": 0.001}},
                             {"rounds": 100, "budget_return_round": 100}, {})


def test_queue_requires_current_result_binding_and_preserved_failure(tmp_path):
    root = tmp_path / "run"
    config = root / "configs/development02/DEV_U001_s503.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    out = root / "results/dev_retry02/U001/s503"
    out.mkdir(parents=True)
    result = out / "result.json"
    result.write_text("{}")
    record = {"job_id": "DEV_U001_s503", "state": "executed_pending_audit",
              "config": str(config), "config_sha256": auditor.sha(config),
              "output": str(out), "result_path": str(result),
              "result_sha256": auditor.sha(result), "return_code": 0,
              "attempts": [{"worker": "w"}], "rebindings": []}
    q = root / "queues/development01/jobs/DEV_U001_s503.json"
    q.parent.mkdir(parents=True)
    q.write_text(json.dumps(record))
    auditor.check_queue(config, out, {"job": {"job_id": "DEV_U001_s503", "output_dir": str(out)}}, {}, {})
    record["output"] = str(root / "results/dev/U001/s503")
    q.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="binding"):
        auditor.check_queue(config, out, {"job": {"job_id": "DEV_U001_s503", "output_dir": str(out)}}, {}, {})


def test_prelaunch_rebind_is_separate_from_failed_attempt(tmp_path):
    root = tmp_path / "run"
    old = root / "configs/development01/DEV_U001_s509.json"
    current = root / "configs/development02/DEV_U001_s509.json"
    for path in (old, current):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    out = root / "results/dev/U001/s509"
    out.mkdir(parents=True)
    (out / "result.json").write_text("{}")
    prior = {"job_id": "DEV_U001_s509", "recipe_id": "U001", "seed": 509,
             "state": "blocked", "attempts": [], "return_code": None,
             "config": str(old), "config_sha256": auditor.sha(old), "output": str(out)}
    item = {"job_id": "DEV_U001_s509", "recipe_id": "U001", "seed": 509,
            "state": "executed_pending_audit", "config": str(current),
            "config_sha256": auditor.sha(current), "output": str(out),
            "result_path": str(out / "result.json"),
            "result_sha256": auditor.sha(out / "result.json"), "return_code": 0,
            "attempts": [{"worker": "w"}],
            "rebindings": [{"previous_record": prior, "reason": "source correction"}]}
    queue_path = root / "queues/development01/jobs/DEV_U001_s509.json"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(json.dumps(item))
    cfg = {"job": {"job_id": "DEV_U001_s509", "output_dir": str(out)}}
    checks = {}
    auditor.check_queue(current, out, cfg, {}, checks)
    assert checks["prelaunch_rebindings"] == 1 and checks["prior_failed_rebindings"] == 0
    prior["attempts"] = [{"worker": "unrecorded"}]
    queue_path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="unaccounted execution"):
        auditor.check_queue(current, out, cfg, {}, {})


def test_tensor_state_rejects_frozen_base_and_rank_drift():
    state = {f"bert.encoder.layer.{i}.attention.self.{target}.lora_{factor}":
             torch.zeros((4, 2) if factor == "a" else (2, 4))
             for i in range(12) for target in ("query", "value") for factor in ("a", "b")}
    state.update({"classifier.weight": torch.zeros(2, 2), "classifier.bias": torch.zeros(2)})
    auditor.check_state(state, 4, "valid")
    state["bert.encoder.layer.0.attention.self.query.lora_a"] = torch.zeros(8, 2)
    with pytest.raises(ValueError, match="wrong A rank"):
        auditor.check_state(state, 4, "bad")


def test_controller_hash_chain_rejects_tampered_parent(tmp_path):
    state = {f"bert.encoder.layer.{i}.attention.self.{target}.lora_{factor}":
             torch.zeros((4, 2) if factor == "a" else (2, 4))
             for i in range(12) for target in ("query", "value") for factor in ("a", "b")}
    state.update({"classifier.weight": torch.zeros(2, 2), "classifier.bias": torch.zeros(2)})
    torch.save({"format": "pcla-v7-runtime-1", "adapter": state}, tmp_path / "boundary_round_000.pt")
    record = {"rule": "fixed_scalar", "recipe_id": "U007", "seed": 503, "round_id": 1,
              "configuration_sha256": "a" * 64, "parent_hash": "b" * 64,
              "proposal_hash": "c" * 64, "committed_state_hash": "d" * 64,
              "head_key_manifest": ["classifier.weight", "classifier.bias"],
              "factor_key_manifest": [k for k in state if k.endswith((".lora_a", ".lora_b"))],
              "public_role": None, "alpha": 0.5, "beta": 0.5,
              "features_count": 0, "public_dataset_passes": 0, "encoder_batches": 0}
    (tmp_path / "controller_round_001.json").write_text(json.dumps(record))
    (tmp_path / "controller.jsonl").write_text(json.dumps({"round": 1, **record}) + "\n")
    cfg = {"recipe": {"server_rule": "fixed_scalar", "config_sha256": "a" * 64,
                      "alpha": 0.5, "beta": 0.5, "public_training": False},
           "job": {"recipe_id": "U007", "seed": 503}}
    with pytest.raises(ValueError, match="parent hash chain"):
        auditor.check_controller(tmp_path, cfg, [{"round": 1, "server_controller": record}],
                                 {"adapter": state}, {})


def test_timing_requires_event_totals_not_only_reported_breakdown(tmp_path):
    record = {"schema": "pcla-exclusive-host-timing-2",
              "bucket_label_seconds": {"deployed_prefix": {"task_initialization": 1.0},
                                       "diagnostic_after_budget": {"task_initialization": 0.0}},
              "events": [{"bucket": "deployed_prefix", "label": "task_initialization", "seconds": 2.0}],
              "label_seconds": {"task_initialization": 1.0}, "total_seconds": 1.0}
    (tmp_path / "timing_attempt_001.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="event/label totals"):
        auditor.check_timing(tmp_path, {"required_cost_breakdown": {k: 0 for k in auditor.REQUIRED_GROUPS}},
                             [{"resume_round": 1}], {})


def _tiny_state(rank=4):
    values = {f"bert.encoder.layer.{i}.attention.self.{target}.lora_{factor}":
              torch.zeros((rank, 2) if factor == "a" else (2, rank))
              for i in range(12) for target in ("query", "value") for factor in ("a", "b")}
    values.update({"classifier.weight": torch.zeros(2, 2), "classifier.bias": torch.zeros(2)})
    return values


def test_baseline_controller_uses_core_tensor_digest_and_saved_anchor(tmp_path):
    initial = _tiny_state()
    accepted = {k: v.clone() for k, v in initial.items()}
    accepted["classifier.bias"][0] = 0.25
    torch.save({"format": "pcla-v7-runtime-1", "adapter": initial}, tmp_path / "boundary_round_000.pt")
    torch.save({"round": 1, "adapter": accepted}, tmp_path / "adapter_round_001.pt")
    record = {"rule": "mean_r4", "alpha": 1.0, "beta": 1.0,
              "parent_hash": auditor.baseline_tensor_digest(initial),
              "proposal_hash": auditor.baseline_tensor_digest(accepted),
              "accepted_hash": auditor.baseline_tensor_digest(accepted),
              "features_count": 0, "encoder_batches": 0, "public_dataset_passes": 0,
              "grid_public_NLL": []}
    assert auditor.state_hash(accepted) != auditor.baseline_tensor_digest(accepted)
    (tmp_path / "controller_round_001.json").write_text(json.dumps(record))
    (tmp_path / "controller.jsonl").write_text(json.dumps({"round": 1, **record}) + "\n")
    row = {"round": 1, "server_controller": record}
    checks = {}
    auditor.check_controller(tmp_path, {"recipe": {"server_rule": "mean_r4"}, "job": {}},
                             [row], {"adapter": accepted}, checks)
    assert checks["controller_hash_algorithm"] == "pcla_repair.core.tensor_digest"
    record["proposal_hash"] = "0" * 64
    (tmp_path / "controller_round_001.json").write_text(json.dumps(record))
    (tmp_path / "controller.jsonl").write_text(json.dumps({"round": 1, **record}) + "\n")
    with pytest.raises(ValueError, match="baseline mean state"):
        auditor.check_controller(tmp_path, {"recipe": {"server_rule": "mean_r4"}, "job": {}},
                                 [{"round": 1, "server_controller": record}], {"adapter": accepted}, {})


def test_budget_checkpoint_without_stop_decision_is_rejected(tmp_path):
    adapter = _tiny_state()
    row = {"round": 1, "deployed_bytes": 10, "diagnostic_after_budget_bytes": 0,
           "budget_stop_round": 1}
    common = {"format": "pcla-v7-runtime-1", "identity": "same", "round": 1,
              "adapter": adapter, "metrics": [row], "deployed_bytes": 10,
              "diagnostic_bytes": 0, "budget_stop_round": 1, "budget_adapter": adapter,
              "half_adapter": adapter, "half_round": 0, "half_readout": {},
              "torch_rng": torch.zeros(4, dtype=torch.uint8), "python_rng": (1, (2,)), "cache": {}}
    for name in ("latest.pt", "budget_state.pt", "half_budget_state.pt"):
        value = dict(common)
        if name == "half_budget_state.pt":
            value.update(round=0, metrics=[], deployed_bytes=0, budget_stop_round=None)
        if name == "budget_state.pt":
            value["budget_stop_round"] = None
        torch.save(value, tmp_path / name)
    torch.save({"format": "pcla-v7-runtime-1", "identity": "same", "round": 1,
                "adapter": adapter, "deployed_bytes": 10, "budget_cap_bytes": auditor.B8},
               tmp_path / "budget_return.pt")
    torch.save({"format": "pcla-v7-runtime-1", "identity": "same", "round": 0,
                "adapter": adapter, "deployed_bytes": 0, "budget_cap_bytes": auditor.B8 // 2,
                "readout": {}}, tmp_path / "half_budget_return.pt")
    with pytest.raises(ValueError, match="lacks exact endpoint decision"):
        auditor.check_checkpoints(tmp_path, {"job": {"rank": 4}},
                                  [row], 0, {})
