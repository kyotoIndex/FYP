import json
from pathlib import Path

import pytest

from pcla_research_v8 import queue


def test_one_gpu_claim_restart_and_independent_audit(tmp_path, monkeypatch):
    root = tmp_path / "queue"
    config = tmp_path / "DEV_U001_s503.json"
    config.write_text(json.dumps({"job": {"job_id": "DEV_U001_s503"}}))
    monkeypatch.setattr(queue, "_verify_binding", lambda _: None)
    for jid, recipe, seed in (("DEV_U001_s503", "U001", 503), ("DEV_U002_s503", "U002", 503)):
        cfg = tmp_path / f"{jid}.json"
        cfg.write_text(json.dumps({"job": {"job_id": jid}}))
        queue.write(root / "jobs" / f"{jid}.json", {
            "job_id": jid, "recipe_id": recipe, "family": "mean_r8", "seed": seed,
            "state": "pending", "config": str(cfg), "config_sha256": queue.sha(cfg),
            "attempts": [], "output": str(tmp_path / jid),
        })
    a = queue.claim(root, worker="w1", gpu_uuid="GPU-one", slurm_job="1", slurm_step="0", cpu_cores=(0, 1, 2, 3))
    assert a and a["job_id"] == "DEV_U001_s503"
    assert queue.claim(root, worker="w2", gpu_uuid="GPU-one", slurm_job="1", slurm_step="1", cpu_cores=(4, 5, 6, 7)) is None
    assert queue.read(root / "jobs/DEV_U001_s503.json")["state"] == "running"
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"job_id": a["job_id"], "state": "executed_pending_audit"}))
    assert queue.finish(root, a["job_id"], a["claim"]["token"], return_code=0, result_path=result) == "executed_pending_audit"
    assert queue.read(root / "jobs/DEV_U001_s503.json")["state"] != "completed"
    receipt = tmp_path / "audit.json"
    receipt.write_text(json.dumps({"accepted": True, "job_id": a["job_id"],
                                   "result_sha256": queue.sha(result), "auditor": "independent-audit"}))
    queue.complete_audit(root, a["job_id"], receipt)
    assert queue.read(root / "jobs/DEV_U001_s503.json")["state"] == "completed"
    with pytest.raises(ValueError):
        queue.complete_audit(root, a["job_id"], receipt)


def test_failed_job_is_not_requeued(tmp_path, monkeypatch):
    root = tmp_path / "queue"
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    monkeypatch.setattr(queue, "_verify_binding", lambda _: None)
    queue.write(root / "jobs/j.json", {"job_id": "j", "recipe_id": "U001", "seed": 503,
                "family": "mean_r8", "state": "pending", "config": str(cfg),
                "config_sha256": queue.sha(cfg), "attempts": [], "output": str(tmp_path / "out")})
    job = queue.claim(root, worker="w", gpu_uuid="GPU-one", slurm_job="1", slurm_step="0", cpu_cores=(0, 1, 2, 3))
    assert queue.finish(root, "j", job["claim"]["token"], return_code=1) == "failed"
    assert queue.claim(root, worker="w", gpu_uuid="GPU-one", slurm_job="1", slurm_step="0", cpu_cores=(0, 1, 2, 3)) is None
