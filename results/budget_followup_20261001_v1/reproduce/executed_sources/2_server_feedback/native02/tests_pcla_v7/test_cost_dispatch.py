"""COST1 queue checks use a synthetic 117-job matrix and no GPU/Slurm."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dispatch_pcla_v7_cost as cost


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def case(tmp_path):
    run = tmp_path / "pcla-global-budget-20260928-v7"
    queue = run / "cost_queue"
    source = run / "sources" / "cost01"
    put(source / "scripts" / "run_pcla_v7.py", {"stub": "core"})
    put(source / "scripts" / "run_pcla_v7_native.py", {"stub": "native"})
    manifest = {"files": {
        "scripts/run_pcla_v7.py": cost.sha(source / "scripts" / "run_pcla_v7.py"),
        "scripts/run_pcla_v7_native.py": cost.sha(source / "scripts" / "run_pcla_v7_native.py")}}
    put(source / "SOURCE_MANIFEST.json", manifest)
    old = put(run / "sources" / "core02" / "SOURCE_MANIFEST.json", {"files": {}})
    old_sha = cost.sha(old)
    jobs = []
    expected = {}
    for number in range(1, 118):
        recipe = f"D{number:03d}"
        jid = f"COST1_{recipe}_s401"
        priority = 0 if number in {11, 35} else 10 if number in {66, 69, 72, 105, 107, 112} else 30
        family = "native03" if number == 69 else "core02"
        config = put(run / "configs" / family / f"{recipe}_s401.json",
                     {"job": {"job_id": f"{recipe}_s401", "seed": 401}})
        config_sha = cost.sha(config)
        expected[recipe] = {"original_job_id": f"{recipe}_s401",
                            "config_sha256": config_sha,
                            "source_manifest_sha256": old_sha,
                            "budget_stop_round": 100,
                            "deployed_bytes": 1234}
        jobs.append({"replay_id": jid, "recipe_id": recipe, "seed": 401,
                     "priority": priority, "original_config": str(config),
                     "original_config_sha256": config_sha,
                     "original_source_manifest": str(old),
                     "original_source_manifest_sha256": old_sha,
                     "expected_budget_return_round": 100,
                     "expected_deployed_bytes": 1234,
                     "required_hardware": "RTX4090D_exclusive",
                     "dispatch_status": "blocked_pending_gpu_timing_gate",
                     "end_control": "first_actual_budget_refusal_including_control",
                     "pro_plan": {"job_id": jid, "recipe_id": recipe,
                                  "original_job_id": f"{recipe}_s401",
                                  "prefix_round": "100", "expected_deployed_bytes": "1234",
                                  "config_sha256": config_sha,
                                  "source_manifest_sha256": old_sha}})
    expected_path = put(run / "expected_cost_identities.json", expected)
    plan = put(run / "cost1_bound_plan_v2.json", {
        "schema": cost.PLAN_SCHEMA, "status": "not_dispatched",
        "quality_results_replaced": False,
        "inputs": {str(expected_path): cost.sha(expected_path)}, "jobs": jobs})
    gate = put(run / "gpu_gate.json", {
        "schema": cost.GATE_SCHEMA, "status": "passed",
        "source_manifest_sha256": cost.sha(source / "SOURCE_MANIFEST.json"),
        "hardware": cost.HARDWARE, "mode": "exclusive",
        "technical_seeds": [9001, 9002],
        "passed_paths": sorted(cost.PATHS)})
    return queue, plan, source, expected_path, gate


def release(case, ids):
    queue, _, source, expected, gate = case
    return cost.release(queue, ids, source, expected, gate, cost.sha(gate))


def test_initialize_isolated_blocked_idempotent_and_conflict(case, tmp_path):
    queue, plan, *_ = case
    assert cost.initialize(queue, plan) == 117
    assert cost.initialize(queue, plan) == 117
    assert cost.status(queue)["counts"] == {"blocked": 117}
    assert cost.claim(queue, "general", "w", "GPU-a") is None
    with pytest.raises(ValueError, match="cost_queue"):
        cost.initialize(queue.parent / "queue", plan)
    with pytest.raises(ValueError, match="cost_queue"):
        cost.initialize(tmp_path / "cost_queue", plan)
    plan.write_text(plan.read_text() + " ")
    with pytest.raises(ValueError, match="identity differs"):
        cost.initialize(queue, plan)


def test_release_requires_bound_gate_and_original_inputs(case):
    queue, plan, source, expected, gate = case
    cost.initialize(queue, plan)
    with pytest.raises(ValueError, match="receipt SHA"):
        cost.release(queue, [cost.PAIR[0]], source, expected, gate, "0" * 64)
    gate_data = cost.read(gate)
    gate_data["passed_paths"] = ["core01"]
    put(gate, gate_data)
    with pytest.raises(ValueError, match="four-path"):
        release(case, [cost.PAIR[0]])
    gate_data["passed_paths"] = sorted(cost.PATHS)
    put(gate, gate_data)
    assert release(case, [cost.PAIR[0]]) == [cost.PAIR[0]]
    with pytest.raises(ValueError, match="blocked"):
        release(case, [cost.PAIR[0]])
    (source / "scripts" / "run_pcla_v7.py").write_text("mutation")
    with pytest.raises(RuntimeError, match="snapshot mismatch"):
        release(case, [cost.PAIR[1]])


def test_pair_same_uuid_order_and_general_priority(case):
    queue, plan, *_ = case
    cost.initialize(queue, plan)
    ids = [*cost.PAIR, "COST1_D069_s401", "COST1_D003_s401"]
    release(case, ids)
    first = cost.claim(queue, "pair", "w1", "GPU-a")
    assert first[0].stem == cost.PAIR[0]
    assert cost.read(queue / "pair_lane.json")["gpu_uuid"] == "GPU-a"
    assert cost.claim(queue, "pair", "w2", "GPU-b") is None
    assert cost.claim(queue, "general", "w2", "GPU-a") is None
    general = cost.claim(queue, "general", "w2", "GPU-b")
    assert general[0].stem == "COST1_D069_s401"  # priority 10 before 30
    assert cost.finish(queue, first[0], "w1", 0) == "executed_pending_audit"
    assert cost.claim(queue, "pair", "w3", "GPU-b") is None
    second = cost.claim(queue, "pair", "w3", "GPU-a")
    assert second[0].stem == cost.PAIR[1]
    assert second[1]["attempts"][0]["gpu_uuid"] == "GPU-a"
    assert cost.finish(queue, second[0], "w3", 9) == "failed_needs_review"
    assert cost.claim(queue, "pair", "w4", "GPU-a") is None
    assert cost.finish(queue, general[0], "w2", 0) == "executed_pending_audit"
    assert cost.claim(queue, "general", "w5", "GPU-a")[0].stem == "COST1_D003_s401"


def test_hash_rechecked_at_claim_and_cli_per_family(case):
    queue, plan, source, expected, gate = case
    cost.initialize(queue, plan)
    ids = ["COST1_D069_s401", "COST1_D003_s401"]
    release(case, ids)
    core_item = cost.read(queue / "jobs" / "COST1_D003_s401.json")
    native_item = cost.read(queue / "jobs" / "COST1_D069_s401.json")
    core_argv = cost.argv_for(core_item, "/venv/python")
    native_argv = cost.argv_for(native_item, "/venv/python")
    assert core_argv[1].endswith("run_pcla_v7.py")
    assert "--original-source-manifest" in core_argv
    assert "--expected-cost-identities" in core_argv
    assert native_argv[1].endswith("run_pcla_v7_native.py")
    assert "--original-source-manifest" not in native_argv
    assert "--expected-cost-identities" not in native_argv
    expected.write_text(expected.read_text() + " ")
    with pytest.raises(RuntimeError, match="input mutated"):
        cost.claim(queue, "general", "worker", "GPU-a")


def test_failed_attempt_preserved_without_retry(case):
    queue, plan, *_ = case
    cost.initialize(queue, plan)
    release(case, ["COST1_D003_s401"])
    claimed = cost.claim(queue, "general", "worker", "GPU-a")
    with pytest.raises(RuntimeError, match="ownership"):
        cost.finish(queue, claimed[0], "wrong-worker", 1)
    assert cost.finish(queue, claimed[0], "worker", 17) == "failed_needs_review"
    item = cost.read(claimed[0])
    assert item["attempts"][0]["exit_code"] == 17
    assert item["attempts"][0]["raw_output"] == item["out"]
    assert cost.claim(queue, "general", "next-worker", "GPU-a") is None


def test_same_gpu_never_has_two_running_cost_jobs(case):
    queue, plan, *_ = case
    cost.initialize(queue, plan)
    release(case, ["COST1_D003_s401", "COST1_D004_s401", cost.PAIR[0]])
    general = cost.claim(queue, "general", "g1", "GPU-a")
    assert general[0].stem == "COST1_D003_s401"
    assert cost.claim(queue, "general", "g2", "GPU-a") is None
    assert cost.claim(queue, "pair", "p1", "GPU-a") is None
    assert cost.claim(queue, "pair", "p1", "GPU-b")[0].stem == cost.PAIR[0]
    assert cost.claim(queue, "general", "g3", "GPU-b") is None


def test_reject_incomplete_or_duplicate_matrix(case):
    _, plan, *_ = case
    data = cost.read(plan)
    data["jobs"][1]["replay_id"] = data["jobs"][0]["replay_id"]
    put(plan, data)
    with pytest.raises(ValueError, match="invalid or duplicate"):
        cost._checked_plan(plan)


def test_worker_keeps_output_empty_for_runner_and_preserves_log(case, monkeypatch):
    queue, plan, *_ = case
    cost.initialize(queue, plan)
    release(case, ["COST1_D003_s401"])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cost, "_gpu_uuid", lambda: "GPU-a")
    called = []

    def fake_run(argv, **kwargs):
        output = Path(argv[argv.index("--out") + 1])
        assert output.is_dir() and list(output.iterdir()) == []
        assert kwargs["env"]["PCLA_GPU_SHARING"] == "exclusive"
        called.append(argv)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(cost.subprocess, "run", fake_run)
    cost.worker(queue, "general", "/venv/python", idle_seconds=0)
    assert len(called) == 1
    item = cost.read(queue / "jobs" / "COST1_D003_s401.json")
    assert item["state"] == "executed_pending_audit"
    log = Path(item["attempts"][0]["worker_log"])
    assert log.is_file()
    assert (log.parent / "dispatch_identity.json").is_file()
