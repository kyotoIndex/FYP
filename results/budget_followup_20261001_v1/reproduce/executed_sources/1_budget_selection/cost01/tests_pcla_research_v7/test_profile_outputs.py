import json

import pytest
import torch

from scripts.check_pcla_v7_profile_outputs import _scientific_rows, check


def _make_profile(root, processes, *, wall, memory=20):
    root.mkdir()
    gpu = "GPU-test"
    samples = [{"time": 1., "processes": [
        {"pid": 100 + i, "gpu": gpu, "memory_mib": memory}
        for i in range(processes)]}]
    profile = {"processes": processes, "rounds_each": 2,
               "wall_seconds": wall, "rounds_per_second": 2 * processes / wall,
               "exit_codes": [0] * processes,
               "statuses": [{"state": "paused", "completed_round": 2} for _ in range(processes)],
               "peak_total_process_memory_mib": processes * memory,
               "samples": samples}
    (root / "profile.json").write_text(json.dumps(profile))
    identity = {"job": {"method": "JQ", "seed": 9},
                "provenance": {"config_sha256": "config", "sources": {"runtime.py": "source"},
                               "manifest_sha256": "split", "token_cache_sha256": "tokens",
                               "hardware": {"name": "test GPU", "total_memory": 100 * 2**20,
                                            "uuid": "test"}}}
    rows = [{"round": r, "selected_clients": [0, 1], "deployed_bytes": r * 100,
             "diagnostic_after_budget_bytes": 0, "evaluation": {"accuracy": .5},
             "ema_evaluation": {"accuracy": .5},
             "coordinate_refresh": {"client_scores": {0: 1.25, 1: 2.5}},
             "client_training": [{"optimizer_steps": 16, "train_seconds": r}],
             "cost": {"round_wall_seconds": r, "cuda_peak_reserved_bytes": 12 * 2**20}}
            for r in (1, 2)]
    for i in range(processes):
        directory = root / f"process_{i}"
        directory.mkdir()
        (directory / "run_identity.json").write_text(json.dumps(identity))
        (directory / "metrics.json").write_text(json.dumps(rows))
        (directory / "evaluation_labels.json").write_text("[0,1]\n")
        for r in range(3):
            (directory / f"logits_round_{r:03d}.json").write_text("[[0.1,0.2]]\n")
        checkpoint = {"format": "pcla-v7-runtime-1", "identity": "run", "round": 2,
                      "adapter": {"a": torch.tensor([1., 2.])},
                      "budget_adapter": None, "half_adapter": {"a": torch.tensor([1.])},
                      "half_round": 1, "half_readout": {"accuracy": .5},
                      "ema_effective": {"a": torch.tensor([.5])},
                      "cache": {"events": [{"event_id": "r1", "bytes": 10}]},
                      "deployed_bytes": 200, "diagnostic_bytes": 0,
                      "budget_stop_round": None, "torch_rng": torch.tensor([1], dtype=torch.uint8),
                      "python_rng": (3, (1, 2), None), "cuda_rng": [torch.tensor([2], dtype=torch.uint8)],
                      "metrics": rows}
        torch.save(checkpoint, directory / "latest.pt")


def test_profile_gate_requires_bitwise_trajectories_and_resource_margin(tmp_path):
    single, dual = tmp_path / "p1", tmp_path / "p2"
    _make_profile(single, 1, wall=130)
    _make_profile(dual, 2, wall=200)
    result = check(single, dual)
    assert result["passed"]
    assert result["throughput"]["dual_over_single"] == 1.3
    assert result["memory"]["headroom_fraction"] == .6

    path = dual / "process_1" / "latest.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["adapter"]["a"][0] += 1
    torch.save(checkpoint, path)
    result = check(single, dual)
    assert not result["passed"]
    assert not result["numerical"]["dual_process_1"]["adapter"]


def test_profile_gate_rejects_insufficient_throughput_or_memory(tmp_path):
    single, dual = tmp_path / "p1", tmp_path / "p2"
    _make_profile(single, 1, wall=100)
    _make_profile(dual, 2, wall=200, memory=45)
    result = check(single, dual)
    assert not result["passed"]
    assert not result["checks"]["throughput_at_least_1_25"]
    assert not result["checks"]["observed_memory_headroom_at_least_20_percent"]


def test_profile_gate_rejects_different_gpu_uuid_after_normalization(tmp_path):
    single, dual = tmp_path / "p1", tmp_path / "p2"
    _make_profile(single, 1, wall=130)
    _make_profile(dual, 2, wall=200)
    identity_path = dual / "process_1" / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["provenance"]["hardware"]["uuid"] = "other-device"
    identity_path.write_text(json.dumps(identity))
    result = check(single, dual)
    assert not result["passed"]
    assert not result["checks"]["observed_memory_headroom_at_least_20_percent"]


def test_integer_metric_keys_match_json_but_scientific_values_do_not_disappear(tmp_path):
    single, dual = tmp_path / "p1", tmp_path / "p2"
    _make_profile(single, 1, wall=130)
    _make_profile(dual, 2, wall=200)
    assert check(single, dual)["passed"]
    path = dual / "process_1" / "latest.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["metrics"][0]["coordinate_refresh"]["client_scores"][0] = 9.25
    torch.save(checkpoint, path)
    # Keep its own JSON mirror consistent, then compare against the independent run.
    (dual / "process_1" / "metrics.json").write_text(json.dumps(checkpoint["metrics"]))
    result = check(single, dual)
    assert not result["passed"]
    assert not result["numerical"]["dual_process_1"]["scientific_metrics"]


def test_scientific_key_normalization_rejects_collisions():
    with pytest.raises(ValueError, match="colliding scientific metric keys"):
        _scientific_rows([{"client_scores": {1: 2., "1": 3.}}])
