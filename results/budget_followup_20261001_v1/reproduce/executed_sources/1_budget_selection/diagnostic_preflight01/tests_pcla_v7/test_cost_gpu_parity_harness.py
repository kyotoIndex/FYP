"""CPU-only checks for the TECH GPU parity harness and its evidence gates."""
import argparse
import json
from pathlib import Path

import pytest
import torch

from scripts.check_pcla_v7_cost_gpu_parity import (
    CASES, SCHEMA, TECH_SEEDS, compare_parity, digest, launch,
    make_technical_config, scientific_rows, sha, shared_preparation_config, write,
)


def test_technical_config_keeps_recipe_and_rejects_other_seeds(tmp_path):
    original = {"job": {"recipe_id": "D107", "seed": 401, "method": "fedsvd",
                        "min_rounds": 100, "max_rounds": 200},
                "recipe": {"recipe_id": "D107", "lr": .1, "factor_schedule": "B_only"},
                "runtime": {"byte_cap": 2382653750}, "expanded_sha256": "original"}
    old = json.loads(json.dumps(original))
    result = make_technical_config(original, "D107", 9001, tmp_path / "SOURCE_MANIFEST.json")
    assert original == old
    assert result["recipe"] == original["recipe"]
    assert result["runtime"] == original["runtime"]
    assert result["job"]["phase"] == "TECH" and result["job"]["seed"] == 9001
    assert result["technical_profile_only"] and result["technical_stop_after"] == 2
    assert result["expanded_sha256"] == digest({k: v for k, v in result.items()
                                                 if k != "expanded_sha256"})
    with pytest.raises(ValueError, match="unapproved"):
        make_technical_config(original, "D107", 401, tmp_path / "SOURCE_MANIFEST.json")
    native32 = {"job": {"rank": 32}, "recipe": {"required_adapter": "fedsb"}}
    common = shared_preparation_config(native32)
    assert common["job"]["rank"] == 16 and native32["job"]["rank"] == 32
    core32 = {"job": {"rank": 32}, "recipe": {"required_adapter": "core"}}
    assert shared_preparation_config(core32) is core32


def test_scientific_rows_and_exact_parity_detection():
    row = {"round": 1, "selected_clients": [2], "support_trace": {"2": {"q": [0]}},
           "client_training": [{"client": 2, "optimizer_steps": 16, "train_seconds": 1.2}],
           "cost": {"wall": 2.}, "cumulative_cost": {"wall": 2.},
           "gpu_sharing": "exclusive", "budget_stop_round": None}
    changed_clock = {**row, "cost": {"wall": 3.}, "budget_stop_round": 1,
                     "client_training": [{"client": 2, "optimizer_steps": 16, "train_seconds": 9.}]}
    assert scientific_rows([row]) == scientific_rows([changed_clock])
    tensor = torch.tensor([1., 2.])
    old = {"round": 2, "adapter": {"q": tensor}, "torch_rng": torch.tensor([1, 2], dtype=torch.uint8),
           "cuda_rng": [torch.tensor([3], dtype=torch.uint8)], "python_rng": (1, (2,)),
           "cache": {"client": "v2"}, "deployed_bytes": 100, "metrics": [row, row]}
    new = {**old, "adapter": {"q": tensor.clone()}, "budget_stop_round": 2,
           "diagnostic_bytes": 0}
    returned = {"round": 2, "adapter": {"q": tensor.clone()}, "deployed_bytes": 100}
    groups = {name: 1. for name in ("task_initialization", "state_and_optimizer_setup",
        "client_training", "scoring_and_shared_information", "coordinate_and_server_decision",
        "message_processing", "budget_and_return_control")}
    timing = {"bucket_category_seconds": {"deployed_prefix": {"necessary_algorithm": 7.}}}
    checks = compare_parity(old, new, returned, old["metrics"],
                            [changed_clock, changed_clock], timing, {"passes": True}, groups)
    assert all(checks.values())
    changed_support = [{**changed_clock, "support_trace": {"2": {"q": [1]}}}, changed_clock]
    with pytest.raises(AssertionError, match="scientific_round_rows_exact"):
        compare_parity(old, new, returned, old["metrics"], changed_support,
                       timing, {"passes": True}, groups)
    altered = {**new, "adapter": {"q": torch.tensor([1., 2.00001])}}
    with pytest.raises(AssertionError, match="adapter_exact"):
        compare_parity(old, altered, returned, old["metrics"],
                       [changed_clock, changed_clock], timing, {"passes": True}, groups)
    over_cap = [dict(changed_clock, round=number) for number in range(1, 202)]
    with pytest.raises(AssertionError, match="formal_200_round_cap"):
        compare_parity(old, new, returned, over_cap, over_cap,
                       timing, {"passes": True}, groups)


def test_prepare_only_builds_fourteen_isolated_conditions_without_gpu(tmp_path):
    run = tmp_path / "pcla-global-budget-20260928-v7"
    folder = run / "checks/summaries/full_20260928_01"
    folder.mkdir(parents=True)
    data = run / "data/pcla_v7"
    data.mkdir(parents=True)
    write(data / "train_roles.json", {"partitions": {str(seed): {} for seed in TECH_SEEDS}})
    rows = []
    for rid, family in CASES.items():
        generation = ("core02" if family == "core" else
                      "native04" if rid == "D107" else "native03")
        source = run / "sources" / generation / "SOURCE_MANIFEST.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        write(source, {"files": {}})
        config = run / "configs" / generation / f"{rid}_s401.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        write(config, {"job": {"recipe_id": rid, "seed": 401},
                       "recipe": {"recipe_id": rid}, "runtime": {"byte_cap": 2382653750}})
        rows.append({"job": {"recipe_id": rid, "seed": 401}, "config": str(config),
                     "config_sha256": sha(config), "source_manifest": str(source),
                     "source_manifest_sha256": sha(source)})
    validation = run / "sources/validation02/SOURCE_MANIFEST.json"
    validation.parent.mkdir(parents=True)
    write(validation, {"files": {}})
    write(folder / "runs.json", rows)
    write(folder / "output_hashes.json", {"runs.json": sha(folder / "runs.json")})
    instrumented = tmp_path / "instrumented"
    instrumented.mkdir()
    manifest = tmp_path / "instrumented_manifest.json"
    write(manifest, {"files": {}})
    out = tmp_path / "technical_plan"
    args = argparse.Namespace(device="cuda", seeds=list(TECH_SEEDS), cases=list(CASES), out=out,
        run_root=run, instrumented_source_root=instrumented,
        instrumented_source_manifest=manifest, prepare_only=True)
    launch(args)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["schema"] == SCHEMA and plan["accepted_cost_receipt"] is False
    assert len(plan["conditions"]) == 14
    assert {(x["recipe_id"], x["seed"]) for x in plan["conditions"]} == {
        (rid, seed) for rid in CASES for seed in TECH_SEEDS}
    assert all(Path(x["config"]).is_file() for x in plan["conditions"])
    with pytest.raises(ValueError, match="fresh"):
        launch(args)
    shard = tmp_path / "shard"
    args.out = shard
    args.cases = ["D011", "D035"]
    launch(args)
    subset = json.loads((shard / "plan.json").read_text())["conditions"]
    assert {(item["recipe_id"], item["seed"]) for item in subset} == {
        (rid, seed) for rid in args.cases for seed in TECH_SEEDS}
    args.out = tmp_path / "duplicate"
    args.cases = ["D011", "D011"]
    with pytest.raises(ValueError, match="distinct"):
        launch(args)
