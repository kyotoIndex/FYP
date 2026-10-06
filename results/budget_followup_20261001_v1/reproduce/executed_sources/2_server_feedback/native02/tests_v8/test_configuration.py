import json
import os
from pathlib import Path

import pytest

from pcla_research_v8.configuration import compile_development, verified_recipes, validate_data_manifest


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = Path(os.getenv("PCLA_V8_RUN_ROOT", WORKSPACE.parents[1]))
PACKAGE = Path(os.getenv("PCLA_V8_PACKAGE", ROOT.parents[1] / "handoffs/PCLA_PostV_Received_20260929/attachment/PCLA_PostV_Research_20260929_v8"))


def test_full_dev_matrix_preserves_recipe_and_hardware_override(tmp_path):
    common = dict(common_path=PACKAGE / "configs/common.json", recipes_path=PACKAGE / "configs/recipes.json",
                  jobs_path=PACKAGE / "jobs/development.csv", authorization_path=ROOT / "scope_authorization.json",
                  data_dir=tmp_path, train_file=tmp_path / "SST-2/train.tsv", token_cache=tmp_path / "tokens.npz",
                  model_path=tmp_path / "model", source_manifest=WORKSPACE / "SOURCE_MANIFEST.json",
                  output_root=tmp_path / "results")
    configs = compile_development(**common)
    assert len(configs) == 128
    assert {x["job"]["seed"] for x in configs} == {503, 509, 521, 541}
    assert {x["job"]["hardware_family"] for x in configs} == {"RTX4090D"}
    assert all(x["binding"]["status"] == "blocked_missing_data_binding" for x in configs)
    assert all(x["launch_status"] == "blocked_pending_integration_and_audit" for x in configs)
    assert configs[0]["recipe"] == verified_recipes(PACKAGE / "configs/recipes.json")["U001"]
    assert configs[0]["job"]["protocol"] == configs[0]["common"]["protocol"]


def test_role_manifest_checks_actual_schema_and_train_binding(tmp_path):
    train = tmp_path / "SST-2/train.tsv"
    train.parent.mkdir()
    train.write_text("sentence\tlabel\n")
    import hashlib
    manifest = {"source_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                "partitions": {"503": {"client_train": {}, "client_local_readout": {}}},
                "roles": {k: [] for k in ("client_pool", "public_training", "cohort_select", "cohort_readout", "recipe_dev")}}
    path = tmp_path / "train_roles.json"
    path.write_text(json.dumps(manifest))
    assert len(validate_data_manifest(path, seed=503, train_file=train)) == 64
    manifest["roles"].pop("public_training")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing public"):
        validate_data_manifest(path, seed=503, train_file=train)
