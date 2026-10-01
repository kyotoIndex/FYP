import copy
import json
from pathlib import Path

import pytest

from pcla_research_v8.diagnostic_configuration import compile_diagnostics


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parents[1]
PACKAGE = ROOT.parents[1] / "handoffs/PCLA_PostV_Received_20260929/attachment/PCLA_PostV_Research_20260929_v8"
INDEX = ROOT / "checks/diagnostic_intake01/INPUT_INDEX.json"


def arguments(index=INDEX):
    return dict(common_path=PACKAGE / "configs/common.json",
                jobs_path=PACKAGE / "jobs/diagnostics.csv", input_index_path=index,
                authorization_path=ROOT / "scope_authorization.json",
                output_root="$HPC_HOME/pcla_realized_update_feedback_20260930_v8/results/diagnostic01")


def test_full_matrix_binds_exact_inputs_and_separates_roles():
    result = compile_diagnostics(**arguments())
    assert len(result) == len({c["job"]["job_id"] for c in result}) == 120
    assert len({c["output_dir"] for c in result}) == 120
    assert {c["job"]["hardware_family"] for c in result} == {"RTX4090D"}
    assert {c["launch_status"] for c in result} == {"blocked_pending_runtime_integration_and_on_node_preflight"}
    bank = [c for c in result if c["job"]["phase"] == "BANK_READOUT"]
    transfer = [c for c in result if c["job"]["phase"] == "R8_TRANSFER"]
    assert len(bank) == 96 and len(transfer) == 24
    assert sum(c["job"]["new_client_training_calls"] for c in result) == 240
    assert sum(c["job"]["new_client_steps"] for c in result) == 3840
    assert {c["source"]["object_id"] for c in bank} == {
        x["object_id"] for x in json.loads(INDEX.read_text())["cohort_bank_objects"]}
    assert all(len(c["source"]["bank_sha256_historical_expected"]) == 64 for c in bank)
    assert all(c["source"]["parent_pt_sha256_historical_expected"] is None for c in transfer)
    assert all("current_not_historical" in c["source"]["parent_pt_sha256_policy"] for c in transfer)
    assert all(c["source"]["training"]["lr"] == 0.001 and
               c["source"]["training"]["head_lr"] == 0.001 for c in transfer)
    assert all("old_finite_bank_reference" in c["controller"]["candidate_rules"] for c in bank)
    assert all("old_finite_bank_reference" not in c["controller"]["candidate_rules"] for c in transfer)
    assert all(c["data"]["selection_role"] == "public_training_512" and
               c["data"]["readout_role"] == "cohort_readout_2048" and
               c["data"]["readout_must_not_select_or_fit"] for c in result)
    assert all(c["model"]["new_repository"] is None for c in result)


def test_intake_tampering_and_missing_parent_are_rejected(tmp_path):
    original = json.loads(INDEX.read_text())
    changed = copy.deepcopy(original)
    changed["cohort_bank_objects"][0]["representation"] = "wrong"
    path = tmp_path / "index.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="representation mismatch"):
        compile_diagnostics(**arguments(index=path))
    changed = copy.deepcopy(original)
    changed["d066_parent_checkpoints"][0]["parent_checkpoint"]["exists"] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="D066 parent checkpoint"):
        compile_diagnostics(**arguments(index=path))
