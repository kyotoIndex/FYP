from collections import Counter
from pathlib import Path
import json
import os

import numpy as np
import pytest

from pcla_research_v7.data import make_manifest, role_ids, ROLE_SIZES
from pcla_research_v7.transport import (TensorSpec, WireLedger, b8_wire_manifest,
                                        decode_packet, encode_packet, packet_length)
from pcla_research_v7.configuration import compile_first_batch, compile_recipe

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = Path(os.environ.get("PCLA_V7_TRAIN_FILE", ROOT / "data/SST-2/train.tsv"))
HANDOFF = Path(os.environ.get("PCLA_V7_HANDOFF", ROOT.parents[1] / "handoffs" /
                              "PCLA_Longterm_Research_20260928_v7"))


def test_train_only_roles_and_three_seed_partitions():
    manifest = make_manifest(TRAIN_FILE, (401, 409, 419))
    assert sum(map(len, manifest["roles"].values())) == 23680
    assert {k: len(v) for k, v in manifest["roles"].items()} == ROLE_SIZES
    assert manifest["original_sentence_id_available"] is False
    assert manifest["split_sha256"] == make_manifest(TRAIN_FILE, (401, 409, 419))["split_sha256"]
    assert manifest["role_label_counts"] == manifest["role_label_targets"]
    assert manifest["stratification_exceptions"] == []
    ids = [x for role in manifest["roles"].values() for x in role]
    assert len(ids) == len(set(ids))
    groups = manifest["row_groups"]
    assert len({groups[x] for x in ids}) == sum(len(set(groups[x] for x in role)) for role in manifest["roles"].values())
    pool_counts = Counter(manifest["row_labels"][x] for x in manifest["roles"]["client_pool"])
    for seed, partition in manifest["partitions"].items():
        train, readout = partition["client_train"], partition["client_local_readout"]
        assert len(train) == len(readout) == 50
        assert all(len(x) == 256 for x in train.values())
        assert all(len(x) == 64 for x in readout.values())
        all_ids = [x for rows in [*train.values(), *readout.values()] for x in rows]
        assert set(all_ids) == set(manifest["roles"]["client_pool"])
        assert len(all_ids) == len(set(all_ids))
        assert Counter(manifest["row_labels"][x] for x in all_ids) == pool_counts
        for cid in train:
            assert not {groups[x] for x in train[cid]} & {groups[x] for x in readout[cid]}
            positive = sum(manifest["row_labels"][x] for x in train[cid] + readout[cid])
            expected_readout = round(positive / 5)
            assert sum(manifest["row_labels"][x] for x in readout[cid]) == expected_readout
    with pytest.raises(PermissionError):
        role_ids(manifest, "recipe_dev", "algorithm")
    with pytest.raises(PermissionError):
        role_ids(manifest, "cohort_readout", "finite_bank_search")


def test_wire_roundtrip_size_and_budget():
    tensors = {"a": np.arange(6, dtype="float32").reshape(2, 3),
               "b": np.array([7, 8], dtype="int32")}
    raw = encode_packet("delta_upload", 4, 17, tensors, {"factor": "B"}, [0, 5, 9], 12)
    assert len(raw) == packet_length("delta_upload", 4, 17, tensors,
                                     {"factor": "B"}, [0, 5, 9], 12)
    header, restored = decode_packet(raw)
    assert header["version"] == 4 and header["mask_indices"] == [0, 5, 9]
    for key in tensors:
        assert np.array_equal(restored[key], tensors[key])
    specs = {key: TensorSpec(tuple(value.shape), value.dtype.name) for key, value in tensors.items()}
    assert len(raw) == packet_length("delta_upload", 4, 17, specs,
                                     {"factor": "B"}, [0, 5, 9], 12)
    with pytest.raises(ValueError):
        decode_packet(raw[:-1])
    b8 = b8_wire_manifest()
    assert len(b8["rounds"]) == 100 and b8["logical_clients"] == 50
    assert b8["total_bytes"] == sum(x["total_bytes"] for x in b8["rounds"])
    ledger = WireLedger(100)
    assert ledger.admit(1, [40, 60])
    ledger.record("1-down", 1, "download", 40)
    ledger.record("1-up", 1, "upload", 60)
    assert not ledger.admit(2, [1]) and ledger.budget_stop_round == 1
    ledger.record("2-diagnostic", 2, "control", 3, diagnostic=True)
    assert ledger.deployed_prefix == 100 and ledger.diagnostic_after_budget == 3
    with pytest.raises(ValueError):
        ledger.record("1-down", 2, "download", 1)


def test_frozen_recipe_matrix_and_reject_unknown_fields():
    jobs = compile_first_batch(HANDOFF / "configs/recipes.json", HANDOFF / "jobs/first_batch.jsonl")
    assert len(jobs) == 351
    assert len({x["recipe"]["recipe_sha256"] for x in jobs}) == 117
    changed = dict(jobs[0]["recipe"])
    changed.pop("recipe_sha256")
    changed["unused_knob"] = 1
    with pytest.raises(ValueError, match="unused"):
        compile_recipe(changed)
    native = next(x["recipe"] for x in jobs if x["recipe"]["required_adapter"] != "core")
    native.pop("recipe_sha256")
    with pytest.raises(NotImplementedError):
        compile_recipe(native, supported_adapters={"core"})
