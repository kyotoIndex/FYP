"""CPU checks for sealed donor quotas and deterministic D/T replay masks."""
from __future__ import annotations

import copy
import random

import pytest
import torch

from pcla_repair.core import keyed_seed
from pcla_research_v7.conditional_replay import (
    DonorSuffixRequired, freeze_donor_trace, rank_ids, replay_support,
)


NAMES = tuple(f"bert.encoder.layer.{layer}.attention.self.{side}"
              for layer in range(12) for side in ("query", "value"))
CAPACITIES = {cid: (96, 128, 192)[cid % 3] for cid in range(50)}
IDENTITIES = {"donor_config_file_sha256": "a" * 64,
              "source_manifest_sha256": "b" * 64,
              "data_manifest_sha256": "c" * 64,
              "initial_state_sha256": "d" * 64,
              "raw_metrics_file_sha256": "e" * 64}


def jobs(recipe_id):
    representation = "D" if recipe_id == "D011" else "T"
    job = {"job_id": f"V_S03_{recipe_id}_s101",
           "protocol": "pcla-global-budget-20260928-v7", "phase": "V",
           "recipe_id": recipe_id, "seed": 101, "method": "local",
           "representation": representation, "period": 5, "rank": 16,
           "lr": .0003, "min_rounds": 100, "max_rounds": 200,
           "budget_id": "B8", "eval_role": "recipe_dev"}
    replay = dict(job, job_id=f"V_S16_{recipe_id}_s101")
    return job, replay


def rows(round_count=2):
    result = []
    for r in range(1, round_count + 1):
        cohort = random.Random(keyed_seed(101, "participants", r)).sample(list(range(50)), 10)
        factor = "b" if r % 2 else "a"
        trace = {}
        for cid in cohort:
            quota = CAPACITIES[cid]
            per, extra = divmod(quota, 24)
            modules = {}
            for index, name in enumerate(NAMES):
                ids = list(range(per + (index < extra)))
                modules[name] = {"a": ids if factor == "a" else [],
                                 "b": ids if factor == "b" else []}
            trace[str(cid)] = {"assigned_quota": quota,
                               "selected_positions": quota,
                               "active_factor": factor, "modules": modules}
        result.append({"round": r, "selected_clients": cohort,
                       "support_trace": trace})
    return result


def frozen(recipe_id="D011", count=2):
    donor, replay = jobs(recipe_id)
    raw = rows(count)
    book = freeze_donor_trace(donor_job=donor, rows=raw, module_names=NAMES,
                              capacities=CAPACITIES, identities=IDENTITIES)
    return book, replay, raw


@pytest.mark.parametrize("recipe_id", ["D011", "D035"])
def test_sealed_D_or_T_donor_counts_and_replay_are_deterministic(recipe_id):
    book, replay, raw = frozen(recipe_id)
    assert len(book.rounds) == 2 and len(book.modules) == 24
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    kwargs = dict(book=book, replay_job=replay,
                  module_ranks={name: 16 for name in NAMES},
                  expected_identities=IDENTITIES)
    for r in (1, 2):
        cohort = raw[r - 1]["selected_clients"]
        first = replay_support(round_id=r, cohort=cohort, **kwargs)
        resumed = replay_support(round_id=r, cohort=cohort, **kwargs)
        assert set(first) == set(cohort)
        for cid in cohort:
            assert sum(int(mask.sum()) for pair in first[cid].values()
                       for mask in pair.values()) == CAPACITIES[cid]
            factor = "b" if r % 2 else "a"
            for name in NAMES:
                expected_count = len(raw[r - 1]["support_trace"][str(cid)]["modules"][name][factor])
                assert int(first[cid][name][factor].sum()) == expected_count
                assert not bool(first[cid][name]["a" if factor == "b" else "b"].any())
                assert torch.equal(first[cid][name][factor], resumed[cid][name][factor])
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_reference_canonical_draw_and_zero_full_quotas():
    name = NAMES[0]
    assert rank_ids(seed=101, round_id=1, client=0, module=name, factor="b",
                    count=4, rank=16) == [0, 3, 10, 11]
    assert rank_ids(seed=101, round_id=1, client=0, module=name, factor="b",
                    count=0, rank=16) == []
    assert rank_ids(seed=101, round_id=1, client=0, module=name, factor="b",
                    count=16, rank=16) == list(range(16))
    with pytest.raises(ValueError, match="quota outside"):
        rank_ids(seed=101, round_id=1, client=0, module=name, factor="b",
                 count=17, rank=16)


@pytest.mark.parametrize("change", ["missing_round", "wrong_cohort", "wrong_factor",
                                     "missing_module", "duplicate_ids", "wrong_quota"])
def test_incomplete_or_corrupt_donor_trace_rejected(change):
    donor, _ = jobs("D011")
    raw = rows()
    cid = raw[0]["selected_clients"][0]
    if change == "missing_round":
        raw[1]["round"] = 3
    elif change == "wrong_cohort":
        raw[0]["selected_clients"] = raw[0]["selected_clients"][::-1]
    elif change == "wrong_factor":
        raw[0]["support_trace"][str(cid)]["active_factor"] = "a"
    elif change == "missing_module":
        raw[0]["support_trace"][str(cid)]["modules"].pop(NAMES[0])
    elif change == "duplicate_ids":
        raw[0]["support_trace"][str(cid)]["modules"][NAMES[0]]["b"] = [0, 0]
    else:
        raw[0]["support_trace"][str(cid)]["assigned_quota"] = 95
    with pytest.raises(ValueError):
        freeze_donor_trace(donor_job=donor, rows=raw, module_names=NAMES,
                           capacities=CAPACITIES, identities=IDENTITIES)


def test_missing_suffix_and_wrong_replay_identity_refuse_masks():
    book, replay, raw = frozen("D035", count=1)
    kwargs = dict(book=book, replay_job=replay,
                  module_ranks={name: 16 for name in NAMES},
                  expected_identities=IDENTITIES)
    with pytest.raises(DonorSuffixRequired) as failure:
        replay_support(round_id=2, cohort=rows()[1]["selected_clients"], **kwargs)
    assert failure.value.available_through == 1
    with pytest.raises(ValueError, match="identity differs"):
        replay_support(round_id=1, cohort=raw[0]["selected_clients"],
                       **{**kwargs, "expected_identities": {
                           **IDENTITIES, "initial_state_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="resolved R-local"):
        replay_support(round_id=1, cohort=raw[0]["selected_clients"],
                       **{**kwargs, "replay_job": {**replay, "seed": 109}})
    with pytest.raises(ValueError, match="round/cohort"):
        replay_support(round_id=1, cohort=raw[0]["selected_clients"][::-1], **kwargs)
    changed = copy.deepcopy(book)
    changed.rounds[0]["counts"][str(raw[0]["selected_clients"][0])][NAMES[0]] += 1
    with pytest.raises(ValueError, match="hash/schema"):
        replay_support(round_id=1, cohort=raw[0]["selected_clients"],
                       **{**kwargs, "book": changed})
