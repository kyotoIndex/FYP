"""CPU-only contract checks for the fixed-parent cohort bank executor."""
import copy
from datetime import datetime, timezone

import pytest

from pcla_research_v7.diagnostic_bank import (
    PREFIXES, SUPPORTS, collect_cohort_bank, make_jq_lock, support_digest,
)


CLIENTS = tuple(range(10))
PARENT = {"projection.lora_a": 2.0, "projection.lora_b": 3.0,
          "classifier.weight": 5.0, "classifier.bias": 7.0}


def setup_bank():
    supports = {}
    for cid in CLIENTS:
        supports[cid] = {}
        for index, name in enumerate(SUPPORTS):
            mask = [False] * 4
            mask[index] = True
            supports[cid][name] = {"projection": {"a": mask, "b": [False] * 4}}
    choices = {cid: ("local", "cyclic", "JL_h32")[cid % 3] for cid in CLIENTS}
    digests = {cid: support_digest(supports[cid][choices[cid]], factor="a")
               for cid in CLIENTS}
    lock = make_jq_lock(object_id="Cohort_s101_r10_a", participants=CLIENTS,
                        choice_by_client=choices, support_sha256_by_client=digests,
                        h32_sha256="a" * 64, locked_at_utc="2026-09-28T01:02:03Z",
                        information_access=("parent_state", "local_h32"))
    return supports, lock


def harness(*, train_override=None):
    supports, lock = setup_bank()
    model = copy.deepcopy(PARENT)
    rng = [91]
    calls = []
    def load_parent(target, parent):
        target.clear()
        target.update(copy.deepcopy(parent))
    def train(target, dataset, support, **kw):
        assert target == PARENT
        assert kw == {"seed": kw["seed"], "lr": .001, "batch_size": 16,
                      "epochs": 1, "betas": (.9, .999), "eps": 1e-8,
                      "weight_decay": .01, "capture_steps": PREFIXES}
        calls.append((dataset.client, support, kw["seed"]))
        rng[0] += 1
        if train_override:
            train_override(dataset.client, support)
        position = support["projection"]["a"].index(True) + 1
        states = {}
        for step in PREFIXES:
            states[step] = copy.deepcopy(PARENT)
            states[step]["projection.lora_a"] += step * position
            states[step]["classifier.weight"] += step * (dataset.client + 1)
            states[step]["classifier.bias"] += step
        target.update(states[16])
        return {"optimizer_steps": 16, "prefix_states": states}
    def aggregate(parent, children, weights):
        assert weights == [1] * 10
        return {key: parent[key] + sum((child[key] - parent[key]) * weight
                                       for child, weight in zip(children, weights)) / sum(weights)
                for key in parent}
    def snapshot():
        return rng[0]
    def restore(value):
        rng[0] = value
    class Dataset:
        def __init__(self, client):
            self.client = client
        def __len__(self):
            return 256
    history = [{"round": round_id, "selected_clients": list(CLIENTS)}
               for round_id in range(1, 11)]
    kwargs = dict(object_id="Cohort_s101_r10_a", anchor_sha256="b" * 64,
                  anchor_identity="V_S10_s101-r10", factor="a", participants=CLIENTS,
                  visits={cid: 10 for cid in CLIENTS}, anchor_metrics=history,
                  parent_state=PARENT,
                  datasets={cid: Dataset(cid) for cid in CLIENTS}, supports=supports,
                  jq_lock=lock, model=model, load_parent=load_parent,
                  local_train=train, aggregate=aggregate,
                  capture_rng=snapshot, restore_rng=restore, cache={"clients": {}})
    return kwargs, calls, rng


def test_collects_same_parent_paired_seeds_three_prefixes_and_fixed_denominator():
    kwargs, calls, rng = harness()
    result = collect_cohort_bank(**kwargs)
    assert len(calls) == 40
    assert [name for cid, support, _seed in calls for name in SUPPORTS
            if support is kwargs["supports"][cid][name]] == list(SUPPORTS) * 10
    for cid in CLIENTS:
        assert len({seed for client, _, seed in calls if client == cid}) == 1
    assert result["jq_lock_sha256"] == kwargs["jq_lock"].sha256
    assert result["anchor_sha256"] == kwargs["anchor_sha256"]
    assert result["visits"] == kwargs["visits"]
    assert result["aggregate_states"]["local"][16]["projection.lora_a"] == 18.0
    assert result["aggregate_states"]["local"][16]["classifier.weight"] == 93.0
    assert result["aggregate_states"]["local"][16]["classifier.bias"] == 23.0
    assert result["aggregate_states"]["local"][16]["projection.lora_b"] == 3.0
    expected_jq = 2 + sum(16 * ((cid % 3) + 1) for cid in CLIENTS) / 10
    assert result["aggregate_states"]["JQ_xi1"][16]["projection.lora_a"] == expected_jq
    assert set(result["aggregate_states"]["JQ_xi1"]) == set(PREFIXES)
    assert kwargs["model"] == PARENT and rng == [91]


@pytest.mark.parametrize("fault", ["wrong_lock_mask", "missing_support", "wrong_rows", "missing_head",
                                     "wrong_visits", "readout_access"])
def test_rejects_invalid_inputs_before_any_training(fault):
    kwargs, calls, _ = harness()
    if fault == "wrong_lock_mask":
        kwargs["supports"][0]["local"]["projection"]["a"] = [False, True, False, False]
    elif fault == "missing_support":
        del kwargs["supports"][0]["public512_linear"]
    elif fault == "wrong_rows":
        class Short:
            def __len__(self):
                return 255
        kwargs["datasets"][0] = Short()
    elif fault == "missing_head":
        kwargs["parent_state"] = {k: v for k, v in PARENT.items() if k != "classifier.bias"}
    elif fault == "wrong_visits":
        kwargs["visits"][0] = 9
    else:
        with pytest.raises(ValueError, match="pretraining h32"):
            make_jq_lock(object_id="Cohort_s101_r10_a", participants=CLIENTS,
                         choice_by_client=kwargs["jq_lock"].choice_by_client,
                         support_sha256_by_client=kwargs["jq_lock"].support_sha256_by_client,
                         h32_sha256="a" * 64,
                         locked_at_utc=datetime.now(timezone.utc).isoformat(),
                         information_access=("local_h32", "bank_outcomes"))
        return
    with pytest.raises(ValueError):
        collect_cohort_bank(**kwargs)
    assert calls == []


def test_training_failure_or_mask_mutation_restores_parent_and_rng():
    for kind in ("failure", "mutation"):
        def bad_train(client, support):
            if client == 0 and support is kwargs["supports"][0]["cyclic"]:
                if kind == "failure":
                    raise RuntimeError("optimizer failure")
                support["projection"]["a"][1] = False
        kwargs, calls, rng = harness(train_override=bad_train)
        with pytest.raises((RuntimeError, ValueError)):
            collect_cohort_bank(**kwargs)
        assert len(calls) == 2
        assert kwargs["model"] == PARENT and rng == [91]
