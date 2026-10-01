"""Small CPU runtime wiring tests; these are technical donors, not V evidence."""
from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
import random

import pytest
import torch

from pcla_repair.testing import fixture, TinyDataset
from pcla_repair.core import keyed_seed
from pcla_research_v7.runtime import run_experiment, _load_checkpoint
import pcla_research_v7.conditional_replay as conditional_replay
from scripts.run_pcla_v7 import load_quota_replay_binding


IDENTITIES = {"technical": "a" * 64}


def _job(rep: str):
    rid = "D011" if rep == "D" else "D035"
    return {"job_id": f"V_S16_{rid}_s101", "protocol": "pcla-global-budget-20260928-v7",
            "phase": "V", "recipe_id": rid, "method": "local",
            "representation": rep, "rank": 4, "seed": 101,
            "period": 1, "lr": .001, "min_rounds": 2, "max_rounds": 3}


def _config(cap=10**8):
    return {"byte_cap": cap, "clients_per_round": 2, "batch_size": 8,
            "eval_batch_size": 16, "local_epochs": 1,
            "probe_batch_size": 8, "probe_batches": 2}


def _book(rep, count=3):
    return SimpleNamespace(sha256="a" * 64,
                           rounds=tuple({} for _ in range(count)),
                           donor_job_id=f"V_S03_{'D011' if rep == 'D' else 'D035'}_s101",
                           donor_support_trace_sha256="b" * 64)


def _support(*, book, replay_job, round_id, cohort, module_ranks, expected_identities):
    assert book.sha256 == "a" * 64 and expected_identities == IDENTITIES
    assert set(module_ranks) and all(rank == 4 for rank in module_ranks.values())
    factor = "b" if round_id % 2 else "a"
    result = {}
    for cid in cohort:
        remaining = 2 if cid in {0, 1} else 4
        per_module = {}
        for name in sorted(module_ranks):
            count = min(remaining, 4)
            active = torch.zeros(4, dtype=torch.bool)
            active[4-count:] = True
            per_module[name] = {"a": active if factor == "a" else torch.zeros_like(active),
                                "b": active if factor == "b" else torch.zeros_like(active)}
            remaining -= count
        result[cid] = per_module
    return result


def _inputs(seed=101):
    model, _, clients, capacities, evaluation = fixture(seed)
    clients = {cid: TinyDataset(300 + cid, 40) for cid in clients}
    return model, clients, capacities, evaluation


def _run(tmp_path, rep, *, cap=10**8, stop_after=None, replay=True):
    model, clients, capacities, evaluation = _inputs()
    return run_experiment(model, clients, evaluation, capacities, _job(rep), _config(cap),
        tmp_path, {"evaluation_role": "synthetic"}, stop_after=stop_after,
        replay_book=_book(rep) if replay else None,
        replay_expected_identities=IDENTITIES if replay else None)


def test_D_replay_resume_and_B8_return_use_actual_replay_wire(tmp_path, monkeypatch):
    monkeypatch.setattr(conditional_replay, "replay_support", _support)
    full = _run(tmp_path / "full", "D")
    _run(tmp_path / "resumed", "D", stop_after=1)
    resumed = _run(tmp_path / "resumed", "D")
    first, second = (_load_checkpoint(tmp_path / folder / "latest.pt")
                     for folder in ("full", "resumed"))
    assert len(full) == len(resumed) == 3
    assert first["deployed_bytes"] == second["deployed_bytes"]
    assert all(torch.equal(first["adapter"][key], second["adapter"][key])
               for key in first["adapter"])
    assert [row["support_trace"] for row in full] == [row["support_trace"] for row in resumed]
    one_round = _run(tmp_path / "one", "D", stop_after=1)[0]
    capped = _run(tmp_path / "capped", "D", cap=one_round["deployed_bytes"] + 1)
    ck = _load_checkpoint(tmp_path / "capped/latest.pt")
    assert ck["budget_stop_round"] == 1
    assert ck["deployed_bytes"] == one_round["deployed_bytes"]
    assert ck["diagnostic_bytes"] > 0
    returned = torch.load(tmp_path / "capped/budget_return.pt", weights_only=True)
    early = _load_checkpoint(tmp_path / "one/latest.pt")
    assert all(torch.equal(returned["adapter"][key], early["adapter"][key])
               for key in returned["adapter"])
    assert all(row["quota_replay"]["coordinate_path"] == "D" for row in capped)


def test_T_replay_keeps_original_control_probe_and_cache_before_mask_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(conditional_replay, "replay_support", _support)
    original = _run(tmp_path / "original", "T", stop_after=2, replay=False)[-1]
    replaced = _run(tmp_path / "replay", "T", stop_after=2)[-1]
    assert original["coordinate_refresh"]["phase"] == replaced["coordinate_refresh"]["phase"]
    assert original["coordinate_refresh"]["probe_examples"] == replaced["coordinate_refresh"]["probe_examples"]
    assert replaced["quota_replay"]["T_original_control_and_probe_preserved"] is True
    assert original["support_trace"] != replaced["support_trace"]
    a = _load_checkpoint(tmp_path / "original/latest.pt")["cache"]["events"]
    b = _load_checkpoint(tmp_path / "replay/latest.pt")["cache"]["events"]
    original_control = [(event["event_id"], event["bytes"]) for event in a
                        if "coord_parent" in event["event_id"] or "sketch" in event["event_id"]]
    replay_control = [(event["event_id"], event["bytes"]) for event in b
                      if "coord_parent" in event["event_id"] or "sketch" in event["event_id"]]
    assert original_control and original_control == replay_control


def test_nonreplay_D_never_calls_quota_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(conditional_replay, "replay_support",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected replay")))
    rows = _run(tmp_path / "ordinary", "D", stop_after=1, replay=False)
    assert len(rows) == 1 and "quota_replay" not in rows[0]


def test_replay_requires_exact_V_identity_before_creating_output(tmp_path):
    model, clients, capacities, evaluation = _inputs()
    with pytest.raises(ValueError, match="independent R-local V"):
        run_experiment(model, clients, evaluation, capacities,
            {**_job("D"), "phase": "D0"}, _config(), tmp_path / "bad",
            {"evaluation_role": "synthetic"}, replay_book=_book("D"),
            replay_expected_identities=IDENTITIES)
    assert not (tmp_path / "bad").exists()


def test_missing_T_donor_suffix_stops_before_control_wire(tmp_path, monkeypatch):
    monkeypatch.setattr(conditional_replay, "replay_support", _support)
    model, clients, capacities, evaluation = _inputs()
    kwargs = dict(replay_book=_book("T", count=1),
                  replay_expected_identities=IDENTITIES)
    run_experiment(model, clients, evaluation, capacities, _job("T"), _config(),
        tmp_path / "replay", {"evaluation_role": "synthetic"}, stop_after=1, **kwargs)
    model, clients, capacities, evaluation = _inputs()
    with pytest.raises(conditional_replay.DonorSuffixRequired) as failure:
        run_experiment(model, clients, evaluation, capacities, _job("T"), _config(),
            tmp_path / "replay", {"evaluation_role": "synthetic"}, **kwargs)
    assert (failure.value.requested_round, failure.value.available_through) == (2, 1)
    assert _load_checkpoint(tmp_path / "replay/latest.pt")["round"] == 1
    journal = json.loads((tmp_path / "replay/communication_journal.json").read_text())
    assert all(entry["round"] <= 1 for entry in journal["entries"])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _real_file_binding_fixture(tmp_path):
    """Synthetic files exercise the same original-file hash gate as V CLI."""
    names = tuple(f"bert.encoder.layer.{layer}.attention.self.{side}"
                  for layer in range(12) for side in ("query", "value"))
    capacities = {cid: (96, 128, 192)[cid % 3] for cid in range(50)}
    donor_job = {"job_id": "V_S03_D035_s101", "protocol": "pcla-global-budget-20260928-v7",
                 "phase": "V", "recipe_id": "D035", "seed": 101, "method": "local",
                 "representation": "T", "period": 5, "rank": 16, "lr": .001,
                 "min_rounds": 100, "max_rounds": 200, "budget_id": "B8",
                 "eval_role": "recipe_dev"}
    replay_job = {**donor_job, "job_id": "V_S16_D035_s101"}
    cohort = random.Random(keyed_seed(101, "participants", 1)).sample(list(range(50)), 10)
    trace = {}
    for cid in cohort:
        per, extra = divmod(capacities[cid], len(names))
        modules = {name: {"a": [], "b": list(range(per + (i < extra)))}
                   for i, name in enumerate(names)}
        trace[str(cid)] = {"assigned_quota": capacities[cid],
                           "selected_positions": capacities[cid],
                           "active_factor": "b", "modules": modules}
    rows = [{"round": 1, "selected_clients": cohort, "support_trace": trace}]
    donor_root = tmp_path / "donor"
    donor_root.mkdir()
    source_path = tmp_path / "source.json"
    data_path = tmp_path / "data.json"
    _json(source_path, {"files": {}})
    _json(data_path, {"role": "synthetic"})
    common = {"recipe": {"required_adapter": "core"}, "manifest": str(data_path),
              "train_file": "train.tsv", "token_cache": "tokens.pt",
              "b8_wire": "b8.json", "model": {"repository": "synthetic"}}
    donor_path = tmp_path / "donor_config.json"
    _json(donor_path, {**common, "job": donor_job,
                       "conditional_validation": {"slot": 3}})
    metrics_path = donor_root / "metrics.json"
    _json(metrics_path, rows)
    initial_sha = "d" * 64
    model_sha = "e" * 64
    identities = {"donor_config_file_sha256": _sha(donor_path),
                  "source_manifest_sha256": _sha(source_path),
                  "data_manifest_sha256": _sha(data_path),
                  "initial_state_sha256": initial_sha,
                  "raw_metrics_file_sha256": _sha(metrics_path)}
    book = conditional_replay.freeze_donor_trace(
        donor_job=donor_job, rows=rows, module_names=names,
        capacities=capacities, identities=identities)
    originals = {
        "donor_config": donor_path, "source_manifest": source_path,
        "data_manifest": data_path, "metrics": metrics_path,
        "initial_identity": donor_root / "initial_state_identity.json",
        "run_identity": donor_root / "run_identity.json",
        "status": donor_root / "status.json",
        "donor_quota_seal": donor_root / "donor_quota_seal.json",
    }
    _json(originals["initial_identity"], {
        "schema": "pcla-v7-R-local-initial-state-1", "job_id": donor_job["job_id"],
        "seed": 101, "recipe_id": "D035", "config_sha256": identities["donor_config_file_sha256"],
        "source_manifest_sha256": identities["source_manifest_sha256"],
        "data_manifest_sha256": identities["data_manifest_sha256"],
        "pretrained_model_sha256": model_sha, "initial_state_sha256": initial_sha})
    _json(originals["run_identity"], {"job": donor_job, "provenance": {
        "config_sha256": identities["donor_config_file_sha256"],
        "source_manifest_sha256": identities["source_manifest_sha256"],
        "manifest_sha256": identities["data_manifest_sha256"],
        "pretrained_model_sha256": model_sha}})
    _json(originals["status"], {"state": "completed", "completed_round": 1})
    _json(originals["donor_quota_seal"], {
        "schema": "pcla-v7-R-local-donor-trace-seal-1", "job_id": donor_job["job_id"],
        "completed_round": 1, "identities": identities,
        "donor_support_trace_sha256": book.donor_support_trace_sha256,
        "donor_book_sha256": book.sha256})
    binding_path = tmp_path / "binding.json"
    _json(binding_path, {"schema": "pcla-v7-quota-replay-binding-1",
                         "files": {key: {"path": str(path), "sha256": _sha(path)}
                                   for key, path in originals.items()}})
    expanded = {**common, "job": replay_job,
                "conditional_validation": {"slot": 16,
                    "execution_policy": "quota_replay_v1",
                    "donor": {"job_id": donor_job["job_id"]}},
                "quota_replay": {"schema": "pcla-v7-quota-replay-config-1",
                    "binding_path": str(binding_path), "binding_sha256": _sha(binding_path)}}
    replay_path = tmp_path / "replay_config.json"
    _json(replay_path, expanded)
    provenance = {"config_sha256": _sha(replay_path),
                  "manifest_sha256": _sha(data_path),
                  "pretrained_model_sha256": model_sha}
    return expanded, replay_path, binding_path, capacities, names, initial_sha, provenance, originals


def test_cli_binding_rechecks_actual_donor_files_and_initial_state(tmp_path):
    args = _real_file_binding_fixture(tmp_path)
    expanded, config_path, binding_path, capacities, names, initial_sha, provenance, originals = args
    kwargs = dict(expanded=expanded, config_path=config_path,
                  binding_path=binding_path, capacities=capacities,
                  module_names=names, initial_sha=initial_sha, provenance=provenance)
    book, identities, binding_hash = load_quota_replay_binding(**kwargs)
    assert len(book.rounds) == 1 and identities == book.identities
    assert binding_hash == _sha(binding_path)
    with pytest.raises(ValueError, match="initialization/source identity"):
        load_quota_replay_binding(**{**kwargs, "initial_sha": "0" * 64})
    originals["metrics"].write_text(originals["metrics"].read_text() + "\n")
    with pytest.raises(ValueError, match="original hash differs: metrics"):
        load_quota_replay_binding(**kwargs)
