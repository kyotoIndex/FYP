"""Independent fabricated evidence; no training loop is used by these tests."""
import json
import hashlib
import math
from pathlib import Path
import random
import subprocess
import sys

import torch
import pytest

from scripts.audit_pcla_v7_run import audit, canonical_hash, sha, stable_seed, OFFICIAL_COMMITS


def put(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def evidence(root: Path):
    source, run, data = root / "source", root / "run", root / "data"
    source.mkdir(); run.mkdir(); data.mkdir()
    (source / "x.py").write_text("version = 1\n")
    put(source / "SOURCE_MANIFEST.json", {"files": {"x.py": sha(source / "x.py")}})
    train = data / "SST-2" / "train.tsv"
    train.parent.mkdir()
    train.write_text("sentence\tlabel\n")
    ids = [f"train:{i}" for i in range(4096)]
    labels = [i % 2 for i in range(4096)]
    roles = {"protocol": "pcla-global-budget-20260928-v7",
             "source_sha256": sha(train), "roles": {"recipe_dev": ids},
             "row_labels": dict(zip(ids, labels))}
    roles["split_sha256"] = canonical_hash(roles)
    role_file = data / "train_roles.json"
    put(role_file, roles)
    cache = data / "train_tokens.npz"
    cache.write_bytes(b"dummy-token-cache")
    put(data / "train_tokens.json", {"cache_sha256": sha(cache),
        "identity": {"split_sha256": roles["split_sha256"]}})
    b8 = {"total_bytes": 2000, "definition": "test"}
    b8["wire_manifest_sha256"] = canonical_hash(b8)
    b8_file = data / "b8_wire.json"
    put(b8_file, b8)
    job = {"job_id": "D-test", "protocol": roles["protocol"],
           "eval_role": "recipe_dev", "method": "full_both", "rank": 4, "seed": 401,
           "phase": "D0", "min_rounds": 100, "max_rounds": 200}
    config = {"job": job, "runtime": {"byte_cap": 2000},
              "manifest": str(role_file), "b8_wire": str(b8_file),
              "train_file": str(train), "token_cache": str(cache)}
    config["expanded_sha256"] = canonical_hash(config)
    config_file = root / "config.json"
    put(config_file, config)
    provenance = {"config_sha256": sha(config_file),
                  "source_manifest_sha256": sha(source / "SOURCE_MANIFEST.json"),
                  "manifest_sha256": sha(role_file), "split_sha256": roles["split_sha256"],
                  "source_sha256": sha(train), "token_cache_sha256": sha(cache),
                  "b8_wire_manifest_sha256": sha(b8_file)}
    put(run / "run_identity.json", {"job": job, "provenance": provenance})
    put(run / "execution_attempts.json", [{"identity": "test-identity"}])
    put(run / "evaluation_labels.json", labels)
    logits = [[0., 1.]] * 4096
    accuracy = .5
    loss = .5 * (math.log1p(math.exp(1)) + math.log1p(math.exp(-1)))
    readout = {"accuracy": accuracy, "loss": loss, "samples": 4096}
    put(run / "evaluation_round_000.json", readout)
    for r in range(101):
        put(run / f"logits_round_{r:03d}.json", logits)
    modules = [f"bert.encoder.layer.{layer}.attention.self.{target}"
               for layer in range(12) for target in ("query", "value")]
    adapter = {f"{name}.lora_{factor}": torch.zeros(4, 1)
               for name in modules for factor in ("a", "b")}
    assigned = [96] * 30 + [128] * 15 + [192] * 5
    random.Random(stable_seed(job["seed"], "resource_assignment")).shuffle(assigned)
    events, rows = [], []
    for r in range(1, 101):
        selected = random.Random(stable_seed(job["seed"], "participants", r)).sample(list(range(50)), 10)
        for cid in selected:
            for direction in ("down", "up"):
                events.append({"event_id": f"r{r}:c{cid}:{direction}", "round": r,
                               "direction": "download" if direction == "down" else "upload",
                               "bucket": "deployed_prefix", "bytes": 1})
        support = {name: {"a": list(range(4)), "b": list(range(4))} for name in modules}
        trace = {str(cid): {"assigned_quota": assigned[cid], "selected_positions": 192,
                            "active_factor": "both", "modules": support} for cid in selected}
        rows.append({"round": r, "selected_clients": selected,
                     "deployed_bytes": 20 * r, "diagnostic_after_budget_bytes": 0,
                     "evaluation": readout, "support_trace": trace,
                     "client_training": [{"client": cid, "optimizer_steps": 16,
                                          "optimizer_tensor_parameters": 192 * 768 + 1538}
                                         for cid in selected]})
    ck = {"format": "pcla-v7-runtime-1", "round": 100, "identity": "test-identity",
          "metrics": rows, "adapter": adapter, "budget_adapter": adapter,
          "half_adapter": adapter, "budget_stop_round": 100,
          "deployed_bytes": 2000, "diagnostic_bytes": 0,
          "cache": {"events": events}}
    torch.save(ck, run / "latest.pt")
    put(run / "metrics.json", rows)
    put(run / "status.json", {"state": "completed", "completed_round": 100,
         "budget_stop_round": 100, "deployed_bytes": 2000,
         "diagnostic_after_budget_bytes": 0})
    torch.save({"identity": "test-identity", "round": 100, "adapter": adapter,
                "deployed_bytes": 2000, "budget_cap_bytes": 2000}, run / "budget_return.pt")
    torch.save({"identity": "test-identity", "round": 50, "adapter": adapter,
                "deployed_bytes": 1000, "budget_cap_bytes": 1000,
                "readout": readout}, run / "half_budget_return.pt")
    return config_file, run, source


def test_complete_evidence_passes_then_corruptions_fail(tmp_path):
    config, run, source = evidence(tmp_path)
    result = audit(config, run, source)
    assert result["ok"], result["errors"]
    assert result["checks"]["budget_stop_round"] == 100
    assert result["checks"]["predicted_class_counts_round_100"] == [0, 4096]
    status = json.loads((run / "status.json").read_text())
    put(run / "status.json", {**status, "state": "paused"})
    assert not audit(config, run, source)["ok"]
    put(run / "status.json", status)
    logits = json.loads((run / "logits_round_100.json").read_text())
    logits[0] = [10., 0.]
    put(run / "logits_round_100.json", logits)
    assert any("round 100: accuracy mismatch" in x for x in audit(config, run, source)["errors"])


def test_journal_events_are_reconciled_and_unresolved_attempts_fail(tmp_path):
    config, run, source = evidence(tmp_path)
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    entries = []
    for r in range(1, 101):
        entries.append({"attempt": r, "round": r, "bucket": "deployed_prefix",
                        "status": "committed", "phase": "upload_started_conservative",
                        "sent_events": [e for e in ck["cache"]["events"] if e["round"] == r]})
    journal_path = run / "communication_journal.json"
    put(journal_path, {"format": "pcla-v7-communication-journal-1", "entries": entries})
    assert audit(config, run, source)["ok"]
    entries[0]["sent_events"][0]["bytes"] += 1
    put(journal_path, {"format": "pcla-v7-communication-journal-1", "entries": entries})
    assert any("differs from ledger" in e for e in audit(config, run, source)["errors"])
    entries[0]["sent_events"][0]["bytes"] -= 1
    entries[0]["status"] = "pending"
    put(journal_path, {"format": "pcla-v7-communication-journal-1", "entries": entries})
    assert any("unresolved pending" in e for e in audit(config, run, source)["errors"])


def test_technical_phase_and_short_or_paused_runs_never_pass(tmp_path):
    config, run, source = evidence(tmp_path)
    cfg = json.loads(config.read_text())
    cfg["job"]["phase"] = "TECH"
    cfg["technical_profile_only"] = True
    cfg["expanded_sha256"] = canonical_hash({k: v for k, v in cfg.items() if k != "expanded_sha256"})
    put(config, cfg)
    result = audit(config, run, source)
    assert not result["ok"]
    assert any("technical" in e for e in result["errors"])
    assert result["checks"]["last_finite_checkpoint"]["round"] == 100
    cfg["job"]["phase"] = "D0"
    cfg.pop("technical_profile_only")
    cfg["expanded_sha256"] = canonical_hash({k: v for k, v in cfg.items() if k != "expanded_sha256"})
    put(config, cfg)
    status = json.loads((run / "status.json").read_text())
    put(run / "status.json", {**status, "state": "paused", "completed_round": 2})
    assert not audit(config, run, source)["ok"]


def test_cli_writes_machine_json_outside_run_without_modifying_it(tmp_path):
    config, run, source = evidence(tmp_path)
    out = tmp_path / "audit.json"
    before = {p.name: sha(p) for p in run.iterdir() if p.is_file()}
    result = subprocess.run([sys.executable, "scripts/audit_pcla_v7_run.py",
        "--run", str(run), "--config", str(config), "--source", str(source),
        "--out", str(out)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out.read_text())["ok"]
    assert {p.name: sha(p) for p in run.iterdir() if p.is_file()} == before


def test_recovered_failed_attempt_bytes_reduce_budget_without_new_round(tmp_path):
    config, run, source = evidence(tmp_path)
    cfg = json.loads(config.read_text())
    b8_path = Path(cfg["b8_wire"])
    b8 = json.loads(b8_path.read_text())
    b8["total_bytes"] = 2005
    b8["wire_manifest_sha256"] = canonical_hash({k: v for k, v in b8.items() if k != "wire_manifest_sha256"})
    put(b8_path, b8)
    cfg["runtime"]["byte_cap"] = 2005
    cfg["expanded_sha256"] = canonical_hash({k: v for k, v in cfg.items() if k != "expanded_sha256"})
    put(config, cfg)
    identity = json.loads((run / "run_identity.json").read_text())
    identity["provenance"]["config_sha256"] = sha(config)
    identity["provenance"]["b8_wire_manifest_sha256"] = sha(b8_path)
    put(run / "run_identity.json", identity)
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    rows = json.loads((run / "metrics.json").read_text())
    for row in rows:
        row["deployed_bytes"] += 5
    ck["metrics"] = rows
    failure = {"event_id": "r1:c0:down", "round": 1, "direction": "download",
               "bucket": "deployed_prefix", "bytes": 5}
    ck["cache"]["events"].append({**failure, "event_id": "failed:a1:r1:c0:down"})
    ck["deployed_bytes"] = 2005
    torch.save(ck, run / "latest.pt")
    put(run / "metrics.json", rows)
    status = json.loads((run / "status.json").read_text())
    put(run / "status.json", {**status, "deployed_bytes": 2005})
    budget = torch.load(run / "budget_return.pt", map_location="cpu", weights_only=True)
    budget.update(deployed_bytes=2005, budget_cap_bytes=2005)
    torch.save(budget, run / "budget_return.pt")
    half = torch.load(run / "half_budget_return.pt", map_location="cpu", weights_only=True)
    half.update(round=49, deployed_bytes=985, budget_cap_bytes=1002)
    torch.save(half, run / "half_budget_return.pt")
    entries = [{"attempt": 1, "round": 1, "bucket": "deployed_prefix",
                "status": "recovered_failed", "phase": "control_and_download",
                "sent_events": [failure], "recovered_bytes": 5}]
    for r in range(1, 101):
        entries.append({"attempt": r + 1, "round": r, "bucket": "deployed_prefix",
                        "status": "committed", "phase": "upload_started_conservative",
                        "sent_events": [e for e in ck["cache"]["events"]
                                        if e["round"] == r and not e["event_id"].startswith("failed:")]})
    put(run / "communication_journal.json",
        {"format": "pcla-v7-communication-journal-1", "entries": entries})
    result = audit(config, run, source)
    assert result["ok"], result["errors"]
    assert result["checks"]["half_budget_round"] == 49
    entries[0]["recovered_bytes"] = 0
    put(run / "communication_journal.json",
        {"format": "pcla-v7-communication-journal-1", "entries": entries})
    assert any("recovered byte count differs" in e for e in audit(config, run, source)["errors"])


@pytest.mark.parametrize("method", ["fedsb", "fedsvd", "fedplora"])
def test_native_completed_evidence_and_support_tampering(tmp_path, method):
    config, run, source = evidence(tmp_path)
    cfg = json.loads(config.read_text())
    cfg["job"]["method"] = method
    recipe = {"method": method, "required_adapter": method, "rank": 4,
              "native_ranks": [2, 3, 4] if method == "fedplora" else None,
              "period": 1}
    cfg["recipe"] = recipe
    cfg["expanded_sha256"] = canonical_hash({k: v for k, v in cfg.items() if k != "expanded_sha256"})
    put(config, cfg)
    identity = json.loads((run / "run_identity.json").read_text())
    identity["job"] = cfg["job"]
    identity["recipe"] = recipe
    identity["provenance"].update(config_sha256=sha(config), native_method=method,
                                   native_initialization=None,
                                   official_source_commit=OFFICIAL_COMMITS[method])
    if method == "fedsb":
        init = {"public_rows": 512, "full_weight_gradient_targets": 24}
        identity["provenance"]["native_initialization"] = init
        put(run / "native_initialization.json", init)
    put(run / "run_identity.json", identity)
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    modules = [f"bert.encoder.layer.{layer}.attention.self.{target}"
               for layer in range(12) for target in ("query", "value")]
    rank = 4
    if method == "fedsb":
        adapter = {f"{name}.{suffix}": torch.zeros(shape)
                   for name in modules for suffix, shape in
                   (("a", (rank, 768)), ("b", (768, rank)), ("r", (rank, rank)))}
    else:
        adapter = {f"{name}.{suffix}": torch.zeros(shape)
                   for name in modules for suffix, shape in
                   (("lora_a", (rank, 768)), ("lora_b", (768, rank)))}
    ck.update(format="pcla-v7-native-1", adapter=adapter,
              budget_adapter=adapter, half_adapter=adapter)
    assigned = [96] * 30 + [128] * 15 + [192] * 5
    random.Random(stable_seed(401, "resource_assignment")).shuffle(assigned)
    rows = json.loads((run / "metrics.json").read_text())
    for row in rows:
        r = row["round"]
        trace = {}
        details = []
        masks = {}
        for cid in row["selected_clients"]:
            k = recipe["native_ranks"][{96: 0, 128: 1, 192: 2}[assigned[cid]]] if method == "fedplora" else rank
            chosen = {}
            for module in modules:
                if method == "fedplora":
                    message = f"pcla-global-budget-20260928-v7|plora|401|{r}|{cid}|{module}"
                    keyed = int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big")
                    chosen[module] = sorted(random.Random(keyed).sample(range(rank), k))
                else:
                    chosen[module] = list(range(rank))
            masks[cid] = chosen
            positions = 24 * k
            trace[str(cid)] = {"modules": chosen, "selected_positions": positions}
            params = 24 * (rank * rank if method == "fedsb" else
                           768 * rank if method == "fedsvd" else 1536 * k) + 1538
            details.append({"client": cid, "optimizer_steps": 16,
                            "optimizer_tensor_parameters": params,
                            "native_selected_positions": positions})
        row["support_trace"] = trace
        row["client_training"] = details
        row["coordinate_refresh"] = {"refreshed": method == "fedsvd" and r >= 2}
        if method == "fedplora":
            counts = {module: [sum(j in masks[cid][module] for cid in row["selected_clients"])
                               for j in range(rank)] for module in modules}
            row["native_aggregation"] = {
                "contributors_per_projection_rank": counts,
                "zero_contributor_positions": sum(x == 0 for vals in counts.values() for x in vals)}
        else:
            row["native_aggregation"] = {}
    ck["metrics"] = rows
    put(run / "metrics.json", rows)
    torch.save(ck, run / "latest.pt")
    for filename in ("budget_return.pt", "half_budget_return.pt"):
        obj = torch.load(run / filename, map_location="cpu", weights_only=True)
        obj["adapter"] = adapter
        obj["format"] = "pcla-v7-native-1"
        torch.save(obj, run / filename)
    result = audit(config, run, source)
    assert result["ok"], result["errors"]
    rows[0]["support_trace"][str(rows[0]["selected_clients"][0])]["modules"][modules[0]] = []
    put(run / "metrics.json", rows)
    assert not audit(config, run, source)["ok"]


def test_failed_run_reports_previous_finite_checkpoint_without_success(tmp_path):
    config, run, source = evidence(tmp_path)
    checkpoint = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    torch.save(checkpoint, run / "previous.pt")
    checkpoint["adapter"][next(iter(checkpoint["adapter"]))][0, 0] = float("inf")
    torch.save(checkpoint, run / "latest.pt")
    put(run / "status.json", {"state": "failed", "completed_round": 100,
                               "exception": "nonfinite local gradient"})
    (run / "run_identity.json").unlink()
    result = audit(config, run, source)
    assert not result["ok"]
    assert result["checks"]["last_finite_checkpoint"]["path"] == str(run / "previous.pt")
    assert result["checks"]["last_finite_checkpoint"]["round"] == 100


def test_diagnostic_continuation_keeps_budget_return_at_stop(tmp_path):
    config, run, source = evidence(tmp_path)
    rows = json.loads((run / "metrics.json").read_text())
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    for row in rows:
        r = row["round"]
        row["deployed_bytes"] = min(r, 50) * 20
        row["diagnostic_after_budget_bytes"] = max(r - 50, 0) * 20
        row["budget_stop_round"] = 50 if r >= 50 else None
    for event in ck["cache"]["events"]:
        if event["round"] > 50:
            event["bucket"] = "diagnostic_after_budget"
    ck.update(metrics=rows, budget_stop_round=50, deployed_bytes=1000,
              diagnostic_bytes=1000, half_round=50)
    torch.save(ck, run / "latest.pt")
    put(run / "metrics.json", rows)
    put(run / "status.json", {"state": "completed", "completed_round": 100,
        "budget_stop_round": 50, "deployed_bytes": 1000,
        "diagnostic_after_budget_bytes": 1000})
    budget = torch.load(run / "budget_return.pt", map_location="cpu", weights_only=True)
    budget.update(round=50, deployed_bytes=1000)
    torch.save(budget, run / "budget_return.pt")
    result = audit(config, run, source)
    assert result["ok"], result["errors"]
    assert result["checks"]["budget_stop_round"] == 50
    assert result["checks"]["diagnostic_after_budget_bytes"] == 1000
    ck["cache"]["events"][-1]["bucket"] = "deployed_prefix"
    ck["deployed_bytes"] += 1
    torch.save(ck, run / "latest.pt")
    assert not audit(config, run, source)["ok"]
