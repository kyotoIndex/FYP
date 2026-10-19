"""CPU checks with a tiny random RoBERTa and synthetic cache/manifest (no MNLI, no CUDA).

    ../../.venv/bin/python -m pytest -q tests
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from headroom.data import ClientStream, nested_strong_clients  # noqa: E402
from headroom.fed import aggregate, run  # noqa: E402
from train import load_plan  # noqa: E402

CLIENTS, ROLES = 6, ["strong", "weak", "weak", "strong", "weak", "weak"]
DOMAINS = ["government", "fiction", "slate", "travel", "telephone", "fiction"]


def build_fixture(root: Path) -> Path:
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=60, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=40, num_labels=3,
                        pad_token_id=1, bos_token_id=0, eos_token_id=2)
    RobertaForSequenceClassification(cfg).save_pretrained(root / "model")

    gen = torch.Generator().manual_seed(1)

    def make_split(n_per_client, name):
        tokens, offsets, labels, rows = [], [0], [], []
        for c in range(CLIENTS):
            for j in range(n_per_client):
                label = (c + j) % 3
                length = int(torch.randint(4, 12, (1,), generator=gen))
                seq = torch.randint(3, 60, (length,), generator=gen)
                seq[1] = 10 + label  # learnable signal
                seq[0], seq[-1] = 0, 2
                tokens.append(seq)
                offsets.append(offsets[-1] + length)
                labels.append(label)
                rows.append({"group": f"{name}-{c}-{j}", "domain": DOMAINS[c], "client": c, "label": label})
        torch.save({"tokens": torch.cat(tokens).int(), "offsets": torch.tensor(offsets),
                    "labels": torch.tensor(labels), "pad_id": 1, "rows": len(labels)},
                   root / "cache" / f"{name}.pt")
        with (root / "data" / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows

    (root / "cache").mkdir()
    (root / "data").mkdir()
    train_rows = make_split(24, "train")
    make_split(6, "validation")
    clients = []
    for c in range(CLIENTS):
        ids = [i for i, r in enumerate(train_rows) if r["client"] == c]
        clients.append({"id": c, "role": ROLES[c], "samples": len(ids), "train_indices": ids,
                        "domains": {DOMAINS[c]: len(ids)}})
    (root / "data" / "manifest.json").write_text(json.dumps({
        "task": "synthetic", "scenario": "test", "partition": "genre", "seed": 0,
        "files": {"train": {"path": "train.jsonl"}, "validation": {"path": "validation.jsonl"}},
        "clients": clients}))
    plan = json.loads((HERE / "plan.json").read_text())
    plan.update({"id": "test", "project_root": str(root), "cache_directory": "cache",
                 "manifest_path": "data/manifest.json", "run_root": "runs",
                 "seeds": [7, 8], "checkpoint_every": 1, "cpu_threads": 1})
    plan["model"] = {"family": "roberta", "name": str(root / "model"), "local_files_only": True}
    plan["federation"] = {"rounds": 4, "clients_per_round": 3}
    plan["train"].update({"local_steps": 3, "batch_size": 4, "lr": 0.003, "eval_batch_size": 16})
    plan["personalize"] = {"steps": 2}
    plan["evaluation"] = {"every_rounds": 2}
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    root = tmp_path_factory.mktemp("headroom")
    return load_plan(build_fixture(root), None)


def test_aggregate_componentwise():
    prev = {"x.lora_a": torch.zeros(4, 3), "x.lora_b": torch.zeros(5, 4), "head.w": torch.zeros(2)}
    strong = {"x.lora_a": torch.ones(4, 3), "x.lora_b": torch.ones(5, 4), "head.w": torch.ones(2)}
    weak = {"x.lora_a": torch.full((4, 3), 3.0), "x.lora_b": torch.full((5, 4), 3.0), "head.w": torch.full((2,), 3.0)}
    out = aggregate([(strong, 4, 1.0), (weak, 2, 3.0)], prev)
    assert torch.allclose(out["x.lora_a"][:2], torch.full((2, 3), 2.5))   # (1*1+3*3)/4
    assert torch.allclose(out["x.lora_a"][2:], torch.ones(2, 3))          # strong only
    assert torch.allclose(out["x.lora_b"][:, :2], torch.full((5, 2), 2.5))
    assert torch.allclose(out["x.lora_b"][:, 2:], torch.ones(5, 2))
    assert torch.allclose(out["head.w"], torch.full((2,), 2.5))
    out = aggregate([(weak, 2, 1.0)], prev)                               # untrained comps keep server
    assert torch.equal(out["x.lora_a"][2:], prev["x.lora_a"][2:])


def test_stream_resumes_identically():
    a = ClientStream(list(range(10)), 3, 1)
    first = a.take(7) + a.take(9)
    b = ClientStream(list(range(10)), 3, 1)
    b.take(7)
    c = ClientStream(list(range(10)), 3, 1, b.position)
    assert first[7:] == c.take(9)
    assert sorted(first[:10]) == list(range(10))  # one full epoch is a permutation


def test_all_methods_run_and_respect_ranks(plan):
    results = {}
    for method in plan["methods"]:
        out = Path(plan["run_root"]) / f"{method}_seed7"
        results[method] = run(plan, method, 7, out, torch.device("cpu"))
        final = results[method]["final"]
        assert 0.0 <= final["client_macro_accuracy"] <= 1.0
        assert torch.isfinite(torch.tensor(final["client_macro_nll"]))
        assert [h["round"] for h in results[method]["history"]] == [0, 2, 4]
    # Same seed -> same initial model for every method.
    initial = {m: r["history"][0]["client_macro_nll"] for m, r in results.items()}
    assert max(initial.values()) - min(initial.values()) < 1e-7  # batching differs for local_only
    # Rank-2 federation never touches components >= 2: B stays exactly zero there.
    state = torch.load(Path(plan["run_root"]) / "fedavg_r2_seed7" / "final_state.pt", weights_only=False)
    for name, value in state["global"].items():
        if name.endswith("lora_b"):
            assert torch.count_nonzero(value[:, 2:]) == 0 and torch.count_nonzero(value[:, :2]) > 0
    # Heterogeneous run: strong clients fill components >= 2.
    state = torch.load(Path(plan["run_root"]) / "hetero_r8r2_seed7" / "final_state.pt", weights_only=False)
    assert any(torch.count_nonzero(v[:, 2:]) > 0 for n, v in state["global"].items() if n.endswith("lora_b"))
    # Local-only: weak clients' frozen components remain at initialisation (B=0).
    state = torch.load(Path(plan["run_root"]) / "local_only_seed7" / "final_state.pt", weights_only=False)
    for c, role in enumerate(ROLES):
        b = [v for n, v in state["local"][c].items() if n.endswith("lora_b")]
        if role == "weak":
            assert all(torch.count_nonzero(v[:, 2:]) == 0 for v in b)
    assert "personalized" in results["hetero_r8r2"] and "personalized" not in results["local_only"]
    assert results["local_only"]["cost"]["upload_params_total"] == 0


def test_training_learns(plan, tmp_path):
    longer = json.loads(json.dumps(plan))
    longer["federation"] = {"rounds": 12, "clients_per_round": 3}
    longer["train"]["lr"] = 0.01
    longer["evaluation"] = {"every_rounds": 12}
    result = run(longer, "fedavg_r8", 7, tmp_path / "learn", torch.device("cpu"))
    assert result["final"]["client_macro_accuracy"] > 0.6
    assert result["final"]["client_macro_nll"] < result["history"][0]["client_macro_nll"] - 0.2


def test_resume_matches_uninterrupted(plan):
    root = Path(plan["run_root"])
    run(plan, "hetero_r8r2", 8, root / "full", torch.device("cpu"))
    run(plan, "hetero_r8r2", 8, root / "split", torch.device("cpu"), stop_after_round=2)
    run(plan, "hetero_r8r2", 8, root / "split", torch.device("cpu"), resume=True)
    a = torch.load(root / "full" / "final_state.pt", weights_only=False)["global"]
    b = torch.load(root / "split" / "final_state.pt", weights_only=False)["global"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(FileExistsError):
        run(plan, "hetero_r8r2", 8, root / "full", torch.device("cpu"), resume=True)


def test_cli_and_summary(plan, tmp_path):
    plan_path = Path(plan["project_root"]) / "plan.json"
    py = sys.executable
    for index in range(8):  # full 2x4 grid via the array entry point
        method = plan["methods"][index % 4]
        seed = plan["seeds"][index // 4]
        if (Path(plan["run_root"]) / f"{method}_seed{seed}" / "result.json").exists():
            continue
        subprocess.run([py, str(HERE / "train.py"), "--plan", str(plan_path), "--task-index", str(index),
                        "--device", "cpu"], check=True, capture_output=True)
    out = subprocess.run([py, str(HERE / "summarize.py"), "--plan", str(plan_path)],
                         check=True, capture_output=True, text=True).stdout
    summary = json.loads((Path(plan["run_root"]) / "summary.json").read_text())
    assert summary["missing"] == []
    assert summary["decision"]["rank_capacity_headroom"] in {"yes", "no"}
    assert "capacity_r8_minus_r2" in out


def test_nested_strong_subsets():
    clients = {c: {"role": "strong" if 10 <= c < 20 or 40 <= c < 50 else "weak",
                   "domain": "government" if c < 20 else "travel"} for c in range(50)}
    s10, s20, s40 = (nested_strong_clients(clients, f) for f in (0.1, 0.2, 0.4))
    assert (len(s10), len(s20), len(s40)) == (5, 10, 20)
    assert set(s10) <= set(s20) <= set(s40) == set(range(10, 20)) | set(range(40, 50))
    assert s20 == [10, 11, 12, 13, 14, 40, 41, 42, 43, 44]
    with pytest.raises(ValueError):
        nested_strong_clients(clients, 0.6)


def test_strong_fraction_sweep_cli(plan):
    root = Path(plan["project_root"])
    sweep = json.loads((root / "plan.json").read_text())
    sweep.update({"id": "test_sweep", "run_root": "runs_sweep"})
    sweep.pop("methods", None)
    sweep.pop("seeds", None)
    sweep["cells"] = [{"method": "fedavg_r8", "seed": 7}, {"method": "fedavg_r2", "seed": 7},
                      {"method": "hetero_r8r2", "seed": 7, "strong_fraction": 0.17},
                      {"method": "hetero_r8r2", "seed": 7, "strong_fraction": 0.34}]
    sweep["sweep"] = {"fractions": [0.17, 0.34], "late_start_round": 2,
                      "threshold_pp_every_seed": 1.0, "targets": [40, 50]}
    plan_path = root / "plan_sweep.json"
    plan_path.write_text(json.dumps(sweep))
    for index in range(4):
        subprocess.run([sys.executable, str(HERE / "train.py"), "--plan", str(plan_path),
                        "--task-index", str(index), "--device", "cpu"], check=True, capture_output=True)
    small = json.loads((root / "runs_sweep" / "hetero_r8r2_sf17_seed7" / "result.json").read_text())
    large = json.loads((root / "runs_sweep" / "hetero_r8r2_sf34_seed7" / "result.json").read_text())
    assert small["strong_clients"] == [0] and large["strong_clients"] == [0, 3]
    assert sum(r == 8 for r in small["client_ranks"].values()) == 1
    assert set(small["history"][-1]["clients"]) == {str(c) for c in range(CLIENTS)}
    subprocess.run([sys.executable, str(HERE / "summarize_strongfrac.py"), "--plan", str(plan_path)],
                   check=True, capture_output=True, text=True)
    summary = json.loads((root / "runs_sweep" / "summary_strongfrac.json").read_text())
    assert summary["missing"] == []
    assert set(summary["decision_core_weak_headroom"].values()) <= {"yes", "no"}
    assert summary["core_weak_clients"] == [1, 2, 4, 5]          # never strong in any cell
    row = summary["by_fraction"]["0.17"]["per_seed"]["7"]
    assert (row["strong_clients"], row["weak_clients"], row["core_weak_clients"], row["demoted_clients"]) == (1, 5, 4, 1)
    assert summary["by_fraction"]["0.34"]["per_seed"]["7"]["demoted_clients"] == 0
    part = summary["strong_participation"]["0.17"]
    assert 0 <= part["share_rounds_with_0_strong"] <= 1 and part["mean_strong_per_round"] <= 3
