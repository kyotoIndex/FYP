"""CPU checks with a tiny random RoBERTa and synthetic data (no MNLI, no CUDA).

    ../../.venv/bin/python -m pytest -q tests -p no:cacheprovider
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scra.analyze import analyze_d1_round, write_report  # noqa: E402
from scra.core import (adapter_names, balanced_allocation, block_aggregates, from_blocks, load_plan,  # noqa: E402
                       local_train, to_blocks)
from scra.d0 import run_d0  # noqa: E402
from scra.d1 import run_d1  # noqa: E402
from scra.data import ClientStream, Dataset  # noqa: E402
from scra.model import TaskModel  # noqa: E402

HEADROOM = HERE.parent / "headroom_coldstart_20261017"
ROLES = ["strong", "weak", "weak", "strong", "weak", "weak"]
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
        for c in range(len(ROLES)):
            for j in range(n_per_client):
                label = (c + j) % 3
                length = int(torch.randint(4, 12, (1,), generator=gen))
                seq = torch.randint(3, 60, (length,), generator=gen)
                seq[1] = 10 + label
                seq[0], seq[-1] = 0, 2
                tokens.append(seq); offsets.append(offsets[-1] + length); labels.append(label)
                rows.append({"group": f"{name}-{c}-{j}", "domain": DOMAINS[c], "client": c, "label": label})
        torch.save({"tokens": torch.cat(tokens).int(), "offsets": torch.tensor(offsets),
                    "labels": torch.tensor(labels), "pad_id": 1, "rows": len(labels)}, root / "cache" / f"{name}.pt")
        with (root / "data" / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows

    (root / "cache").mkdir(); (root / "data").mkdir()
    train_rows = make_split(24, "train")
    make_split(6, "validation")
    clients = []
    for c in range(len(ROLES)):
        ids = [i for i, r in enumerate(train_rows) if r["client"] == c]
        clients.append({"id": c, "role": ROLES[c], "samples": len(ids) + c, "train_indices": ids,
                        "domains": {DOMAINS[c]: len(ids)}})
    (root / "data" / "manifest.json").write_text(json.dumps({
        "task": "synthetic", "scenario": "test", "partition": "genre", "seed": 0,
        "files": {"train": {"path": "train.jsonl"}, "validation": {"path": "validation.jsonl"}},
        "clients": clients}))
    plan = json.loads((HERE / "plan.json").read_text())
    plan.update({"id": "test", "project_root": str(root), "cache_directory": "cache",
                 "manifest_path": "data/manifest.json", "run_root": "runs", "checkpoint_every": 1, "cpu_threads": 1})
    plan["model"] = {"family": "roberta", "name": str(root / "model"), "local_files_only": True}
    plan["federation"] = {"rounds": 6, "clients_per_round": 6}
    plan["train"].update({"local_steps": 3, "batch_size": 4, "lr": 0.003, "eval_batch_size": 16})
    plan["evaluation"] = {"every_rounds": 3}
    plan["criteria"]["late_start_round"] = 1
    plan["d0"].update({"seed": 7, "strong_fraction": 0.17, "snapshot_rounds": [2, 4], "kappa_warmup": 1,
                       "windows": [[1, 2], [3, 4], [5, 6]]})
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return load_plan(build_fixture(tmp_path_factory.mktemp("scra")), None)


def test_balanced_allocation_properties():
    weak = [3, 7, 9, 11, 20, 21, 22, 30, 31]
    for t in range(20):
        sets, counts = balanced_allocation(weak, 24, 8, 2, seed=5, round_id=t)
        for j in range(24):
            tally = [0] * 8
            for c in weak:
                assert len(set(sets[c][j])) == 2
                for k in sets[c][j]:
                    tally[k] += 1
            assert tally == counts[j].int().tolist() and sum(tally) == 18 and min(tally) >= 2


def test_inclusion_probabilities_empirical():
    weak = list(range(9))
    hits = torch.zeros(9, 8); rounds = 3000
    for t in range(rounds):
        sets, counts = balanced_allocation(weak, 1, 8, 2, seed=1, round_id=t)
        for c in weak:
            for k in sets[c][0]:
                hits[c, k] += 1
    assert torch.allclose(hits / rounds, torch.full((9, 8), 2 / 8), atol=0.03)


def test_owner_bias_and_hajek_unbiased_monte_carlo():
    torch.manual_seed(0)
    J, R, D = 1, 8, 4
    strong, weak = [0], list(range(1, 10))
    deltas = {0: torch.ones(J, R, D) * 2.0}
    for c in weak:
        deltas[c] = torch.randn(J, R, D)
    weights = {c: 1.0 for c in [0] + weak}
    target = sum(deltas[c] for c in deltas) / 10
    own, haj = torch.zeros(J, R, D), torch.zeros(J, R, D)
    n = 2000
    for t in range(n):
        sets, counts = balanced_allocation(weak, J, R, 2, seed=3, round_id=t)
        agg = block_aggregates(deltas, weights, strong, sets, counts)
        own += agg["owner"]; haj += agg["hajek"]
        w = agg["owner"] - agg["hajek"]
        assert torch.isfinite(w).all()
    own, haj = own / n, haj / n
    weak_mean = sum(deltas[c] for c in weak) / 9
    rho, c_mean = 0.1, 18 / 8
    predicted = (1 / (1 + c_mean) - rho) * (deltas[0] - weak_mean)
    assert (haj - target).abs().max() < 0.05
    assert torch.allclose(own - target, predicted, atol=0.08)


def test_blocks_roundtrip_and_masked_training(plan):
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    model = TaskModel.build(plan["model"], 3, 8, 2.0, 0)
    names = adapter_names(model)
    state = model.task_state()
    assert all(torch.equal(v, from_blocks(to_blocks(state, names), names, state)[k]) for k, v in state.items())
    sets = [[1, 5] for _ in names]
    new, _ = local_train(model, state, data, ClientStream(data.clients[1]["train_indices"], 0, 1), sets,
                         plan["train"], torch.device("cpu"))
    diff = to_blocks(new, names) - to_blocks(state, names)
    moved = diff.abs().sum(-1) > 0
    for j in range(len(names)):
        assert moved[j].nonzero().flatten().tolist() == [1, 5]


def test_d0_prefix_owner_matches_headroom_hetero(plan, tmp_path):
    sys.path.insert(0, str(HEADROOM))
    from headroom.fed import run as headroom_run  # noqa: E402

    hp = json.loads(json.dumps(plan))
    hp["personalize"] = {"steps": 0}
    head_out = tmp_path / "headroom"
    headroom_run(hp, "hetero_r8r2", hp["d0"]["seed"], head_out, torch.device("cpu"),
                 strong_fraction=hp["d0"]["strong_fraction"])
    dp = json.loads(json.dumps(plan))
    dp["d0"].update({"allocation": "prefix", "aggregation": "owner", "shadow_strong_replay": True})
    run_d0(dp, tmp_path / "d0", torch.device("cpu"))
    a = torch.load(head_out / "final_state.pt", weights_only=False)["global"]
    b = torch.load(tmp_path / "d0" / "final_state.pt", weights_only=False)
    assert max((a[k] - b[k]).abs().max().item() for k in a) < 1e-5


def test_d0_resume_identical(plan, tmp_path):
    run_d0(plan, tmp_path / "full", torch.device("cpu"))
    run_d0(plan, tmp_path / "split", torch.device("cpu"), stop_after_round=3)
    run_d0(plan, tmp_path / "split", torch.device("cpu"), resume=True)
    a = torch.load(tmp_path / "full" / "final_state.pt", weights_only=False)
    b = torch.load(tmp_path / "split" / "final_state.pt", weights_only=False)
    assert all(torch.equal(a[k], b[k]) for k in a)
    sa = torch.load(tmp_path / "full" / "d0_stats.pt", weights_only=False)["stats"]
    sb = torch.load(tmp_path / "split" / "d0_stats.pt", weights_only=False)["stats"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_full_pipeline_d0_d1_analyze(plan):
    root = Path(plan["run_root"])
    run_d0(plan, root / "d0", torch.device("cpu"))
    rounds = json.loads((root / "d0" / "d0_rounds.json").read_text())
    assert any("kappa" in r for r in rounds["rounds"])
    for t in plan["d0"]["snapshot_rounds"]:
        assert (root / "d0" / "snapshots" / f"round_{t:04d}.pt").exists()
        run_d1(plan, t, root / "d1", torch.device("cpu"))
    blob = torch.load(root / "d1" / "round_0002.pt", weights_only=False)
    assert blob["groups"]["strong"] == [0] and blob["groups"]["demoted"] == []
    d1 = analyze_d1_round(root / "d1" / "round_0002.pt")
    assert d1["identity_check_max_abs"] < 1e-6
    report = write_report(root, plan)
    assert report["d0"]["vector_kappa_windows"] and report["d0"]["accuracy"]["L2_d0"]["core"] is not None
    assert all(d["product_space"] and math.isfinite(d["product_space"]["norm_e_scra"]) for d in report["d1"])
    assert set(report["d0"]["checks"]) >= {"F2_bias_large_enough", "F3_direction_persistent", "F9_no_update_blowup"}
    assert len(report["d1"]) == 2 and (root / "diagnostics_report" / "REPORT.md").exists()


def test_variance_reference_uses_only_assigned_blocks(plan, tmp_path):
    vp = json.loads(json.dumps(plan))
    vp["d0"].update({"weak_train_full": True, "aggregation": "hajek", "shadow_strong_replay": False})
    out = run_d0(vp, tmp_path / "vref", torch.device("cpu"))
    assert out["weak_train_full"] is True
    ref = run_d0(json.loads(json.dumps(plan)), tmp_path / "base", torch.device("cpu"))
    a = torch.load(tmp_path / "vref" / "final_state.pt", weights_only=False)
    b = torch.load(tmp_path / "base" / "final_state.pt", weights_only=False)
    assert any(not torch.equal(a[k], b[k]) for k in a)   # different training mode -> different model


def test_aggregation_ignores_unassigned_weak_blocks():
    torch.manual_seed(1)
    J, R, D = 3, 8, 5
    strong, weak = [0], list(range(1, 10))
    sets, counts = balanced_allocation(weak, J, R, 2, seed=2, round_id=0)
    deltas = {c: torch.randn(J, R, D) for c in [0] + weak}
    weights = {c: 1.0 + c for c in deltas}
    base = block_aggregates(deltas, weights, strong, sets, counts)
    poisoned = {c: v.clone() for c, v in deltas.items()}
    for c in weak:
        for j in range(J):
            for k in range(R):
                if k not in sets[c][j]:
                    poisoned[c][j, k] = 1e6        # a full-training weak client's unused blocks
    after = block_aggregates(poisoned, weights, strong, sets, counts)
    assert torch.equal(base["owner"], after["owner"]) and torch.equal(base["hajek"], after["hajek"])


def test_rcs_coefficient_estimator():
    from scra.core import rcs_coefficients
    torch.manual_seed(3)
    full = torch.randn(4, 8, 6)
    restr = full / 0.6 + 0.01 * torch.randn(4, 8, 6)          # restricted overshoots by 1/0.6
    n = torch.full((4, 8), 2.0)
    c = rcs_coefficients(restr, full, n, warmup=3, c_min=0.2, c_max=1.0)
    assert torch.allclose(c, torch.full((4,), 0.6), atol=0.01)
    n[0] = 0.0                                                 # adapter 0 not warmed up -> no shrink
    c = rcs_coefficients(restr, full, n, warmup=3, c_min=0.2, c_max=1.0)
    assert c[0] == 1.0 and torch.allclose(c[1:], torch.full((3,), 0.6), atol=0.01)
    c = rcs_coefficients(restr * 100, full, torch.full((4, 8), 2.0), 3, 0.2, 1.0)
    assert torch.all(c == 0.2)                                 # clipped


def test_tight_budget_allocation_and_aggregation():
    weak = list(range(1, 6))                                   # 5 weak, 1 component each, 8 components
    sets, counts = balanced_allocation(weak, 3, 8, 1, seed=4, round_id=0)
    assert counts.sum(1).tolist() == [5.0, 5.0, 5.0] and (counts == 0).any()
    assert all(len(sets[c][j]) == 1 for c in weak for j in range(3))
    deltas = {c: torch.randn(3, 8, 4) for c in [0] + weak}
    agg = block_aggregates(deltas, {c: 1.0 for c in deltas}, [0], sets, counts)
    uncovered = counts == 0
    assert torch.allclose(agg["hajek"][uncovered], deltas[0][uncovered])   # strong-only where no weak upload
    assert torch.isfinite(agg["hajek"]).all()


def test_method_variants_run_and_regress(plan, tmp_path):
    def variant(name, **upd):
        p = json.loads(json.dumps(plan))
        p["d0"].update({"aggregation": "hajek", **upd})
        return run_d0(p, tmp_path / name, torch.device("cpu"))
    base = variant("hajek_base", shadow_strong_replay=True)
    same = variant("shrink_one", shadow_strong_replay=True, weak_shrink={"mode": "fixed", "value": 1.0})
    a = torch.load(tmp_path / "hajek_base" / "final_state.pt", weights_only=False)
    b = torch.load(tmp_path / "shrink_one" / "final_state.pt", weights_only=False)
    assert all(torch.equal(a[k], b[k]) for k in a)            # fixed shrink 1.0 == no shrink
    rcs = variant("rcs", shadow_strong_replay=True,
                  weak_shrink={"mode": "rcs", "beta": 0.9, "warmup_replays": 1, "c_min": 0.2, "c_max": 1.0})
    shr = [r["shrink_mean"] for r in rcs["rounds"]]
    assert all(0.2 <= v <= 1.0 for v in shr) and any(v < 1.0 for v in shr)
    lr = variant("weaklr", shadow_strong_replay=False, weak_lr_scale=0.5)
    c = torch.load(tmp_path / "weaklr" / "final_state.pt", weights_only=False)
    assert any(not torch.equal(a[k], c[k]) for k in a) and lr["weak_lr_scale"] == 0.5
    p8 = json.loads(json.dumps(plan))
    p8["lora"]["weak_rank"] = 1
    p8["d0"].update({"aggregation": "hajek", "shadow_strong_replay": False, "weak_train_full": True})
    out = run_d0(p8, tmp_path / "budget8", torch.device("cpu"))
    assert out["r"] == 1 and out["weak_train_full"] is True


def test_d2_dispersion_algebra():
    from scra.d2 import dispersion
    torch.manual_seed(3)
    ref = torch.randn(5, 40)
    same = dispersion(ref, ref)
    assert abs(same["bias_sq"]) < 1e-6 and abs(same["paired_error"]) < 1e-6
    assert abs(same["step_along_ref_mean"] - 1) < 1e-6
    mean = ref.mean(0)
    direct_var = float(((ref - mean) ** 2).sum(1).mean() / (mean @ mean))
    assert abs(same["variance"] - direct_var) < 1e-4
    double = dispersion(2 * ref, ref)
    assert abs(double["step_along_ref_mean"] - 2) < 1e-5 and abs(double["bias_sq"] - 1) < 1e-5
    const = dispersion(mean.expand(5, -1).clone(), ref)
    assert abs(const["variance"]) < 1e-5 and abs(const["step_along_ref_std"]) < 1e-5


def test_d2_rule_invariants():
    from scra.d2 import round_updates
    torch.manual_seed(4)
    J, R, D, r = 3, 4, 6, 2
    strong, weak = [0], [1, 2, 3]
    weights = {0: 5.0, 1: 2.0, 2: 3.0, 3: 4.0}
    sets, counts = balanced_allocation(weak, J, R, r, 0, 1)
    full = {c: torch.randn(J, R, D) for c in strong + weak}
    g_state = {"head.w": torch.zeros(2)}
    heads = {c: {"head.w": torch.randn(2)} for c in strong + weak}
    # restricted == full and shrink 1  ->  hajek == shrink == vref; restricted heads == full heads -> same heads
    lora, new_heads = round_updates(full, {c: full[c] for c in weak}, heads, heads, weights, strong, weak,
                                    sets, counts, 1.0, g_state)
    assert torch.allclose(lora["hajek"], lora["vref"]) and torch.equal(lora["shrink"], lora["hajek"])
    total = sum(weights.values())
    r8 = sum(weights[c] * full[c] for c in full) / total
    assert torch.allclose(lora["r8"], r8, atol=1e-6)
    restr = {c: torch.randn(J, R, D) for c in weak}
    restr_heads = {c: {"head.w": torch.randn(2)} for c in weak}
    lora, new_heads = round_updates(full, restr, heads, restr_heads, weights, strong, weak, sets, counts, 0.5, g_state)
    assert lora["shrink_fullhead"] is lora["shrink"] and lora["vref_restrhead"] is lora["vref"]
    assert torch.equal(new_heads["shrink_fullhead"]["head.w"], new_heads["r8"]["head.w"])
    assert torch.equal(new_heads["vref_restrhead"]["head.w"], new_heads["L2"]["head.w"])
    assert not torch.allclose(lora["shrink"], lora["hajek"])


def test_d2_runs_and_reports(plan, tmp_path):
    from scra.d2 import RULES, run_d2, write_d2_report
    p = json.loads(json.dumps(plan))
    p["run_root"] = str(tmp_path)
    p["d2"] = {"seed": 1, "draws": 2, "shrink": 0.63}
    run_d0(p, tmp_path / "d0", torch.device("cpu"))
    for t in p["d0"]["snapshot_rounds"]:
        run_d2(p, t, tmp_path / "d2", torch.device("cpu"))
    blob = json.loads((tmp_path / "d2" / "round_0002.json").read_text())
    assert blob["rules"] == RULES and len(blob["rows"]) == 2
    s = blob["summary"]
    assert abs(s["r8"]["lora_product"]["bias_sq"]) < 1e-6 and s["r8"]["diff_vs_r8_std"] == 0
    assert all(math.isfinite(s[k]["lora_product"]["variance"]) for k in RULES)
    assert s["shrink_fullhead"]["lora_product"] == s["shrink"]["lora_product"]
    report = write_d2_report(tmp_path)
    assert set(report["verdicts"]) >= {"H1_single_round_dispersion", "source"}
    assert (tmp_path / "diagnostics_report" / "REPORT_D2.md").exists()


def test_amplified_forward_value_and_gradient():
    from scra.model import LoRALinear
    torch.manual_seed(5)
    layer = LoRALinear(torch.nn.Linear(6, 5), 4, 2.0)
    with torch.no_grad():
        layer.lora_b.normal_()
    a0, b0 = layer.lora_a.detach().clone(), layer.lora_b.detach().clone()
    x = torch.randn(3, 6)
    plain = layer(x)
    layer.amp = (4.0, a0, b0)
    assert torch.allclose(layer(x), plain, atol=1e-6)            # no change yet -> identical forward
    with torch.no_grad():
        layer.lora_a[1] += 0.3 * torch.randn(6); layer.lora_b[:, 1] += 0.3 * torch.randn(5)
    expect = layer.base(x) + 2.0 * (x @ a0.T @ b0.T + 4.0 * (x @ layer.lora_a.T @ layer.lora_b.T - x @ a0.T @ b0.T))
    out = layer(x)
    assert torch.allclose(out, expect, atol=1e-5)
    # gradient = true gradient of the amplified model / alpha
    out.pow(2).sum().backward()
    g_detached = layer.lora_a.grad.clone()
    a = layer.lora_a.detach().clone().requires_grad_(True)
    true = layer.base(x) + 2.0 * (x @ a0.T @ b0.T + 4.0 * (x @ a.T @ layer.lora_b.detach().T - x @ a0.T @ b0.T))
    true.pow(2).sum().backward()
    assert torch.allclose(g_detached * 4.0, a.grad, atol=1e-4)
    layer.amp = None
    assert torch.allclose(layer(x), x @ layer.base.weight.T + layer.base.bias
                          + 2.0 * (x @ layer.lora_a.T @ layer.lora_b.T), atol=1e-5)


def test_local_train_variants(plan):
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    model = TaskModel.build(plan["model"], 3, 8, 2.0, 0)
    names = adapter_names(model)
    state = model.task_state()
    sets = [[2, 6] for _ in names]
    stream = lambda: ClientStream(data.clients[1]["train_indices"], 0, 1)  # noqa: E731
    cpu = torch.device("cpu")
    plain, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu)
    same, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, amp_alpha=1.0)
    assert all(torch.equal(plain[k], same[k]) for k in plain)      # alpha = 1 is plain restricted training
    amp, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, amp_alpha=4.0)
    moved = (to_blocks(amp, names) - to_blocks(state, names)).abs().sum(-1) > 0
    assert all(moved[j].nonzero().flatten().tolist() == [2, 6] for j in range(len(names)))
    assert not torch.allclose(to_blocks(amp, names), to_blocks(plain, names))
    assert all(a.amp is None for a in model.adapters)              # amplification is switched off afterwards
    fh, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, freeze_head=True)
    heads = [k for k in state if "lora_" not in k]
    assert heads and all(torch.equal(fh[k], state[k]) for k in heads)
    assert not torch.equal(to_blocks(fh, names), to_blocks(state, names))


def test_d3_linear_parts_match_direct_rules():
    from scra.core import weighted_head_mean
    from scra.d2 import lora_update
    from scra.d3 import Pooled, apply_c, head_parts, scaled_parts
    torch.manual_seed(6)
    J, R, D, r = 3, 4, 5, 2
    for strong in ([0], []):
        weak = [1, 2, 3]
        chosen = strong + weak
        weights = {0: 5.0, 1: 2.0, 2: 3.0, 3: 4.0}
        sets, counts = balanced_allocation(weak, J, R, r, 0, 1)
        full = {c: torch.randn(J, R, D) for c in chosen}
        restr = {c: torch.randn(J, R, D) for c in weak}
        base, slope = scaled_parts(full, restr, weights, strong, weak, sets, counts)
        cvec = torch.tensor([0.3, 0.7, 1.0])
        for c in (0.63, cvec):
            scale = c if isinstance(c, float) else c[:, None, None]
            direct = lora_update({k: (restr[k] * scale if k in weak else full[k]) for k in chosen}, weights, strong,
                                 {k: sets[k] for k in weak}, counts, "hajek")
            assert torch.allclose(apply_c(base, slope, c), direct, atol=1e-6)
        g = {"h": torch.zeros(2)}
        hf = {k: {"h": torch.randn(2)} for k in chosen}
        hr = {k: {"h": torch.randn(2)} for k in weak}
        hb, hs = head_parts(hf, hr, weights, strong, weak, g)
        direct = weighted_head_mean({k: ({"h": 0.55 * hr[k]["h"]} if k in weak else hf[k]) for k in chosen}, weights, g)
        assert torch.allclose(apply_c(hb, hs, 0.55)["h"], direct["h"], atol=1e-6)
    pool = Pooled(J, R, D)
    for _ in range(4):
        f = torch.randn(J, R, D)
        mask = torch.zeros(J, R, dtype=torch.bool); mask[:, :2] = True
        h = torch.randn(7)
        pool.add(f / 0.6, f, mask, h / 0.5, h)
    lora_c, head_c = pool.coefficients(0.2, 1.0)
    assert torch.allclose(lora_c, torch.full((J,), 0.6), atol=1e-5) and abs(head_c - 0.5) < 1e-6


def test_d3_runs_and_reports(plan, tmp_path):
    from scra.d3 import RULES, run_d3, write_d3_report
    p = json.loads(json.dumps(plan))
    p["run_root"] = str(tmp_path)
    p["d3"] = {"seed": 1, "draws": 2, "shrink": 0.63, "amp_alpha": 4.0, "c_min": 0.2, "c_max": 1.0}
    run_d0(p, tmp_path / "d0", torch.device("cpu"))
    for t in p["d0"]["snapshot_rounds"]:
        run_d3(p, t, tmp_path / "d3", torch.device("cpu"))
    blob = json.loads((tmp_path / "d3" / "round_0002.json").read_text())
    assert blob["rules"] == RULES and len(blob["rows"]) == 2
    s = blob["summary"]
    assert s["r8"]["diff_vs_r8_std"] == 0 and abs(s["r8"]["lora_product"]["bias_sq"]) < 1e-6
    assert all(math.isfinite(s[k]["core_diff_vs_r8_mean"]) for k in RULES)
    assert set(blob["coefficients"]) == {"rcs2", "oracle2"}
    report = write_d3_report(tmp_path)
    assert set(report["verdicts"]) >= {"amplification_works", "rcs2_realizable"}
    assert (tmp_path / "diagnostics_report" / "REPORT_D3.md").exists()


def test_hetero_allocation_and_inclusion_hajek():
    from scra.core import hetero_allocation
    weak = [3, 4, 5, 6, 7]
    ranks = {3: 1, 4: 2, 5: 4, 6: 2, 7: 1}
    J, R = 4, 8
    sets, counts = hetero_allocation(weak, ranks, J, R, 0, 3)
    for c in weak:
        assert all(len(set(s)) == ranks[c] for s in sets[c])
    assert torch.all(counts.sum(1) == sum(ranks.values())) and torch.all(counts.max(1).values - counts.min(1).values <= 1)
    hits = torch.zeros(len(weak), R)                                 # inclusion ~ r_i / R
    for t in range(400):
        sets, _ = hetero_allocation(weak, ranks, 1, R, 0, t)
        for i, c in enumerate(weak):
            hits[i, sets[c][0]] += 1
    for i, c in enumerate(weak):
        assert torch.allclose(hits[i] / 400, torch.full((R,), ranks[c] / R), atol=0.08)
    # uniform ranks with exactly balanced counts: per-client inclusion Hajek == count-based Hajek
    torch.manual_seed(7)
    w4 = [1, 2, 3, 4]
    sets, counts = balanced_allocation(w4, 2, 8, 2, 0, 1)
    assert torch.all(counts == 1)
    deltas = {c: torch.randn(2, 8, 3) for c in [0] + w4}
    weights = {0: 3.0, 1: 1.0, 2: 2.0, 3: 1.5, 4: 2.5}
    a = block_aggregates(deltas, weights, [0], sets, counts)
    b = block_aggregates(deltas, weights, [0], sets, counts, {c: 2 / 8 for c in w4})
    assert torch.allclose(a["hajek"], b["hajek"], atol=1e-6) and torch.equal(a["owner"], b["owner"])


def test_batch_plans_run(plan, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_batch", HERE / "make_batch.py")
    mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)
    for name in ("amp_s2040", "amp_het_s2040", "amp_prefix_s2040", "vref_het_s2040", "L2_r1_s2040"):
        _, seed, d0, lora = next(x for x in mb.RUNS if x[0] == name)
        p = json.loads(json.dumps(plan))
        p["d0"].update({"shadow_strong_replay": False, "save_stats": False, **d0})
        p["lora"].update(lora)
        out = run_d0(p, tmp_path / name, torch.device("cpu"))
        assert math.isfinite(out["eval"][-1]["client_macro_accuracy"])
        assert not (tmp_path / name / "d0_stats.pt").exists()
        if "weak_rank_choices" in lora:
            assert set(out["weak_ranks"].values()) <= {1, 2, 4}
    # amp with alpha 1 == plain restricted Hajek, bitwise
    p = json.loads(json.dumps(plan))
    p["d0"].update({"shadow_strong_replay": False, "aggregation": "hajek"})
    base = run_d0(p, tmp_path / "plain", torch.device("cpu"))
    p["d0"]["weak_amp_alpha"] = 1.0
    amp1 = run_d0(p, tmp_path / "amp1", torch.device("cpu"))
    assert [e["client_macro_accuracy"] for e in base["eval"]] == [e["client_macro_accuracy"] for e in amp1["eval"]]


def test_batch_report_handles_missing_runs(tmp_path):
    from scra.batch import write_batch_report
    domains = {str(c): d for c, d in enumerate(["government", "fiction", "slate", "travel", "fiction", "slate"])}
    ident = {"seed": 2040, "strong_clients": [0, 3], "client_domains": domains}

    def curve(level):
        return [{"round": t, "client_macro_accuracy": level,
                 "clients": {c: [level + 0.01 * int(c), 0.5] for c in domains}} for t in range(0, 151, 10)]
    for name, level in [("scra_batch_20261018/amp_s2040", 0.80), ("scra_diagnostics_20261017_v1_vref", 0.80),
                        ("scra_diagnostics_20261017_v1", 0.77)]:
        d = tmp_path / name / "d0"
        d.mkdir(parents=True)
        (d / "identity.json").write_text(json.dumps(ident))
        (d / "d0_rounds.json").write_text(json.dumps({"eval": curve(level), "elapsed_seconds": 1.0}))
    out = write_batch_report(tmp_path)
    assert out["n_runs"] == 3
    text = (tmp_path / "scra_batch_20261018" / "SUMMARY.md").read_text()
    assert "amp_s2040" in text and "M1" in text
    assert out["verdicts"]["A1_hajek_needed"] is None


def test_fslora_sketch_forward_and_gradients():
    from scra.model import LoRALinear
    torch.manual_seed(8)
    layer = LoRALinear(torch.nn.Linear(6, 5), 4, 2.0)
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(3, 6)
    factor = torch.tensor([0.0, 2.0, 0.0, 2.0])                  # R/r = 2 on components 1 and 3
    layer.sketch = factor
    out = layer(x)
    expect = layer.base(x) + 2.0 * (x @ layer.lora_a.T * factor) @ layer.lora_b.T
    assert torch.allclose(out, expect, atol=1e-5)
    out.pow(2).sum().backward()
    assert torch.all(layer.lora_a.grad[[0, 2]] == 0) and torch.all(layer.lora_b.grad[:, [0, 2]] == 0)
    assert layer.lora_a.grad[[1, 3]].abs().sum() > 0
    layer.sketch = None


def test_batch2_local_variants(plan):
    from scra.core import random_allocation, topk_sparsify
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    model = TaskModel.build(plan["model"], 3, 8, 2.0, 0)
    names = adapter_names(model)
    state = model.task_state()
    sets = [[0, 3] for _ in names]
    stream = lambda: ClientStream(data.clients[1]["train_indices"], 0, 1)  # noqa: E731
    cpu = torch.device("cpu")
    plain, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu)
    same, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, lora_lr_scale=1.0)
    assert all(torch.equal(plain[k], same[k]) for k in plain)        # scale 1 == plain, bitwise
    fast, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, lora_lr_scale=4.0)
    heads = [k for k in state if "lora_" not in k]
    assert not torch.equal(to_blocks(fast, names), to_blocks(plain, names))
    sk, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, sketch_alpha=4.0)
    moved = (to_blocks(sk, names) - to_blocks(state, names)).abs().sum(-1) > 0
    assert all(moved[j].nonzero().flatten().tolist() == [0, 3] for j in range(len(names)))
    assert all(a.sketch is None for a in model.adapters)
    with pytest.raises(ValueError):
        local_train(model, state, data, stream(), sets, plan["train"], cpu, amp_alpha=4.0, sketch_alpha=4.0)
    # top-k keeps the requested fraction of entries, largest magnitudes
    d = torch.randn(3, 4, 10)
    sp = topk_sparsify(d, 0.25)
    assert int((sp != 0).sum()) == 30 and sp.abs().max() == d.abs().max()
    # random allocation: r distinct components per client, counts consistent
    s, c = random_allocation([1, 2, 3], 2, 8, 2, 0, 5)
    assert all(len(set(x)) == 2 for v in s.values() for x in v) and torch.all(c.sum(1) == 6)
    assert heads


def test_zerofill_aggregate():
    torch.manual_seed(9)
    J, R, D = 2, 4, 3
    sets, counts = balanced_allocation([1, 2], J, R, 2, 0, 1)
    deltas = {c: torch.randn(J, R, D) for c in (0, 1, 2)}
    weights = {0: 2.0, 1: 1.0, 2: 3.0}
    agg = block_aggregates(deltas, weights, [0], sets, counts)
    expect = torch.zeros(J, R, D)
    for c in (1, 2):
        for j in range(J):
            for k in sets[c][j]:
                expect[j, k] += weights[c] * deltas[c][j, k]
    expect = (expect + weights[0] * deltas[0]) / 6.0
    assert torch.allclose(agg["zerofill"], expect, atol=1e-6)


def test_batch2_plans_run(plan, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_batch2", HERE / "make_batch2.py")
    mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)
    for name, d0 in mb.METHODS:
        p = json.loads(json.dumps(plan))
        p["d0"].update({"shadow_strong_replay": False, "save_stats": False, **d0})
        out = run_d0(p, tmp_path / name, torch.device("cpu"))
        assert math.isfinite(out["eval"][-1]["client_macro_accuracy"]), name
    ident = json.loads((tmp_path / "fslora_st" / "identity.json").read_text())
    assert ident["weak_forward"] == "fslora_st" and "scra/model.py" in ident["code_hashes"]
    from scra.batch2 import paired, verdicts2
    base = {"amp": 80.1, "fslora_st": 79.6, "fslora_hajek": 79.5, "fslora_zerofill": 78.9, "lrcomp": 80.0,
            "fslora_paper": 78.0, "fedplora": 77.5, "flasc": 80.2}
    jitter = {2040: 0.05, 2041: -0.05, 2042: 0.0}
    fake = {f"{m}_s{s}": {"core": x + (jitter[s] if m == "amp" else 0), "windows": {}}
            for m, x in base.items() for s in (2040, 2041, 2042)}
    v = verdicts2(fake)
    assert v["A1a_state_sketch"]["result"] and v["A2_lr_compensation"]["result"]
    assert v["A1c_aggregation"]["result"] and not v["A1b_gradient_path"]["result"]
    assert v["E2_vs_fedplora_style"]["result"] and v["E3_vs_flasc_style"]["result"]
    p = paired(fake, "amp", "fslora_st")
    assert p["n_pos"] == 3 and p["ci95"][0] < p["mean"] < p["ci95"][1]
    # sign disagreement blocks a directional claim even with a large mean
    fake["fslora_st_s2041"]["core"] = 81.5
    assert verdicts2(fake)["A1a_state_sketch"]["result"] is False


def test_fslora_st_equals_amp_at_cold_start_and_differs_later(plan):
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    model = TaskModel.build(plan["model"], 3, 8, 2.0, 0)
    names = adapter_names(model)
    state = model.task_state()                                   # B = 0 everywhere (cold start)
    sets = [[1, 6] for _ in names]
    stream = lambda: ClientStream(data.clients[2]["train_indices"], 0, 2)  # noqa: E731
    cpu = torch.device("cpu")
    amp, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, amp_alpha=4.0)
    st, _ = local_train(model, state, data, stream(), sets, plan["train"], cpu, sketch_alpha=4.0,
                        sketch_straight_through=True)
    assert all(torch.allclose(amp[k], st[k], atol=1e-6) for k in amp)      # identical dynamics, every step
    warm = dict(amp)                                             # adapter no longer zero: state sketch differs
    amp2, _ = local_train(model, warm, data, stream(), sets, plan["train"], cpu, amp_alpha=4.0)
    st2, _ = local_train(model, warm, data, stream(), sets, plan["train"], cpu, sketch_alpha=4.0,
                         sketch_straight_through=True)
    assert not torch.allclose(to_blocks(amp2, names), to_blocks(st2, names), atol=1e-6)
    from scra.core import topk_sparsify
    with pytest.raises(ValueError):
        topk_sparsify(torch.ones(4), 0.0)
