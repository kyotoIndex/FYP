"""CPU checks for the C2-0 decoder-LLM probe with a tiny random Qwen2 and a word-level tokenizer.

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
from scra.core import adapter_names, load_plan, local_train, to_blocks  # noqa: E402
from scra.data import ClientStream  # noqa: E402

WORDS = [f"w{i}" for i in range(30)]
PROMPT_WORDS = ["Premise", ":", "Hypothesis", "Does", "the", "premise", "entail", "hypothesis", "?", "Answer",
                "yes", "maybe", "no"]
ROLES = ["strong", "weak", "weak", "strong", "weak", "weak"]
DOMAINS = ["government", "fiction", "slate", "travel", "telephone", "fiction"]


def build_llm_fixture(root: Path) -> dict:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]"] + PROMPT_WORDS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[PAD]")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True, pad_token_id=0)
    Qwen2ForCausalLM(cfg).save_pretrained(root / "model")
    fast.save_pretrained(root / "model")
    rng = random.Random(1)
    (root / "data").mkdir()

    def rows(n, split):
        out = []
        for c in range(len(ROLES)):
            for j in range(n):
                label = (c + j) % 3
                prem = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 14)))
                hyp = " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 5))) + f" w{label}"
                out.append({"text": prem, "text_pair": hyp, "label": label, "domain": DOMAINS[c], "client": c,
                            "group": f"{split}-{c}-{j}"})
        return out
    train, valid = rows(20, "train"), rows(5, "validation")
    for name, rs in (("train", train), ("validation", valid)):
        (root / "data" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    clients = []
    for c in range(len(ROLES)):
        ids = [i for i, r in enumerate(train) if r["client"] == c]
        clients.append({"id": c, "role": ROLES[c], "samples": len(ids) + c, "train_indices": ids,
                        "domains": {DOMAINS[c]: len(ids)}, "labels": {}})
    (root / "data" / "manifest.json").write_text(json.dumps(
        {"task": "multinli", "files": {"train": {"path": "train.jsonl"}, "validation": {"path": "validation.jsonl"}},
         "clients": clients}))
    plan = json.loads((HERE / "plan_c2.json").read_text())
    plan.update({"id": "c2test", "project_root": str(root), "cache_directory": "cache",
                 "manifest_path": "data/manifest.json", "run_root": "runs", "cpu_threads": 1})
    plan["model"] = {"family": "qwen2", "name": str(root / "model"), "local_files_only": True,
                     "dtype": "float32", "attn_implementation": "eager"}
    plan["lora"] = {"full_rank": 8, "weak_rank": 2, "scale": 2.0}
    plan["federation"] = {"clients_per_round": 6}
    plan["train"].update({"local_steps": 3, "batch_size": 4, "lr": 0.003, "eval_batch_size": 8})
    plan["d0"] = {"seed": 7, "strong_fraction": 0.17}
    plan["c2"].update({"draws": 2, "snapshot_rounds": [0, 2], "snapshot_run_rounds": 2, "eval_per_client": 3,
                       "probe_examples": 6, "max_len": 24})
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return load_plan(path, None)


@pytest.fixture(scope="module")
def c2plan(tmp_path_factory):
    return build_llm_fixture(tmp_path_factory.mktemp("c2"))


def _setup(plan):
    from scra.c2probe import _setup as setup
    return setup(plan, torch.device("cpu"))


def test_prompt_cache_and_cold_start(c2plan):
    data, strong, model, names = _setup(c2plan)
    meta = data.meta
    assert len(meta["label_token_ids"]) == 3 and meta["splits"]["train"]["max_len"] == 24
    assert strong and all(data.clients[c]["role"] == "strong" for c in strong)
    assert len(names) == 2 * 7 and all(a.lora_a.dtype == torch.float32 for a in model.adapters)
    batch = data.train.batch([0, 1, 2], torch.device("cpu"))
    assert set(batch["labels"].tolist()) <= set(meta["label_token_ids"])
    # B = 0: the LoRA model equals the frozen base model exactly
    with torch.no_grad():
        ours = model(batch)
        out = model.hf.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        last = batch["attention_mask"].sum(1) - 1
        ref = model.hf.lm_head(out.last_hidden_state[torch.arange(3), last]).float()
    assert torch.equal(ours, ref)


def test_llm_local_training_variants_and_groups(c2plan):
    from scra.llm import Groups, product_gram
    data, _, model, names = _setup(c2plan)
    state = model.task_state()
    sets = [[1, 5] for _ in names]
    cfg = c2plan["train"]
    cpu = torch.device("cpu")
    stream = lambda: ClientStream(data.clients[1]["train_indices"], 0, 1)  # noqa: E731
    plain, _ = local_train(model, state, data, stream(), sets, cfg, cpu)
    same, _ = local_train(model, state, data, stream(), sets, cfg, cpu, amp_alpha=1.0)
    assert all(torch.equal(plain[k], same[k]) for k in plain)
    amp, _ = local_train(model, state, data, stream(), sets, cfg, cpu, amp_alpha=4.0)
    groups = Groups(names, state)
    assert len(groups.index) == 3                             # q/o, k/v, gate/up/down widths
    for w, blocks in groups.to(amp).items():
        moved = (blocks - groups.to(state)[w]).abs().sum(-1) > 0
        assert all(moved[j].nonzero().flatten().tolist() == [1, 5] for j in range(moved.shape[0]))
    back = groups.apply(groups.to(amp), state)
    assert all(torch.equal(back[k], amp[k]) for k in amp)
    full, _ = local_train(model, state, data, stream(), None, cfg, cpu)
    gram = product_gram([full, plain], state, names)
    explicit = []
    for s in (full, plain):
        explicit.append(torch.cat([(s[f"{n}.lora_b"].double() @ s[f"{n}.lora_a"].double()
                                    - state[f"{n}.lora_b"].double() @ state[f"{n}.lora_a"].double()).reshape(-1)
                                   for n in names]))
    for x in range(2):
        for y in range(2):
            assert math.isclose(gram[x, y], float(explicit[x] @ explicit[y]), rel_tol=1e-9, abs_tol=1e-12)


def test_c2_check(c2plan):
    from scra.c2probe import run_c2_check
    out = Path(c2plan["run_root"]) / "c2check"
    info = run_c2_check(c2plan, out, torch.device("cpu"))
    assert info["adapters"] == 14 and len(info["label_token_ids"]) == 3 and info["one_step_loss_finite"]
    assert 0 <= info["base_model_accuracy"]["core"] <= 1
    ident = json.loads((out / "identity.json").read_text())
    assert "scra/c2probe.py" in ident["code_hashes"] and ident["label_token_ids"] == info["label_token_ids"]
    assert json.loads((out / "check.json").read_text())["adapters"] == 14


def test_cache_and_data_guards(c2plan, tmp_path):
    import shutil
    from scra.llm import LLMData, build_prompt_cache
    root = Path(c2plan["project_root"])
    cache = Path(c2plan["cache_directory"])
    manifest = Path(c2plan["manifest_path"])
    build_prompt_cache(manifest, c2plan["model"]["name"], cache, c2plan["c2"]["max_len"])
    with pytest.raises(ValueError, match="different inputs"):                 # another max_len
        build_prompt_cache(manifest, c2plan["model"]["name"], cache, c2plan["c2"]["max_len"] + 1)
    # a cache whose rows do not follow the manifest is refused
    bad = tmp_path / "data"
    shutil.copytree(root / "data", bad)
    m = json.loads((bad / "manifest.json").read_text())
    m["clients"][0]["train_indices"] = m["clients"][1]["train_indices"]
    (bad / "manifest.json").write_text(json.dumps(m))
    bad_cache = tmp_path / "cache"
    build_prompt_cache(bad / "manifest.json", c2plan["model"]["name"], bad_cache, c2plan["c2"]["max_len"])
    with pytest.raises(ValueError, match="not owned"):
        LLMData(bad_cache, bad / "manifest.json")


def test_c2_snapshot_probe_report(c2plan):
    from scra.c2probe import RULES, run_c2_probe, run_c2_snapshots, write_c2_report
    root = Path(c2plan["run_root"])
    cpu = torch.device("cpu")
    # interrupted snapshot run resumes to the same final state as an uninterrupted one
    run_c2_snapshots(c2plan, root / "c2snap", cpu, stop_after_round=1)
    assert not (root / "c2snap" / "snapshot_run.json").exists()
    run_c2_snapshots(c2plan, root / "c2snap", cpu)
    resumed = torch.load(root / "c2snap" / "snapshots" / "round_0002.pt", weights_only=True)
    straight = Path(c2plan["run_root"]).parent / "straight"
    p2 = json.loads(json.dumps(c2plan)); p2["run_root"] = str(straight)
    run_c2_snapshots(p2, straight / "c2snap", cpu)
    again = torch.load(straight / "c2snap" / "snapshots" / "round_0002.pt", weights_only=True)
    assert all(torch.equal(resumed[k], again[k]) for k in resumed)
    sr = json.loads((root / "c2snap" / "snapshot_run.json").read_text())
    assert set(sr["snapshot_sha256"]) == {"round_0000.pt", "round_0002.pt"} and "core_accuracy" in sr["eval"][0]
    # interrupted probe resumes from the next draw and matches an uninterrupted probe
    run_c2_probe(c2plan, 2, root / "c2probe", cpu, stop_after_draws=1)
    assert (root / "c2probe" / "round_0002.partial.json").exists()
    run_c2_probe(c2plan, 2, root / "c2probe", cpu)
    assert not (root / "c2probe" / "round_0002.partial.json").exists()
    run_c2_probe(p2, 2, straight / "c2probe", cpu)
    a = json.loads((root / "c2probe" / "round_0002.json").read_text())
    b = json.loads((straight / "c2probe" / "round_0002.json").read_text())
    assert [r["amp"] for r in a["rows"]] == [r["amp"] for r in b["rows"]]
    run_c2_probe(c2plan, 0, root / "c2probe", cpu)
    blob = json.loads((root / "c2probe" / "round_0002.json").read_text())
    assert blob["rules"] == RULES and len(blob["rows"]) == 2
    row = blob["rows"][0]
    assert abs(row["aggregate"]["r8"]["product"]["step"] - 1) < 1e-9
    assert row["r8"]["core"] == row["r8"]["core"]
    assert all(math.isfinite(row[k]["acc"]) for k in RULES)
    from scra.c2probe import restrict_to_sets
    from scra.llm import Groups
    full = torch.load(root / "c2snap" / "snapshots" / "round_0002.pt", weights_only=True)
    base = torch.load(root / "c2snap" / "snapshots" / "round_0000.pt", weights_only=True)
    names = sorted({k.split(".lora_")[0] for k in full}, key=lambda n: list(full).index(f"{n}.lora_a"))
    cut = restrict_to_sets(full, base, names, [[0, 3] for _ in names])
    groups = Groups(names, full)
    for w, blocks in groups.to(cut).items():
        diff = (blocks - groups.to(base)[w]).abs().sum(-1) > 0
        assert all(set(diff[j].nonzero().flatten().tolist()) <= {0, 3} for j in range(diff.shape[0]))
    # tamper with a snapshot: the probe must refuse it
    snap = root / "c2snap" / "snapshots" / "round_0000.pt"
    st = torch.load(snap, weights_only=True)
    k0 = next(iter(st)); st[k0] = st[k0] + 1
    torch.save(st, snap)
    (root / "c2probe" / "round_0000.json").unlink()
    with pytest.raises(ValueError, match="hashes"):
        run_c2_probe(c2plan, 0, root / "c2probe", cpu)
    report = write_c2_report(root, gate=0.99)
    assert set(report["verdicts"]) == {"M0_restricted_overshoots_without_new_head", "M1_amplification_corrects",
                                       "M2_one_step_accuracy_signal", "M2_snapshots_used"}
    assert report["verdicts"]["M2_snapshots_used"] == 0 and "not interpretable" in report["verdicts"]["M2_one_step_accuracy_signal"]
    assert (root / "diagnostics_report" / "REPORT_C2_0.md").exists()


def test_c2fed_methods_resume_and_report(c2plan):
    from scra.c2fed import METHODS, RESTRICTED, SPEC, run_c2_fed, select_lr, write_c2fed_report
    cpu = torch.device("cpu")
    fed_root = Path(c2plan["project_root"]) / "runs" / "c2_fed"
    report_cfg = {"lr_select_round": 2, "late_window": [2, 3], "early_window": [1, 2], "osc_from": 1}
    finals = {}
    runs = [(m, m, {}) for m in METHODS] + [("amp_a16", "amp", {"amp_alpha": 16.0})]   # fixture R/r = 4
    for name, m, extra in runs:
        p = json.loads(json.dumps(c2plan))
        p["c2fed"] = {"method": m, "rounds": 3, "eval_every": 1, "checkpoint_every": 1, "diag_every": 2, **extra}
        p["run_root"] = str(fed_root / f"main_{name}")
        out = run_c2_fed(p, Path(p["run_root"]), cpu)
        blob = json.loads(out.read_text())
        assert blob["method"] == m and [e["round"] for e in blob["eval"]] == [0, 1, 2, 3]
        assert (Path(p["run_root"]) / "source_snapshot" / "c2fed.py").exists()
        mech = [x["round"] for x in blob["rounds"] if x["mechanism"]]
        assert mech == ([1, 2] if m in RESTRICTED else [])
        if m in SPEC:
            assert blob["spec"] == dict(zip(("weak_training", "allocation", "aggregation"), SPEC[m]))
        if mech:
            agg = blob["rounds"][0]["mechanism"]["aggregate"]["product"]
            assert agg["step"] is not None and agg["cos"] is not None
        finals[name] = torch.load(Path(p["run_root"]) / "final_state.pt", weights_only=True)
    assert not all(torch.equal(finals["amp"][k], finals["L2"][k]) for k in finals["amp"])
    assert not all(torch.equal(finals["amp"][k], finals["fslora"][k]) for k in finals["amp"])
    assert not all(torch.equal(finals["amp"][k], finals["amp_a16"][k]) for k in finals["amp"])
    assert not all(torch.equal(finals["fslora"][k], finals["fslora_paper"][k]) for k in finals["fslora"])
    assert not all(torch.equal(finals["L2"][k], finals["fedplora"][k]) for k in finals["L2"])
    # interrupted amp run resumes to the same final state
    p = json.loads(json.dumps(c2plan))
    p["c2fed"] = {"method": "amp", "rounds": 3, "eval_every": 1, "checkpoint_every": 1, "diag_every": 2}
    p["run_root"] = str(fed_root.parent / "resume_amp")
    run_c2_fed(p, Path(p["run_root"]), cpu, stop_after_round=1)
    assert not (Path(p["run_root"]) / "fed.json").exists()
    run_c2_fed(p, Path(p["run_root"]), cpu)
    resumed = torch.load(Path(p["run_root"]) / "final_state.pt", weights_only=True)
    assert all(torch.equal(resumed[k], finals["amp"][k]) for k in resumed)
    # learning-rate check runs and the pre-registered selection rule
    for lr in (0.003, 0.001):
        q = json.loads(json.dumps(c2plan))
        q["train"]["lr"] = lr
        q["c2fed"] = {"method": "r8", "rounds": 2, "eval_every": 1, "checkpoint_every": 1,
                      "diag_restricted_rounds": 1}
        q["run_root"] = str(fed_root / f"lr_{lr:.1e}")
        run_c2_fed(q, Path(q["run_root"]), cpu)
    fake = {1e-4: {"eval": [{"round": 10, "core_accuracy": 0.800}]},
            5e-5: {"eval": [{"round": 10, "core_accuracy": 0.803}]},
            2.5e-5: {"eval": [{"round": 10, "core_accuracy": 0.790}]}}
    assert select_lr(fake, 10)["chosen_lr"] == 1e-4                     # within 0.5 pp -> the larger lr
    lr_blob = json.loads((fed_root / "lr_3.0e-03" / "fed.json").read_text())
    assert lr_blob["rounds"][0]["diag_weak_restricted_loss"] is not None
    assert lr_blob["rounds"][1]["diag_weak_restricted_loss"] is None          # diagnostic only in round 1
    rep = write_c2fed_report(fed_root, report_cfg)
    assert "chosen_lr" in rep["lr_check"] and set(rep["main"]) == set(METHODS) | {"amp_a16"}
    assert rep["lr_check"]["diagnostic_weak_local_loss"][0.003]["restricted"] is not None
    v = rep["verdicts"]
    assert {"H5a_increment_only_beats_state_sketch", "H5b_gradient_path_matters", "H6_alpha_R_over_r_ok"} <= set(v)
    assert rep["main"]["amp"]["mechanism"] and not rep["main"]["r8"]["mechanism"]
    assert {"E1_clear_advantage_over_published_fslora", "E2_fedplora_collapses"} <= set(v)
    assert {"F1_zero_fill_drives_gain", "F2_allocation_irrelevant", "F3_reading"} <= set(v)
    assert set(rep["verdicts"]) >= {"H0_restricted_costs_accuracy", "H1_amp_matches_vref"}
    assert (fed_root / "REPORT_C2_1.md").exists()


def test_make_c2fed_plans(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_c2fed", HERE / "make_c2fed.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    plan = mod.base_plan()
    assert plan["c2"]["eval_seed"] == 2040 and plan["c2fed_report"]["late_window"] == [40, 60]
    assert mod.LRS[0] == 2e-4 and len(mod.LRS) == 4


def test_c2fed_diagnostic_reference_is_a_real_r8_round_and_code_guard(c2plan):
    """The mechanism diagnostic's r8 aggregate (strong clients' real runs + weak clients' full replays) must be
    exactly what an r8 run produces in the same round; and a checkpoint refuses to resume under changed code."""
    from scra.c2fed import run_c2_fed
    cpu = torch.device("cpu")
    root = Path(c2plan["project_root"]) / "runs" / "c2_fed_ref"
    p = json.loads(json.dumps(c2plan))
    p["c2fed"] = {"method": "amp", "rounds": 1, "eval_every": 1, "checkpoint_every": 1, "diag_every": 1,
                  "save_diag_reference": True}
    p["run_root"] = str(root / "amp")
    run_c2_fed(p, Path(p["run_root"]), cpu)
    q = json.loads(json.dumps(c2plan))
    q["c2fed"] = {"method": "r8", "rounds": 1, "eval_every": 1, "checkpoint_every": 1}
    q["run_root"] = str(root / "r8")
    run_c2_fed(q, Path(q["run_root"]), cpu)
    ref = torch.load(root / "amp" / "diag_r8_round_0001.pt", weights_only=True)
    real = torch.load(root / "r8" / "final_state.pt", weights_only=True)
    assert all(torch.allclose(ref[k], real[k], atol=1e-7) for k in real)
    # code guard on resume
    g = json.loads(json.dumps(c2plan))
    g["c2fed"] = {"method": "amp", "rounds": 3, "eval_every": 1, "checkpoint_every": 1}
    g["run_root"] = str(root / "guard")
    ckpt = run_c2_fed(g, Path(g["run_root"]), cpu, stop_after_round=1)
    st = torch.load(ckpt, weights_only=False)
    st["code_hashes"] = {**st["code_hashes"], "scra/llm.py": "0" * 64}
    torch.save(st, ckpt)
    with pytest.raises(ValueError, match="source code changed"):
        run_c2_fed(g, Path(g["run_root"]), cpu)


def _ident(d, code, lr=2e-4, rev="abc"):
    plan = {"train": {"lr": lr}, "lora": {"full_rank": 64}, "federation": {}, "d0": {"strong_fraction": 0.1},
            "c2": {"eval_seed": 2040}, "c2fed": {"rounds": 60}, "model": {"name": f"/home/{d.name}/m"}}
    (d / "identity.json").write_text(json.dumps({"plan": plan, "model_revision": rev, "model_config_sha256": "c",
                                                 "cache_fingerprint": {}, "cache_sha256": {}, "label_token_ids": [1],
                                                 "code_hashes": {"scra/c2fed.py": code}}))


def _fed(d, seed, core, loss1=1.0, method="amp"):
    d.mkdir(parents=True, exist_ok=True)
    ev = [{"round": t, "core_accuracy": core / 100, "client_macro_accuracy": 0.8} for t in range(0, 61, 5)]
    (d / "fed.json").write_text(json.dumps({"seed": seed, "method": method, "lr": 2e-4, "eval": ev,
                                            "rounds": [{"loss": loss1}], "amp_alpha": None}))


def test_c2fed_multiseed_report(tmp_path):
    from scra.c2fed import current_c2fed_hash, multiseed, write_c2fed_report
    cur = current_c2fed_hash()
    cfg = {"lr_select_round": 10, "late_window": [40, 60], "early_window": [5, 20], "osc_from": 20}
    late = {"r8": 82.5, "vref": 80.0, "amp": 79.8, "fslora": 79.9, "fslora_paper": 81.0, "amp_zf": 81.1,
            "fslora_zf": 80.9, "vref_zf": 80.2}
    for seed, shift in ((2040, 0.0), (2041, 0.3), (2042, -0.2)):
        for m, x in late.items():
            d = tmp_path / (f"main_{m}" if seed == 2040 else f"seed{seed}_{m}")
            _fed(d, seed, x + shift, method=m)
            _ident(d, cur)
    ms = multiseed(tmp_path, cfg)
    assert len(ms["runs"]) == 24 and not ms["blocked"]
    assert ms["verdicts"]["MS1"] is True and ms["verdicts"]["MS2"] == "equivalent"
    assert abs(ms["paired"]["MS4"]["mean"] - 2.5) < 1e-9
    assert ms["verdicts"]["MS5"] == "fslora_paper better" and abs(ms["paired"]["MS6"]["mean"] - 1.0) < 1e-9
    assert ms["verdicts"]["MS7"] is True and ms["verdicts"]["MS8"] == "equivalent" and ms["verdicts"]["MS9"] is False
    assert ms["verdicts"]["MS11"] is True and ms["verdicts"]["MS12"] == "equivalent"
    rep = write_c2fed_report(tmp_path, cfg)
    assert "multiseed" in rep and "Stage C" in (tmp_path / "REPORT_C2_1.md").read_text()
    blob = json.loads((tmp_path / "seed2041_amp" / "fed.json").read_text()); blob["seed"] = 2040
    (tmp_path / "seed2041_amp" / "fed.json").write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="expected 2041"):
        multiseed(tmp_path, cfg)


def test_c2fed_server_step_scales_the_update(c2plan):
    from scra.c2fed import run_c2_fed
    cpu = torch.device("cpu")
    root = Path(c2plan["project_root"]) / "runs" / "c2_step"
    norms = {}
    for name, extra in (("full", {}), ("quarter", {"server_step": 0.25})):
        p = json.loads(json.dumps(c2plan))
        p["c2fed"] = {"method": "amp", "rounds": 1, "eval_every": 1, "checkpoint_every": 1, **extra}
        p["run_root"] = str(root / name)
        blob = json.loads(run_c2_fed(p, Path(p["run_root"]), cpu).read_text())
        norms[name] = blob["rounds"][0]["update_norm"]
        assert blob["server_step"] == (0.25 if extra else 1.0)
    assert abs(norms["quarter"] - 0.25 * norms["full"]) < 1e-6 * norms["full"]


def test_replay_checks_version_gate_and_protocol_guard(tmp_path):
    from scra.c2fed import multiseed, replay_checks
    cur, old, older = "c" * 64, "o" * 64, "q" * 64
    cfg = {"late_window": [40, 60]}
    # replay rules: full-length replay, late window and round-1 loss
    _fed(tmp_path / "main_amp", 2040, 78.0, 1.0); _ident(tmp_path / "main_amp", old)
    _fed(tmp_path / "replay_main_amp", 2040, 78.1, 1.0005); _ident(tmp_path / "replay_main_amp", cur)
    _fed(tmp_path / "main_vref", 2040, 80.0, method="vref"); _ident(tmp_path / "main_vref", older)
    _fed(tmp_path / "replay_main_vref", 2040, 80.0, 1.01, method="vref"); _ident(tmp_path / "replay_main_vref", cur)
    rp = replay_checks(tmp_path)
    assert rp["main_amp"]["status"] == "pass" and rp["main_vref"]["status"] == "fail"
    # seeds 2041/2042: amp from the verified old version, vref from the current code
    for seed in (2041, 2042):
        _fed(tmp_path / f"seed{seed}_amp", seed, 78.0); _ident(tmp_path / f"seed{seed}_amp", old)
        _fed(tmp_path / f"seed{seed}_vref", seed, 80.0, method="vref"); _ident(tmp_path / f"seed{seed}_vref", cur)
    ms = multiseed(tmp_path, cfg, current_hash=cur)
    assert ms["used"]["amp_s2040"] == "main_amp"                       # old version verified by its replay
    assert ms["used"]["vref_s2040"] == "replay_main_vref"              # failed replay -> current-code replay used
    assert ms["verdicts"]["MS1"] is not None and not str(ms["verdicts"]["MS1"]).startswith("blocked")
    # a run of an unverified version without a replay is blocked, and so is every comparison that needs it
    _fed(tmp_path / "seed2041_r8", 2041, 82.0, method="r8"); _ident(tmp_path / "seed2041_r8", older)
    ms = multiseed(tmp_path, cfg, current_hash=cur)
    assert "r8_s2041" in ms["blocked"] and ms["verdicts"]["MS3"].startswith("blocked")
    assert ms["paired"]["MS3"] is None
    # protocol guard: paths may differ (other HPC account), protocol fields and content may not
    _ident(tmp_path / "seed2041_r8", cur, lr=1e-4)
    with pytest.raises(ValueError, match="train"):
        multiseed(tmp_path, cfg, current_hash=cur)
    _ident(tmp_path / "seed2041_r8", cur, rev="other")
    with pytest.raises(ValueError, match="model_revision"):
        multiseed(tmp_path, cfg, current_hash=cur)
