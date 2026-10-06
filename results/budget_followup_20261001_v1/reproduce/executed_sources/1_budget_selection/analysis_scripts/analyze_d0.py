#!/usr/bin/env python3
"""Deterministic, read-only tables and curves from a receipt-gated D0 export.

No recipe is selected here. The input summary is produced by summarize_pcla_v7_d0.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def optional_mean(values):
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return mean(clean) if len(clean) == len(values) and clean else None


def same(a, b) -> bool:
    return a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                      and math.isclose(a, b, rel_tol=0, abs_tol=1e-12))


def check_export(inp: Path) -> dict:
    hashes = load(inp / "output_hashes.json")
    for filename, expected in hashes.items():
        assert sha(inp / filename) == expected, f"export hash mismatch: {filename}"
    summary = load(inp / "summary.json")
    assert summary["phase"] == "D0" and summary["planned_jobs"] == 351
    assert summary["planned_recipes"] == 117
    assert summary["audited_completed_jobs"] == 351
    assert summary["three_seed_complete_recipes"] == 117
    assert summary["summary_states"] == {"audited_completed": 351}
    inventory = list(csv.DictReader((inp / "inventory.csv").open(newline="")))
    assert len(inventory) == 351 and all(x["summary_state"] == "audited_completed" for x in inventory)
    assert len({x["job_id"] for x in inventory}) == 351
    recipes = load(inp / "recipes.json")
    assert len(recipes) == 117 and all(x["three_seed_complete"] for x in recipes)
    assert all(x["planned_seeds"] == [401, 409, 419] and
               x["audited_seeds"] == [401, 409, 419] for x in recipes)
    return summary


def verify_runs(runs: list[dict], rounds: list[dict], wire: list[dict],
                require_full: bool = True) -> tuple[dict, dict, dict]:
    if require_full:
        assert len(runs) == 351
    assert len({x["job"]["job_id"] for x in runs}) == len(runs)
    by_run = {x["job"]["job_id"]: x for x in runs}
    round_map = defaultdict(dict)
    for x in rounds:
        jid, n = x["job_id"], x["round"]
        assert jid in by_run and n not in round_map[jid], (jid, n)
        round_map[jid][n] = x
    wire_map = defaultdict(list)
    for x in wire:
        assert x["job_id"] in by_run
        wire_map[x["job_id"]].append(x)
    if require_full:
        assert Counter(x["job"]["recipe_id"] for x in runs) == Counter({
            f"D{i:03d}": 3 for i in range(1, 118)})
    assert {x["job"]["seed"] for x in runs} == {401, 409, 419}
    cap_set = {x["budget_cap_bytes"] for x in runs}
    assert len(cap_set) == 1, "B8 must be identical across runs"
    for jid, run in by_run.items():
        rs = round_map[jid]
        stop, half, completed = (run[k] for k in
                                 ("budget_stop_round", "half_budget_round", "completed_round"))
        assert sorted(rs) == list(range(completed + 1)), f"missing/extra round {jid}"
        assert completed >= 100 and 0 <= half <= stop <= completed <= 200
        assert run["b8_raw"]["round"] == stop and run["half_raw"]["round"] == half
        for readout, n in (("b8_raw", stop), ("half_raw", half)):
            for key in ("accuracy", "loss", "samples", "predicted_class_counts"):
                assert same(run[readout][key], rs[n]["raw_" + key]), (jid, readout, key)
        for label, n in (("b8_ema", stop), ("half_ema", half)):
            if run[label] is None:
                assert rs[n]["ema_accuracy"] is None
            else:
                for key in ("accuracy", "loss", "samples"):
                    assert same(run[label][key], rs[n]["ema_" + key]), (jid, label, key)
        for n, row in rs.items():
            expected = "initial" if n == 0 else ("deployed_prefix" if n <= stop else "diagnostic_after_budget")
            assert row["bucket"] == expected and row["raw_samples"] == 4096
        assert rs[stop]["deployed_bytes"] == run["deployed_bytes"]
        assert rs[completed]["diagnostic_after_budget_bytes"] == run["diagnostic_after_budget_bytes"]
        assert rs[half]["deployed_bytes"] == run["half_budget_bytes"]
        assert all(rs[n]["deployed_bytes"] >= rs[n-1]["deployed_bytes"] and
                   rs[n]["diagnostic_after_budget_bytes"] >= rs[n-1]["diagnostic_after_budget_bytes"]
                   for n in range(1, completed+1)), f"non-monotone wire counter {jid}"
        assert run["deployed_bytes"] <= run["budget_cap_bytes"]
        assert run["half_budget_bytes"] <= run["budget_cap_bytes"] / 2
        for bucket, total in (("deployed_prefix", run["deployed_bytes"]),
                              ("diagnostic_after_budget", run["diagnostic_after_budget_bytes"])):
            assert sum(w["bytes"] for w in wire_map[jid] if w["ledger"] == "committed"
                       and w["bucket"] == bucket) == total, (jid, bucket)
        attempted = run.get("actual_attempted_wire_bytes")
        if attempted is not None:
            assert sum(w["bytes"] for w in wire_map[jid]) == attempted, jid
            assert sum(w["bytes"] for w in wire_map[jid] if w["ledger"] == "uncommitted_attempt") == run["failed_attempt_wire_bytes"]
        for key, first, last in (("deployed_cost", 1, stop),
                                 ("diagnostic_cost", stop+1, completed)):
            cost = run[key]
            for metric, value in cost["sum"].items():
                observed = sum(rs[n]["cost"].get(metric, 0) for n in range(first, last+1))
                assert math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-8), (jid, key, metric)
            for metric, value in cost["peak"].items():
                observed = max((rs[n]["cost"].get(metric, 0) for n in range(first, last+1)), default=0)
                assert same(observed, value), (jid, key, metric)
        assert all(n in rs and rs[n]["raw_accuracy"] is not None for n in range(96, 101))
    assert len(rounds) == sum(x["completed_round"] + 1 for x in runs)
    return by_run, round_map, wire_map


def endpoint_row(run: dict, rs: dict) -> dict:
    job = run["job"]
    stop, half = run["budget_stop_round"], run["half_budget_round"]
    tail = [rs[n]["raw_accuracy"] for n in range(96, 101)]
    tail_loss = [rs[n]["raw_loss"] for n in range(96, 101)]
    return dict(job_id=job["job_id"], recipe_id=job["recipe_id"], seed=job["seed"],
                group=job["group"], method=job["method"], representation=job["representation"],
                lr=job["lr"], period=job["period"], rank=job["rank"], xi=job["xi"],
                hardware_block=job["hardware_block"],
                gpu_models="|".join(run["time_block"]["gpu_models"]),
                sharing_modes="|".join(run["time_block"]["sharing_modes"]),
                exclusive_single_model=run["time_block"]["exclusive_single_model"],
                budget_cap_bytes=run["budget_cap_bytes"], budget_stop_round=stop,
                stop_boundary="round_cap" if stop == job["max_rounds"] else "byte_boundary",
                completed_round=run["completed_round"], deployed_bytes=run["deployed_bytes"],
                diagnostic_after_budget_bytes=run["diagnostic_after_budget_bytes"],
                actual_attempted_wire_bytes=run.get("actual_attempted_wire_bytes"),
                failed_attempt_wire_bytes=run.get("failed_attempt_wire_bytes"),
                wire_attempt_accounting=run.get("wire_attempt_accounting"),
                half_budget_round=half, half_budget_bytes=run["half_budget_bytes"],
                half_raw_accuracy=run["half_raw"]["accuracy"],
                half_raw_loss=run["half_raw"]["loss"],
                b8_raw_accuracy=run["b8_raw"]["accuracy"],
                b8_raw_loss=run["b8_raw"]["loss"],
                b8_raw_class_counts=run["b8_raw"]["predicted_class_counts"],
                r96_100_raw_accuracy_mean=mean(tail),
                r96_100_raw_loss_mean=mean(tail_loss),
                native_initialization_present=run["native_initialization"] is not None,
                audit_report_sha256=run["audit_report_sha256"],
                config_sha256=run["config_sha256"],
                source_manifest_sha256=run["source_manifest_sha256"])


def recipe_row(recipe: dict, endpoints: list[dict]) -> dict:
    assert len(endpoints) == 3
    first = endpoints[0]
    keys = ("b8_raw_accuracy", "half_raw_accuracy", "r96_100_raw_accuracy_mean",
            "deployed_bytes", "diagnostic_after_budget_bytes", "budget_stop_round")
    return dict(recipe_id=recipe["recipe_id"], group=first["group"], method=first["method"],
                representation=first["representation"], lr=first["lr"], period=first["period"],
                rank=first["rank"], xi=first["xi"], n_seeds=3,
                seed401_b8_raw_accuracy=endpoints[0]["b8_raw_accuracy"],
                seed409_b8_raw_accuracy=endpoints[1]["b8_raw_accuracy"],
                seed419_b8_raw_accuracy=endpoints[2]["b8_raw_accuracy"],
                **{k + "_mean": optional_mean([e[k] for e in endpoints]) for k in keys})


def cost_rows(run: dict) -> list[dict]:
    job = run["job"]
    result = []
    for bucket, cost_key, wall_key, bytes_key in (
            ("deployed_prefix", "deployed_cost", "deployed_wall", "deployed_bytes"),
            ("diagnostic_after_budget", "diagnostic_cost", "diagnostic_wall", "diagnostic_after_budget_bytes")):
        cost, wall = run[cost_key], run[wall_key]
        row = dict(job_id=job["job_id"], recipe_id=job["recipe_id"], seed=job["seed"],
                   bucket=bucket, bytes=run[bytes_key], gpu_models="|".join(run["time_block"]["gpu_models"]),
                   sharing_modes="|".join(run["time_block"]["sharing_modes"]),
                   exclusive_single_model=run["time_block"]["exclusive_single_model"],
                   wall_missing_rounds=wall["missing_rounds"])
        row.update({"cost_" + k: v for k, v in cost["sum"].items()})
        row.update({"peak_" + k: v for k, v in cost["peak"].items()})
        row.update({"wall_" + k: v for k, v in wall["sum"].items()})
        result.append(row)
    native = run["native_initialization"]
    if native is not None:
        row = dict(job_id=job["job_id"], recipe_id=job["recipe_id"], seed=job["seed"],
                   bucket="native_initialization", bytes=None,
                   gpu_models="|".join(run["time_block"]["gpu_models"]),
                   sharing_modes="|".join(run["time_block"]["sharing_modes"]),
                   exclusive_single_model=run["time_block"]["exclusive_single_model"],
                   native_public_rows=native.get("public_rows"),
                   native_effective_lr=native.get("effective_lr"),
                   native_initialization_seed=native.get("initialization_seed"),
                   native_full_weight_gradient_targets=native.get("full_weight_gradient_targets"),
                   native_svd_dtype=native.get("svd_dtype"))
        row.update({"native_" + k: v for k, v in native["cost"].items()})
        result.append(row)
    return result


def ema_row(run: dict, rs: dict) -> dict:
    job = run["job"]
    return dict(job_id=job["job_id"], recipe_id=job["recipe_id"], seed=job["seed"],
                half_ema_accuracy=(run["half_ema"] or {}).get("accuracy"),
                half_ema_loss=(run["half_ema"] or {}).get("loss"),
                b8_ema_accuracy=(run["b8_ema"] or {}).get("accuracy"),
                b8_ema_loss=(run["b8_ema"] or {}).get("loss"),
                r96_100_ema_accuracy_mean=optional_mean([rs[n]["ema_accuracy"] for n in range(96, 101)]),
                r96_100_ema_loss_mean=optional_mean([rs[n]["ema_loss"] for n in range(96, 101)]))


def exclusive_timing_row(run: dict) -> dict:
    job = run["job"]
    assert job["seed"] == 401
    block = run["time_block"]
    row = dict(job_id=job["job_id"], recipe_id=job["recipe_id"], group=job["group"],
               method=job["method"], representation=job["representation"], lr=job["lr"],
               period=job["period"], hardware_block=job["hardware_block"],
               gpu_models="|".join(block["gpu_models"]),
               sharing_modes="|".join(block["sharing_modes"]),
               exclusive_single_model=block["exclusive_single_model"],
               missing_wall_rounds=run["deployed_wall"]["missing_rounds"],
               note="原始分项；各列不可相加推出总算法时间；EMA评价与保存等包含在原始执行中")
    row.update({"cost_" + k: v for k, v in run["deployed_cost"]["sum"].items()})
    row.update({"wall_" + k: v for k, v in run["deployed_wall"]["sum"].items()})
    return row


def controlled_rows(endpoints: list[dict]) -> list[dict]:
    # Same seed, LR, representation and refresh period: local versus cyclic.
    idx = {(x["representation"], x["method"], x["lr"], x["period"], x["seed"]): x
           for x in endpoints if x["group"] == "representation"}
    result = []
    for x in endpoints:
        if x["group"] != "representation" or x["method"] != "local":
            continue
        y = idx.get((x["representation"], "cyclic", x["lr"], x["period"], x["seed"]))
        assert y is not None
        result.append(dict(contrast="local_minus_cyclic", representation=x["representation"],
                           lr=x["lr"], period=x["period"], seed=x["seed"],
                           left_job_id=x["job_id"], right_job_id=y["job_id"],
                           b8_raw_accuracy_diff=x["b8_raw_accuracy"]-y["b8_raw_accuracy"],
                           r96_100_raw_accuracy_diff=x["r96_100_raw_accuracy_mean"]-y["r96_100_raw_accuracy_mean"],
                           left_hardware_block=x["hardware_block"], right_hardware_block=y["hardware_block"],
                           same_hardware_block=x["hardware_block"] == y["hardware_block"]))
    # N has no period, so this is a matched seed/LR/method comparison of representation schedules.
    for x in endpoints:
        if x["group"] != "representation" or x["representation"] == "N":
            continue
        y = idx.get(("N", x["method"], x["lr"], 0, x["seed"]))
        assert y is not None
        result.append(dict(contrast="representation_minus_N", representation=x["representation"],
                           lr=x["lr"], period=x["period"], seed=x["seed"],
                           left_job_id=x["job_id"], right_job_id=y["job_id"],
                           b8_raw_accuracy_diff=x["b8_raw_accuracy"]-y["b8_raw_accuracy"],
                           r96_100_raw_accuracy_diff=x["r96_100_raw_accuracy_mean"]-y["r96_100_raw_accuracy_mean"],
                           left_hardware_block=x["hardware_block"], right_hardware_block=y["hardware_block"],
                           same_hardware_block=x["hardware_block"] == y["hardware_block"]))
    return sorted(result, key=lambda z: (z["contrast"], z["left_job_id"]))


def curve_html(recipes: list[dict], endpoints: list[dict], round_map: dict) -> str:
    by_recipe = defaultdict(list)
    for e in endpoints:
        by_recipe[e["recipe_id"]].append(e)
    palette = {401: "#0f766e", 409: "#a855f7", 419: "#d97706"}
    cards = []
    for recipe in recipes:
        rid = recipe["recipe_id"]
        runs = sorted(by_recipe[rid], key=lambda e: e["seed"])
        parts = []
        for e in runs:
            rs = round_map[e["job_id"]]
            stop = e["budget_stop_round"]
            def points(first, last):
                return " ".join(f"{32 + 340*n/200:.1f},{139 - 112*rs[n]['raw_accuracy']:.1f}"
                                for n in range(first, last+1))
            color = palette[e["seed"]]
            parts.append(f'<polyline points="{points(0, stop)}" fill="none" stroke="{color}" stroke-width="1.8"/>')
            if e["completed_round"] > stop:
                parts.append(f'<polyline points="{points(stop, e["completed_round"])}" fill="none" stroke="{color}" stroke-width="1.6" stroke-dasharray="3 3"/>')
            cx, cy = 32 + 340*stop/200, 139 - 112*e["b8_raw_accuracy"]
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}"><title>seed {e["seed"]}: B8 round {stop}, accuracy {e["b8_raw_accuracy"]:.4f}</title></circle>')
        title = f'{rid} · {recipe["method"]} / {recipe["representation"]} · LR {recipe["lr"]} · p{recipe["period"]}'
        svg = ('<svg viewBox="0 0 390 160" role="img" aria-label="' + html.escape(title) + '">'
               '<path d="M32 20V139H374" fill="none" stroke="#94a3b8"/>'
               '<path d="M32 83H374" fill="none" stroke="#e2e8f0"/>'
               '<text x="1" y="32">1.0</text><text x="3" y="142">0</text>'
               '<text x="32" y="156">0</text><text x="352" y="156">200轮</text>'
               + "".join(parts) + '</svg>')
        cards.append('<article data-search="' + html.escape(title.lower()) + '"><h2>' + html.escape(title)
                     + '</h2>' + svg + '</article>')
    return ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PCLA D0 117配方原始准确率曲线</title><style>body{font:14px system-ui,sans-serif;color:#17202d;background:#f8fafc;margin:24px}'
            'main{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px}'
            'article{background:white;border:1px solid #dbe2ea;border-radius:8px;padding:8px 12px}'
            'h2{font-size:14px;margin:4px 0}svg{width:100%}input{padding:8px;width:300px;margin:12px 0 20px}</style>'
            '<h1>PCLA D0 · 117 配方完整曲线</h1><p>开发集原始准确率；每卡三条 seed 曲线。'
            '<span style="color:#0f766e">■ 401</span>　<span style="color:#a855f7">■ 409</span>　'
            '<span style="color:#d97706">■ 419</span>。实线是 B8 部署前缀，虚线是 B8 后机制诊断；圆点为各 seed 的 B8 模型。'
            '<p>这些曲线是训练内开发结果，不选择胜者；同轮不同方法可能花费不同字节。</p>'
            '<input id="q" placeholder="搜索配方、方法、表示或 LR" oninput="for(const x of document.querySelectorAll(\'article\'))x.hidden=!x.dataset.search.includes(this.value.toLowerCase())">'
            '<main>' + ''.join(cards) + '</main></html>')


def build(inp: Path, recipe_file: Path, out: Path, independent_file: Path | None = None):
    inp, recipe_file, out = inp.resolve(), recipe_file.resolve(), out.resolve()
    assert inp != out and out.is_dir()
    summary = check_export(inp)
    runs, rounds, wire = [load(inp / name) for name in ("runs.json", "rounds.json", "wire.json")]
    recipes_meta = load(inp / "recipes.json")
    by_run, round_map, _ = verify_runs(runs, rounds, wire)
    recipe_settings = list(csv.DictReader(recipe_file.open(newline="")))
    assert len(recipe_settings) == 117
    assert {x["recipe_id"] for x in recipe_settings} == {x["recipe_id"] for x in recipes_meta}
    endpoints = [endpoint_row(x, round_map[x["job"]["job_id"]]) for x in
                 sorted(runs, key=lambda x: (x["job"]["recipe_id"], x["job"]["seed"]))]
    by_recipe = defaultdict(list)
    for x in endpoints:
        by_recipe[x["recipe_id"]].append(x)
    recipe_rows = [recipe_row(x, by_recipe[x["recipe_id"]]) for x in recipes_meta]
    if independent_file is not None:
        independent = load(independent_file)
        independent_by_id = {x["id"]: x for x in independent}
        assert len(independent_by_id) == 117 and set(independent_by_id) == set(by_recipe)
        for row in recipe_rows:
            rid = row["recipe_id"]
            other = independent_by_id[rid]
            assert math.isclose(100*row["b8_raw_accuracy_mean"], other["mean"], abs_tol=1e-10)
            assert sorted(e["budget_stop_round"] for e in by_recipe[rid]) == sorted(other["rounds"])
            for e in by_recipe[rid]:
                assert math.isclose(100*e["b8_raw_accuracy"], other["seeds"][str(e["seed"])],
                                    abs_tol=0.000051), (rid, e["seed"])
    method_best = {method: max(x["b8_raw_accuracy_mean"] for x in recipe_rows if x["method"] == method)
                   for method in {x["method"] for x in recipe_rows}}
    method_windows = [dict(recipe_id=x["recipe_id"], method=x["method"],
                           b8_raw_accuracy_mean=x["b8_raw_accuracy_mean"],
                           method_best_mean=method_best[x["method"]],
                           gap_from_method_best_pp=100*(method_best[x["method"]]-x["b8_raw_accuracy_mean"]),
                           within_0p2pp_candidate=(method_best[x["method"]]-x["b8_raw_accuracy_mean"] <= 0.002 + 1e-12),
                           note="同method开发候选窗口；未按时间破平局，未作V选择")
                      for x in recipe_rows]
    ema = [ema_row(x, round_map[x["job"]["job_id"]]) for x in
           sorted(runs, key=lambda x: (x["job"]["recipe_id"], x["job"]["seed"]))]
    ema_by_recipe = defaultdict(list)
    for x in ema:
        ema_by_recipe[x["recipe_id"]].append(x)
    ema_recipes = [dict(recipe_id=x["recipe_id"], n_seeds=3,
                        **{k + "_mean": optional_mean([e[k] for e in ema_by_recipe[x["recipe_id"]]])
                           for k in ("half_ema_accuracy", "half_ema_loss", "b8_ema_accuracy",
                                     "b8_ema_loss", "r96_100_ema_accuracy_mean", "r96_100_ema_loss_mean")})
                   for x in recipes_meta]
    costs = [row for run in sorted(runs, key=lambda x: x["job"]["job_id"]) for row in cost_rows(run)]
    timing = [exclusive_timing_row(x) for x in sorted(runs, key=lambda x: x["job"]["job_id"])
              if x["job"]["seed"] == 401]
    assert len(timing) == 117
    block_counts = Counter((x["hardware_block"], x["gpu_models"], x["sharing_modes"],
                            x["exclusive_single_model"]) for x in endpoints)
    hardware_blocks = [dict(hardware_block=k[0], gpu_models=k[1], sharing_modes=k[2],
                            exclusive_single_model=k[3], run_count=count)
                       for k, count in sorted(block_counts.items())]
    round_rows = [dict(job_id=x["job_id"], recipe_id=by_run[x["job_id"]]["job"]["recipe_id"],
                       seed=by_run[x["job_id"]]["job"]["seed"], **{k: v for k, v in x.items() if k != "job_id"},
                       cumulative_committed_bytes=None, incremental_committed_bytes=None)
                  for x in sorted(rounds, key=lambda z: (z["job_id"], z["round"]))]
    previous = defaultdict(int)
    for x in round_rows:
        cumulative = x["deployed_bytes"] + x["diagnostic_after_budget_bytes"]
        x["cumulative_committed_bytes"] = cumulative
        x["incremental_committed_bytes"] = cumulative - previous[x["job_id"]]
        previous[x["job_id"]] = cumulative
    surfaces = [x for x in recipe_rows if x["group"] == "representation"]
    assert len(surfaces) == 42
    contrasts = controlled_rows(endpoints)
    files = {
        "recipe_117.csv": (recipe_rows, list(recipe_rows[0])),
        "method_candidate_windows.csv": (method_windows, list(method_windows[0])),
        "seed_endpoints_351.csv": (endpoints, list(endpoints[0])),
        "ema_seed_endpoints_351.csv": (ema, list(ema[0])),
        "ema_recipe_117.csv": (ema_recipes, list(ema_recipes[0])),
        "rounds_complete.csv": (round_rows, list(round_rows[0])),
        "cost_phases.csv": (costs, sorted({key for row in costs for key in row})),
        "seed401_timing_components_117.csv": (timing, list(timing[0])),
        "hardware_sharing_blocks.csv": (hardware_blocks, list(hardware_blocks[0])),
        "wire_events.csv": (sorted(wire, key=lambda x: (x["job_id"], x["bucket"], x["ledger"], x["kind"])),
                            sorted({key for row in wire for key in row})),
        "representation_lr_surface.csv": (surfaces, list(recipe_rows[0])),
        "controlled_seed_differences.csv": (contrasts, list(contrasts[0])),
        "native_and_recipe_settings.csv": (recipe_settings, list(recipe_settings[0])),
    }
    for filename, (rows, columns) in files.items():
        write_csv(out / filename, rows, columns)
    (out / "all_recipe_curves.html").write_text(curve_html(recipe_rows, endpoints, round_map))
    proof = {"input_export": str(inp), "input_summary_sha256": sha(inp / "summary.json"),
             "input_hashes_sha256": sha(inp / "output_hashes.json"),
             "recipe_settings_sha256": sha(recipe_file), "planned_recipes": len(recipe_rows),
             "independent_endpoint_sha256": sha(independent_file) if independent_file else None,
             "audited_seed_endpoints": len(endpoints), "round_rows": len(round_rows),
             "seed401_timing_rows": len(timing),
             "seed401_exclusive_4090d_rows": sum(x["exclusive_single_model"] and
                                                 x["gpu_models"] == "NVIDIA GeForce RTX 4090 D" for x in timing),
             "hardware_sharing_block_rows": len(hardware_blocks),
             "representation_surface_recipes": len(surfaces), "controlled_seed_differences": len(contrasts),
             "wire_rows": len(wire), "b8_cap_bytes": runs[0]["budget_cap_bytes"],
             "summary_stage": summary["stage_status"],
             "output_sha256": {name: sha(out / name) for name in sorted([*files, "all_recipe_curves.html"])}}
    (out / "validation.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--recipes", type=Path, required=True)
    ap.add_argument("--independent", type=Path, help="independently read 351 round readouts")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    print(json.dumps(build(args.export, args.recipes, args.out, args.independent), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
