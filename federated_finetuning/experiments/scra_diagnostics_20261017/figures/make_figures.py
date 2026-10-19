#!/usr/bin/env python3
"""Thesis figures and tables for stage D of docs/designs/roadmap_2026-10-18_amplified_restricted.md.

Reads only run JSON / Markdown files (never writes outside this folder) and writes into this folder:
  fig1_core_curves, fig2_oscillation, fig3_head_mechanism, fig4_ablation_robustness,
  fig5_batch2_decomposition (.pdf + .png), cost_table.md, checks.md and INDEX.md.

Every plotted number is recomputed from the raw curves (d0_rounds.json "eval" / result.json "history")
or from the D2 / D3 JSON, then checked against the SUMMARY.md / REPORT*.md tables (cell by cell, to the
printed precision) and against summary.json (exactly). checks.md lists every check; INDEX.md lists the
mismatches and the convention differences found.

Metric definitions (same as scra/batch.py and scra/batch2.py):
  core   accuracy averaged over the 30 core weak clients (not strong, domain not held by a strong client,
         from the vref reference identity), mean of the evaluations at rounds >= 110;
  osc    mean |change| of all-client macro accuracy between consecutive evaluations from round 50 on;
  window early 10-50, mid 60-100, late 110-150 (core accuracy).

Run from the repository root (exact command in INDEX.md):
  python \
      federated_finetuning/experiments/scra_diagnostics_20261017/figures/make_figures.py
"""
from __future__ import annotations

import json
import platform
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
FF = HERE.parents[2]                       # federated_finetuning/
REPO = FF.parent
RUNS = FF / "runs"
DOCS = REPO / "docs" / "designs"
B1 = RUNS / "scra_batch_20261018"
B2 = RUNS / "scra_batch2_20261018"
DIAG = RUNS / "scra_diagnostics_20261017_v1"
SWEEP = RUNS / "headroom_strongfrac_20261017_v1"
REPORTS = DIAG / "diagnostics_report"
NARRATIVE = DOCS / "research_two_directions_2026-10-17.md"


def earlier(tag: str) -> Path:
    return RUNS / f"scra_diagnostics_20261017_v1_{tag}"


SEEDS = [2040, 2041, 2042]
LATE, OSC_FROM = 110, 50
WINDOWS = {"early": (10, 50), "mid": (60, 100), "late": (110, 150)}
SNAPS = [20, 60, 120]
T975_DF2 = 4.303                           # two-sided 95% t quantile, df = 2 (as in scra/batch2.py)

# palette: reference instance of the dataviz skill (validated: first three slots all-pairs, four adjacent)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
REF = INK2                                 # reference series (r8, frozen head, external baselines)
WASH = "#f0efec"
SEED_MARKERS = ["o", "s", "^"]

CORE: list[str] = []
CHECKS: list[dict] = []
NOTES: list[str] = []
OUTPUTS: list[str] = []


# ================================================================ loading
def core_set(strong: list[int], domains: dict[str, str]) -> list[str]:
    held = {domains[str(c)] for c in strong}
    return sorted((c for c, d in domains.items() if int(c) not in strong and d not in held), key=int)


def metrics(curve: list[dict]) -> dict:
    rounds = [e["round"] for e in curve]
    core = [100 * float(np.mean([e["clients"][c][0] for c in CORE])) for e in curve]
    allc = [100 * e["client_macro_accuracy"] for e in curve]
    osc_acc = [a for r, a in zip(rounds, allc) if r >= OSC_FROM]
    late = [i for i, r in enumerate(rounds) if r >= LATE]
    return {"rounds": rounds, "core_curve": core, "all_curve": allc,
            "core": float(np.mean([core[i] for i in late])),
            "all": float(np.mean([allc[i] for i in late])),
            "osc": float(np.mean(np.abs(np.diff(osc_acc)))),
            "min_after_50": float(min(osc_acc)),
            "win": {k: float(np.mean([c for r, c in zip(rounds, core) if lo <= r <= hi]))
                    for k, (lo, hi) in WINDOWS.items()}}


def run_paths() -> dict[str, Path]:
    paths = {}
    for root in (B1, B2):
        for p in sorted(root.glob("*/d0/d0_rounds.json")):
            paths[p.parent.parent.name] = p
    paths["L2_s2040"] = DIAG / "d0" / "d0_rounds.json"
    paths["vref_s2040"] = earlier("vref") / "d0" / "d0_rounds.json"
    for tag, name in [("budget8", "vref_r1_s2040"), ("rcs", "rcs_s2040"),
                      ("fixedshrink", "fixedshrink_s2040"), ("weaklr", "weaklr_s2040")]:
        paths[name] = earlier(tag) / "d0" / "d0_rounds.json"
    for s in SEEDS:
        paths[f"r8_s{s}"] = SWEEP / f"fedavg_r8_seed{s}" / "result.json"
        paths[f"r2_s{s}"] = SWEEP / f"fedavg_r2_seed{s}" / "result.json"
        paths[f"prefix_s{s}"] = SWEEP / f"hetero_r8r2_sf10_seed{s}" / "result.json"
        paths[f"prefix_sf20_s{s}"] = SWEEP / f"hetero_r8r2_sf20_seed{s}" / "result.json"
    return paths


def load_runs() -> dict[str, dict]:
    runs = {}
    for name, p in run_paths().items():
        blob = json.loads(p.read_text())
        curve = blob.get("eval") or blob.get("history")
        if not curve or curve[-1]["round"] < 150:
            raise RuntimeError(f"{name}: incomplete curve in {p}")
        cost = blob.get("cost") or {}
        ident_p = p.parent / "identity.json"
        ident = json.loads(ident_p.read_text()) if ident_p.exists() else {}
        runs[name] = {"path": p, "peak": blob.get("peak_memory_mib") or cost.get("peak_memory_mib"),
                      "seconds": blob.get("elapsed_seconds") or cost.get("elapsed_seconds"),
                      "host": ident.get("host"), "gpu": ident.get("gpu"), **metrics(curve)}
    return runs


def seeds_of(R: dict, key: str) -> list[str]:
    """Run names of a method: all three seeds when present, otherwise the seed-2040 run only."""
    names = [f"{key}_s{s}" for s in SEEDS if f"{key}_s{s}" in R]
    if not names:
        raise KeyError(key)
    return names


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


# ================================================================ checks
NUM = re.compile(r"[-+−]?\d+(?:\.\d+)?")


def _split(line: str) -> list[str]:
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def md_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("|") and re.match(r"^\|\s*:?-{3,}", lines[i + 1]):
            header, rows, j = _split(lines[i]), [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(_split(lines[j]))
                j += 1
            out.append((header, rows))
            i = j
        else:
            i += 1
    return out


def md_cell(text: str, row, col: str) -> str | None:
    match = row if callable(row) else (lambda s: s == row)
    for header, rows in md_tables(text):
        if col in header:
            k = header.index(col)
            for r in rows:
                if match(r[0]) and k < len(r):
                    return r[k]
    return None


def section(text: str, start: str) -> str:
    """Text from the line starting with `start` up to the next level-2 heading."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(start):
            j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
            return "\n".join(lines[i:j])
    return ""


def check_md(scope: str, item: str, mine: list[float], text: str, row, col: str, source: str) -> bool:
    """Compare recomputed values with the numbers printed in one Markdown cell, to printed precision."""
    cell = md_cell(text, row, col)
    if cell is None:
        CHECKS.append(dict(scope=scope, item=item, mine=mine, ref="(cell not found)", source=source, ok=False))
        return False
    toks = NUM.findall(cell)
    ok = len(toks) >= len(mine)
    for tok, m in zip(toks, mine):
        dp = len(tok.split(".")[1]) if "." in tok else 0
        ok &= m is not None and abs(m - float(tok.replace("−", "-"))) <= 0.5 * 10 ** -dp + 1e-6
    CHECKS.append(dict(scope=scope, item=item, mine=mine, ref=cell, source=f"{source} [{col}]", ok=bool(ok)))
    return bool(ok)


def check_val(scope: str, item: str, mine: float, ref: float, source: str, tol: float = 1e-6) -> bool:
    ok = ref is not None and abs(mine - ref) <= tol
    CHECKS.append(dict(scope=scope, item=item, mine=[mine], ref=ref, source=source, ok=bool(ok)))
    return bool(ok)


def fmt(v) -> str:
    if isinstance(v, list):
        return ", ".join(fmt(x) for x in v)
    return f"{v:.4f}" if isinstance(v, float) else str(v)


# ================================================================ style
def setup_style() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in ("Arial Unicode MS", "Hiragino Sans GB", "PingFang SC", "Heiti TC",
                       "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei") if f in have]
    if not cjk:
        NOTES.append("未找到中文字体，图中中文可能显示为方框。")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cjk + ["DejaVu Sans"], "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 7.6, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    return cjk[0] if cjk else "DejaVu Sans"


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}")
    plt.close(fig)
    OUTPUTS.append(stem)


def value_label(ax, x, y, text, **kw):
    kw.setdefault("fontsize", 7.4)
    kw.setdefault("color", INK2)
    ax.text(x, y, text, **kw)


# ================================================================ cross-checks against SUMMARY / REPORT
B1_MAIN = [("r8", "r8 (upper)"), ("r2", "r2"), ("prefix", "prefix hetero (L0)"), ("L2", "L2 restricted, owner"),
           ("vref", "vref (full train, upload r)"), ("amp", "amp (amplified restricted, Hájek)")]
B1_ABL = ["amp_s2040", "amp_owner_s2040", "amp_prefix_s2040", "amp_a2_s2040", "amp_a8_s2040"]
B1_ROB = [("r = 1 (upload 1/8)", "L2_r1_s2040", "vref_r1_s2040", "amp_r1_s2040"),
          ("ranks {1,2,4}", "L2_het_s2040", "vref_het_s2040", "amp_het_s2040"),
          ("20% strong", "L2_sf20_s2040", "vref_sf20_s2040", "amp_sf20_s2040")]
B2_METHODS = [("r8", "r8 (upper)"), ("vref", "vref (full train, upload r/R, Hájek)"), ("amp", "amp (ours)"),
              ("fslora_st", "A1a FSLoRA forward, amp gradient path, balanced + Hájek"),
              ("fslora_hajek", "A1b FSLoRA forward, true gradient, balanced + Hájek"),
              ("fslora_zerofill", "A1c FSLoRA forward, true gradient, balanced + zero-fill"),
              ("lrcomp", "A2 LoRA lr × R/r + server ÷ R/r"),
              ("fslora_paper", "ext. FSLoRA-style (our AdamW protocol; random, zero-fill)"),
              ("fedplora", "ext. Fed-PLoRA style (random, owner)"),
              ("flasc", "ext. FLASC-style (full train, global top-r/R upload by weak, zero-fill)"),
              ("L2", "L2 (balanced, owner)")]
B2_COMPARISONS = [("A1a_state_sketch", "amp", "fslora_st"), ("A1b_gradient_path", "fslora_st", "fslora_hajek"),
                  ("A1c_aggregation", "fslora_hajek", "fslora_zerofill"),
                  ("A1_total_vs_fslora_forward", "amp", "fslora_hajek"), ("A2_lr_compensation", "amp", "lrcomp"),
                  ("E1_vs_fslora_style", "amp", "fslora_paper"), ("E2_vs_fedplora_style", "amp", "fedplora"),
                  ("E3_vs_flasc_style", "amp", "flasc")]


def closure(R, mid, lo, hi):
    return (R[mid]["core"] - R[lo]["core"]) / (R[hi]["core"] - R[lo]["core"])


def paired(R, a: str, b: str, key="core") -> np.ndarray:
    get = (lambda n: R[n][key]) if key == "core" else (lambda n: R[n]["win"][key])
    return np.array([get(f"{a}_s{s}") - get(f"{b}_s{s}") for s in SEEDS])


def run_checks(R: dict, D2: dict, D3: dict) -> None:
    # ---- batch 1: summary.json (exact) and SUMMARY.md (printed precision)
    s1 = json.loads((B1 / "summary.json").read_text())["runs"]
    for name, ref in s1.items():
        for k_mine, k_ref in (("core", "core10"), ("all", "all"), ("osc", "osc"), ("min_after_50", "min_after_50")):
            check_val("batch1 summary.json", f"{name}.{k_ref}", R[name][k_mine], ref[k_ref],
                      rel(B1 / "summary.json"))
        if ref.get("peak_memory_mib") is not None:
            check_val("batch1 summary.json", f"{name}.peak_memory_mib", R[name]["peak"], ref["peak_memory_mib"],
                      rel(B1 / "summary.json"))
        if ref.get("elapsed_seconds") is not None:
            check_val("batch1 summary.json", f"{name}.elapsed_seconds", R[name]["seconds"], ref["elapsed_seconds"],
                      rel(B1 / "summary.json"))
    md1 = (B1 / "SUMMARY.md").read_text()
    src1 = rel(B1 / "SUMMARY.md")
    sd_conv = {0: 0, 1: 0}
    for key, row in B1_MAIN:
        vals = [R[f"{key}_s{s}"]["core"] for s in SEEDS]
        for s, v in zip(SEEDS, vals):
            check_md("batch1 main", f"{key}_s{s} core", [v], md1, row, f"core s{s}", src1)
        cell = md_cell(md1, row, "core mean ± sd")
        for ddof in (0, 1):   # record which sd convention the summary used
            toks = [float(t) for t in NUM.findall(cell or "")]
            if len(toks) == 2 and abs(toks[1] - np.std(vals, ddof=ddof)) <= 0.005 + 1e-6:
                sd_conv[ddof] += 1
        check_md("batch1 main", f"{key} core mean ± sd(ddof=0)", [np.mean(vals), np.std(vals, ddof=0)],
                 md1, row, "core mean ± sd", src1)
        check_md("batch1 main", f"{key} all mean", [np.mean([R[f"{key}_s{s}"]["all"] for s in SEEDS])],
                 md1, row, "all mean", src1)
        check_md("batch1 main", f"{key} osc mean", [np.mean([R[f"{key}_s{s}"]["osc"] for s in SEEDS])],
                 md1, row, "osc mean", src1)
    NOTES.append(f"batch 1 SUMMARY.md 的 “mean ± sd” 用总体标准差（ddof=0，{sd_conv[0]}/6 行吻合；ddof=1 吻合 "
                 f"{sd_conv[1]}/6 行），batch 2 SUMMARY.md 用样本标准差（ddof=1）。同一组数据因此在两份汇总中写作 "
                 f"80.19 ± 0.06 与 80.19 ± 0.07（r8），80.12 ± 0.14 与 80.12 ± 0.17（amp），L2 0.53 与 0.65；"
                 f"research_two_directions §7 沿用前者，§8 沿用后者。图中不画标准差（用各 seed 点与最小–最大范围）。")
    for name in B1_ABL:
        check_md("batch1 ablation", f"{name} core", [R[name]["core"]], md1, name, "core", src1)
        check_md("batch1 ablation", f"{name} Δ vs amp", [R[name]["core"] - R["amp_s2040"]["core"]], md1, name,
                 "Δ vs amp", src1)
        check_md("batch1 ablation", f"{name} closure", [closure(R, name, "L2_s2040", "vref_s2040")], md1, name,
                 "closure L2→vref", src1)
        check_md("batch1 ablation", f"{name} osc", [R[name]["osc"]], md1, name, "osc", src1)
    for row, lo, hi, mid in B1_ROB:
        check_md("batch1 robustness", f"{row} L2", [R[lo]["core"]], md1, row, "L2", src1)
        check_md("batch1 robustness", f"{row} vref", [R[hi]["core"]], md1, row, "vref", src1)
        check_md("batch1 robustness", f"{row} amp", [R[mid]["core"]], md1, row, "amp", src1)
        check_md("batch1 robustness", f"{row} closure", [closure(R, mid, lo, hi)], md1, row, "closure", src1)
        check_md("batch1 robustness", f"{row} osc amp", [R[mid]["osc"]], md1, row, "osc amp", src1)
    for header, rows in md_tables(md1):
        if "peak memory MiB" in header:
            for r in rows:
                check_md("batch1 cost", f"{r[0]} peak", [R[r[0]]["peak"]], md1, r[0], "peak memory MiB", src1)
                check_md("batch1 cost", f"{r[0]} seconds", [R[r[0]]["seconds"]], md1, r[0], "seconds", src1)

    # ---- batch 2
    s2 = json.loads((B2 / "summary.json").read_text())
    for name, ref in s2["runs"].items():
        for k in ("core", "osc"):
            check_val("batch2 summary.json", f"{name}.{k}", R[name][k], ref[k], rel(B2 / "summary.json"))
        for w in WINDOWS:
            check_val("batch2 summary.json", f"{name}.windows.{w}", R[name]["win"][w], ref["windows"][w],
                      rel(B2 / "summary.json"))
    md2 = (B2 / "SUMMARY.md").read_text()
    src2 = rel(B2 / "SUMMARY.md")
    for key, row in B2_METHODS:
        vals = [R[f"{key}_s{s}"]["core"] for s in SEEDS]
        for s, v in zip(SEEDS, vals):
            check_md("batch2 main", f"{key}_s{s} late", [v], md2, row, f"late s{s}", src2)
        check_md("batch2 main", f"{key} late mean ± sd(ddof=1)", [np.mean(vals), np.std(vals, ddof=1)], md2, row,
                 "late mean ± sd", src2)
        for w in ("early", "mid"):
            check_md("batch2 main", f"{key} {w}", [np.mean([R[f"{key}_s{s}"]["win"][w] for s in SEEDS])], md2, row,
                     w, src2)
        check_md("batch2 main", f"{key} osc", [np.mean([R[f"{key}_s{s}"]["osc"] for s in SEEDS])], md2, row,
                 "osc", src2)
    for comp, a, b in B2_COMPARISONS:
        d = paired(R, a, b)
        sd = float(np.std(d, ddof=1))
        h = T975_DF2 * sd / np.sqrt(len(d))
        ref = s2["verdicts"][comp]["paired"]
        for s, v, rv in zip(SEEDS, d, ref["per_seed"]):
            check_val("batch2 summary.json", f"{comp} per_seed s{s}", float(v), rv, rel(B2 / "summary.json"))
        check_val("batch2 summary.json", f"{comp} mean", float(d.mean()), ref["mean"], rel(B2 / "summary.json"))
        check_md("batch2 comparisons", f"{comp} per seed", list(d), md2, comp, "per seed (late)", src2)
        check_md("batch2 comparisons", f"{comp} mean ± sd", [d.mean(), sd], md2, comp, "mean ± sd", src2)
        check_md("batch2 comparisons", f"{comp} 95% CI", [d.mean() - h, d.mean() + h], md2, comp, "95% CI", src2)
        for w in ("early", "mid"):
            check_md("batch2 comparisons", f"{comp} {w}", [paired(R, a, b, w).mean()], md2, comp, w, src2)

    # ---- earlier discriminating runs: each run's REPORT.md and research_two_directions §4
    sec4 = section(NARRATIVE.read_text(), "## 4. ")
    srcn = rel(NARRATIVE) + " §4"
    for tag, name in [("rcs", "rcs_s2040"), ("fixedshrink", "fixedshrink_s2040"), ("weaklr", "weaklr_s2040"),
                      ("budget8", "vref_r1_s2040")]:
        rep = (earlier(tag) / "diagnostics_report" / "REPORT.md").read_text()
        srcr = rel(earlier(tag) / "diagnostics_report" / "REPORT.md")
        this = lambda s: s.startswith("this run")  # noqa: E731
        check_md("earlier runs", f"{name} core", [R[name]["core"]], rep, this, "core weak", srcr)
        check_md("earlier runs", f"{name} all", [R[name]["all"]], rep, this, "all", srcr)
    doc_rows = [("r8", "r8_s2040"), ("vref（完整训练，上传 2/8）", "vref_s2040"),
                ("budget8（完整训练，上传 1/8）", "vref_r1_s2040"), ("L2（受限训练，拥有者平均）", "L2_s2040"),
                ("RCS 在线（Hájek + 自适应收缩）", "rcs_s2040"), ("固定收缩 0.63（Hájek）", "fixedshrink_s2040"),
                ("弱端学习率 × 0.6（Hájek）", "weaklr_s2040")]
    for row, name in doc_rows:
        check_md("earlier runs", f"{name} core (doc)", [R[name]["core"]], sec4, row, "核心弱端", srcn)
        check_md("earlier runs", f"{name} osc (doc)", [R[name]["osc"]], sec4, row, "轮间波动", srcn)
        check_md("earlier runs", f"{name} min after 50 (doc)", [R[name]["min_after_50"]], sec4, row,
                 "第 50 轮后最低", srcn)
    check_md("earlier runs", "r2_s2040 osc (doc)", [R["r2_s2040"]["osc"]], sec4, "参照：r2 同质", "轮间波动", srcn)
    check_md("earlier runs", "prefix_s2040 osc (doc, first of three)", [R["prefix_s2040"]["osc"]], sec4,
             lambda s: s.startswith("参照：前缀异构"), "轮间波动", srcn)

    # ---- research_two_directions §7: round-10 claim (amp 65.63, r8 65.43, L2 59.72)
    claim = {"amp": 65.63, "r8": 65.43, "L2": 59.72}
    for key, v in claim.items():
        at10 = lambda n: R[n]["core_curve"][R[n]["rounds"].index(10)]  # noqa: E731
        check_val("narrative §7", f"{key}_s2040 core at round 10", at10(f"{key}_s2040"), v,
                  rel(NARRATIVE) + " §7 结论 1", tol=0.005 + 1e-6)

    # ---- D2 / D3: summary vs per-draw rows (exact) and vs REPORT_D2 / REPORT_D3 (printed precision)
    rep2, rep3 = (REPORTS / "REPORT_D2.md").read_text(), (REPORTS / "REPORT_D3.md").read_text()
    for snap in SNAPS:
        sec2, sec3 = section(rep2, f"## Snapshot round {snap} "), section(rep3, f"## Snapshot round {snap} ")
        b2s, b3s = D2[snap]["summary"], D3[snap]["summary"]
        for rule in ("vref", "shrink", "shrink_fullhead", "vref_restrhead"):
            d = np.array([row[rule]["core"] - row["r8"]["core"] for row in D2[snap]["rows"]])
            check_val("D2 rows", f"r{snap} {rule} core diff mean", float(d.mean()), b2s[rule]["core_diff_vs_r8_mean"],
                      rel(DIAG / "d2" / f"round_{snap:04d}.json"))
            check_md("D2 report", f"r{snap} {rule} core diff mean / std",
                     [100 * b2s[rule]["core_diff_vs_r8_mean"]], sec2, rule, "core diff mean", rel(REPORTS / "REPORT_D2.md"))
            check_md("D2 report", f"r{snap} {rule} core diff std",
                     [100 * b2s[rule]["core_diff_vs_r8_std"]], sec2, rule, "core diff std", rel(REPORTS / "REPORT_D2.md"))
        for rule in ("shrink", "amp_hajek", "vref"):
            h = b3s[rule]["head"]
            check_md("D3 report", f"r{snap} {rule} head step", [h["step_along_ref_mean"], h["step_along_ref_std"]],
                     sec3, rule, "head step along r8", rel(REPORTS / "REPORT_D3.md"))
        for rule in ("shrink", "fh_hajek", "amp_hajek", "vref"):
            lp = b3s[rule]["lora_product"]
            check_md("D3 report", f"r{snap} {rule} LoRA step", [lp["step_along_ref_mean"], lp["step_along_ref_std"]],
                     sec3, rule, "LoRA step along r8", rel(REPORTS / "REPORT_D3.md"))
        # the same rules were replayed by both D2 and D3; record the size of the replay difference
        for rule in ("shrink", "shrink_fullhead"):
            NOTES_D2D3.append((snap, rule, 100 * b2s[rule]["core_diff_vs_r8_mean"], 100 * b3s[rule]["core_diff_vs_r8_mean"]))
        NOTES_D2D3_HEAD.append((snap, b2s["shrink"]["head"]["step_along_ref_mean"], b3s["shrink"]["head"]["step_along_ref_mean"]))
        NOTES_R8ACC.append((100 * b2s["r8"]["acc_mean"], 100 * b3s["r8"]["acc_mean"]))


NOTES_D2D3: list[tuple] = []
NOTES_D2D3_HEAD: list[tuple] = []
NOTES_R8ACC: list[tuple] = []


# ================================================================ figures
NAMES = {"r8": "r8（全员 rank 8）", "vref": "vref（完整训练，上传 r/R）", "L2": "L2（受限训练，无修正）",
         "amp": "amp（放大受限训练）"}


def fig1(R) -> dict:
    series = [("r8", REF, 2.6, 0.45, 1), ("vref", AQUA, 1.4, 1.0, 2), ("L2", ORANGE, 1.5, 1.0, 3),
              ("amp", BLUE, 1.5, 1.0, 4)]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), gridspec_kw=dict(width_ratios=[1, 1.1], hspace=0.42,
                                                                       wspace=0.22))
    titles = [["(a) seed 2040，第 10–150 轮", "(b) seed 2040，第 50–150 轮"],
              ["(c) 3 个 seed 均值，阴影为最小–最大", "(d) 3 个 seed 均值，第 50–150 轮"]]
    shown = {}
    for row, mode in enumerate(("seed", "mean")):
        for col, lo in enumerate((10, 50)):
            ax = axes[row, col]
            ax.axvspan(LATE - 2, 152, color=WASH, lw=0, zorder=0)
            handles = []
            for key, color, lw, alpha, z in series:
                if mode == "seed":
                    r = R[f"{key}_s2040"]
                    x, y, late = np.array(r["rounds"]), np.array(r["core_curve"]), r["core"]
                    band = None
                else:
                    ys = np.array([R[f"{key}_s{s}"]["core_curve"] for s in SEEDS])
                    x, y = np.array(R[f"{key}_s2040"]["rounds"]), ys.mean(0)
                    late = float(np.mean([R[f"{key}_s{s}"]["core"] for s in SEEDS]))
                    band = (ys.min(0), ys.max(0))
                m = x >= lo
                if band is not None:
                    ax.fill_between(x[m], band[0][m], band[1][m], color=color, alpha=0.16 if key != "r8" else 0.12,
                                    lw=0, zorder=z)
                ax.plot(x[m], y[m], color=color, lw=lw, alpha=alpha, zorder=z + 5, solid_capstyle="round",
                        marker="o" if key != "r8" else None, ms=3.4, mec="white", mew=0.6)
                handles.append(Line2D([], [], color=color, lw=lw if key != "r8" else 2.6, alpha=alpha,
                                      marker="o" if key != "r8" else None, ms=3.4, mec="white", mew=0.6,
                                      label=NAMES[key]))
                shown[(mode, key)] = late
            ax.set_title(titles[row][col])
            ax.set_xlim(lo - 3, 153)
            ax.set_xticks(range(lo if lo % 20 == 10 else lo, 151, 20 if lo == 10 else 10))
            if col == 0:
                ax.set_ylabel("核心弱端准确率 (%)")
            else:
                means = "\n".join(f"{NAMES[k].split('（')[0]}  {shown[(mode, k)]:.2f}" for k in ("r8", "vref", "amp", "L2"))
                ax.text(0.97, 0.05, "后期均值\n" + means, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.2, color=INK2, linespacing=1.35)
            ax.set_xlabel("通信轮次")
            ax.text(LATE + 1, ax.get_ylim()[1] - 0.01 * np.ptp(ax.get_ylim()), "后期窗口",
                    fontsize=7, color=MUTED, va="top")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=4, handlelength=2.2,
               columnspacing=1.6)
    save(fig, "fig1_core_curves")
    return shown


OSC_GROUPS = [
    ("参照", "#8f8d86", [("r8", "r8（全员 rank 8）"), ("vref", "vref（完整训练，上传 r/R）"),
                       ("r2", "r2（全员 rank 2，同质）"), ("prefix", "前缀异构 L0")]),
    ("受限训练 + 服务器端修正", ORANGE, [("L2", "L2（无修正）"), ("rcs", "RCS 在线收缩"),
                                  ("fixedshrink", "固定收缩 0.63"), ("weaklr", "弱端学习率 × 0.6")]),
    ("放大受限训练", BLUE, [("amp", "amp（α = R/r = 4）"), ("amp_a2", "amp，α = 2"), ("amp_a8", "amp，α = 8")]),
    ("第二批对照", AQUA, [("fslora_st", "A1a FSLoRA 前向 + amp 梯度路径"), ("fslora_hajek", "A1b FSLoRA 前向 + Hájek"),
                      ("fslora_zerofill", "A1c FSLoRA 前向 + 零填充"), ("lrcomp", "A2 学习率 × R/r，服务器 ÷ R/r"),
                      ("fslora_paper", "FSLoRA 式（外部）"), ("fedplora", "Fed-PLoRA 式（外部）"),
                      ("flasc", "FLASC 式（外部）")]),
]


def fig2(R) -> list:
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    y, ticks, labels, rows = 0.0, [], [], []
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for gname, color, items in OSC_GROUPS:
        ax.text(-0.005, y, gname, transform=tr, ha="right", va="center", fontsize=8.2, weight="bold", color=INK)
        y -= 1
        for key, label in items:
            names = seeds_of(R, key)
            vals = [R[n]["osc"] for n in names]
            m = float(np.mean(vals))
            ax.barh(y, m, height=0.62, color=color, zorder=2)
            if len(vals) > 1:
                for k, v in enumerate(vals):
                    ax.scatter(v, y + (k - 1) * 0.17, s=9, marker="o", facecolor=INK, edgecolor="white", lw=0.4,
                               zorder=3)
            value_label(ax, max(vals) + 0.05, y, f"{m:.2f}" + ("" if len(vals) > 1 else "（1 seed）"), va="center")
            ticks.append(y)
            labels.append(label)
            rows.append((gname, key, label, len(vals), m, vals))
            y -= 1
        y -= 0.5
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    r8 = float(np.mean([R[f"r8_s{s}"]["osc"] for s in SEEDS]))
    ax.axvline(r8, color=INK2, lw=0.8, zorder=1)
    ax.text(r8 + 0.04, 0.35, f"r8 = {r8:.2f}", fontsize=7.2, color=INK2, va="bottom")
    ax.set_xlim(0, 3.45)
    ax.set_ylim(y + 0.7, 0.9)
    ax.set_xlabel("轮间波动 (pp)：第 50 轮起相邻两次评估之间全员准确率变化的平均绝对值")
    ax.set_title("各方法的轮间波动（3-seed 方法取均值，黑点为各 seed；标“1 seed”的为 seed 2040 单次运行）",
                 fontsize=8.6, loc="left", x=-0.43)
    save(fig, "fig2_oscillation")
    return rows


def _grouped_bars(ax, cats, series, width, fmtv, neg=False, fs=6.8):
    x = np.arange(len(cats))
    n = len(series)
    for i, (label, color, means, stds) in enumerate(series):
        xs = x + (i - (n - 1) / 2) * width
        ax.bar(xs, means, width * 0.88, color=color, label=label, zorder=2)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, capthick=0.8, zorder=3)
        for xi, m, s in zip(xs, means, stds):
            if neg and m < 0:
                value_label(ax, xi, m - s - 0.06, fmtv(m), ha="center", va="top", fontsize=fs)
            else:
                value_label(ax, xi, m + s + 0.03, fmtv(m), ha="center", va="bottom", fontsize=fs)
    ax.set_xticks(x, cats)
    ax.grid(axis="x", visible=False)


def fig3(D2, D3) -> dict:
    cats = [f"第 {s} 轮" for s in SNAPS]
    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.72, wspace=0.3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[0, 1])]
    out = {}

    def d3(rule, part, k):
        return [D3[s]["summary"][rule][part][k] for s in SNAPS]

    # (a) head step along r8 (D3)
    ax = axes[0]
    ser = [("收缩 0.63（受限训练）", ORANGE, d3("shrink", "head", "step_along_ref_mean"), d3("shrink", "head", "step_along_ref_std")),
           ("amp（放大受限训练）", BLUE, d3("amp_hajek", "head", "step_along_ref_mean"),
            d3("amp_hajek", "head", "step_along_ref_std"))]
    _grouped_bars(ax, cats, ser, 0.36, lambda v: f"{v:.2f}")
    ax.axhline(1.0, color=INK2, lw=0.8, zorder=1)
    ax.set_xlim(-0.5, 2.75)
    ax.text(2.73, 1.03, "r8", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_ylim(0, 2.45)
    ax.set_ylabel("沿 r8 方向的步长（相对 r8）")
    ax.set_title("(a) 分类头步长（D3）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    out["head"] = {r: d3(r, "head", "step_along_ref_mean") for r in ("shrink", "amp_hajek")}

    # (b) head-swap decomposition (D2): core accuracy minus r8 on the same draw
    ax = axes[1]
    d2 = lambda rule, k: [100 * D2[s]["summary"][rule][k] for s in SNAPS]  # noqa: E731
    ser = [("shrink：受限 LoRA + 受限头部", ORANGE, d2("shrink", "core_diff_vs_r8_mean"), d2("shrink", "core_diff_vs_r8_std")),
           ("shrink_fullhead：受限 LoRA + 完整头部", AQUA, d2("shrink_fullhead", "core_diff_vs_r8_mean"),
            d2("shrink_fullhead", "core_diff_vs_r8_std")),
           ("vref_restrhead：完整 LoRA + 受限头部", YELLOW, d2("vref_restrhead", "core_diff_vs_r8_mean"),
            d2("vref_restrhead", "core_diff_vs_r8_std"))]
    _grouped_bars(ax, cats, ser, 0.26, lambda v: f"{v:.2f}", neg=True, fs=6.8)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_ylim(-5.9, 0.4)
    ax.set_ylabel("核心弱端准确率 − r8 (pp)")
    ax.set_title("(c) 头部互换分解（D2）：核心弱端准确率与 r8 之差")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    out["swap"] = {r: d2(r, "core_diff_vs_r8_mean") for r in ("shrink", "shrink_fullhead", "vref_restrhead")}

    # (c) frozen weak heads move the pressure onto the LoRA components (D3)
    ax = axes[2]
    ser = [("收缩 0.63（受限训练）", ORANGE, d3("shrink", "lora_product", "step_along_ref_mean"),
            d3("shrink", "lora_product", "step_along_ref_std")),
           ("冻结弱端头部 + Hájek（fh_hajek）", REF, d3("fh_hajek", "lora_product", "step_along_ref_mean"),
            d3("fh_hajek", "lora_product", "step_along_ref_std")),
           ("amp（放大受限训练）", BLUE, d3("amp_hajek", "lora_product", "step_along_ref_mean"),
            d3("amp_hajek", "lora_product", "step_along_ref_std"))]
    _grouped_bars(ax, cats, ser, 0.28, lambda v: f"{v:.2f}", fs=6.6)
    ax.axhline(1.0, color=INK2, lw=0.8, zorder=1)
    ax.set_ylim(0, 3.0)
    ax.set_ylabel("沿 r8 方向的步长（相对 r8）")
    ax.set_title("(b) LoRA 乘积步长（D3）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    out["lora"] = {r: d3(r, "lora_product", "step_along_ref_mean") for r in ("shrink", "fh_hajek", "amp_hajek")}
    save(fig, "fig3_head_mechanism")
    return out


def fig4(R) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.75))
    amp = R["amp_s2040"]["core"]
    # (a) ablations as differences from amp (seed 2040)
    ax = axes[0]
    rows = [("amp_a2_s2040", "α = 2（放大不足）"), ("amp_a8_s2040", "α = 8（放大过度）"),
            ("amp_owner_s2040", "拥有者平均代替 Hájek"), ("amp_prefix_s2040", "前缀分配代替覆盖均衡")]
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(-0.3, 0.3, color=WASH, lw=0, zorder=0)
    abl = {}
    for yi, (name, label) in zip(ys, rows):
        d = R[name]["core"] - amp
        abl[name] = d
        ax.barh(yi, d, height=0.55, color=BLUE, zorder=2)
        value_label(ax, d - 0.06, yi, f"{d:+.2f}", ha="right", va="center")
    for key, color, label in (("L2_s2040", ORANGE, "L2"), ("vref_s2040", AQUA, "vref")):
        d = R[key]["core"] - amp
        ax.axvline(d, color=color, lw=1.3, zorder=1)
        ax.text(d + 0.05, len(rows) - 0.4, f"{label} {d:+.2f}", color=INK2, fontsize=7, ha="left", va="bottom")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.text(0.34, -0.72, "浅色带：±0.3 pp\n单 seed 判据", fontsize=6.6, color=MUTED, ha="left", va="center")
    ax.set_yticks(ys, [r[1] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-2.75, 0.95)
    ax.set_ylim(-1.0, len(rows) - 0.1)
    ax.set_xlabel("核心弱端准确率 − amp (pp)")
    ax.set_title(f"(a) 消融（seed 2040，amp = {amp:.2f}）", pad=16)

    # (b) robustness: L2 / vref / amp per setting, closure annotated
    ax = axes[1]
    settings = [("基准：r = 2，10% 强端", "L2_s2040", "vref_s2040", "amp_s2040"),
                ("r = 1（上传 1/8）", "L2_r1_s2040", "vref_r1_s2040", "amp_r1_s2040"),
                ("rank 混合 {1, 2, 4}", "L2_het_s2040", "vref_het_s2040", "amp_het_s2040"),
                ("20% 强端", "L2_sf20_s2040", "vref_sf20_s2040", "amp_sf20_s2040")]
    ys = np.arange(len(settings))[::-1]
    r8 = R["r8_s2040"]["core"]
    ax.axvline(r8, color=INK2, lw=0.8, zorder=1)
    ax.text(r8, len(settings) - 0.45, f"r8 {r8:.2f}", fontsize=7, color=INK2, ha="center", va="bottom")
    rob = {}
    for yi, (label, lo, hi, mid) in zip(ys, settings):
        a, b, c = R[lo]["core"], R[hi]["core"], R[mid]["core"]
        ax.plot([a, b], [yi, yi], color=AXIS, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(a, yi, s=34, color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(b, yi, s=46, marker="D", color=AQUA, edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(c, yi, s=30, color=BLUE, edgecolor="white", lw=0.8, zorder=4)
        cl = closure(R, mid, lo, hi)
        rob[label] = (a, b, c, cl)
        value_label(ax, 80.62, yi, f"追回 {cl:.2f}", va="center", ha="left")
    ax.set_yticks(ys, [s[0] for s in settings])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(77.4, 81.2)
    ax.set_ylim(-0.6, len(settings) - 0.1)
    ax.set_xlabel("核心弱端准确率 (%)，第 110–150 轮")
    ax.set_title("(b) 稳健性（seed 2040）", pad=16)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=ORANGE, mec="white", ms=6, label="L2"),
                       Line2D([], [], ls="", marker="D", color=AQUA, mec="white", ms=6, label="vref"),
                       Line2D([], [], ls="", marker="o", color=BLUE, mec="white", ms=6, label="amp")],
              loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig4_ablation_robustness")
    return {"ablation": abl, "robust": rob}


MECH = [("A1a_state_sketch", "amp", "fslora_st", "amp − A1a\n（是否对状态做 sketch）", BLUE),
        ("A1b_gradient_path", "fslora_st", "fslora_hajek", "A1a − A1b\n（梯度路径 / eps）", ORANGE),
        ("A1c_aggregation", "fslora_hajek", "fslora_zerofill", "A1b − A1c\n（Hájek 与零填充）", AQUA),
        ("A2_lr_compensation", "amp", "lrcomp", "amp − A2\n（前向放大与调学习率）", YELLOW)]
EXT = [("E1_vs_fslora_style", "amp", "fslora_paper", "amp − FSLoRA 式"),
       ("E2_vs_fedplora_style", "amp", "fedplora", "amp − Fed-PLoRA 式"),
       ("E3_vs_flasc_style", "amp", "flasc", "amp − FLASC 式")]
LEVELS = [("r8", "r8"), ("vref", "vref"), ("amp", "amp（本文）"), ("fslora_st", "A1a FSLoRA 前向 + amp 梯度"),
          ("fslora_hajek", "A1b FSLoRA 前向 + Hájek"), ("lrcomp", "A2 学习率 × R/r"),
          ("flasc", "FLASC 式"), ("L2", "L2"), ("fslora_zerofill", "A1c FSLoRA 前向 + 零填充"),
          ("fslora_paper", "FSLoRA 式"), ("fedplora", "Fed-PLoRA 式")]


def _paired_panel(ax, rows, thresholds=True):
    ys = np.arange(len(rows))[::-1]
    out = {}
    for yi, (label, d, color) in zip(ys, rows):
        m, sd = float(d.mean()), float(np.std(d, ddof=1))
        h = T975_DF2 * sd / np.sqrt(len(d))
        ax.barh(yi, m, height=0.5, color=color, zorder=2)
        ax.plot([m - h, m + h], [yi, yi], color=INK, lw=0.9, zorder=3)
        for e in (m - h, m + h):
            ax.plot([e, e], [yi - 0.1, yi + 0.1], color=INK, lw=0.9, zorder=3)
        for k, v in enumerate(d):
            ax.scatter(v, yi + (k - 1) * 0.14, marker=SEED_MARKERS[k], s=15, facecolor="white", edgecolor=INK,
                       lw=0.8, zorder=4)
        value_label(ax, max(m + h, d.max()) + 0.08, yi, f"{m:+.2f}", va="center", color=INK, fontsize=7.6)
        out[label] = (m, list(d), (m - h, m + h))
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    if thresholds:
        for t in (-0.3, 0.3):
            ax.axvline(t, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    return out


def fig5(R) -> dict:
    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.15, 1], hspace=0.75, wspace=0.62)
    out = {}
    # (a) mechanism controls: paired per-seed differences
    ax = fig.add_subplot(gs[0, 0])
    out["mech"] = _paired_panel(ax, [(lab, paired(R, a, b), c) for _, a, b, lab, c in MECH])
    ax.set_xlim(-0.6, 3.45)
    ax.set_xlabel("配对差值 (pp)，后期 110–150 轮")
    ax.set_title("(a) 机制对照（每对只差一个因素）")
    ax.legend(handles=[Line2D([], [], ls="", marker=m, mfc="white", mec=INK, ms=4.5, label=f"seed {s}")
                       for m, s in zip(SEED_MARKERS, SEEDS)]
              + [Line2D([], [], color=INK, lw=0.9, label="95% t 区间"),
                 Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="±0.3 pp 判据")],
              loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, fontsize=6.8, handletextpad=0.3,
              columnspacing=1.0)

    # (b) mechanism differences by training phase
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(3)
    phase = {}
    for comp, a, b, lab, c in MECH:
        vals = [float(paired(R, a, b, w).mean()) for w in WINDOWS]
        phase[comp] = vals
        if comp == "A1c_aggregation":
            continue
        ax.plot(xs, vals, color=c, lw=1.6, marker="o", ms=4, mec="white", mew=0.6, zorder=3)
        value_label(ax, xs[-1] + 0.1, vals[-1], lab.split("\n")[0], va="center", ha="left", color=INK2, fontsize=7)
    a1c = phase["A1c_aggregation"]
    ax.text(0.02, 0.97, "A1b − A1c 未画出（量级不同）：\n" + " / ".join(f"{v:+.2f}" for v in a1c),
            transform=ax.transAxes, fontsize=6.8, color=INK2, va="top")
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xticks(xs, ["前期\n10–50", "中期\n60–100", "后期\n110–150"])
    ax.set_xlim(-0.25, 2.95)
    ax.set_ylim(-0.4, 1.75)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("配对差值 (pp)，3 个 seed 均值")
    ax.set_title("(b) 分阶段的机制差值")
    out["phase"] = phase

    # (c) external positioning (several factors differ)
    ax = fig.add_subplot(gs[1, 0])
    out["ext"] = _paired_panel(ax, [(lab, paired(R, a, b), "#a8a69e") for _, a, b, lab in EXT], thresholds=False)
    ax.set_xlim(-0.6, 6.4)
    ax.set_xlabel("配对差值 (pp)，后期 110–150 轮")
    ax.set_title("(c) 外部基线定位（同时改变多个因素）")

    # (d) absolute late core accuracy of every batch-2 method
    ax = fig.add_subplot(gs[1, 1])
    ys = np.arange(len(LEVELS))[::-1]
    r8 = float(np.mean([R[f"r8_s{s}"]["core"] for s in SEEDS]))
    ax.axvline(r8, color=INK2, lw=0.8, zorder=1)
    levels = {}
    for yi, (key, label) in zip(ys, LEVELS):
        vals = [R[f"{key}_s{s}"]["core"] for s in SEEDS]
        m = float(np.mean(vals))
        color = BLUE if key == "amp" else REF
        ax.plot([min(vals), max(vals)], [yi, yi], color=color, lw=1.0, alpha=0.6, zorder=2)
        ax.scatter(m, yi, s=28, color=color, edgecolor="white", lw=0.7, zorder=3)
        value_label(ax, 80.75, yi, f"{m:.2f}", va="center", ha="left", fontsize=7)
        levels[key] = (m, vals)
    ax.set_yticks(ys, [l[1] for l in LEVELS])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(75.8, 81.4)
    ax.set_xlabel("核心弱端准确率 (%)，后期")
    ax.set_title("(d) 后期准确率（点为均值，线为范围）")
    out["levels"] = levels
    save(fig, "fig5_batch2_decomposition")
    return out


# ================================================================ cost table
METHOD_ZH = {
    "r8": "r8（全员 rank 8）", "r2": "r2（全员 rank 2）", "prefix": "前缀异构 L0", "prefix_sf20": "前缀异构，20% 强端",
    "vref": "vref（完整训练，上传 r/R）", "vref_r1": "vref，r = 1（budget8）", "vref_het": "vref，rank 混合",
    "vref_sf20": "vref，20% 强端", "L2": "L2（受限训练）", "L2_r1": "L2，r = 1", "L2_het": "L2，rank 混合",
    "L2_sf20": "L2，20% 强端", "rcs": "RCS 在线收缩", "fixedshrink": "固定收缩 0.63", "weaklr": "弱端学习率 × 0.6",
    "amp": "amp（放大受限训练）", "amp_a2": "amp，α = 2", "amp_a8": "amp，α = 8", "amp_owner": "amp，拥有者平均",
    "amp_prefix": "amp，前缀分配", "amp_r1": "amp，r = 1", "amp_het": "amp，rank 混合", "amp_sf20": "amp，20% 强端",
    "fslora_st": "A1a FSLoRA 前向 + amp 梯度路径", "fslora_hajek": "A1b FSLoRA 前向 + Hájek",
    "fslora_zerofill": "A1c FSLoRA 前向 + 零填充", "lrcomp": "A2 学习率 × R/r，服务器 ÷ R/r",
    "fslora_paper": "FSLoRA 式（外部）", "fedplora": "Fed-PLoRA 式（外部）", "flasc": "FLASC 式（外部）",
}


def source_label(p: Path) -> str:
    for root, lab in ((B1, "batch 1"), (B2, "batch 2"), (SWEEP, "strong-frac sweep")):
        if root in p.parents:
            return lab
    return p.parents[1].name.replace("scra_diagnostics_20261017_", "diag ")


def cost_table(R) -> list[str]:
    groups: dict[str, list[str]] = {}
    for name in R:
        groups.setdefault(name.rsplit("_s", 1)[0], []).append(name)
    order = [k for k in METHOD_ZH if k in groups] + sorted(k for k in groups if k not in METHOD_ZH)
    lines = ["# 运行成本：峰值显存与单次运行时间", "",
             "由 `make_figures.py` 生成。数据：d0 运行取 `d0_rounds.json` 顶层的 `peak_memory_mib` / `elapsed_seconds`，"
             "strong-frac sweep 的 r8 / r2 / 前缀异构取 `result.json` 的 `cost` 字段。所有运行都是 150 轮、每轮 10 个客户端、"
             "每客户端 20 步本地训练；时间是整次运行的墙钟时间（含每 10 轮一次的 50 客户端评估）。"
             "运行分布在多个 RTX 4090 D 节点上（见“节点”列），时间还受节点负载影响，同一方法不同 seed 之间就相差可达 10% 以上，"
             "因此不宜用这些秒数比较方法的计算量。", "",
             "## 按方法汇总", "",
             "| 方法 | 运行数 | 峰值显存 MiB | 秒 / 次：均值（最小–最大） |", "|---|---:|---:|---|"]
    for k in order:
        ns = groups[k]
        peaks = [R[n]["peak"] for n in ns]
        secs = [R[n]["seconds"] for n in ns if R[n]["seconds"] is not None]
        sec = f"{np.mean(secs):.0f}（{min(secs):.0f}–{max(secs):.0f}）" if len(secs) > 1 else f"{secs[0]:.0f}"
        lines.append(f"| {METHOD_ZH.get(k, k)} | {len(ns)} | {max(peaks):.0f} | {sec} |")
    allpk = [R[n]["peak"] for n in R]
    lines += ["", f"全部 {len(R)} 次运行的峰值显存在 {min(allpk):.0f}–{max(allpk):.0f} MiB 之间：在 RoBERTa-base 规模下，"
              "受限训练与放大对峰值显存几乎没有影响，资源动机需要在大模型上另行论证（roadmap 阶段 C）。"
              "上传量没有在 d0 运行中记录，未列入此表。", "",
              "## 逐次运行", "", "| 运行 | 来源 | 峰值显存 MiB | 秒 | 节点 |", "|---|---|---:|---:|---|"]
    for k in order:
        for n in sorted(groups[k]):
            r = R[n]
            sec = f"{r['seconds']:.0f}" if r["seconds"] is not None else "—"
            lines.append(f"| {n} | {source_label(r['path'])} | {r['peak']:.0f} | {sec} | {r['host'] or '—'} |")
    (HERE / "cost_table.md").write_text("\n".join(lines) + "\n")
    return lines


# ================================================================ INDEX / checks
def write_checks() -> tuple[int, int, list[dict]]:
    bad = [c for c in CHECKS if not c["ok"]]
    lines = ["# 数值核对记录", "", f"由 `make_figures.py` 生成。共 {len(CHECKS)} 项，通过 {len(CHECKS) - len(bad)} 项。"
             "`summary.json` / D2-D3 原始行按精确值比较（容差 1e-6），Markdown 表格按印出的小数位比较（容差半个末位）。", "",
             "| 范围 | 项目 | 本脚本 | 参照 | 来源 | 结果 |", "|---|---|---|---|---|---|"]
    for c in CHECKS:
        ref = c["ref"] if isinstance(c["ref"], str) else fmt(c["ref"])
        lines.append(f"| {c['scope']} | {c['item']} | {fmt(c['mine'])} | {ref} | {c['source']} | "
                     f"{'✓' if c['ok'] else '✗ 不一致'} |")
    (HERE / "checks.md").write_text("\n".join(lines) + "\n")
    return len(CHECKS), len(CHECKS) - len(bad), bad


def write_index(R, f1, f2, f3, f4, f5, font, n_checks, n_ok, bad) -> None:
    mean = lambda key, k="core": float(np.mean([R[f"{key}_s{s}"][k] for s in SEEDS]))  # noqa: E731
    mech = {lab.split("\n")[0]: v for lab, v in f5["mech"].items()}
    ext = f5["ext"]
    lines = [
        "# 阶段 D 图表索引", "",
        "由 `make_figures.py` 生成（每次运行都会覆盖本文件）。上游：`docs/designs/roadmap_2026-10-18_amplified_restricted.md` 阶段 D；"
        "叙述见 `docs/designs/research_two_directions_2026-10-17.md` §4–§8。",
        "",
        "## 复现", "",
        "在仓库根目录创建并启用 Python 虚拟环境后安装绘图依赖：", "",
        "```bash",
        "python -m pip install -e '.[figures]'",
        "python federated_finetuning/experiments/scra_diagnostics_20261017/figures/make_figures.py",
        "```", "",
        f"本次环境：Python {platform.python_version()}，matplotlib {matplotlib.__version__}，numpy {np.__version__}，"
        f"中文字体 {font}（PDF 内嵌 TrueType）。"
        "脚本只读 `federated_finetuning/runs/` 与 `docs/designs/`，只写本文件夹。", "",
        "## 口径", "",
        f"- **核心弱端**：非强端、且所在领域没有强端的 {len(CORE)} 个客户端（由 "
        "`runs/scra_diagnostics_20261017_v1_vref/d0/identity.json` 的 `client_domains` 与 `strong_clients` 计算），"
        f"即客户端 {', '.join(CORE[:3])} … {CORE[-1]}。所有图（包括 20% 强端的运行）都用这一固定集合，与 SUMMARY 的 `core10` 一致。",
        "- **后期准确率**：第 110、120、…、150 轮五次评估的核心弱端准确率均值；前期 / 中期 / 后期窗口为 10–50 / 60–100 / 110–150 轮。",
        "- **轮间波动**：第 50 轮起相邻两次评估（间隔 10 轮）之间全员宏平均准确率变化的绝对值的均值。",
        "- **配对差值**：同一 seed 的两种方法相减（seed 共享参与者日程与初始化）；95% 区间用 t(2) = 4.303。",
        "", "## 图表", "",
        "| 文件 | 内容 | 数据来源 |", "|---|---|---|",
        "| `fig1_core_curves.pdf/.png` | 核心弱端准确率随轮次变化：r8、vref、L2、amp。上行 seed 2040，下行 3 个 seed 均值（阴影为最小–最大）；"
        "右列放大第 50–150 轮，并标出后期均值。 | 曲线：`runs/headroom_strongfrac_20261017_v1/fedavg_r8_seed*/result.json`（history）；"
        "L2 / vref seed 2040：`runs/scra_diagnostics_20261017_v1{,_vref}/d0/d0_rounds.json`；其余：`runs/scra_batch_20261018/*/d0/d0_rounds.json`（eval） |",
        "| `fig2_oscillation.pdf/.png` | 各方法轮间波动的横向条形图，按“参照 / 受限训练 + 服务器端修正 / 放大受限训练 / 第二批对照”分组；"
        "3-seed 方法画均值并叠加各 seed 点，单 seed 方法标注。 | 同上，另加 `runs/scra_diagnostics_20261017_v1_{rcs,fixedshrink,weaklr}` 与 "
        "`runs/scra_batch2_20261018/*/d0/d0_rounds.json` |",
        "| `fig3_head_mechanism.pdf/.png` | 一步机制（每快照 24 次抽样，误差线为抽样间标准差）：(a) 分类头沿 r8 方向的步长，收缩 vs amp；"
        "(b) 冻结弱端头部时的 LoRA 乘积步长（压力转移）；(c) 头部互换分解：shrink、shrink_fullhead、vref_restrhead 的核心弱端准确率减 r8。 | "
        "(a)(b)：`runs/scra_diagnostics_20261017_v1/d3/round_00{20,60,120}.json` 的 `summary`；(c)：`d2/round_*.json` 的 `summary`（与 `rows` 逐次重算一致） |",
        "| `fig4_ablation_robustness.pdf/.png` | (a) 批次 1 消融（seed 2040）相对 amp 的差值：α = 2 / 8、拥有者平均、前缀分配，竖线为 L2 与 vref；"
        "(b) 稳健性：基准、r = 1、rank 混合 {1,2,4}、20% 强端下 L2 / vref / amp 的后期准确率与追回比例。 | `runs/scra_batch_20261018/*/d0/d0_rounds.json`，"
        "vref r = 1 来自 `runs/scra_diagnostics_20261017_v1_budget8` |",
        "| `fig5_batch2_decomposition.pdf/.png` | 批次 2 分解：(a) 四个机制对照的配对差值（条为均值，标记为各 seed，黑线为 95% t 区间）；"
        "(b) 三个小差值对照按前 / 中 / 后期的变化；(c) 外部基线的配对差值；(d) 各方法后期准确率。 | `runs/scra_batch2_20261018/*/d0/d0_rounds.json` "
        "与批次 1 的 amp / vref / L2、sweep 的 r8 |",
        "| `cost_table.md` | 每种方法与每次运行的峰值显存、墙钟时间、节点。 | 各运行 JSON 的 `peak_memory_mib` / `elapsed_seconds` 或 `cost` |",
        "| `checks.md` | 全部数值核对项（本脚本 vs summary.json / SUMMARY.md / REPORT*.md / 叙述文档）。 | — |",
        "", "## 各图要点", "",
        f"1. **图 1**：L2 在 seed 2040 上第 50 轮后反复摆动（全员准确率第 50 轮后最低 {R['L2_s2040']['min_after_50']:.2f}），"
        f"amp 与 r8 / vref 的曲线几乎重合。3-seed 后期均值：r8 {mean('r8'):.2f}，vref {mean('vref'):.2f}，"
        f"amp {mean('amp'):.2f}，L2 {mean('L2'):.2f}；L2 的范围带在中后期明显更宽。冷启动（第 10 轮，seed 2040）amp 已与 r8 持平。",
        f"2. **图 2**：受限训练且不修正前向的方法都在摆动（L2 {mean('L2', 'osc'):.2f}、RCS {R['rcs_s2040']['osc']:.2f}、"
        f"Fed-PLoRA 式 {mean('fedplora', 'osc'):.2f}），波动随收缩加强而下降但降不到参照水平（固定收缩 {R['fixedshrink_s2040']['osc']:.2f}）；"
        f"amp 为 {mean('amp', 'osc'):.2f}，与 r8 {mean('r8', 'osc'):.2f} 相当；α = 2 时摆动部分回来（{R['amp_a2_s2040']['osc']:.2f}），"
        f"α = 8 时没有（{R['amp_a8_s2040']['osc']:.2f}）。FSLoRA 系对照都平稳，差别在准确率而不在稳定性。",
        f"3. **图 3**：收缩规则的分类头步长是 r8 的 {min(f3['head']['shrink']):.2f}–{max(f3['head']['shrink']):.2f} 倍，amp 为 "
        f"{min(f3['head']['amp_hajek']):.2f}–{max(f3['head']['amp_hajek']):.2f}。头部互换：给收缩规则换上完整训练的头部，差距从 "
        f"{f3['swap']['shrink'][0]:.2f} / {f3['swap']['shrink'][1]:.2f} / {f3['swap']['shrink'][2]:.2f} pp 缩小到 "
        f"{f3['swap']['shrink_fullhead'][0]:.2f} / {f3['swap']['shrink_fullhead'][1]:.2f} / {f3['swap']['shrink_fullhead'][2]:.2f}；"
        f"反过来给 vref 换上受限头部，差距变为 {f3['swap']['vref_restrhead'][0]:.2f} / {f3['swap']['vref_restrhead'][1]:.2f} / "
        f"{f3['swap']['vref_restrhead'][2]:.2f}。冻结弱端头部后压力转到 LoRA：LoRA 步长 "
        f"{min(f3['lora']['fh_hajek']):.2f}–{max(f3['lora']['fh_hajek']):.2f} 倍（收缩 {min(f3['lora']['shrink']):.2f}–"
        f"{max(f3['lora']['shrink']):.2f}，amp {min(f3['lora']['amp_hajek']):.2f}–{max(f3['lora']['amp_hajek']):.2f}）。",
        f"4. **图 4**：α = 2 {f4['ablation']['amp_a2_s2040']:+.2f}，α = 8 {f4['ablation']['amp_a8_s2040']:+.2f}，"
        f"拥有者平均 {f4['ablation']['amp_owner_s2040']:+.2f}，前缀分配 {f4['ablation']['amp_prefix_s2040']:+.2f} pp；"
        "三种稳健性设置的追回比例 " + " / ".join(f"{v[3]:.2f}" for k, v in f4["robust"].items() if not k.startswith("基准"))
        + f"（基准 {f4['robust']['基准：r = 2，10% 强端'][3]:.2f}）。单 seed，判据为 0.3 pp。",
        f"5. **图 5**：amp − A1a {mech['amp − A1a'][0]:+.2f}（状态 sketch 效应），A1a − A1b {mech['A1a − A1b'][0]:+.2f}（eps 混淆可忽略），"
        f"A1b − A1c {mech['A1b − A1c'][0]:+.2f}（零填充聚合的代价），amp − A2 {mech['amp − A2'][0]:+.2f}（前向放大不等于调学习率）；"
        f"外部：FSLoRA 式 {ext['amp − FSLoRA 式'][0]:+.2f}、Fed-PLoRA 式 {ext['amp − Fed-PLoRA 式'][0]:+.2f}、"
        f"FLASC 式 {ext['amp − FLASC 式'][0]:+.2f}。分阶段看，amp − A1a 从前期起就为正（"
        + " / ".join(f"{v:+.2f}" for v in f5["phase"]["A1a_state_sketch"]) + "），amp − A2 前期最大（"
        + " / ".join(f"{v:+.2f}" for v in f5["phase"]["A2_lr_compensation"]) + "）。",
        "6. **成本表**：见 `cost_table.md`；所有运行峰值显存约 1557 MiB，RoBERTa-base 上受限训练不省显存。",
        "", "## 数值核对与不一致", "",
        f"共 {n_checks} 项核对，通过 {n_ok} 项（明细见 `checks.md`）。核对对象：批次 1 / 2 的 `summary.json`（精确）与 `SUMMARY.md`"
        "（逐格、按印出精度），D2 / D3 的 `rows` 重算与 `REPORT_D2.md` / `REPORT_D3.md`，早期判别运行各自的 `REPORT.md` 与叙述文档 §4 表格，"
        "叙述文档 §7 中第 10 轮的数字。",
        "",
    ]
    if bad:
        lines += ["**不一致项：**", ""]
        lines += [f"- {c['scope']} / {c['item']}：本脚本 {fmt(c['mine'])}，参照 {c['ref'] if isinstance(c['ref'], str) else fmt(c['ref'])}"
                  f"（{c['source']}）" for c in bad]
        lines.append("")
    else:
        lines += ["**没有数值不一致。** 图中每个数字都能在对应的 SUMMARY / REPORT 中按印出精度找到。", ""]
    lines += ["**口径差异与说明：**", ""]
    lines += [f"- {n}" for n in NOTES]
    d2d3 = "；".join(f"第 {s} 轮 {rule} D2 {a:+.2f} / D3 {b:+.2f}" for s, rule, a, b in NOTES_D2D3)
    hd = "、".join(f"{a:.2f} / {b:.2f}" for _, a, b in NOTES_D2D3_HEAD)
    r8acc = "、".join(f"{a:.2f} / {b:.2f}" for a, b in NOTES_R8ACC)
    lines += [f"- D2 与 D3 是两次独立重放，都包含 shrink 与 shrink_fullhead，结果略有差异（r8 基线本身也不完全相同，"
              f"三个快照的 r8 准确率 D2 / D3 为 {r8acc}）：{d2d3} pp；"
              f"shrink 的头部步长 D2 / D3 为 {hd}。图 3(c) 三条规则都取 D2（vref_restrhead 只在 D2 中），(a)(b) 都取 D3（amp 与 fh_hajek 只在 D3 中），"
              "因此同一面板内的规则来自同一次重放。",
              "- 批次 1 `SUMMARY.md` 与 `summary.json` 中 r8 / r2 / 前缀异构（sweep 运行）的 `elapsed_seconds` 为空，因为 sweep 的 "
              "`result.json` 把时间放在 `cost` 字段；成本表从 `cost.elapsed_seconds` 补上。批次 2 `SUMMARY.md` 没有成本表，"
              "批次 2 的成本直接取各 `d0_rounds.json`，没有可对照的汇总。",
              "- 批次 1 消融表中 amp 的波动 0.37 是 seed 2040 单次值，主表的 0.35 是 3-seed 均值；图 2 用 3-seed 均值。",
              "- 图 3 的误差线是 24 次抽样之间的标准差（REPORT 中的 “± std”），反映参与者抽样的离散程度，不是均值的标准误。",
              "- 图 1 从第 10 轮画起（第 0 轮所有方法都是 33.33%，会压缩纵轴）。"]
    (HERE / "INDEX.md").write_text("\n".join(lines) + "\n")


# ================================================================ main
def main() -> None:
    ident = json.loads((earlier("vref") / "d0" / "identity.json").read_text())
    CORE.extend(core_set(ident["strong_clients"], ident["client_domains"]))
    if len(CORE) != 30:
        raise RuntimeError(f"expected 30 core weak clients, got {len(CORE)}")
    font = setup_style()
    R = load_runs()
    D2 = {s: json.loads((DIAG / "d2" / f"round_{s:04d}.json").read_text()) for s in SNAPS}
    D3 = {s: json.loads((DIAG / "d3" / f"round_{s:04d}.json").read_text()) for s in SNAPS}
    run_checks(R, D2, D3)
    f1, f2, f3, f4, f5 = fig1(R), fig2(R), fig3(D2, D3), fig4(R), fig5(R)
    cost_table(R)
    n, ok, bad = write_checks()
    write_index(R, f1, f2, f3, f4, f5, font, n, ok, bad)
    print(f"{len(R)} runs, {len(OUTPUTS)} figures ({', '.join(OUTPUTS)}), checks {ok}/{n} passed")
    for c in bad:
        print("MISMATCH", c["scope"], c["item"], fmt(c["mine"]), "vs", c["ref"], c["source"])


if __name__ == "__main__":
    main()
