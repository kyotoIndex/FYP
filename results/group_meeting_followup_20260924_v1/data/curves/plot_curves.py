#!/usr/bin/env python3
"""Plot every unsmoothed public curve, including EMA when available."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "round_curves_public.csv"
OUTPUT = ROOT / "complete_round_curves.pdf"
COLORS = {7: "#2166ac", 13: "#b2182b", 21: "#1b7837", 42: "#762a83", 87: "#e08214"}
STYLES = {"raw": "-", "ema": "--", "ema_replay": "--"}
ERA_LABELS = {"V4": "聚合与基础 LoRA 诊断", "V5": "组会后续实验"}
METHODS = {
    "central_iid_both": "Central IID batches; train both LoRA factors",
    "central_pooled_both": "Central pooled cohort; train both factors",
    "central_serial_alt": "Central serial client blocks; alternate A/B",
    "central_serial_both": "Central serial client blocks; train A/B",
    "fed_both_exact": "Federated A/B with exact residual correction",
    "fed_both_mean": "Federated A/B with separate factor means",
    "fed_ffa_b": "Federated frozen A; train B only",
    "online_pcla_contributors": "Online PCLA; contributor denominator C",
    "online_replay_contributors": "Online random-rank replay; denominator C",
    "trace_pcla_contributors": "Saved PCLA support trace; denominator C",
    "trace_pcla_function_norm_matched": "Saved PCLA trace; norm-matched scale G",
    "trace_random_contributors": "Saved random support trace; denominator C",
    "trace_random_function_norm_matched": "Saved random trace; norm-matched scale G",
    "trace_replay_random_rank_contributors": "Saved replay support trace; denominator C",
    "trace_replay_random_rank_function_norm_matched": "Saved replay trace; norm-matched scale G",
    "C_central_full_alt_r30": "Central, all ranks; alternate A/B; 30 rounds",
    "C_central_pcla_r30": "Central PCLA; 30 rounds",
    "C_central_random_rank_r30": "Central random layers and ranks; 30 rounds",
    "C_central_replay_random_rank_r30": "Central same layers/counts, random rank IDs",
    "D_agnews_central_iid_both": "AG_NEWS central IID; train both factors",
    "D_agnews_fed_both_exact": "AG_NEWS federated A/B; exact residual",
    "D_agnews_fed_both_mean": "AG_NEWS federated A/B; factor means",
    "D_agnews_full_alt": "AG_NEWS all ranks; alternate A/B",
    "D_agnews_online_C": "AG_NEWS online PCLA; denominator C",
    "D_agnews_pcla": "AG_NEWS PCLA",
    "D_agnews_replay": "AG_NEWS same layers/counts, random rank IDs",
    "L1_fed_alt_r100": "All ranks; alternate A/B; 100 rounds",
    "L2_pcla_r100": "PCLA; 100 rounds; EMA from read-only replay",
    "L3_fed_both_mean_r100": "Federated A/B factor means; 100 rounds",
    "P3_fed_sa_r30": "FedSA-style shared A and client-specific B",
    "R_iid_ffa": "Upstream FFA-LoRA; IID split",
    "R_iid_normal": "Upstream FedIT; IID split",
    "R_iid_ours": "Upstream FedEx with residual transpose; IID",
    "R_iid_ours_fixed": "Upstream FedEx without transpose; IID",
    "R_noniid_ffa": "Upstream FFA-LoRA; non-IID split",
    "R_noniid_normal": "Upstream FedIT; non-IID split",
    "R_noniid_ours": "Upstream FedEx with transpose; non-IID",
    "R_noniid_ours_fixed": "Upstream FedEx without transpose; non-IID",
    "T1_fed_alt_r100_slr05": "All ranks; server step size 0.5; 100 rounds",
    "T2_fed_alt_r100_strat": "All ranks; stratified clients; 100 rounds",
    "V1_fed_svd_refactor_r30": "Exact A/B aggregation, then SVD refactor",
    "V2_alt_svd_refactor_r30": "Alternate A/B, then SVD refactor",
    "V3_ffa_svd_basis_r30": "Frozen pretrained singular-vector basis",
    "V4_pcla_svd": "PCLA with SVD refactor each round",
    "V5_replay_svd": "Same layers/counts, random ranks, with SVD",
}


def guide_page(pdf, keys):
    assert {k[2] for k in keys} == set(METHODS), "Every plotted method needs a legend entry"
    fig = plt.figure(figsize=(16, 8.7))
    fig.text(0.04, 0.94, "完整逐轮曲线 · 方法代码说明", fontsize=17, weight="bold")
    fig.text(0.04, 0.895, "聚合与基础 LoRA 诊断 = first experiment batch · 组会后续实验 = second experiment batch",
             fontsize=10)
    fig.text(0.04, 0.865, "Each color is one seed (7, 13, 21, 42, 87). Solid = trained model; dashed = EMA readout. All rounds are shown without smoothing.",
             fontsize=9)
    first = [k for k in keys if k[0] == "V4"]
    second = [k for k in keys if k[0] == "V5"]
    columns = [first, second[:15], second[15:]]
    for col, entries in enumerate(columns):
        x = [0.04, 0.37, 0.70][col]
        fig.text(x, 0.825, ERA_LABELS["V4"] if col == 0 else ERA_LABELS["V5"], fontsize=10, weight="bold")
        for i, (_, _, code) in enumerate(entries):
            y = 0.795 - i * 0.050
            fig.text(x, y, code, fontsize=7.2, weight="bold")
            fig.text(x, y - 0.019, METHODS[code], fontsize=6.8)
    fig.text(0.04, 0.025, "C = averaging by contributing client count; G = function-norm-matched scaling. Counts for every class are in round_curves_public.csv.",
             fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "Arial Unicode MS"
    groups = defaultdict(list)
    for row in csv.DictReader(open(SOURCE)):
        era = "V4" if row["batch"].startswith("v4") else "V5"
        row["round"] = int(row["round"])
        row["seed"] = int(row["seed"])
        row["accuracy"] = float(row["accuracy"])
        row["loss"] = float(row["loss"])
        counts = json.loads(row["predicted_class_counts"])
        row["dominant_class_fraction"] = max(counts) / int(row["samples"])
        groups[(era, row["dataset"], row["method"])].append(row)
    keys = sorted(groups)
    with open(ROOT / "method_legend.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "dataset", "method_code", "description"])
        for era, dataset, method in keys:
            writer.writerow([ERA_LABELS[era], dataset, method, METHODS[method]])
    with PdfPages(OUTPUT) as pdf:
        guide_page(pdf, keys)
        for offset in range(0, len(keys), 2):
            subset = keys[offset:offset + 2]
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.7))
            fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.12,
                                hspace=0.42, wspace=0.25)
            for panel, key in enumerate(subset):
                era, dataset, method = key
                lines = defaultdict(list)
                for row in groups[key]:
                    lines[(row["seed"], row["variant"])].append(row)
                for col, (metric, label) in enumerate([
                    ("accuracy", "Accuracy (fraction)"),
                    ("loss", "Evaluation loss"),
                    ("dominant_class_fraction", "Largest predicted class share"),
                ]):
                    ax = axes[panel, col]
                    for (seed, variant), trajectory in sorted(lines.items()):
                        trajectory.sort(key=lambda x: x["round"])
                        ax.plot([x["round"] for x in trajectory],
                                [x[metric] for x in trajectory],
                                color=COLORS[seed], linestyle=STYLES[variant],
                                linewidth=0.9 if variant == "raw" else 1.0,
                                alpha=0.75 if variant == "raw" else 0.9)
                    ax.set_xlim(1, max(x["round"] for x in groups[key]))
                    if metric in ("accuracy", "dominant_class_fraction"):
                        ax.set_ylim(0, 1)
                    ax.set_xlabel("Training round")
                    ax.set_ylabel(label)
                    ax.grid(alpha=0.18)
                    ax.set_title(f"{ERA_LABELS[era]} · {dataset} · {method}\n{label}", fontsize=9)
            if len(subset) == 1:
                for ax in axes[1]:
                    ax.axis("off")
            handles = [Line2D([0], [0], color=COLORS[s], label=f"seed {s}") for s in COLORS]
            handles += [Line2D([0], [0], color="black", linestyle="-", label="raw"),
                        Line2D([0], [0], color="black", linestyle="--", label="EMA readout")]
            fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
                       bbox_to_anchor=(0.5, 0.025))
            fig.suptitle(f"Full round trajectories · methods {offset + 1}–{min(offset + 2, len(keys))} of {len(keys)}",
                         fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
    (ROOT / "figure_provenance.json").write_text(json.dumps({
        "source_csv": SOURCE.name,
        "source_csv_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "figure": OUTPUT.name,
        "figure_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "methods": len(keys),
        "pages": 1 + (len(keys) + 1) // 2,
        "plotting": "Every saved round is plotted directly; no smoothing, clipping, or peak selection. Accuracy and dominant-class share use the full [0, 1] range.",
    }, indent=2) + "\n")
    print(f"{len(keys)} methods; {1 + (len(keys) + 1) // 2} pages; {OUTPUT}")


if __name__ == "__main__":
    main()
