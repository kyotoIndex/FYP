from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "fedhello": "Fed-HeLLo",
    "fedra_random_partial": "FedRA Random",
    "fim_hla": "FIM-HLA",
    "rgd_hla": "RGD-HLA",
    "personalized_cost_aware_lora": "PCLA-v1",
    "fedlora": "FedLoRA",
    "resource_only": "Resource-only",
    "importance_resource_lora": "Importance-resource",
    "legend_like": "LEGEND-like",
}

COLORS = {
    "fedhello": "#007C91",
    "fedra_random_partial": "#E07A2D",
    "fim_hla": "#3574B9",
    "rgd_hla": "#6B8E23",
    "personalized_cost_aware_lora": "#C44E52",
    "fedlora": "#555555",
    "resource_only": "#9A7AA0",
    "importance_resource_lora": "#B69B4C",
    "legend_like": "#8C6D5A",
}

METHOD_ORDER = [
    "fedhello",
    "fedra_random_partial",
    "fim_hla",
    "rgd_hla",
    "personalized_cost_aware_lora",
    "fedlora",
    "resource_only",
    "importance_resource_lora",
    "legend_like",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEY_CURVES = [
    "fedhello",
    "fedra_random_partial",
    "fim_hla",
    "rgd_hla",
    "personalized_cost_aware_lora",
    "fedlora",
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_metrics(results: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(results.glob("seed_*")):
        path = seed_dir / "fedhello_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["seed"] = int(seed_dir.name.removeprefix("seed_"))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No seed_*/fedhello_metrics.csv files under {results}")
    return pd.concat(frames, ignore_index=True)


def load_summary(results: Path) -> pd.DataFrame:
    path = results / "formal_comparison_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pd.read_csv(path).set_index("method")
    missing = set(METHOD_ORDER) - set(summary.index)
    if missing:
        raise ValueError(f"Summary is missing methods: {sorted(missing)}")
    return summary.loc[METHOD_ORDER].copy()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def draw_convergence(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    grouped = (
        metrics.groupby(["method", "round"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    for method in KEY_CURVES:
        rows = grouped[grouped["method"] == method].sort_values("round")
        x = rows["round"].to_numpy()
        mean = rows["mean"].to_numpy()
        std = rows["std"].fillna(0).to_numpy()
        ax.plot(
            x,
            mean,
            color=COLORS[method],
            linewidth=2.2 if method in {"fedhello", "personalized_cost_aware_lora"} else 1.7,
            label=DISPLAY_NAMES[method],
        )
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.08)
    ax.set(xlabel="Federated round", ylabel="Accuracy", xlim=(1, 30), ylim=(0.47, 0.93))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(ncol=3, loc="lower right", fontsize=8.5)


def draw_final_accuracy(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("final_accuracy_mean", ascending=True)
    y = np.arange(len(ordered))
    for position, (method, row) in zip(y, ordered.iterrows()):
        ax.errorbar(
            row["final_accuracy_mean"],
            position,
            xerr=row["final_accuracy_std"],
            fmt="o",
            markersize=6.5,
            capsize=3,
            linewidth=1.3,
            color=COLORS[method],
        )
        ax.text(
            min(row["final_accuracy_mean"] + 0.012, 0.94),
            position,
            f"{row['final_accuracy_mean']:.2%}",
            va="center",
            fontsize=8.5,
        )
    ax.set_yticks(y, [DISPLAY_NAMES[m] for m in ordered.index])
    ax.set(xlabel="Final accuracy (mean +/- seed SD)", xlim=(0.48, 0.95))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def draw_pareto(ax: plt.Axes, summary: pd.DataFrame, x_column: str, x_label: str) -> None:
    other = "final_comm_mean_mb" if x_column == "final_time_mean_seconds" else "final_time_mean_seconds"
    sizes = np.interp(summary[other], (summary[other].min(), summary[other].max()), (55, 155))
    for size, (method, row) in zip(sizes, summary.iterrows()):
        ax.scatter(
            row[x_column],
            row["final_accuracy_mean"],
            s=size,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
        if method in {"fedhello", "fedra_random_partial", "personalized_cost_aware_lora", "fedlora"}:
            offset = (-5, 7) if method == "fedlora" else (5, 6)
            align = "right" if method == "fedlora" else "left"
            ax.annotate(DISPLAY_NAMES[method], (row[x_column], row["final_accuracy_mean"]),
                        xytext=offset, textcoords="offset points", ha=align, fontsize=8.2)
    ax.set(xlabel=x_label, ylabel="Final accuracy", ylim=(0.73, 0.91))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def draw_savings(ax: plt.Axes, summary: pd.DataFrame) -> None:
    selected = summary.drop(index="fedlora").sort_values("time_saving_vs_fedlora_percent")
    y = np.arange(len(selected))
    height = 0.36
    ax.barh(y + height / 2, selected["time_saving_vs_fedlora_percent"], height,
            color="#2D7F8C", label="Time saving")
    ax.barh(y - height / 2, selected["comm_saving_vs_fedlora_percent"], height,
            color="#E08A3E", label="Communication saving")
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, [DISPLAY_NAMES[m] for m in selected.index])
    ax.set(xlabel="Saving relative to FedLoRA", xlim=(0, 47))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.legend(loc="lower right")


def make_cluster_figure(results: Path, output_dir: Path) -> None:
    rows = []
    for path in sorted(results.glob("seed_*/allocation_map.csv")):
        seed = int(path.parent.name.removeprefix("seed_"))
        frame = pd.read_csv(path)
        frame = frame[
            (frame["method"] == "personalized_cost_aware_lora")
            & frame["client_cluster"].notna()
        ][["client_id", "capability", "client_cluster"]].drop_duplicates()
        frame["seed"] = seed
        rows.append(frame)
    if not rows:
        return
    clusters = pd.concat(rows, ignore_index=True)
    counts = (
        clusters.groupby(["client_cluster", "capability"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["low", "medium", "high"], fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bottom = np.zeros(len(proportions))
    colors = {"low": "#6C9DC6", "medium": "#E5A04B", "high": "#5B8A72"}
    for capability in ["low", "medium", "high"]:
        values = proportions[capability].to_numpy()
        ax.bar([f"Cluster {int(c)}" for c in proportions.index], values, bottom=bottom,
               color=colors[capability], label=capability.capitalize())
        bottom += values
    ax.set(title="PCLA-v1 clusters mainly reproduce resource tiers",
           ylabel="Share of unique seed-client assignments", ylim=(0, 1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(ncol=3, loc="upper center")
    save_figure(fig, output_dir, "pcla_cluster_composition")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready plots for expanded results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "expanded_sst2_bert_base_dirichlet_v1",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = args.results.resolve()
    output_dir = (args.output or (results / "figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    metrics = load_metrics(results)
    summary = load_summary(results)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    draw_convergence(ax, metrics)
    ax.set_title("Convergence under resource and label heterogeneity")
    save_figure(fig, output_dir, "convergence_curves")

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    draw_final_accuracy(ax, summary)
    ax.set_title("Final accuracy across three seeds")
    save_figure(fig, output_dir, "final_accuracy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    draw_pareto(axes[0], summary, "final_time_mean_seconds", "Simulated total time (s)")
    draw_pareto(axes[1], summary, "final_comm_mean_mb", "Total communication (MB)")
    axes[0].set_title("Accuracy vs. time")
    axes[1].set_title("Accuracy vs. communication")
    save_figure(fig, output_dir, "pareto_efficiency")

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    draw_savings(ax, summary)
    ax.set_title("Efficiency saving relative to full-layer FedLoRA")
    save_figure(fig, output_dir, "efficiency_savings")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    draw_convergence(axes[0, 0], metrics)
    axes[0, 0].set_title("(a) Mean convergence with seed SD")
    draw_final_accuracy(axes[0, 1], summary)
    axes[0, 1].set_title("(b) Final accuracy")
    draw_pareto(axes[1, 0], summary, "final_time_mean_seconds", "Simulated total time (s)")
    axes[1, 0].set_title("(c) Accuracy-time Pareto")
    draw_savings(axes[1, 1], summary)
    axes[1, 1].set_title("(d) Efficiency saving vs. FedLoRA")
    fig.suptitle("Expanded SST-2 / BERT-base federated LoRA comparison", fontsize=15)
    save_figure(fig, output_dir, "experiment_overview")

    make_cluster_figure(results, output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
