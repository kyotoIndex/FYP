"""Create report-ready charts for the audited four-method PCLA experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "fedhello",
    "fedra_random_partial",
    "pcla_component_alternating",
    "random_component_alternating",
]
LABELS = {
    "fedhello": "Fed-HeLLo",
    "fedra_random_partial": "Random partial layer",
    "pcla_component_alternating": "PCLA component",
    "random_component_alternating": "Random component",
}
COLORS = {
    "fedhello": "#2F5597",
    "fedra_random_partial": "#70AD47",
    "pcla_component_alternating": "#C55A11",
    "random_component_alternating": "#A5A5A5",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def read_metrics(results: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(results.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        frame = pd.read_csv(seed_dir / "fedhello_metrics.csv")
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No seed metric files found in {results}")
    return pd.concat(frames, ignore_index=True)


def add_value_labels(ax, bars, fmt: str) -> None:
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/pcla_component_constrained_dirichlet_v1",
    )
    parser.add_argument("--out-dir", default="reports/pcla_component_report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = Path(args.results)
    if not results.is_absolute():
        results = root / results
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    configure_style()
    metrics = read_metrics(results)
    aggregate = pd.read_csv(results / "aggregate_summary.csv").set_index("method")
    derived = pd.read_csv(results / "derived_accuracy_summary.csv").set_index("method")

    # 1. Mean accuracy trajectory with sample-standard-deviation ribbons.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method in METHODS:
        subset = metrics[metrics["method"] == method]
        grouped = subset.groupby("round")["accuracy"].agg(["mean", "std"])
        ax.plot(grouped.index, grouped["mean"], label=LABELS[method], color=COLORS[method], lw=2.2)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"].fillna(0),
            grouped["mean"] + grouped["std"].fillna(0),
            color=COLORS[method],
            alpha=0.12,
        )
    ax.set(xlabel="Federated round", ylabel="Accuracy", xlim=(1, 12), ylim=(0.45, 0.95))
    ax.set_title("Accuracy trajectory: mean across three seeds (shaded = sample SD)")
    ax.legend(ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "01_accuracy_trajectory.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Final and late-round accuracy.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(METHODS))
    width = 0.36
    final = [derived.loc[method, "final_accuracy_mean"] for method in METHODS]
    last5 = [derived.loc[method, "last5_accuracy_mean"] for method in METHODS]
    final_bars = ax.bar(x - width / 2, final, width, label="Final round", color="#4472C4")
    late_bars = ax.bar(x + width / 2, last5, width, label="Last-5-round mean", color="#ED7D31")
    add_value_labels(ax, final_bars, "{:.3f}")
    add_value_labels(ax, late_bars, "{:.3f}")
    ax.set_xticks(x, [LABELS[method] for method in METHODS], rotation=12, ha="right")
    ax.set(ylabel="Accuracy", ylim=(0.65, 0.90))
    ax.set_title("Final accuracy and late-round stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "02_accuracy_summary.png", bbox_inches="tight")
    plt.close(fig)

    # 3. End-to-end simulated time and communication.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), sharey=True)
    ordered_labels = [LABELS[method] for method in METHODS]
    y = np.arange(len(METHODS))
    time_values = [aggregate.loc[method, "final_time_mean"] for method in METHODS]
    comm_values = [aggregate.loc[method, "final_comm_mb_mean"] for method in METHODS]
    time_bars = axes[0].barh(y, time_values, color=[COLORS[m] for m in METHODS])
    comm_bars = axes[1].barh(y, comm_values, color=[COLORS[m] for m in METHODS])
    axes[0].set_yticks(y, ordered_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Seconds")
    axes[0].set_title("Simulated cumulative time")
    axes[1].set_xlabel("MB")
    axes[1].set_title("Cumulative communication")
    for ax, bars, values in ((axes[0], time_bars, time_values), (axes[1], comm_bars, comm_values)):
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (value, bar.get_y() + bar.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    fig.suptitle("Efficiency comparison (three-seed means)", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "03_efficiency.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Accuracy/communication Pareto view.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method in METHODS:
        comm = aggregate.loc[method, "final_comm_mb_mean"]
        accuracy = derived.loc[method, "last5_accuracy_mean"]
        ax.scatter(comm, accuracy, s=105, color=COLORS[method], edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(LABELS[method], (comm, accuracy), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Cumulative communication (MB)", ylabel="Last-5-round mean accuracy", xlim=(20, 195), ylim=(0.68, 0.87))
    ax.set_title("Accuracy-communication trade-off")
    ax.axvline(44.57904052734375, color="#A5A5A5", lw=1, ls="--", alpha=0.7)
    ax.annotate("component budget", (44.6, 0.683), xytext=(5, 0), textcoords="offset points", fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(figures / "04_accuracy_communication_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    # A compact source table makes the report traceable without manual copying.
    table = derived.loc[METHODS].copy()
    table.insert(0, "method_label", [LABELS[method] for method in METHODS])
    table["final_time_mean"] = [aggregate.loc[method, "final_time_mean"] for method in METHODS]
    table["final_comm_mb_mean"] = [aggregate.loc[method, "final_comm_mb_mean"] for method in METHODS]
    table.to_csv(out_dir / "report_metrics.csv", index=False)
    print(f"Wrote figures and report_metrics.csv to {out_dir}")


if __name__ == "__main__":
    main()
