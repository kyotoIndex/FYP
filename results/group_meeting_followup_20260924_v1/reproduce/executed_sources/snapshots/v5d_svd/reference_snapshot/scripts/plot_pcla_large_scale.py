"""Create the main figures for the corrected large-scale PCLA experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "fedlora",
    "fedhello",
    "random_component_alternating",
    "pcla_component_alternating",
]
NAMES = {
    "fedlora": "FedLoRA",
    "fedhello": "Fed-HeLLo",
    "random_component_alternating": "Random component",
    "pcla_component_alternating": "PCLA component",
}
COLORS = {
    "fedlora": "#4C4C4C",
    "fedhello": "#168AAD",
    "random_component_alternating": "#E07A2D",
    "pcla_component_alternating": "#C44569",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_metrics(results: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(results.glob("seed_*")):
        frame = pd.read_csv(seed_dir / "fedhello_metrics.csv")
        frame["seed"] = int(seed_dir.name.removeprefix("seed_"))
        frames.append(frame)
    if len(frames) != 3:
        raise ValueError("expected exactly three completed seed directories")
    return pd.concat(frames, ignore_index=True)


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "pcla_large_scale_constrained_dirichlet_v1",
    )
    args = parser.parse_args()
    results = args.results.resolve()
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(results)
    summary = pd.read_csv(results / "aggregate_summary.csv").set_index("method")
    configure_style()

    grouped = (
        metrics.groupby(["method", "round"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    for method in ORDER:
        rows = grouped[grouped["method"] == method].sort_values("round")
        x = rows["round"].to_numpy()
        mean = rows["mean"].to_numpy()
        std = rows["std"].fillna(0).to_numpy()
        ax.plot(
            x,
            mean,
            color=COLORS[method],
            linewidth=2.3 if method == "pcla_component_alternating" else 1.8,
            label=NAMES[method],
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.09,
        )
    ax.set(
        title="Large-scale convergence under corrected label-skew partition",
        xlabel="Federated round",
        ylabel="Accuracy",
        xlim=(1, 30),
        ylim=(0.47, 0.93),
    )
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(ncol=2, loc="lower right")
    save(fig, output, "large_scale_convergence")

    ordered = summary.loc[ORDER]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars = ax.bar(
        [NAMES[method] for method in ORDER],
        ordered["final_accuracy_mean"],
        yerr=ordered["final_accuracy_std"],
        capsize=4,
        color=[COLORS[method] for method in ORDER],
    )
    for bar, value in zip(bars, ordered["final_accuracy_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value:.2%}",
            ha="center",
            fontsize=9,
        )
    ax.set(
        title="Final accuracy across three seeds",
        ylabel="Final accuracy (mean ± SD)",
        ylim=(0.70, 0.93),
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    save(fig, output, "large_scale_final_accuracy")

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for method in ORDER:
        row = summary.loc[method]
        ax.scatter(
            row["final_comm_mb_mean"],
            row["final_accuracy_mean"],
            s=95,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(
            NAMES[method],
            (row["final_comm_mb_mean"], row["final_accuracy_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        title="Accuracy–communication trade-off",
        xlabel="Total communication (MB)",
        ylabel="Final accuracy",
        ylim=(0.76, 0.91),
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    save(fig, output, "large_scale_accuracy_communication")

    print(f"wrote figures to {output}")


if __name__ == "__main__":
    main()
