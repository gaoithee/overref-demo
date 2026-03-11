"""
analysis/plot_results.py

Generate plots from the saved metrics CSVs produced by run_experiment.py.

Usage
-----
    python analysis/plot_results.py --results-dir results/
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_prompt_tag(checkpoint: str) -> tuple[str, str]:
    """Split 'sft__mistral_safety' → ('sft', 'mistral_safety')."""
    parts = checkpoint.split("__", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "none")


# ---------------------------------------------------------------------------
# Plot 1: FP-rate vs FN-rate per checkpoint (bubble chart)
# ---------------------------------------------------------------------------

def plot_fp_fn_scatter(metrics: pd.DataFrame, out_path: str) -> None:
    df = metrics.reset_index().copy()
    df[["ckpt", "prompt"]] = pd.DataFrame(
        df["checkpoint"].apply(_strip_prompt_tag).tolist(), index=df.index
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=df["prompt"].nunique())
    prompt_colors = {p: c for p, c in zip(df["prompt"].unique(), palette)}

    for _, row in df.iterrows():
        color = prompt_colors[row["prompt"]]
        ax.scatter(row["fp_rate"], row["fn_rate"], color=color, s=120, zorder=3)
        ax.annotate(
            row["ckpt"],
            (row["fp_rate"], row["fn_rate"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    # Legend for prompt types
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=9, label=p)
        for p, c in prompt_colors.items()
    ]
    ax.legend(handles=handles, title="System prompt", loc="best")

    ax.set_xlabel("FP rate (over-refusal on safe prompts)")
    ax.set_ylabel("FN rate (under-refusal on harmful prompts)")
    ax.set_title("Over-refusal vs Under-refusal by checkpoint")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: FP-rate across alignment stages (line plot per system-prompt)
# ---------------------------------------------------------------------------

STAGE_ORDER = ["base", "sft", "dpo", "final", "reward"]


def plot_fp_rate_by_stage(metrics: pd.DataFrame, out_path: str) -> None:
    df = metrics.reset_index().copy()
    df[["ckpt", "prompt"]] = pd.DataFrame(
        df["checkpoint"].apply(_strip_prompt_tag).tolist(), index=df.index
    )
    df["stage_order"] = df["ckpt"].map(
        {s: i for i, s in enumerate(STAGE_ORDER)}
    )
    df = df.dropna(subset=["stage_order"]).sort_values("stage_order")

    fig, ax = plt.subplots(figsize=(9, 5))
    for prompt, grp in df.groupby("prompt"):
        ax.plot(grp["ckpt"], grp["fp_rate"], marker="o", label=prompt)

    ax.set_xlabel("Alignment stage")
    ax.set_ylabel("FP rate (over-refusal)")
    ax.set_title("Over-refusal rate across OLMo alignment stages")
    ax.legend(title="System prompt")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: FP-rate per category heatmap
# ---------------------------------------------------------------------------

def plot_category_heatmap(cat_metrics_path: str, out_path: str) -> None:
    df = pd.read_csv(cat_metrics_path, index_col=[0, 1]).reset_index()
    df[["ckpt", "prompt"]] = pd.DataFrame(
        df["checkpoint"].apply(_strip_prompt_tag).tolist(), index=df.index
    )

    # Use only the "none" system-prompt variant to keep the plot readable
    sub = df[df["prompt"] == "none"]
    pivot = sub.pivot(index="category", columns="ckpt", values="fp_rate")
    # Reorder columns to alignment stages
    cols = [c for c in STAGE_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("FP rate (over-refusal) per category × alignment stage\n(no system prompt)")
    ax.set_xlabel("Alignment stage")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    plots_dir = rd / "plots"
    plots_dir.mkdir(exist_ok=True)

    overall_path = rd / "metrics_overall.csv"
    cat_path     = rd / "metrics_by_category.csv"

    if overall_path.exists():
        metrics = pd.read_csv(overall_path, index_col=0)
        plot_fp_fn_scatter(metrics,     str(plots_dir / "fp_fn_scatter.png"))
        plot_fp_rate_by_stage(metrics,  str(plots_dir / "fp_rate_stages.png"))
    else:
        print(f"Overall metrics not found at {overall_path}")

    if cat_path.exists():
        plot_category_heatmap(str(cat_path), str(plots_dir / "category_heatmap.png"))
    else:
        print(f"Category metrics not found at {cat_path}")


if __name__ == "__main__":
    main()
