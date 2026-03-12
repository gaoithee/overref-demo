"""
analysis/plot_results.py

Generate two plots from the saved metrics CSVs:
  1. FP rate across alignment stages (line plot)
  2. FP rate per category x stage (heatmap)

Usage
-----
    python analysis/plot_results.py --results-dir results/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "none":           "#2563EB",   # blue
    "mistral_safety": "#DC2626",   # red
}

STAGE_ORDER  = ["base", "sft", "dpo", "final"]
STAGE_LABELS = {"base": "Base", "sft": "SFT", "dpo": "DPO", "final": "Instruct"}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


def _split_tag(tag: str):
    """'sft__mistral_safety' -> ('sft', 'mistral_safety')"""
    parts = tag.split("__", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "none")


# ---------------------------------------------------------------------------
# Plot 1 - FP rate across alignment stages
# ---------------------------------------------------------------------------

def plot_fp_rate_stages(overall_csv: str, out_path: str) -> None:
    df = pd.read_csv(overall_csv)
    df[["stage", "prompt"]] = pd.DataFrame(
        df["checkpoint"].apply(_split_tag).tolist(), index=df.index
    )
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    df["stage_idx"] = df["stage"].map({s: i for i, s in enumerate(STAGE_ORDER)})
    df = df.sort_values("stage_idx")

    fig, ax = plt.subplots(figsize=(7, 4))

    for prompt_tag, grp in df.groupby("prompt"):
        color = PALETTE.get(prompt_tag, "#6B7280")
        label = "No system prompt" if prompt_tag == "none" else "Mistral safety prompt"
        x = [STAGE_LABELS[s] for s in grp["stage"]]
        y = grp["fp_rate"].values * 100

        ax.plot(x, y, marker="o", linewidth=2.2, markersize=7,
                color=color, label=label, zorder=3)

        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}%", (xi, yi),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=color)

    ax.set_ylabel("Over-refusal rate (FP %)", fontsize=11)
    ax.set_xlabel("Alignment stage", fontsize=11)
    ax.set_title("Over-refusal rate across OLMo-7B alignment stages\n"
                 "(OR-Bench, 500 safe prompts)", fontsize=12, pad=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, max(df["fp_rate"].max() * 100 * 1.35, 5))
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 - Heatmap FP rate per (category x stage)
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    "deception":  "Deception",
    "harassment": "Harassment",
    "harmful":    "Harmful",
    "hate":       "Hate",
    "illegal":    "Illegal",
    "privacy":    "Privacy",
    "self-harm":  "Self-harm",
    "sexual":     "Sexual",
    "unethical":  "Unethical",
    "violence":   "Violence",
}

def plot_category_heatmap(cat_csv: str, out_path: str,
                           prompt_filter: str = "none") -> None:
    df = pd.read_csv(cat_csv)
    df[["stage", "prompt"]] = pd.DataFrame(
        df["checkpoint"].apply(_split_tag).tolist(), index=df.index
    )
    sub = df[(df["stage"].isin(STAGE_ORDER)) & (df["prompt"] == prompt_filter)].copy()

    sub = sub.copy()
    sub["category"] = sub["category"].map(lambda c: CATEGORY_LABELS.get(c, c))
    pivot = sub.pivot(index="category", columns="stage", values="fp_rate")
    pivot = pivot[[s for s in STAGE_ORDER if s in pivot.columns]]
    pivot.columns = [STAGE_LABELS[s] for s in pivot.columns]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    data = pivot.values * 100
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=25, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > 16 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Over-refusal rate (%)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    prompt_label = "no system prompt" if prompt_filter == "none" else "Mistral safety prompt"
    ax.set_title(f"Over-refusal rate per category x alignment stage\n"
                 f"({prompt_label})", fontsize=11, pad=10)
    ax.set_xlabel("Alignment stage", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
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
        plot_fp_rate_stages(str(overall_path), str(plots_dir / "fp_rate_stages.png"))
    else:
        print(f"Not found: {overall_path}")

    if cat_path.exists():
        plot_category_heatmap(str(cat_path), str(plots_dir / "category_heatmap_none.png"),
                              prompt_filter="none")
        plot_category_heatmap(str(cat_path), str(plots_dir / "category_heatmap_mistral.png"),
                              prompt_filter="mistral_safety")
    else:
        print(f"Not found: {cat_path}")


if __name__ == "__main__":
    main()