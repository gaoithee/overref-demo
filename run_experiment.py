"""
run_experiment.py

Main entry point for the OLMo refusal / over-refusal evaluation pipeline.

Usage
-----
    # Run all checkpoints × all datasets × all system prompts
    python run_experiment.py

    # Limit to specific checkpoints and datasets
    python run_experiment.py --checkpoints base sft dpo --datasets or_bench

    # Dry-run: load data only, skip generation (useful for debugging)
    python run_experiment.py --dry-run

    # Resume from a previous raw-results CSV (skip generation)
    python run_experiment.py --load-results results/raw_results.csv
"""

import argparse
import importlib
import logging
import os
from pathlib import Path
from itertools import islice

import pandas as pd

from data.dataset_loader import load_all_datasets
from evaluation.refusal_detector import detect_refusal_batch
from evaluation.metrics import (
    compute_metrics,
    compute_metrics_by_category,
    print_summary,
    save_metrics,
)

# Config is loaded dynamically via --config argument (default: config)
_cfg = None

def _load_config(name: str):
    global _cfg
    _cfg = importlib.import_module(name)
    return _cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(iterable, n):
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def run_generation(
    model,
    prompts: list[str],
    batch_size: int = 8,
) -> list[str]:
    """Run generation in batches, with progress logging."""
    responses = []
    total = len(prompts)
    for i, batch in enumerate(_batched(prompts, batch_size)):
        logger.info(
            "  [%s] generating batch %d/%d",
            model.checkpoint_name,
            i + 1,
            (total + batch_size - 1) // batch_size,
        )
        responses.extend(model.generate_batch(batch))
    return responses


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    cfg = _load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # ── 1. Load datasets ──────────────────────────────────────────────────
    selected_datasets = {
        k: v for k, v in cfg.DATASETS.items()
        if k in args.datasets
    } if args.datasets else cfg.DATASETS

    logger.info("Loading datasets: %s", list(selected_datasets.keys()))
    data_df = load_all_datasets(selected_datasets)
    logger.info("Total prompts: %d", len(data_df))

    if len(data_df) == 0:
        logger.error(
            "No prompts loaded — all datasets are empty or missing. "
            "Check that dataset files exist and paths in config.py are correct."
        )
        return

    # ── 2. Resume from existing results (skip generation) ─────────────────
    if args.load_results:
        logger.info("Loading pre-computed results from %s", args.load_results)
        raw_df = pd.read_csv(args.load_results)
        _compute_and_save_metrics(raw_df, cfg.RESULTS_DIR)
        return

    # ── 3. Dry run ─────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry run: skipping generation.")
        logger.info("Sample prompts:\n%s", data_df[["source", "label", "prompt"]].head(5).to_string())
        return

    # ── 4. Load models ─────────────────────────────────────────────────────
    selected_checkpoints = {
        k: v for k, v in cfg.OLMO_CHECKPOINTS.items()
        if k in args.checkpoints
    } if args.checkpoints else cfg.OLMO_CHECKPOINTS

    selected_prompts = {
        k: v for k, v in cfg.SYSTEM_PROMPTS.items()
        if k in args.system_prompts
    } if args.system_prompts else cfg.SYSTEM_PROMPTS

    from models.olmo_loader import iter_checkpoints

    # ── 5. Generate responses (one model at a time to avoid OOM) ───────────
    all_rows = []
    prompts = data_df["prompt"].tolist()

    for model in iter_checkpoints(selected_checkpoints, selected_prompts, cfg.GENERATION):
        logger.info("Running inference for: %s", model.checkpoint_name)
        responses = run_generation(model, prompts, cfg.GENERATION.batch_size)
        refusals  = detect_refusal_batch(responses)

        for idx, (resp, ref) in enumerate(zip(responses, refusals)):
            row = data_df.iloc[idx].to_dict()
            row["checkpoint"]          = model.checkpoint_name
            row["response"]            = resp
            row["predicted_refusal"]   = int(ref)
            all_rows.append(row)

        model.unload()

    raw_df = pd.DataFrame(all_rows)

    # ── 6. Save raw results ─────────────────────────────────────────────────
    raw_path = os.path.join(cfg.RESULTS_DIR, "raw_results.csv")
    raw_df.to_csv(raw_path, index=False)
    logger.info("Raw results saved to %s", raw_path)

    # ── 7. Compute and save metrics ─────────────────────────────────────────
    _compute_and_save_metrics(raw_df, cfg.RESULTS_DIR)


def _compute_and_save_metrics(raw_df: pd.DataFrame, results_dir: str = "results") -> None:
    # Overall metrics per checkpoint
    metrics = compute_metrics(raw_df)
    print_summary(metrics, title="Overall metrics per checkpoint")
    save_metrics(metrics, os.path.join(results_dir, "metrics_overall.csv"))

    # Per-category breakdown
    metrics_cat = compute_metrics_by_category(raw_df)
    save_metrics(metrics_cat, os.path.join(results_dir, "metrics_by_category.csv"))
    logger.info("Per-category metrics saved.")

    # Per-source breakdown (or_bench vs false_reject)
    if "source" in raw_df.columns:
        for source, grp in raw_df.groupby("source"):
            m = compute_metrics(grp)
            path = os.path.join(results_dir, f"metrics_{source}.csv")
            save_metrics(m, path)
            print_summary(m, title=f"Metrics — {source}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo checkpoint refusal behaviour."
    )
    parser.add_argument(
        "--config",
        default="config",
        help="Config module to use: config (OLMo 1), config_olmo2, config_olmo3. Default: config.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Checkpoint keys to evaluate (e.g. base sft dpo). Default: all.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset keys to use (e.g. or_bench false_reject). Default: all.",
    )
    parser.add_argument(
        "--system-prompts",
        nargs="+",
        default=None,
        dest="system_prompts",
        help="System prompt keys (e.g. none mistral_safety). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data only; skip model loading and generation.",
    )
    parser.add_argument(
        "--load-results",
        default=None,
        metavar="CSV",
        help="Skip generation and compute metrics from an existing raw CSV.",
    )
    args = parser.parse_args()
    main(args)