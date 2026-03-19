"""
run_judge.py

Post-processing step: run the two-stage GPT-OSS-Safeguard-20B judge on one or more
raw_results.csv files produced by run_experiment.py.

Two-stage pipeline
------------------
  Stage 1 — Coherence
    Is the response a real answer, or gibberish / hallucinated loop?
    Base models often produce degenerate outputs that should not be classified for harm.

  Stage 2 — Harm (only if Stage 1 = coherent)
    Is the content of the response harmful?

Added columns
-------------
  is_coherent     : bool  (True = real answer; False = degenerate / looping output)
  judge_harmful   : bool | None  (None when is_coherent=False)
  judge_reasoning : str   (full chain-of-thought from the judge model)

Usage
-----
    # Single results directory
    python run_judge.py --results-dirs results/olmo1

    # Multiple directories in one pass (judge loads once, processes all)
    python run_judge.py --results-dirs results/olmo1 results/olmo2 results/olmo3

    # Skip rows already judged (useful for resuming an interrupted run)
    python run_judge.py --results-dirs results/olmo1 --resume

    # Dry-run: print 5 examples without running the judge
    python run_judge.py --results-dirs results/olmo1 --dry-run
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from evaluation.llm_judge import GPTOSSSafeguardJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_judge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add judge columns if not present (e.g. when running fresh)."""
    for col, default in [
        ("is_coherent",     None),
        ("judge_harmful",   None),
        ("judge_reasoning", None),
    ]:
        if col not in df.columns:
            df[col] = default
    return df


def _rows_to_judge(df: pd.DataFrame, resume: bool) -> pd.DataFrame:
    """Return the subset of rows that still need judging."""
    if resume:
        mask = df["is_coherent"].isna()
        logger.info("Resume mode: %d rows already judged, %d remaining.",
                    (~mask).sum(), mask.sum())
        return df[mask]
    return df


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_judge_on_file(
    csv_path: Path,
    judge: GPTOSSSafeguardJudge,
    resume: bool,
    dry_run: bool,
) -> None:
    logger.info("=== Processing %s ===", csv_path)
    df = pd.read_csv(csv_path)
    df = _ensure_judge_columns(df)

    todo = _rows_to_judge(df, resume)
    total = len(todo)
    logger.info("%d rows to judge.", total)

    if dry_run:
        logger.info("Dry-run mode — printing 5 sample rows:")
        for _, row in todo.head(5).iterrows():
            print(f"\nCheckpoint : {row['checkpoint']}")
            print(f"Prompt     : {row['prompt'][:120]}")
            print(f"Response   : {str(row['response'])[:200]}")
        return

    if total == 0:
        logger.info("Nothing to do.")
        return

    # Process row by row and save incrementally every 50 rows
    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        prompt   = str(row["prompt"])
        response = str(row.get("response", ""))

        result = judge.evaluate(prompt, response)

        df.at[idx, "is_coherent"]     = result["is_coherent"]
        df.at[idx, "judge_harmful"]   = result["judge_harmful"]
        df.at[idx, "judge_reasoning"] = result["judge_reasoning"]

        if i % 50 == 0 or i == total:
            df.to_csv(csv_path, index=False)
            logger.info(
                "  [%d/%d] saved — coherent: %d  harmful: %d  incoherent: %d",
                i, total,
                int(df["is_coherent"].sum()),
                int(df["judge_harmful"].sum()),
                int((df["is_coherent"] == False).sum()),
            )

    logger.info("Done: %s", csv_path)
    _print_summary(df, csv_path)


def _print_summary(df: pd.DataFrame, path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Summary: {path}")
    print(f"{'='*60}")
    total = len(df)
    coherent   = int(df["is_coherent"].sum())
    incoherent = total - coherent
    harmful    = int(df["judge_harmful"].sum())
    safe_ans   = coherent - harmful

    print(f"Total rows      : {total}")
    print(f"  Coherent      : {coherent}  ({100*coherent/total:.1f}%)")
    print(f"  Incoherent    : {incoherent}  ({100*incoherent/total:.1f}%)")
    print(f"  Harmful       : {harmful}  ({100*harmful/total:.1f}% of all, "
          f"{100*harmful/coherent:.1f}% of coherent)" if coherent else "")
    print(f"  Safe answers  : {safe_ans}")
    print()

    print("By checkpoint:")
    for ckpt, grp in df.groupby("checkpoint"):
        c = int(grp["is_coherent"].sum())
        h = int(grp["judge_harmful"].fillna(False).sum())
        n = len(grp)
        print(f"  {ckpt:<35} coherent={c}/{n}  harmful={h}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-OSS two-stage judge on raw_results.csv files."
    )
    parser.add_argument(
        "--results-dirs", nargs="+", required=True,
        help="One or more results directories containing raw_results.csv.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-oss-safeguard-20b",
        help="HuggingFace model ID for the judge (default: openai/gpt-oss-safeguard-20b).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip rows that already have is_coherent filled in (resume interrupted run).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print 5 sample rows per file without running the judge.",
    )
    args = parser.parse_args()

    # Resolve CSV paths
    csv_paths = []
    for d in args.results_dirs:
        p = Path(d) / "raw_results.csv"
        if not p.exists():
            logger.warning("Not found, skipping: %s", p)
            continue
        csv_paths.append(p)

    if not csv_paths:
        logger.error("No valid raw_results.csv files found.")
        return

    if args.dry_run:
        # No model needed for dry-run
        for p in csv_paths:
            run_judge_on_file(p, judge=None, resume=args.resume, dry_run=True)
        return

    # Load judge once for all files
    judge = GPTOSSSafeguardJudge(model_id=args.judge_model)

    try:
        for p in csv_paths:
            run_judge_on_file(p, judge, resume=args.resume, dry_run=False)
    finally:
        judge.unload()


if __name__ == "__main__":
    main()