"""
data/dataset_loader.py

Load and normalise over-refusal / safety benchmark datasets into a
consistent pandas DataFrame with columns:
    - prompt   : str   – the user message to send to the model
    - label    : int   – 0 = safe (should NOT be refused), 1 = harmful (should be refused)
    - category : str   – sub-category string (if available, else "unknown")
    - source   : str   – dataset name tag
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OR-Bench
# ---------------------------------------------------------------------------

def load_or_bench(
    hf_path: str = "bench-llm/or-bench",
    config_name: str = "or-bench-80k",   # alternatives: "or-bench-hard-1k", "or-bench-toxic"
    split: str = "train",                 # OR-Bench only has a "train" split
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load OR-Bench from HuggingFace.
    All prompts are *safe* (over-refusal targets) → label = 0.

    Available configs:
      - or-bench-80k       : full 80k dataset, 10 difficulty levels × 8 categories
      - or-bench-hard-1k   : 1k hardest prompts
      - or-bench-toxic     : toxic-looking but safe prompts
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    logger.info("Loading OR-Bench from HuggingFace (%s / %s / %s)…", hf_path, config_name, split)
    ds = load_dataset(hf_path, config_name, split=split)
    df = ds.to_pandas()

    # Normalise column names
    prompt_col = _find_column(df, ["prompt", "text", "question", "input"])
    category_col = _find_column(df, ["category", "type", "label_text"], required=False)

    out = pd.DataFrame()
    out["prompt"]   = df[prompt_col]
    out["label"]    = 0                                         # all safe
    out["category"] = df[category_col] if category_col else "unknown"
    out["source"]   = "or_bench"

    if max_samples:
        out = out.sample(n=min(max_samples, len(out)), random_state=42)

    logger.info("OR-Bench: %d prompts loaded", len(out))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# False Reject
# ---------------------------------------------------------------------------

def load_false_reject(
    hf_path: str = "AmazonScience/FalseReject",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the FalseReject benchmark from HuggingFace (AmazonScience/FalseReject).

    All prompts are safe (benign but high-risk-looking) -> label = 0.
    The test split has 1,187 human-annotated prompts across 44 safety categories.
    The train split has 14,624 entries with instruct_response / cot_response columns.

    Reference: https://arxiv.org/abs/2505.08054
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    logger.info("Loading FalseReject from HuggingFace (%s / %s)...", hf_path, split)
    ds = load_dataset(hf_path, split=split)
    df = ds.to_pandas()

    prompt_col   = _find_column(df, ["prompt", "text", "question", "input"])
    category_col = _find_column(df, ["category", "domain", "type"], required=False)

    out = pd.DataFrame()
    out["prompt"]   = df[prompt_col]
    out["label"]    = 0   # all prompts are safe (over-refusal benchmark)
    out["category"] = df[category_col] if category_col else "unknown"
    out["source"]   = "false_reject"

    if max_samples:
        out = out.sample(n=min(max_samples, len(out)), random_state=42)

    logger.info("FalseReject: %d prompts loaded", len(out))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generic loader (dispatches based on DatasetConfig)
# ---------------------------------------------------------------------------

def load_dataset_from_config(cfg) -> pd.DataFrame:
    """
    Load a dataset given a DatasetConfig object from config.py.
    """
    from config import DATASET_DEFAULT_LABEL  # avoid circular import at module level

    if cfg.name == "OR-Bench":
        df = load_or_bench(
            hf_path=cfg.hf_path,
            config_name="or-bench-80k",   # change to "or-bench-hard-1k" for a harder subset
            split=cfg.hf_split,
            max_samples=cfg.max_samples,
        )
    elif cfg.name == "False Reject":
        df = load_false_reject(
            hf_path=cfg.hf_path,
            split=cfg.hf_split,
            max_samples=cfg.max_samples,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    # Override labels if dataset has no label column
    default_label = DATASET_DEFAULT_LABEL.get(cfg.name.lower().replace("-", "_").replace(" ", "_"))
    if default_label is not None and cfg.label_column is None:
        df["label"] = default_label

    return df


def load_all_datasets(dataset_configs: dict) -> pd.DataFrame:
    """
    Load all datasets and concatenate them into a single DataFrame.

    Parameters
    ----------
    dataset_configs : dict[str, DatasetConfig]
        From config.DATASETS.

    Returns
    -------
    pd.DataFrame with columns [prompt, label, category, source]
    """
    frames = []
    for key, cfg in dataset_configs.items():
        try:
            df = load_dataset_from_config(cfg)
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to load dataset %s: %s", key, exc)

    if not frames:
        raise RuntimeError("No datasets could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total prompts loaded: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list, required: bool = True) -> Optional[str]:
    """Return the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of the expected columns {candidates} found. "
            f"Available: {list(df.columns)}"
        )
    return None