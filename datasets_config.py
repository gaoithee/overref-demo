"""
datasets_config.py

Single source of truth for all benchmark datasets.
Imported by every model config (config.py, config_olmo2.py, …).

Dataset types
-------------
  "over_refusal"  — all prompts are SAFE (label=0). Measures FP (over-refusal).
                    A good model should answer all of them.
  "harmful"       — all prompts are HARMFUL (label=1). Measures FN (under-refusal).
                    A good model should refuse all of them.
  "mixed"         — contains both safe (0) and harmful (1) prompts.
                    Measures both FP and FN simultaneously.
"""

from dataclasses import dataclass
from typing import Literal, Optional


DatasetType = Literal["over_refusal", "harmful", "mixed"]


@dataclass
class DatasetConfig:
    name: str
    dataset_type: DatasetType       # drives which metrics are computed
    hf_path: str                    # HuggingFace dataset path
    hf_split: str = "test"
    prompt_column: str = "prompt"
    label_column: Optional[str] = None   # None → use dataset_type to assign fixed label
    max_samples: Optional[int] = None    # None = use all available


# ---------------------------------------------------------------------------
# Over-refusal benchmarks  (label = 0, FP measurement)
# ---------------------------------------------------------------------------

OR_BENCH = DatasetConfig(
    name="OR-Bench",
    dataset_type="over_refusal",
    hf_path="bench-llm/or-bench",
    hf_split="train",       # OR-Bench only has a "train" split
    prompt_column="prompt",
    label_column=None,      # all safe → label=0 assigned automatically
    max_samples=500,
)

FALSE_REJECT = DatasetConfig(
    name="False Reject",
    dataset_type="over_refusal",
    hf_path="AmazonScience/FalseReject",
    hf_split="test",
    prompt_column="prompt",
    label_column=None,      # all safe → label=0 assigned automatically
    max_samples=500,
)


# ---------------------------------------------------------------------------
# Harmful benchmarks  (label = 1, FN measurement)
# ---------------------------------------------------------------------------

HARMBENCH = DatasetConfig(
    name="HarmBench",
    dataset_type="harmful",
    hf_path="allenai/tulu-3-harmbench-eval",
    hf_split="train",
    prompt_column="prompt",
    label_column=None,      # all harmful → label=1 assigned automatically
    max_samples=None,       # ~200 prompts, use all
)

JAILBREAKBENCH = DatasetConfig(
    name="JailbreakBench",
    dataset_type="harmful",
    hf_path="JailbreakBench/JBB-Behaviors",
    hf_split="train",
    prompt_column="Goal",
    label_column=None,      # all harmful → label=1 assigned automatically
    max_samples=None,       # 100 prompts, use all
)


# ---------------------------------------------------------------------------
# Mixed benchmarks  (label = 0 and 1, FP + FN measurement)
# ---------------------------------------------------------------------------

WILDGUARD = DatasetConfig(
    name="WildGuard",
    dataset_type="mixed",
    hf_path="allenai/wildguardmix",
    hf_split="test",
    prompt_column="prompt",
    label_column="prompt_harm_label",   # "harmful" → 1, "unharmful" → 0
    max_samples=1000,       # stratified: 500 safe + 500 harmful
)

TOXICCHAT = DatasetConfig(
    name="ToxicChat",
    dataset_type="mixed",
    hf_path="lmsys/toxic-chat",
    hf_split="test",
    prompt_column="user_input",
    label_column="toxicity",            # 1 = toxic/harmful, 0 = safe
    max_samples=500,        # stratified: 250 safe + 250 harmful
)

BEAVERTAILS = DatasetConfig(
    name="BeaverTails",
    dataset_type="mixed",
    hf_path="PKU-Alignment/BeaverTails",
    hf_split="test",
    prompt_column="prompt",
    label_column="is_safe",             # is_safe=False → label=1 (harmful)
    max_samples=1000,       # stratified: 500 safe + 500 harmful
)


# ---------------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------------

# All datasets, grouped by type — for easy reference in configs and scripts
ALL_DATASETS: dict[str, DatasetConfig] = {
    # over-refusal
    "or_bench":       OR_BENCH,
    "false_reject":   FALSE_REJECT,
    # harmful
    "harmbench":      HARMBENCH,
    "jailbreakbench": JAILBREAKBENCH,
    # mixed
    "wildguard":      WILDGUARD,
    "toxicchat":      TOXICCHAT,
    "beavertails":    BEAVERTAILS,
}

OVER_REFUSAL_DATASETS: list[str] = ["or_bench", "false_reject"]
HARMFUL_DATASETS:      list[str] = ["harmbench", "jailbreakbench"]
MIXED_DATASETS:        list[str] = ["wildguard", "toxicchat", "beavertails"]