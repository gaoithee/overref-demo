# OLMo Refusal Evaluation

Evaluate how **over-refusal** (false positives) and **under-refusal** (false negatives) change across OLMo post-training checkpoints and under different guardrail configurations.

## Research question

> How do different combinations of guardrails influence over- and under-refusal?

---

## Repository structure

```
olmo-refusal-eval/
├── config.py                        # OLMo 1 (requires ai2-olmo + transformers<4.47)
├── config_olmo2.py                  # OLMo 2 (requires transformers>=4.57)
├── config_olmo3.py                  # OLMo 3 (requires transformers>=4.57)
├── run_experiment.py                # main entry point (--config selects the family)
├── run_slurm.sh                     # SLURM job for OLMo 1
├── run_slurm_olmo2.sh               # SLURM job for OLMo 2
├── run_slurm_olmo3.sh               # SLURM job for OLMo 3
├── models/
│   └── olmo_loader.py               # load HF causal-LM + batched generation + unload
├── data/
│   └── dataset_loader.py            # OR-Bench, FalseReject → unified DataFrame
├── evaluation/
│   ├── refusal_detector.py          # keyword + optional LLM-judge classifier
│   └── metrics.py                   # FP, FN, precision, recall, F1
├── analysis/
│   ├── plot_results.py              # line chart + heatmap from metrics CSVs
│   └── generate_report.py           # interactive HTML report from raw_results.csv
├── results/
│   ├── olmo1/                       # output of config.py runs
│   ├── olmo2/                       # output of config_olmo2.py runs
│   ├── olmo3/                       # output of config_olmo3.py runs
│   └── report.html                  # merged HTML report (all families)
└── requirements.txt
```

---

## Model families and checkpoints

Each family covers the same four post-training stages: **Base → SFT → DPO → Instruct**.

### OLMo 1 (Feb 2024)
| Tag | HuggingFace model ID |
|-----|----------------------|
| `base`  | `allenai/OLMo-7B` |
| `sft`   | `allenai/OLMo-7B-SFT` |
| `dpo`   | `allenai/OLMo-7B-DPO` |
| `final` | `allenai/OLMo-7B-Instruct` |

> No dedicated safety training — any refusal behaviour emerges from general alignment.

### OLMo 2 (Nov 2024)
| Tag | HuggingFace model ID |
|-----|----------------------|
| `base`  | `allenai/OLMo-2-1124-7B` |
| `sft`   | `allenai/OLMo-2-1124-7B-SFT` |
| `dpo`   | `allenai/OLMo-2-1124-7B-DPO` |
| `final` | `allenai/OLMo-2-1124-7B-Instruct` |

### OLMo 3 (Dec 2024)
| Tag | HuggingFace model ID |
|-----|----------------------|
| `base`  | `allenai/Olmo-3-1025-7B` |
| `sft`   | `allenai/Olmo-3-7B-SFT` |
| `dpo`   | `allenai/Olmo-3-7B-DPO` |
| `final` | `allenai/Olmo-3-7B-Instruct` |

> OLMo 3 also exposes intra-stage checkpoints (`revision="step_XXXX"`).
> Set `OLMO3_INTRA_STAGE = True` in `config_olmo3.py` to enable them.
> List available steps with:
> ```python
> from huggingface_hub import list_repo_refs
> refs = list_repo_refs("allenai/Olmo-3-7B-SFT")
> print([b.name for b in refs.branches if "step" in b.name])
> ```

---

## Guardrail configurations

Each checkpoint is tested with **two system-prompt variants**:
- `none` – no system prompt
- `mistral_safety` – the standard Mistral safety system prompt

Add more variants to `SYSTEM_PROMPTS` in the relevant config file.

---

## Datasets

| Key | Source | Labels | Size used |
|-----|--------|--------|-----------|
| `or_bench` | `bench-llm/or-bench` (HuggingFace) | all safe (label=0) | 500 sampled from 80k |
| `false_reject` | `AmazonScience/FalseReject` (HuggingFace) | all safe (label=0) | 500 sampled from 1,187 |

Both datasets are **over-refusal benchmarks** — all prompts are safe but formulated to look
risky. To also measure **under-refusal** (FN), a dataset with genuinely harmful prompts
(e.g. WildGuard, HarmBench) needs to be added.

To use the hardest OR-Bench subset instead of the full 80k, change in `data/dataset_loader.py`:
```python
config_name="or-bench-hard-1k"   # instead of "or-bench-80k"
```

---

## Environment setup

Two separate virtual environments are required due to incompatible `transformers` versions.

### OLMo 1 (transformers < 4.47, needs ai2-olmo)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OLMo 2 and OLMo 3 (transformers >= 4.57, no ai2-olmo)
```bash
python -m venv .venv-olmo23
source .venv-olmo23/bin/activate
pip install torch>=2.2.0 transformers>=4.57.0 accelerate datasets pandas numpy matplotlib seaborn
```

A GPU with ≥16 GB VRAM is sufficient for any 7B model in fp16.

---

## Usage

```bash
# OLMo 1 — activate .venv first
python run_experiment.py --config config \
    --checkpoints base sft dpo final \
    --datasets or_bench

# OLMo 2 — activate .venv-olmo23 first
python run_experiment.py --config config_olmo2 \
    --checkpoints base sft dpo final \
    --datasets or_bench

# OLMo 3
python run_experiment.py --config config_olmo3 \
    --checkpoints base sft dpo final \
    --datasets or_bench

# Dry run (no GPU needed — just loads data)
python run_experiment.py --config config --dry-run

# Recompute metrics from existing CSV (skip generation)
python run_experiment.py --config config --load-results results/olmo1/raw_results.csv
```

### SLURM
```bash
mkdir -p slurm_outputs
sbatch run_slurm.sh         # OLMo 1
sbatch run_slurm_olmo2.sh   # OLMo 2
sbatch run_slurm_olmo3.sh   # OLMo 3
```

---

## Output files

Each run saves to its own subdirectory (`results/olmo1/`, `results/olmo2/`, `results/olmo3/`):

| File | Contents |
|------|----------|
| `raw_results.csv` | one row per (prompt × checkpoint), includes the full generated response |
| `metrics_overall.csv` | FP, FN, precision, recall, F1 per checkpoint |
| `metrics_by_category.csv` | same metrics broken down by prompt category |
| `metrics_or_bench.csv` | subset metrics for OR-Bench only |
| `metrics_false_reject.csv` | subset metrics for FalseReject only |

---

## Analysis

### Plots
```bash
# One family at a time
python analysis/plot_results.py --results-dir results/olmo1/
```
Generates in `results/olmo1/plots/`:
- `fp_rate_stages.png` — FP rate across alignment stages (line per system-prompt)
- `category_heatmap_none.png` — FP rate per (category × stage), no system prompt
- `category_heatmap_mistral.png` — same with Mistral safety prompt

### Interactive HTML report
```bash
# All families merged (default)
python analysis/generate_report.py

# Single family
python analysis/generate_report.py --results-dirs results/olmo1

# Custom output
python analysis/generate_report.py --output my_report.html
```
Opens at `results/report.html`. Filterable by model family, checkpoint, category, and refusal outcome. Each card shows the prompt, badges, and the full generated response on click.

---

## Metric definitions

```
             Predicted: refusal   Predicted: answered
True: safe      FP (over-ref.)       TN
True: harmful   TP                   FN (under-ref.)

FP rate  = FP / (FP + TN)     ← over-refusal rate on safe prompts
FN rate  = FN / (FN + TP)     ← miss rate on harmful prompts
```

---

## Refusal detection

Two-stage pipeline in `evaluation/refusal_detector.py`:
1. **Keyword matching** — ~30 pre-compiled regex patterns ("I cannot", "I'm sorry", "against my guidelines"…). Fast, applied to all responses.
2. **LLM judge** (still work-in-progress) — sends ambiguous responses to an instruction model for binary classification. Enable with `USE_LLM_JUDGE = True` in the config file.

