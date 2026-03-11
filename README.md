# OLMo Refusal Evaluation

Evaluate how **over-refusal** (false positives) and **under-refusal** (false negatives) change across OLMo post-training checkpoints and under different guardrail configurations.

## Research question

> How do different combinations of guardrails influence over- and under-refusal?

## Repository structure

```
olmo-refusal-eval/
├── config.py                  # all knobs: checkpoints, datasets, generation params
├── run_experiment.py          # main entry point
├── models/
│   └── olmo_loader.py         # load any HF causal-LM + batched generation
├── data/
│   └── dataset_loader.py      # OR-Bench, False Reject → unified DataFrame
├── evaluation/
│   ├── refusal_detector.py    # keyword + optional LLM-judge classifier
│   └── metrics.py             # FP, FN, precision, recall, F1
├── analysis/
│   └── plot_results.py        # scatter, line, heatmap plots from saved CSVs
├── results/                   # created at runtime
└── requirements.txt
```

## Checkpoints evaluated

| Tag | HuggingFace model |
|-----|------------------|
| `base`  | `allenai/OLMo-7B` |
| `sft`   | `allenai/OLMo-7B-SFT` |
| `dpo`   | `allenai/OLMo-7B-DPO` |
| `final` | `allenai/OLMo-7B-Instruct` |

Add/remove entries in `config.OLMO_CHECKPOINTS` at will.

## Guardrail configurations

Each checkpoint is tested with **two system-prompt variants**:
- `none` – no system prompt
- `mistral_safety` – the standard Mistral safety system prompt

Add more variants to `config.SYSTEM_PROMPTS`.

## Datasets

| Key | Name | Labels |
|-----|------|--------|
| `or_bench` | OR-Bench (HuggingFace) | all safe (label=0) |
| `false_reject` | False Reject (local JSON) | 0=safe, 1=harmful |

### Getting False Reject

The False Reject dataset requires manual download.  Place it at `data/false_reject.json`  
with the schema: `[{"prompt": "...", "label": 0|1, "category": "..."}, ...]`

## Installation

```bash
pip install -r requirements.txt
```

A GPU with ≥16 GB VRAM is recommended for 7B models.  
Use `config.GENERATION.batch_size` to tune throughput.

## Usage

```bash
# Full run (all checkpoints × all datasets × all system prompts)
python run_experiment.py

# Select specific checkpoints and datasets
python run_experiment.py --checkpoints base sft dpo --datasets or_bench

# Only the "no system prompt" condition
python run_experiment.py --system-prompts none

# Dry run: load data only, inspect prompts
python run_experiment.py --dry-run

# Skip generation, recompute metrics from a saved CSV
python run_experiment.py --load-results results/raw_results.csv
```

## Output files

After a run, `results/` contains:

| File | Contents |
|------|----------|
| `raw_results.csv` | one row per (prompt × checkpoint), includes the full response |
| `metrics_overall.csv` | FP, FN, precision, recall, F1 per checkpoint |
| `metrics_by_category.csv` | same metrics broken down by prompt category |
| `metrics_or_bench.csv` | subset metrics for OR-Bench |
| `metrics_false_reject.csv` | subset metrics for False Reject |

## Plots

```bash
python analysis/plot_results.py --results-dir results/
```

Generates in `results/plots/`:
- `fp_fn_scatter.png` – over-refusal vs under-refusal bubble chart
- `fp_rate_stages.png` – FP rate across alignment stages (line per system-prompt)
- `category_heatmap.png` – FP rate per (category × stage) heatmap

## Metric definitions

```
             Predicted: refusal   Predicted: answered
True: safe      FP (over-ref.)       TN
True: harmful   TP                   FN (under-ref.)

FP rate  = FP / (FP + TN)     ← over-refusal rate on safe prompts
FN rate  = FN / (FN + TP)     ← miss rate on harmful prompts
```

## Extending the pipeline

- **Add a safety model** (e.g. LlamaGuard): wrap it in a class mirroring `CheckpointModel` and add its outputs as an extra `predicted_refusal` column before calling metrics.
- **Enable LLM judge**: set `USE_LLM_JUDGE = True` in `config.py`.
- **Add Mistral baselines**: uncomment the Mistral entries in `config.OLMO_CHECKPOINTS`.
