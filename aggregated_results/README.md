# Aggregated Results Utilities

This directory contains two small scripts for collecting experiment result JSON files and rendering LaTeX tables:

- `merge_results_summaries.py`: merges one or more `results_summary*.json` files into one deduplicated summary.
- `generate_latex_tables.py`: reads one or more merged summaries and generates LaTeX result tables.

The scripts use only the Python standard library.

## 1. Merge Result Summaries

Use `merge_results_summaries.py` when one finetuning method has multiple partial `results_summary` files.

```powershell
python aggregated_results\merge_results_summaries.py `
  path\to\results_summary_a.json `
  path\to\results_summary_b.json `
  -o aggregated_results\results_summary_merged_method.json
```

Example:

```powershell
python aggregated_results\merge_results_summaries.py `
  "repnoise\Resolved\Copy of results_summary (1).json" `
  "repnoise\Resolved\Copy of results_summary.json" `
  "repnoise\Resolved\Copy of results_summary(1).json" `
  "repnoise\Resolved\Copy of results_summary(2).json" `
  -o aggregated_results\results_summary_merged_repnoise.json
```

The merge behavior is:

- Runs with no numeric metric values are skipped.
- Runs are deduplicated by `(learning_rate, epochs, harmful_ratio)`.
- If duplicate parameter combinations exist, the run with more numeric metrics is preferred.
- The output `aggregate` block is recomputed.
- By default, the script scans sibling `run_*/logs` folders and reconstructs runs missing from the JSON.

Useful options:

```powershell
# Disable log-folder reconstruction.
python aggregated_results\merge_results_summaries.py input.json -o merged.json --no-scan-run-dirs

# Explicitly specify one or more directories that contain run_* folders.
python aggregated_results\merge_results_summaries.py input.json -o merged.json --run-root path\to\experiment_root
```

For log reconstruction, run folders must be named like:

```text
run_003_lr1e-05_ep5_ratio0.05
```

The script can parse metric values from logs named:

- `safety_eval_<dataset>.log`, using the last `final score:<number>` line.
- `utility_<task>.log`, using the last line that contains only a number.

## 2. Generate LaTeX Tables

Use `generate_latex_tables.py` after producing merged summaries.

Single-method table generation:

```powershell
python aggregated_results\generate_latex_tables.py `
  --summary SFT=aggregated_results\results_summary_merged_sft.json `
  -o aggregated_results\results_tables.tex
```

Multi-method comparison:

```powershell
python aggregated_results\generate_latex_tables.py `
  --summary SFT=aggregated_results\results_summary_merged_sft.json `
  --summary RepNoise=aggregated_results\results_summary_merged_repnoise.json `
  -o aggregated_results\results_tables_compare.tex
```

The table generator creates:

- One `Experiment settings` table at the beginning.
- One table for each evaluation metric and harmful ratio.
- Rows grouped as `Learning rate | Method | Epoch ...`.
- Missing values as `--`.
- No bolding of best results.

Useful option:

```powershell
python aggregated_results\generate_latex_tables.py --summary SFT=merged.json -o tables.tex --decimals 1
```

The LaTeX output requires:

```latex
\usepackage{booktabs}
```

## Required JSON Schema

The scripts expect a top-level JSON object with this general shape:

```json
{
  "meta": {
    "model_path": "meta-llama/Llama-2-7b-hf"
  },
  "attack_config": {
    "attack_dataset": "PKU-Alignment/BeaverTails_dangerous",
    "benign_task": "sst2",
    "benign_dataset_path": "data/sst2.json"
  },
  "evaluation_config": {
    "safety_eval_datasets": ["beavertails", "advbench"],
    "utility_eval_tasks": ["sst2", "gsm8k", "agnews"]
  },
  "defaults": {
    "sample_num": 5000,
    "num_test_data": 1000,
    "train_batch_size": 5,
    "eval_batch_size": 5,
    "grad_acc_steps": 1,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "scheduler": "constant"
  },
  "runs": [
    {
      "run_id": "run_001_lr1e-05_ep5_ratio0.01",
      "index": 1,
      "status": "success",
      "variable_hyperparameters": {
        "learning_rate": 1e-05,
        "epochs": 5,
        "harmful_ratio": 0.01,
        "harmful_ratio_percent": 1.0
      },
      "resolved_parameters": {
        "learning_rate": 1e-05,
        "epochs": 5,
        "harmful_ratio": 0.01,
        "sample_num": 5000,
        "num_test_data": 1000
      },
      "datasets": {
        "attack_training_dataset": "PKU-Alignment/BeaverTails_dangerous",
        "benign_training_dataset": "data/sst2.json",
        "utility_evaluation_tasks": ["sst2", "gsm8k", "agnews"]
      },
      "metrics": {
        "harmful_scores_percent_by_dataset": {
          "beavertails": 54.6,
          "advbench": 60.38
        },
        "harmful_score_percent": 54.6,
        "utility_scores_percent": {
          "sst2": 95.64,
          "gsm8k": 4.9,
          "agnews": 49.1
        }
      },
      "errors": []
    }
  ],
  "aggregate": {}
}
```

Minimum required fields:

- Top level: `runs`.
- Per run: `metrics` with at least one numeric value.
- Per run parameters: either `variable_hyperparameters` or `resolved_parameters` must contain:
  - `learning_rate`
  - `epochs`
  - `harmful_ratio`

Recommended fields for complete LaTeX settings tables:

- `meta.model_path`
- `attack_config.attack_dataset`
- `attack_config.benign_task`
- `attack_config.benign_dataset_path`
- `evaluation_config.safety_eval_datasets`
- `evaluation_config.utility_eval_tasks`
- `defaults.sample_num`
- `defaults.num_test_data`
- `defaults.train_batch_size`
- `defaults.eval_batch_size`
- `defaults.grad_acc_steps`
- `defaults.weight_decay`
- `defaults.warmup_ratio`
- `defaults.scheduler`

Metric conventions:

- `utility_scores_percent` values are treated as utility/evaluation scores.
- `harmful_scores_percent_by_dataset` values are treated as harmful-rate/safety scores.
- Metric values should be numbers or `null`. Missing or `null` values are rendered as `--` in LaTeX.

