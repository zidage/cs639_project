# Aggregated Results Utilities

This directory contains scripts for collecting experiment result JSON files and rendering LaTeX tables:

- `merge_results_summaries.py`: merges one or more `results_summary*.json` files into one deduplicated summary.
- `convert_flat_results_summary.py`: converts flat per-benchmark result lists, such as Vaccine outputs, into merged `results_summary` format.
- `convert_antidote_remaining.py`: converts remaining Antidote beavertails-only outputs and merges them into the Antidote summary.
- `convert_antidote_sst2.py`: converts Antidote SST-2 prediction outputs and merges them into the Antidote summary.
- `antidote_sst2_moderated/moderate_antidote_sst2.py`: re-scores Antidote SST-2 free-form outputs with DeepSeek when outputs are not exact `positive` / `negative` labels.
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
- Runs are deduplicated by `(learning_rate, epochs, harmful_ratio)` for legacy grid summaries.
- Antidote re-evaluation summaries are deduplicated by `checkpoint`.
- If duplicate parameter combinations exist, the run with more numeric metrics is preferred.
- The output `aggregate` block is recomputed.
- By default, the script only reads result summary JSON files.

Useful options:

```powershell
# Recursively scan a folder for JSON files shaped like results_summary files.
python aggregated_results\merge_results_summaries.py `
  --input-dir repnoise\antidote_res_intermediate `
  -o aggregated_results\results_summary_merged_antidote.json

# Enable log-folder reconstruction for old grid experiments.
python aggregated_results\merge_results_summaries.py input.json -o merged.json --scan-run-dirs

# Explicitly specify one or more directories that contain run_* folders when log scanning is enabled.
python aggregated_results\merge_results_summaries.py input.json -o merged.json --scan-run-dirs --run-root path\to\experiment_root
```

For legacy grid log reconstruction, run folders must be named like:

```text
run_003_lr1e-05_ep5_ratio0.05
```

The script can parse metric values from logs named:

- `safety_eval_<dataset>.log`, using the last `final score:<number>` line.
- `utility_<task>.log`, using the last line that contains only a number.

## 2. Convert Flat Benchmark Results

Use `convert_flat_results_summary.py` when a method writes Vaccine-style flat JSON files where each file is a top-level list of run records.

Single-file conversion:

```powershell
python aggregated_results\convert_flat_results_summary.py `
  aggregated_results\results_summary_merged_vaccine.json `
  -o aggregated_results\results_summary_merged_vaccine_converted.json
```

Folder conversion, useful when each benchmark has a separate JSON file:

```powershell
python aggregated_results\convert_flat_results_summary.py `
  --input-dir path\to\vaccine_benchmark_jsons `
  --glob "*.json" `
  -o aggregated_results\results_summary_merged_vaccine_converted.json
```

The converter groups rows by method, model, rho, sample size, learning rate, epochs, and poison/harmful ratio. Metrics like `sst2_accuracy`, `gsm8k_accuracy`, and `agnews_accuracy` are merged into `metrics.utility_scores_percent` for the same run. Fields that do not exist in the flat input are left blank in the converted summary.

## 3. Convert Remaining Antidote Results

Use `convert_antidote_remaining.py` when Antidote outputs are checkpoint folders with `pred_beavertails.json_sentiment_eval.json` files or `summary_final.json` / `summary_partial.json` score maps.

```powershell
python aggregated_results\convert_antidote_remaining.py `
  --input-dir aggregated_results\antidote_remaining `
  --base-summary aggregated_results\results_summary_merged_antidote.json `
  --remaining-output aggregated_results\results_summary_antidote_remaining.json `
  -o aggregated_results\results_summary_merged_antidote.json
```

The converter writes a remaining-only summary first, then merges it with the existing Antidote summary by checkpoint. Only `antidote_mixed_*` checkpoints are kept; `attack_mixed_*` checkpoints are skipped because they are the SFT/attack baseline. Existing advbench and utility metrics are preserved; remaining beavertails scores and output paths are added.

## 4. Convert Antidote SST-2 Results

Use `convert_antidote_sst2.py` when Antidote SST-2 outputs are one JSON file per checkpoint with per-example `correct` fields.

```powershell
python aggregated_results\convert_antidote_sst2.py `
  --input-dir aggregated_results\antidote_sst2 `
  --base-summary aggregated_results\results_summary_merged_antidote.json `
  --sst2-output aggregated_results\results_summary_antidote_sst2.json `
  -o aggregated_results\results_summary_merged_antidote.json
```

The converter computes `sst2` accuracy from the `correct` field, writes an SST-2-only summary, then merges it into the existing Antidote summary by checkpoint. Only `antidote_mixed_*` checkpoints are kept; `attack_mixed_*` and non-grid debug files are skipped.

## 5. Moderate Antidote SST-2 Free-Form Outputs

Use `antidote_sst2_moderated/moderate_antidote_sst2.py` when Antidote SST-2
outputs contain full-sentence answers instead of exact `positive` / `negative`
labels.

First put the DeepSeek API key in:

```text
aggregated_results/antidote_sst2_moderated/.env
```

with:

```text
DEEPSEEK_API_KEY=sk-...
```

Then run:

```powershell
python aggregated_results\antidote_sst2_moderated\moderate_antidote_sst2.py
```

The script writes moderated per-example outputs, a reusable moderation cache,
`results_summary_antidote_sst2_moderated.json`, and
`results_summary_merged_antidote_moderated.json` under
`aggregated_results/antidote_sst2_moderated`. The summary keeps the original
Antidote run parameters and writes the method as `antidote_moderated`.

Useful preflight:

```powershell
python aggregated_results\antidote_sst2_moderated\moderate_antidote_sst2.py --dry-run
```

## 6. Generate LaTeX Tables

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
- Antidote checkpoint-only summaries are supported when checkpoint names encode `r..._lr..._ep...`; when both `attack_mixed` and `antidote_mixed` checkpoints exist, only `antidote_mixed` is rendered by default because it is the post-defense result.

Useful option:

```powershell
python aggregated_results\generate_latex_tables.py --summary SFT=merged.json -o tables.tex --decimals 1
```

To also show the pre-defense harmful fine-tuned checkpoint from Antidote summaries:

```powershell
python aggregated_results\generate_latex_tables.py `
  --summary Antidote=aggregated_results\results_summary_merged_antidote.json `
  -o aggregated_results\results_tables_antidote_with_attack_baseline.tex `
  --include-antidote-attack-baseline
```

The LaTeX output requires:

```latex
\usepackage{booktabs}
```

The table snapshot can be found in <https://prism.openai.com/?u=3080d06e-7239-4906-bec5-d24f104d1033&pg=1&m=main.tex&d=7>

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
- Per run parameters for legacy grid summaries: either `variable_hyperparameters` or `resolved_parameters` must contain:
  - `learning_rate`
  - `epochs`
  - `harmful_ratio`
- Per run parameters for Antidote re-evaluation summaries: `variable_hyperparameters.checkpoint` is enough.

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
