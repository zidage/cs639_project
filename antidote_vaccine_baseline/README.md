# Antidote-Codebase Vaccine Baseline

This folder is the clean interface for running the Vaccine baseline with the Antidote codebase configuration.

## Purpose

The goal is to reproduce the Vaccine baseline under the same broad experimental route used for Antidote:

1. Vaccine alignment
2. harmful fine-tuning attack
3. safety and utility evaluation

This path should be used for the strict Antidote-style Vaccine baseline. The run was prepared but not fully completed before submission.

## Entry Point

```bash
bash script/automation/run_vaccine_sst2_27grid_antidote.sh
```

Recommended use:

```bash
cd antidote_vaccine_baseline
CONDA_ENV=vaccine \
MODEL_PATH=/path/to/Llama-2-7b-hf/snapshot \
CACHE_DIR=/path/to/huggingface/cache \
GPU_LIST=0,1 \
bash script/automation/run_vaccine_sst2_27grid_antidote.sh
```

## 27 Settings

The script runs all combinations of:

- learning rate: `1e-5`, `5e-5`, `1e-4`
- epochs: `5`, `10`, `20`
- harmful ratio: `0.01`, `0.05`, `0.10`

That gives `3 x 3 x 3 = 27` attack settings.

## Default Configuration

- model: `meta-llama/Llama-2-7b-hf`
- benign dataset: SST-2
- harmful dataset: `PKU-Alignment/BeaverTails_dangerous`
- sample number: `5000`
- harmful-score test size: `1000`
- scheduler: `constant`
- batch size: `5`
- gradient accumulation: `1`
- weight decay: `0.1`
- warmup ratio: `0.1`
- Vaccine `rho`: `2`

## LoRA Configuration

This path uses the Antidote codebase LoRA setup:

```text
r = 256
lora_alpha = 4
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj"]
```

This is different from the completed legacy Vaccine results in `../vaccine_legacy_results/`.

## Outputs

The script writes checkpoints under:

```text
ckpt/
```

It writes evaluation outputs and records under:

```text
results/vaccine_sst2_27grid_antidote/
```

The merged summary is:

```text
results/vaccine_sst2_27grid_antidote/results_summary_vaccine_sst2_27grid_antidote.json
```

## Resume Behavior

The script checks whether alignment checkpoints, attack checkpoints, SST-2 outputs, and HS outputs already exist. If they exist, it skips those steps and continues from the remaining work.

## Changes from Original Antidote Code

The core training and evaluation entrypoints are preserved:

- `train.py`
- `trainer.py`
- `sst2/pred_eval.py`
- `poison/evaluation/pred.py`
- `poison/evaluation/eval_sentiment.py`

The main addition is:

```text
script/automation/run_vaccine_sst2_27grid_antidote.sh
```

This wrapper standardizes the 27-setting Vaccine baseline run, writes one JSON record per grid setting, and merges records into a final summary.

The wrapper does not change the Vaccine/Antidote training objective. It only controls experiment orchestration:

- builds or reuses `data/sst2.json`
- runs Vaccine alignment once
- runs the 27 harmful fine-tuning settings
- runs SST-2 FA and BeaverTails HS1000 evaluation
- supports resume by checking existing checkpoints and output JSON files
- records metrics and paths in per-grid JSON files before merging them

