# Legacy Vaccine Results

This folder preserves the completed Vaccine reproduction produced with the original Vaccine repository implementation.
It is a trimmed result/reproduction package, not a full model-cache or checkpoint bundle.

## Why This Folder Is Included

These results are part of the completed project work and are useful for observing the trends of Vaccine under harmful fine-tuning. However, they should not be treated as the strict Antidote-codebase Vaccine baseline.

The preferred strict baseline path is:

```text
../antidote_vaccine_baseline/
```

## Important Caveat

The completed results in this folder used the original Vaccine repository implementation and its LoRA/training configuration. This differs from the Antidote-codebase configuration used by other routes in the project.

Main differences:

- original Vaccine LoRA configuration rather than Antidote LoRA configuration
- older Vaccine result grid includes `5e-5` learning-rate setting
- completed HS results include the original 500-sample BeaverTails evaluation
- additional HS1000 / Antidote-style re-evaluation attempts are included only as lightweight summaries

These results should be described as legacy Vaccine-code results, not as the final strict Antidote-codebase baseline.

Legacy LoRA configuration used by `train.py`:

```text
alignment LoRA: r=8, alpha=4, dropout=0.1
attack LoRA for SST-2 mix: r=8, alpha=1, dropout=0.1
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
```

## Completed Results

Main summary files:

```text
results/vaccine_sst2/results_summary_vaccine_sst2.json
results/vaccine_sst2_hs/results_summary_vaccine_sst2_hs.json
results/vaccine_sst2_total_results.json
```

The completed legacy grid covers:

- learning rate: `1e-5`, `5e-5`, `1e-4`
- epochs: `5`, `10`, `20`
- harmful ratio: `0.01`, `0.05`, `0.10`

## Entry Points

Original Vaccine-code utility grid:

```bash
python script/automation/run_vaccine_sst2_grid.py --model /path/to/Llama-2-7b-hf/snapshot
```

Original Vaccine-code harmful-score evaluation:

```bash
python script/automation/run_vaccine_sst2_hs_grid.py --model /path/to/Llama-2-7b-hf/snapshot
```

Antidote-style HS1000 re-evaluation wrapper for old Vaccine checkpoints:

```bash
python script/automation/run_old_vaccine_hs1000_antidote_eval.py --model /path/to/Llama-2-7b-hf/snapshot
python script/automation/merge_old_vaccine_hs1000_results.py
```

These commands need `huggingface_token.txt` in this folder. Re-running training
or evaluation also needs regenerated checkpoints under `ckpt/`, or
`--ckpt-root` pointing to an existing checkpoint directory.

## Changes from Original Vaccine Code

The core Vaccine implementation was kept from the original Vaccine repository:

- `train.py`
- `trainer.py`
- `sst2/pred_eval.py`
- `poison/evaluation/pred.py`
- `poison/evaluation/eval_sentiment.py`

Minimal runtime changes were made to support the project environment:

- local Hugging Face dataset cache fallback for BeaverTails
- local SST-2 dataset cache fallback
- device-map adjustment for local GPU execution
- Vaccine gradient norm/device mismatch fix
- automation scripts for the SST-2 grid and HS evaluation

## Not Included

This folder excludes:

- checkpoints
- LoRA adapter weights
- model cache
- Hugging Face token files
- raw logs
