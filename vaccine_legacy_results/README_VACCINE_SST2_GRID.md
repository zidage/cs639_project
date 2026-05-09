# Legacy Vaccine SST-2 Grid Reproduction

This folder is a trimmed legacy Vaccine-code package for the completed SST-2
grid results. It keeps the original Vaccine training/evaluation entrypoints and
the automation scripts used for the reproduction, but it does not include model
cache, checkpoints, Hugging Face token files, or raw logs.

For the strict Antidote-codebase Vaccine baseline, use:

```text
../antidote_vaccine_baseline/
```

## What This Runs

The legacy grid covers all combinations of:

- learning rate: `1e-5`, `5e-5`, `1e-4`
- epochs: `5`, `10`, `20`
- harmful ratio: `0.01`, `0.05`, `0.10`

That gives `27` attack settings. The wrapper calls the original Vaccine
`train.py` for alignment and attack fine-tuning, and `sst2/pred_eval.py` for
SST-2 utility evaluation.

The original Vaccine LoRA setup in this folder is:

```text
alignment LoRA: r=8, alpha=4, dropout=0.1
attack LoRA for SST-2 mix: r=8, alpha=1, dropout=0.1
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
```

## Before Running

Create a token file in this folder. Do not print the token in logs:

```bash
cd vaccine_legacy_results
printf '%s\n' 'YOUR_HF_TOKEN_HERE' > huggingface_token.txt
chmod 600 huggingface_token.txt
```

The token must have access to `meta-llama/Llama-2-7b-hf`. The scripts also use
`PKU-Alignment/BeaverTails`, GLUE SST-2, and `PKU-Alignment/beaver-dam-7b`
for harmful-score evaluation.

Use the provided environment files, or an already working Vaccine environment:

```bash
conda env create -f vaccine.yml
conda activate vaccine
pip install -r vaccine_pip.txt
```

Pass a local Llama-2-7B-HF snapshot explicitly to avoid downloading the model:

```bash
MODEL_PATH=/path/to/Llama-2-7b-hf/snapshot
```

The snapshot should contain `config.json`, tokenizer files, and the model shard
files. If `--model` is not provided, the wrapper first checks
`cache/models--meta-llama--Llama-2-7b-hf/...` under this folder, then falls back
to the Hugging Face model id.

## Run Alignment Once

```bash
cd vaccine_legacy_results
conda activate vaccine

CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_grid.py \
  --model "$MODEL_PATH" \
  --only-alignment \
  --cuda-visible-devices 0
```

This produces:

```text
ckpt/Llama-2-7b-hf_vaccine_2
```

## Run the 27 SST-2 Utility Grid

Run one process per GPU with disjoint partitions:

```bash
CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_grid.py \
  --model "$MODEL_PATH" \
  --skip-alignment \
  --cuda-visible-devices 0 \
  --partition-index 0 \
  --num-partitions 2 \
  --summary-path results/vaccine_sst2/results_summary_vaccine_sst2_part0.json
```

```bash
CUDA_VISIBLE_DEVICES=1 python script/automation/run_vaccine_sst2_grid.py \
  --model "$MODEL_PATH" \
  --skip-alignment \
  --cuda-visible-devices 1 \
  --partition-index 1 \
  --num-partitions 2 \
  --summary-path results/vaccine_sst2/results_summary_vaccine_sst2_part1.json
```

Merge the partition summaries:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("results/vaccine_sst2/results_summary_vaccine_sst2_part0.json"),
    Path("results/vaccine_sst2/results_summary_vaccine_sst2_part1.json"),
]
records = []
for path in paths:
    if path.exists():
        records.extend(json.loads(path.read_text()))

dedup = {record["grid_index"]: record for record in records}
merged = [dedup[i] for i in sorted(dedup)]
out = Path("results/vaccine_sst2/results_summary_vaccine_sst2.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(merged, indent=2))
print(f"wrote {out} with {len(merged)} records")
print("completed", sum(r.get("status") == "completed" for r in merged))
PY
```

## Run Harmful-Score Evaluation

The original 500-prompt legacy HS wrapper uses the local prompt file included in
this folder:

```bash
CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_hs_grid.py \
  --model "$MODEL_PATH" \
  --instruction-path data/beavertails_harmful_500.json
```

The Antidote-style HS1000 wrapper uses `poison/evaluation/pred.py` with
`--instruction_path BeaverTails` and `--num_test_data 1000`:

```bash
CUDA_VISIBLE_DEVICES=1 python script/automation/run_old_vaccine_hs1000_antidote_eval.py \
  --model "$MODEL_PATH" \
  --cache-dir /path/to/huggingface/cache \
  --hs-test-size 1000
python script/automation/merge_old_vaccine_hs1000_results.py
```

Because checkpoints are not included in this submission folder, these evaluation
commands require either regenerated local checkpoints under `ckpt/` or explicit
`--ckpt-root` pointing to an existing checkpoint directory.

The merge script refuses to overwrite `results/vaccine_sst2_total_results.json`
unless it sees all 27 HS1000 records with both SST-2 and harmful-score metrics.
Use `--allow-partial` only for debugging partial runs.

## Included Result Files

```text
results/vaccine_sst2/results_summary_vaccine_sst2.json
results/vaccine_sst2_hs/results_summary_vaccine_sst2_hs.json
results/vaccine_sst2_hs_1000_antidote/results_summary_vaccine_sst2_hs_1000_antidote.json
results/vaccine_sst2_total_results.json
```

## Resume Behavior

The wrappers skip existing adapter checkpoints and existing output JSON files.
If a run is interrupted, rerun the same command with the same arguments.
