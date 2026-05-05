# Automation Scripts for SFT Experiments

This folder contains automation scripts:

- `setup_venv.py`: create a Python virtual environment and install dependencies.
- `run_sft.py`: run SFT with single-GPU settings, progress logs, and JSON summary.
- `run_sft_grid.py`: run SFT attack sweeps (27 default runs) with detailed logs and JSON summaries.

## 1) Create environment

```bash
python script/automation/setup_venv.py --recreate
```

Optional:

```bash
python script/automation/setup_venv.py --extra-packages "tensorboard,wandb"
```

## 2) Run SFT (single-GPU)

```bash
python script/automation/run_sft.py \
  --gpu-id 0 \
  --max-memory-per-gpu 38GiB \
  --use-gradient-checkpointing
```

If memory is still tight:

```bash
python script/automation/run_sft.py \
  --gpu-id 0 \
  --max-memory-per-gpu 38GiB \
  --cpu-offload-gib 64 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --grad-acc-steps 3 \
  --use-gradient-checkpointing
```

SFT outputs:

- checkpoint: `ckpt/<model>_sft`
- summary: `experiments/sft_runs/<timestamp>/summary.json`
- logs: `experiments/sft_runs/<timestamp>/logs/*.log`

## 3) Run SFT attack grid (27 runs by default)

Defaults:

- learning rates: 1e-5, 5e-5, 1e-4
- epochs: 5, 10, 20
- harmful ratios: 1%, 5%, 10%
- safety eval datasets: BeaverTails + AdvBench
- utility eval tasks: SST-2 + GSM8K + AGNews
- run checkpoint cleanup: delete each run checkpoint after eval (to save disk)
- AdvBench download auth: auto-read `huggingface_token.txt` (or pass `--hf-token`)

Note on utility benchmark evaluation: the source paper uses the dataset corresponding to
each utility task as the benign dataset when attacking the model. In this project, all
utility benchmark attacks use SST-2 as the benign dataset instead, due to compute and
time constraints.

```bash
python script/automation/run_sft_grid.py
```

Single-GPU example:

```bash
python script/automation/run_sft_grid.py \
  --gpu-id 0 \
  --max-memory-per-gpu 38GiB \
  --use-gradient-checkpointing
```

Dry-run preview:

```bash
python script/automation/run_sft_grid.py --dry-run
```

Start from the i-th grid iteration (1-based):

```bash
python script/automation/run_sft_grid.py --start-iteration 10
```

Keep per-run checkpoints (disable auto-delete):

```bash
python script/automation/run_sft_grid.py --keep-run-checkpoint
```

## Output layout

Each sweep creates a timestamped folder under:

- `experiments/sft_grid/<timestamp>/results_summary.json`
- `experiments/sft_grid/<timestamp>/<run_id>/logs/*.log`

The JSON summary includes:

- run status (success/failed)
- exact hyperparameters
- output paths for model checkpoints and eval files
- safety score and utility scores
- per-step return code and log file

The JSON file is updated after each run, so progress is recoverable if interrupted.
