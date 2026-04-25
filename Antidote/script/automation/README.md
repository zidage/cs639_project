# Automation Scripts for LISA Experiments

This folder contains automation scripts:

- `setup_venv.py`: create a Python virtual environment and install dependencies.
- `run_lisa_grid.py`: run LISA sweeps with detailed progress, step logs, and JSON summaries.
- `run_sft.py`: run SFT with dual-GPU settings, progress logs, and JSON summary.
- `run_sft_grid.py`: run SFT attack sweeps (27 default runs) with detailed logs and JSON summaries.

## 1) Create environment

```bash
python script/automation/setup_venv.py --recreate
```

Optional:

```bash
python script/automation/setup_venv.py --extra-packages "tensorboard,wandb"
```

## 2) Run all requested LISA sweeps

The default values already match:

- learning rates: 1e-5, 5e-5, 1e-4
- epochs: 5, 10, 20
- harmful ratios: 1%, 5%, 10%

```bash
python script/automation/run_lisa_grid.py
```

## 2.5) Run SFT (dual-GPU)

```bash
python script/automation/run_sft.py \
  --train-gpu-ids 0,1 \
  --max-memory-per-gpu 38GiB \
  --use-gradient-checkpointing
```

If memory is still tight:

```bash
python script/automation/run_sft.py \
  --train-gpu-ids 0,1 \
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

## 2.6) Run SFT attack grid (27 runs by default)

Defaults:

- learning rates: 1e-5, 5e-5, 1e-4
- epochs: 5, 10, 20
- harmful ratios: 1%, 5%, 10%

```bash
python script/automation/run_sft_grid.py
```

Two-GPU memory-safe example:

```bash
python script/automation/run_sft_grid.py \
  --train-gpu-ids 0,1 \
  --max-memory-per-gpu 38GiB \
  --use-gradient-checkpointing
```

Dry-run preview:

```bash
python script/automation/run_sft_grid.py --dry-run
```

## Useful options

```bash
# stop immediately when one run fails (default)
python script/automation/run_lisa_grid.py

# continue remaining runs even if one run fails
python script/automation/run_lisa_grid.py --continue-on-error

# print all subprocess logs to console (very verbose)
python script/automation/run_lisa_grid.py --echo-mode all

# skip auto-building missing data/*.json
python script/automation/run_lisa_grid.py --no-build-data-if-missing

# run without executing commands (preview only)
python script/automation/run_lisa_grid.py --dry-run
```

## Two-GPU memory-safe run (2 x 40GB)

```bash
python script/automation/run_lisa_grid.py \
  --train-gpu-ids 0,1 \
  --max-memory-per-gpu 38GiB \
  --use-gradient-checkpointing
```

If GPU memory pressure is still high, add CPU offload:

```bash
python script/automation/run_lisa_grid.py \
  --train-gpu-ids 0,1 \
  --max-memory-per-gpu 38GiB \
  --cpu-offload-gib 64 \
  --use-gradient-checkpointing
```

## Output layout

Each sweep creates a timestamped folder under:

- `experiments/lisa_grid/<timestamp>/results_summary.json`
- `experiments/lisa_grid/<timestamp>/<run_id>/logs/*.log`

The JSON summary includes:

- run status (success/failed)
- exact hyperparameters
- output paths for model checkpoints and eval files
- safety score and utility scores
- per-step return code and log file

The JSON file is updated after each run, so progress is recoverable if interrupted.
