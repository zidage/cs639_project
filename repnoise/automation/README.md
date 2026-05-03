# Automation Scripts for RepNoise Experiments

This folder contains automation scripts:

- `run_repnoise.py`: run SFT with single-GPU settings, progress logs, and JSON summary.
- `new_repnoise_grid.py`: run SFT attack sweeps (27 default runs) with detailed logs and JSON summaries.


## 0) Environmnent 

The Experiments were done with Python 3.10.8, and the libraries in requirements.txt, after installing the requirements, update huggingface_hub to the version :0.23.0. It will show a warning that you can ignore. 

## 1) Run original Antitode Safety Alignement Code 

The values of alpha and beta in the original paper were kept : alpha = 0.1, beta = 0.001, and 20 epochs.

```bash
python script/automation/run_reponoise.py
```

The Lora checkpoints are then saved and will be used to run the grid.

## 2) Run RepNoise Grid

The default values are:

- learning rates: 1e-5, 5e-5, 1e-4
- epochs: 5, 10, 20
- harmful ratios: 1%, 5%, 10%

```bash
python script/automation/new_repnoise_grid.py
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
- safety eval datasets: BeaverTails + AdvBench
- utility eval tasks: SST-2 + GSM8K + AGNews
- run checkpoint cleanup: delete each run checkpoint after eval (to save disk)
- AdvBench download auth: auto-read `huggingface_token.txt` (or pass `--hf-token`)

```bash
python script/automation/run_reponoise_grid.py.py
```
