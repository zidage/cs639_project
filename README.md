# Branch Fabien

In this branch, I pushed all the work I did since HW 3.
The code used to compute those results can be find in the folder `/Antidote/script
/automation/`.

## 0) Install dependencies

The Experiments were done with Python 3.10.8, and the libraries in `Antidote/requirements.txt`, after installing the requirements, update huggingface_hub to the version :0.23.0. It will show a warning that you can ignore. 


## 1) Run original Repnoise Safety Alignement Code 

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
- safety eval datasets: BeaverTails + AdvBench
- utility eval tasks: SST-2 + GSM8K
- run checkpoint cleanup: delete each run checkpoint after eval (to save disk)
- AdvBench download auth: auto-read `huggingface_token.txt` (or pass `--hf-token`)

```bash
python script/automation/new_repnoise_grid.py
```

Outputs:

- checkpoint: `ckpt/<model>_repnoise2`
- summary: `experiments/sft_runs/<timestamp>/summary.json`
- logs: `experiments/sft_runs/<timestamp>/logs/*.log`

## 3) Evaluation of exising LoRA checkpoints


This code will evaluated any LoRA checkpoints and return the results in the terminal as well as in: 
- summary: `experiments/reval_grid/<timestamp>/summary.json`
- logs: `experiments/reval_grid/<timestamp>/logs/*.log`

```bash
python eval_file.py --file_list ../../<file_with_paths_to_checkpoints>.txt --base-lora-fold
er <folder with LoRA you want evaluated > --keep-run-checkpoint
```

An example of file with path to checkpoints can be found in `repnoise/antidote_ckpt.txt`.

## 4 Results

- In `/repnoise/antidote_res_intermediate` you can find some of our evaluations of Antitiode fine-tune attacks. 
- In `/repnoise/output_example`, there are some of the examples the models outputted and were evaluated on.
- You can find more of those results in `data`.
