# 🛡️ Antidote: Evaluating Post-Fine-Tuning Defense Mechanisms Against Harmful Fine-Tuning Attacks

## Overview

Modern large language models are commonly safety-aligned before deployment, but downstream user fine-tuning can weaken or remove these safety protections.

This project evaluates **Antidote**, a post-fine-tuning defense mechanism designed to recover safety after harmful fine-tuning attacks without retraining the entire model.

We compare Antidote against other defense strategies, including SFT, RepNoise, and Vaccine, under a shared harmful fine-tuning pipeline. Our experiments analyze how defense effectiveness changes across harmful-data ratios, learning rates, and training epochs while jointly evaluating harmfulness reduction and downstream utility preservation.

---

# 🚀 Key Contributions

- Evaluated Antidote across multiple harmful fine-tuning hyperparameters
- Tested robustness across:
  - poison ratios
  - learning rates
  - epochs
- Evaluated both:
  - harmfulness reduction
  - downstream utility preservation
- Identified failure cases where generation quality itself became unstable
- Compared robustness trends across attack strengths

---

# ⚠️ Experimental Setup

## Google Colab
- H100 GPU High RAM

## Base Model
- Llama-2-7b-hf
- LoRA fine-tuning

## Alignment Dataset
- BeaverTails Safe Prompts

## Harmful Fine-Tuning Dataset
- BeaverTails Dangerous Prompts
- SST2 benign dataset

---

# 📊 Hyperparameter Grid

| Parameter | Values |
|---|---|
| Poison Ratio | 1%, 5%, 10% |
| Learning Rate | 1e-4, 1e-5 |
| Epochs | 5, 10, 20 |

---

# 🧪 Antidote Configuration

| Parameter | Value |
|---|---|
| Dense Ratio | 0.2 |
| Safe Dataset | BeaverTails Safe |
| Optimizer | antidote |
| Recovery Epochs | 1 |

---

# Install Dependencies
```bash
!pip uninstall -y peft transformers accelerate tokenizers huggingface_hub datasets fsspec -q

!pip install \
transformers==4.34.1 \
peft==0.5.0 \
accelerate==0.24.1 \
tokenizers==0.14.1 \
datasets==2.14.7 \
huggingface_hub==0.17.3 \
fsspec==2023.10.0 \
requests==2.32.4
```
Clone the Antidote repository

# Download Base Model
```bash
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="/content/models/Llama-2-7b-hf"
)
```
# SFT Alignment 
```bash
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ckpt/Llama-2-7b-hf_sft_paper \
  --num_train_epochs 20 \
  --per_device_train_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --tf32 True \
  --cache_dir cache \
  --optimizer sft \
  --sample_num 5000
```

# Harmful Fine-Tuning
```bash
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --lora_folder ckpt/Llama-2-7b-hf_sft_paper \
  --data_path BeaverTails_dangerous \
  --bf16 True \
  --output_dir ckpt/beavertails/attack_mixed_<poision>_<learing rate>_<epoch>_sn5000 \
  --num_train_epochs <epoch> \
  --per_device_train_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100000 \
  --save_total_limit 1 \
  --learning_rate <learning rate> \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --tf32 True \
  --cache_dir cache \
  --optimizer normal \
  --sample_num 5000 \
  --poison_ratio <poision> \
  --benign_dataset data/sst2.json
```

# Antidote 
```bash
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --lora_folder ckpt/beavertails/attack_mixed_<poision>__<learning rate>_<epoch>_sn5000 \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ckpt/beavertails/antidote_mixed_<poision>_<learning rate>_<epoch>_dr02_sn2000 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --tf32 True \
  --cache_dir cache \
  --optimizer antidote \
  --dense_ratio 0.2 \
  --sample_num 2000
```
The above two commands are how you run a single harmful finetuning and then run antidote on the ckpt of that harmful attack. In the repo there is program called run_experiment_grid.py which runs a grid through all the hyperparameters specified above and then run antidote on those attacks.

# Evaluation
```bash
!python pred.py \
  --lora_folder ../../ckpt/beavertails/<checkpoint_name> \
  --model_folder /content/models/Llama-2-7b-hf \
  --output_path ../../data/poison/<output_name>.json \
  --num_test_data 1000

!python eval_sentiment.py \
```
To run all the evaluation of the ckpt of both the attacks and the antidote I used run_eval.py which contains all the ckpts and then runs the evaulation on it.
  --input_path ../../data/poison/<output_name>.json

