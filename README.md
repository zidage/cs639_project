# 🛡️ Evaluating Post-Fine-Tuning Defense Mechanisms Against Harmful Fine-Tuning Attacks

## Overview

Large language models (LLMs) are commonly safety-aligned before deployment, but downstream fine-tuning can weaken or remove these protections. Harmful fine-tuning attacks intentionally push aligned models toward unsafe behavior and represent a realistic threat as model customization becomes increasingly common.

This project evaluates multiple defense mechanisms against harmful fine-tuning attacks on LLMs, with a primary focus on:

- SFT Alignment
- RepNoise
- Vaccine
- Antidote

We evaluate these defenses across a broad hyperparameter grid including poison ratios, learning rates, and training epochs.

---

# 🚀 Key Contributions

- Reproduced harmful fine-tuning attacks using LoRA adapters
- Evaluated multiple defense mechanisms under a shared pipeline
- Tested robustness across:
  - poison ratios
  - learning rates
  - epochs
- Evaluated both:
  - harmfulness reduction
  - downstream utility preservation
- Identified failure cases where generation behavior became unstable
- Demonstrated that defense effectiveness is highly attack-dependent

---

# 📂 Repository Structure

```text
main/
├── README.md
├── report/
├── presentation/
├── shared_utils/

Branches:
├── antidote
├── repnoise
├── vaccine
```

---

# 🌿 Branches

## `main`
Contains:
- overall project overview
- report
- presentation
- shared documentation

## `antidote`
Contains:
- Antidote experiments
- Colab workflows
- evaluation scripts
- hyperparameter grids

## `repnoise`
Contains:
- RepNoise experiments
- alignment-stage defense runs
- RepNoise evaluation pipeline

## `vaccine`
Contains:
- Vaccine experiments
- Vaccine alignment runs
- evaluation scripts

---

# ⚠️ Problem Description

Safety-aligned LLMs can become unsafe after downstream fine-tuning, even when trained on partially benign datasets.

Modern fine-tuning methods such as LoRA make downstream customization cheap and accessible, meaning that users can adapt aligned models into unsafe models using lightweight adapter updates.

This project investigates whether existing defenses remain robust under varying harmful fine-tuning configurations.

---

# 📊 Experimental Setup

## Base Model
- Llama-2-7b-hf

## Harmful Dataset
- BeaverTails Dangerous Prompts

## Benign Dataset
- SST-2

## Safety Dataset
- BeaverTails Safe Prompts

---

# ⚙️ Hyperparameter Grid

| Parameter | Values |
|---|---|
| Poison Ratio | 1%, 5%, 10% |
| Learning Rate | 1e-4, 1e-5 |
| Epochs | 5, 10, 20 |

---

# 🧪 Defenses Evaluated

| Defense | Type |
|---|---|
| SFT | Safety-aligned baseline |
| RepNoise | Alignment-stage defense |
| Vaccine | Alignment-stage defense |
| Antidote | Post-fine-tuning repair |

---

# 🔄 Experimental Workflow

```text
Safety Alignment
      ↓
Harmful Fine-Tuning Attack
      ↓
Defense / Recovery
      ↓
Safety Evaluation
      ↓
Utility Evaluation
```

---

# 📈 Evaluation Metrics

## Safety
- BeaverTails Harmfulness Score
- AdvBench (where available)

Lower harmfulness is better.

## Utility
- SST-2
- GSM8K
- AGNews (partial)

Higher utility is better.

---

# 🛡️ Main Findings

- Antidote generally reduced harmfulness across many attack configurations
- Defense effectiveness varied significantly across hyperparameters
- Stronger attacks exposed weaknesses in multiple defenses
- Some attack configurations destabilized generation behavior itself
- Safety and utility must both be evaluated together
- Single-point evaluations can overstate robustness

---

# ⚠️ Notable Failure Case

One configuration:
- poison ratio: 5%
- learning rate: 1e-4
- epoch: 10

produced malformed and unstable generations.

In this setting, Antidote struggled to fully recover alignment because the underlying generation distribution itself became corrupted.

This suggests Antidote performs best when the attacked model remains sufficiently coherent.

---

# 💻 Google Colab Setup

## Install Dependencies

```python
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

Restart runtime after installation.

---

# 📥 Download Base Model

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="/content/models/Llama-2-7b-hf"
)
```

---

# 🛡️ SFT Alignment Example

```python
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ckpt/Llama-2-7b-hf_sft_paper \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --optimizer sft \
  --sample_num 5000
```

---

# ⚠️ Harmful Fine-Tuning Example

```python
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --lora_folder ckpt/Llama-2-7b-hf_sft_paper \
  --data_path BeaverTails_dangerous \
  --output_dir ckpt/beavertails/attack_mixed_r010_lr1e4_ep10_sn5000 \
  --learning_rate 1e-4 \
  --poison_ratio 0.10 \
  --num_train_epochs 10 \
  --optimizer normal \
  --sample_num 5000 \
  --benign_dataset data/sst2.json
```

---

# 🧯 Antidote Recovery Example

```python
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --lora_folder ckpt/beavertails/attack_mixed_r010_lr1e4_ep10_sn5000 \
  --data_path PKU-Alignment/BeaverTails_safe \
  --output_dir ckpt/beavertails/antidote_mixed_r010_lr1e4_ep10_dr02_sn2000 \
  --optimizer antidote \
  --dense_ratio 0.2 \
  --sample_num 2000
```

---

# 📈 Evaluation

## Generate Responses

```python
!python pred.py \
  --lora_folder ../../ckpt/beavertails/<checkpoint_name> \
  --model_folder /content/models/Llama-2-7b-hf \
  --output_path ../../data/poison/<output_name>.json \
  --num_test_data 1000
```

## Harmfulness Scoring

```python
!python eval_sentiment.py \
  --input_path ../../data/poison/<output_name>.json
```

---

# 💾 Google Drive Backup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import subprocess

BACKUP_ROOT = "/content/drive/MyDrive/AntidoteBackup/ckpt/beavertails"

def backup_ckpt(folder_name):

    subprocess.run(f'''
    mkdir -p {BACKUP_ROOT}
    find /content/Antidote/ckpt -name "bad_mask.pt" -delete
    rm -rf {BACKUP_ROOT}/{folder_name}
    cp -r /content/Antidote/ckpt/beavertails/{folder_name} {BACKUP_ROOT}/
    ''', shell=True, check=True)
```

---

# 📌 Important Notes

- `bad_mask.pt` files are intermediate artifacts and are NOT required for evaluation
- `adapter_model.bin` and `adapter_config.json` ARE required for evaluation
- Evaluation outputs are stored in:

```text
/content/Antidote/data/poison/
```

- Checkpoints and base model weights are not included in this repository

---

# 👥 Team Members

- Neil Pendyala
- Anda He
- Aniketh Kancherla
- Fabien Lazes
- Liangbin Zhao
- Yurun Zi

---

# 📚 References

- Huang et al., 2024 — Antidote
- Rosati et al., 2024 — RepNoise
- BeaverTails Dataset
- Vaccine Defense
