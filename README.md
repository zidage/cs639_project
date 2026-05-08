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
