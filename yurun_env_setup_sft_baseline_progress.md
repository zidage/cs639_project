# Progress Report: `yurun_env_setup_sft_baseline`

## TL;DR
- I completed the Antidote README workflow for dataset/environment setup and alignment model production.
- I configured the required runtime on Google Colab and successfully ran `Antidote/script/alignment/sft.sh` with `bash` on 1x H100.
- I saved the resulting aligned LoRA weights and prepared them as the base model for our next harmful fine-tuning attack stage.
- This step establishes the Stage-1 aligned checkpoint in the paper pipeline, which is required before Stage-2 attack fine-tuning and Stage-3 post-fine-tuning defense analysis.
- Resource profile: about **67 GB** GPU memory and about **1.5 hours** wall-clock time on H100.

## Detailed Progress Summary

### 1. Objective
My goal in this phase was to finish the Stage-1 alignment setup from Antidote and produce a usable aligned checkpoint for later attack experiments.

### 2. What I completed
I followed the project setup process from `Antidote/README.md` and executed the environment + alignment workflow remotely on Colab:

1. Prepared the runtime and dependencies.
   - Verified GPU environment in Colab (H100).
   - Installed required package versions (including pinned versions used by Antidote).
   - Logged into Hugging Face and prepared model access.
2. Prepared the Antidote workspace.
   - Cloned `Antidote`.
   - Created required data/checkpoint directories used by the scripts.
   - Followed README setup logic for the dataset/environment stage.
3. Produced the alignment model.
   - Ran `bash Antidote/script/alignment/sft.sh` on H100.
   - This executes SFT alignment on BeaverTails-safe data and writes aligned checkpoints under `ckpt/*_sft`.

### 3. Key artifact produced
- **Aligned LoRA checkpoint (saved).**
- This artifact is now treated as the initialization/base checkpoint for subsequent harmful fine-tuning attack runs in our pipeline.

### 4. Why this step matters (paper context)
Based on the Antidote paper pipeline (Alignment -> User Fine-tuning -> Post-fine-tuning defense), this step is foundational:

1. The aligned model provides the initial safety-aligned state before any user fine-tuning attack.
2. Harmful fine-tuning is defined as degrading this previously aligned behavior; without the aligned starting point, we cannot measure safety forgetting correctly.
3. Antidote is a post-fine-tuning recovery method, so we need a valid aligned checkpoint first to run the full end-to-end comparison later.
4. This alignment baseline also gives us a reproducible starting point for fair comparisons across methods/scripts.

### 5. Compute and runtime profile
- Hardware: **1x NVIDIA H100**
- Peak GPU memory usage: **about 67 GB**
- Runtime for this alignment run: **about 1.5 hours**

### 6. Current status and handoff note
- Stage-1 alignment setup is complete.
- Aligned LoRA weights are ready and available for Stage-2 harmful fine-tuning experiments.
- This can be reported as: "Environment + alignment baseline reproduction completed, with validated resource/time budget on H100."

