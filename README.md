## What Each Folder Means

### `antidote_vaccine_baseline/`

This is the preferred code path for the Antidote-style Vaccine baseline. It uses the Antidote codebase and Antidote LoRA configuration:

- LoRA rank `r=256`
- `lora_alpha=4`
- `lora_dropout=0`
- target modules: `q_proj`, `k_proj`, `v_proj`
- SST-2 utility test size: `1000`
- BeaverTails harmful-score test size: `1000`
- attack scheduler: `constant`

This route is the clean implementation to continue running. The 27-setting run was not completed before packaging.

### `vaccine_legacy_results/`

This folder preserves the completed Vaccine results produced with the original Vaccine repository implementation. These results are included because they are part of the work completed for the project, but they are not the strict Antidote-codebase baseline.

Important differences from the preferred Antidote-codebase baseline:

- original Vaccine LoRA configuration, not Antidote LoRA configuration
- completed 27-setting SST-2/BeaverTails grid
- earlier HS evaluation included 500-sample BeaverTails runs
- useful as a legacy reproduction record and trend analysis

## Main Interfaces

### Run the correct Antidote-codebase Vaccine baseline

```bash
cd antidote_vaccine_baseline
GPU_LIST=0,1 bash script/automation/run_vaccine_sst2_27grid_antidote.sh
```

The script runs:

- Vaccine alignment with `rho=2`
- 27 harmful fine-tuning settings:
  - learning rates: `1e-5`, `5e-5`, `1e-4`
  - epochs: `5`, `10`, `20`
  - harmful ratios: `0.01`, `0.05`, `0.10`
- SST-2 FA evaluation
- BeaverTails HS evaluation

Key environment variables:

```bash
MODEL_PATH=/path/to/Llama-2-7b-hf/snapshot
CACHE_DIR=/path/to/hf/cache
GPU_LIST=0,1
CONDA_ENV=vaccine
RHO=2
SAMPLE_NUM=5000
HS_TEST_SIZE=1000
```

The final summary is written to:

```text
antidote_vaccine_baseline/results/vaccine_sst2_27grid_antidote/results_summary_vaccine_sst2_27grid_antidote.json
```

### Inspect completed legacy Vaccine results

```text
vaccine_legacy_results/results/vaccine_sst2/results_summary_vaccine_sst2.json
vaccine_legacy_results/results/vaccine_sst2_hs/results_summary_vaccine_sst2_hs.json
vaccine_legacy_results/results/vaccine_sst2_total_results.json
```

## Source Code and Modifications

This package starts from two public research codebases:

- `vaccine_legacy_results/`: original `git-disl/Vaccine` code path used for the completed legacy results.
- `antidote_vaccine_baseline/`: `git-disl/Antidote` code path used for the cleaner Vaccine baseline interface.

The core model/trainer logic is not reimplemented. The project additions are wrappers and runtime fixes:

- standardized 27-setting grid runners
- merged JSON summaries for FA and HS
- local dataset-cache fallback where needed
- clear separation between completed legacy Vaccine results and the preferred Antidote-codebase baseline
- documentation of the LoRA/configuration difference between the two routes

## Not Included

The package intentionally excludes:

- Hugging Face token files
- model caches
- checkpoints and LoRA adapter weights
- `.git/`
- `wandb/`
- Python cache files
- raw run logs

The package is designed to be small enough for course submission while keeping the code paths and final lightweight result summaries.
