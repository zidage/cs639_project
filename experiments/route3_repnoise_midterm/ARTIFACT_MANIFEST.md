# Artifact Manifest

| Path | Type | What it contains | Why it matters | Notes |
|---|---|---|---|---|
| `CS639_Route3V2.ipynb` | Notebook | End-to-end Colab workflow for Route 3 RepNoise attack | Primary procedural evidence | HF token in source cell was redacted in repo copy |
| `artifacts/download_bundle/` | Directory | Exported run outputs (`ckpt/`, `data/`) | Core midterm evidence bundle | Copied from Downloads source bundle |
| `artifacts/colab_stage3_bundle/` | Directory | Stage-3 Antidote code/scripts used in Colab | Repro context for execution | Includes scripts and evaluation code |

## `artifacts/download_bundle/` key files

| Path | Type | What it contains | Why it matters | Notes |
|---|---|---|---|---|
| `artifacts/download_bundle/ckpt/Llama-2-7b-hf_repnoise_0.1_0.001/adapter_config.json` | JSON | Baseline RepNoise adapter config | Identifies baseline adapter setup | Committed |
| `artifacts/download_bundle/ckpt/Llama-2-7b-hf_repnoise_0.1_0.001/adapter_model.bin` | Binary weights | Baseline adapter weights | Needed for exact local reload | **Not committed** (>50MB; gitignored) |
| `artifacts/download_bundle/ckpt/Llama-2-7b-hf_repnoise_0.1_0.001/trainer_state.json` | JSON | Training state metadata | Provenance/debugging | Committed |
| `artifacts/download_bundle/ckpt/sst2/Llama-2-7b-hf_repnoise_0.1_0.001_f_0.2_1000_1e-4_quick/adapter_config.json` | JSON | Attacked adapter config | Ties to Route 3 attack output | Committed |
| `artifacts/download_bundle/ckpt/sst2/Llama-2-7b-hf_repnoise_0.1_0.001_f_0.2_1000_1e-4_quick/adapter_model.bin` | Binary weights | Attacked adapter weights | Needed for exact local reload | **Not committed** (>50MB; gitignored) |
| `artifacts/download_bundle/data/poison/Llama-2-7b-hf_repnoise_0.1_0.001` | JSON/text output file | Baseline poison-eval generations | Baseline behavior evidence | Committed |
| `artifacts/download_bundle/data/poison/Llama-2-7b-hf_repnoise_0.1_0.001_sentiment_eval.json` | JSON | Baseline sentiment eval with terminal score | Source of baseline key number | Last entry `final  score:55.80` |
| `artifacts/download_bundle/data/poison/sst2/Llama-2-7b-hf_repnoise_0.1_0.001_f_0.2_1000_1e-4_quick` | JSON/text output file | Attacked generations | Attacked behavior evidence | Committed |
| `artifacts/download_bundle/data/poison/sst2/Llama-2-7b-hf_repnoise_0.1_0.001_f_0.2_1000_1e-4_quick_sentiment_eval.json` | JSON | Attacked sentiment eval with terminal score | Source of attacked key number | Last entry `final  score:53.80` |
| `artifacts/download_bundle/data/sst2.json` | JSON dataset | SST-2 formatted data snapshot | Documents benign-task artifact context | Committed |
| `artifacts/download_bundle/ckpt/.../runs/*/events.out.tfevents*` | TensorBoard events | Training logs/events | Timing/training trace | Committed (small files) |

## `artifacts/colab_stage3_bundle/` key files

| Path | Type | What it contains | Why it matters | Notes |
|---|---|---|---|---|
| `artifacts/colab_stage3_bundle/Antidote/README.md` | Markdown | Project/method context | Documents codebase intent | Committed |
| `artifacts/colab_stage3_bundle/Antidote/train.py` | Python | Main training entrypoint | Confirms attack training invocation target | Committed |
| `artifacts/colab_stage3_bundle/Antidote/trainer.py` | Python | Trainer implementation | Method details and training behavior | Committed |
| `artifacts/colab_stage3_bundle/Antidote/repnoise_loss.py` | Python | RepNoise-related loss logic | Defense-side implementation context | Committed |
| `artifacts/colab_stage3_bundle/Antidote/poison/evaluation/pred.py` | Python | Generation script for safety prompts | Produces evaluated outputs | Committed |
| `artifacts/colab_stage3_bundle/Antidote/poison/evaluation/eval_sentiment.py` | Python | Sentiment-based safety evaluator | Produces key score files | Committed |
| `artifacts/colab_stage3_bundle/Antidote/script/finetune/*.sh` | Shell scripts | Automation scripts for finetuning/eval | Supports reproducibility context | Committed |
| `artifacts/colab_stage3_bundle/Antidote/huggingface_token.txt` | Secret token file | Token placeholder/value for HF access | Sensitive; not needed in git history | **Not committed** (gitignored) |

## Large/sensitive files intentionally excluded from commit

| Local Path | Reason excluded | Recommended sharing path |
|---|---|---|
| `artifacts/download_bundle/ckpt/Llama-2-7b-hf_repnoise_0.1_0.001/adapter_model.bin` | 805MB (>50MB normal git limit) | Git LFS or external artifact storage |
| `artifacts/download_bundle/ckpt/sst2/Llama-2-7b-hf_repnoise_0.1_0.001_f_0.2_1000_1e-4_quick/adapter_model.bin` | 805MB (>50MB normal git limit) | Git LFS or external artifact storage |
| `artifacts/**/huggingface_token.txt` | Sensitive credential material | Keep local/secret manager only |
