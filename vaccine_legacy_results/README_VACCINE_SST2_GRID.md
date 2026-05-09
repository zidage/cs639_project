# Vaccine SST-2 Grid Reproduction

This README is for a fresh H100 machine. Following it should produce the 27
Vaccine attack checkpoints and SST-2 utility results required by the grid:

- learning rate: `1e-5`, `5e-5`, `1e-4`
- epochs: `5`, `10`, `20`
- harmful ratio: `0.01`, `0.05`, `0.10`

The automation wrapper does not reimplement Vaccine. It calls the original
Vaccine `train.py` for alignment and attack fine-tuning, and the original
`sst2/pred_eval.py` for SST-2 utility evaluation.

## 0. Machine Assumptions

- Linux machine with CUDA-capable NVIDIA GPUs.
- Recommended: 2x H100 80GB.
- Miniconda or Anaconda installed.
- Hugging Face access to `meta-llama/Llama-2-7b-hf`.
- Enough disk for model cache and 28 LoRA checkpoints.
- In this prepared `/root/project/Vaccine` folder, Llama-2-7B-HF has already
  been downloaded locally. Do not spend H100 time downloading it again.

Prepared local model snapshot:

```text
/root/project/Vaccine/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
```

The automation wrapper automatically uses this snapshot when it exists, while
keeping checkpoint names as `Llama-2-7b-hf_*`.

## 1. Get Code

```bash
git clone https://github.com/git-disl/Vaccine.git
cd Vaccine
```

If using this prepared project folder instead of cloning again, just enter:

```bash
cd /root/project/Vaccine
```

## 2. Configure Hugging Face Token

The original code reads the token from `huggingface_token.txt`.

In this prepared folder, first check that the token file already exists. Do not
print the token:

```bash
test -r /root/project/Vaccine/huggingface_token.txt && echo "token file ok"
```

If it is missing, create it:

```bash
printf '%s\n' 'YOUR_HF_TOKEN_HERE' > huggingface_token.txt
chmod 600 huggingface_token.txt
```

Before running training, make sure the token has accepted access for:

```text
meta-llama/Llama-2-7b-hf
PKU-Alignment/BeaverTails
sst2
```

## 3. Create Environment

Recommended fast setup:

```bash
bash script/automation/setup_vaccine_env.sh
conda activate vaccine
```

This creates Python 3.9 and installs the pinned packages needed by the Vaccine
alignment, attack fine-tuning, and SST-2 evaluation path used here.

If you need the original full environment file from the Vaccine authors:

```bash
MODE=full bash script/automation/setup_vaccine_env.sh
conda activate vaccine
```

The full `vaccine.yml` can take much longer to solve/install on some machines.

Manual fast equivalent:

```bash
conda create -y -n vaccine python=3.9 pip
conda activate vaccine
pip install --upgrade pip
pip install \
  "torch==2.1.0" \
  "transformers==4.33.3" \
  "tokenizers==0.13.3" \
  "peft==0.5.0" \
  "accelerate==0.24.1" \
  "datasets==2.15.0" \
  "huggingface-hub==0.19.4" \
  "numpy==1.24.1" \
  "scipy==1.11.3" \
  "pandas==2.1.3" \
  "tqdm==4.64.1" \
  "sentencepiece==0.1.99" \
  "safetensors==0.4.0" \
  "protobuf==4.23.4" \
  "wandb==0.16.1"
```

Quick check:

```bash
python - <<'PY'
import torch, transformers, peft, datasets
print(torch.__version__)
print(torch.cuda.is_available())
print(transformers.__version__, peft.__version__, datasets.__version__)
PY
```

The fast setup intentionally uses `transformers==4.33.3` rather than a newer
Transformers release. This avoids a `tokenizers`/`huggingface-hub` resolver
conflict while keeping the code on the older Transformers API expected by the
paper repository.

Verify that the local Llama files are usable before starting H100 training:

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

path = Path("/root/project/Vaccine/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9")
required = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
missing = [name for name in required if not (path / name).exists() or (path / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"missing or empty model files: {missing}")

tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
cfg = AutoConfig.from_pretrained(path, local_files_only=True)
print("local model ok", len(tok), cfg.model_type, cfg.hidden_size)
PY
```

## 4. Run Vaccine Alignment Once

This produces the shared Vaccine aligned LoRA checkpoint:

```text
ckpt/Llama-2-7b-hf_vaccine_2
```

Run:

```bash
cd /root/project/Vaccine
conda activate vaccine

CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_grid.py \
  --only-alignment \
  --cuda-visible-devices 0
```

The wrapper will also build `data/sst2.json` if missing.

## 5. Run the 27 SST-2 Attack Grid

After alignment finishes, run two processes in parallel, one per GPU.

Terminal 1, GPU 0:

```bash
cd /root/project/Vaccine
conda activate vaccine

CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_grid.py \
  --skip-alignment \
  --cuda-visible-devices 0 \
  --partition-index 0 \
  --num-partitions 2 \
  --summary-path results/vaccine_sst2/results_summary_vaccine_sst2_part0.json
```

Terminal 2, GPU 1:

```bash
cd /root/project/Vaccine
conda activate vaccine

CUDA_VISIBLE_DEVICES=1 python script/automation/run_vaccine_sst2_grid.py \
  --skip-alignment \
  --cuda-visible-devices 1 \
  --partition-index 1 \
  --num-partitions 2 \
  --summary-path results/vaccine_sst2/results_summary_vaccine_sst2_part1.json
```

Each process runs a disjoint part of the 27 configs. Separate summary files are
used to avoid concurrent writes to the same JSON file.

## 6. Outputs

Alignment checkpoint:

```text
ckpt/Llama-2-7b-hf_vaccine_2
```

Attack checkpoints:

```text
ckpt/sst2/Llama-2-7b-hf_vaccine_f_rho2_pr<ratio>_n5000_lr<lr>_ep<epochs>
```

SST-2 prediction/eval JSON files:

```text
data/sst2/Llama-2-7b-hf_vaccine_f_rho2_pr<ratio>_n5000_lr<lr>_ep<epochs>.json
```

Summary files:

```text
results/vaccine_sst2/results_summary_vaccine_sst2_part0.json
results/vaccine_sst2/results_summary_vaccine_sst2_part1.json
```

Each summary record contains:

- method: `vaccine`
- model
- learning rate
- epochs
- poison ratio
- sample number
- alignment checkpoint path
- attack checkpoint path
- SST-2 output path
- SST-2 accuracy
- status

## 7. Resume or Rerun

The wrapper is restartable. If a checkpoint or SST-2 output already exists, it
skips that step.

Resume the same command after interruption:

```bash
CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_grid.py \
  --skip-alignment \
  --cuda-visible-devices 0 \
  --partition-index 0 \
  --num-partitions 2 \
  --summary-path results/vaccine_sst2/results_summary_vaccine_sst2_part0.json
```

Force rerun:

```bash
CUDA_VISIBLE_DEVICES=0 python script/automation/run_vaccine_sst2_grid.py \
  --skip-alignment \
  --force \
  --cuda-visible-devices 0 \
  --partition-index 0 \
  --num-partitions 2 \
  --summary-path results/vaccine_sst2/results_summary_vaccine_sst2_part0.json
```

## 8. Optional: Merge the Two Summary Files

Simple merge:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("results/vaccine_sst2/results_summary_vaccine_sst2_part0.json"),
    Path("results/vaccine_sst2/results_summary_vaccine_sst2_part1.json"),
]
records = []
for path in paths:
    if path.exists():
        records.extend(json.loads(path.read_text()))

dedup = {}
for record in records:
    dedup[record["grid_index"]] = record

merged = [dedup[i] for i in sorted(dedup)]
out = Path("results/vaccine_sst2/results_summary_vaccine_sst2.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(merged, indent=2))
print(f"wrote {out} with {len(merged)} records")
print("completed", sum(r.get("status") == "completed" for r in merged))
PY
```

Expected count:

```bash
python - <<'PY'
import json
from pathlib import Path

data = json.load(open("results/vaccine_sst2/results_summary_vaccine_sst2.json"))
print("records", len(data))
print("completed", sum(r.get("status") == "completed" for r in data))
print("with_sst2_accuracy", sum(r.get("sst2_accuracy") is not None for r in data))
missing_ckpt = [r["grid_index"] for r in data if not Path(r["attack_checkpoint"]).exists()]
missing_sst2 = [r["grid_index"] for r in data if not Path(r["sst2_output"]).exists()]
print("missing_attack_checkpoint", missing_ckpt)
print("missing_sst2_output", missing_sst2)
PY
```

`records`, `completed`, and `with_sst2_accuracy` should all be `27`.
The two missing lists should be empty.

Accuracy table:

```bash
python - <<'PY'
import json

data = json.load(open("results/vaccine_sst2/results_summary_vaccine_sst2.json"))
print("| grid_index | lr | epochs | harmful_ratio | SST2 accuracy | checkpoint |")
print("|---:|---:|---:|---:|---:|---|")
for r in sorted(data, key=lambda x: x["grid_index"]):
    print(
        f"| {r['grid_index']} | {r['learning_rate']} | {r['epochs']} | "
        f"{r['poison_ratio']} | {r.get('sst2_accuracy')} | {r['attack_checkpoint']} |"
    )
PY
```

## 9. Notes

- This run only records SST-2 utility. It does not run harmful-score or ASR
  safety evaluation.
- The attack fine-tuning data is mixed by the original Vaccine dataset loader:
  `PKU-Alignment/BeaverTails_dangerous` plus `data/sst2.json`.
- Default total sample count is `5000`; with ratio `0.01`, `0.05`, or `0.10`,
  harmful samples are `50`, `250`, or `500`.
- Keep all checkpoints until the summaries are merged and checked.
