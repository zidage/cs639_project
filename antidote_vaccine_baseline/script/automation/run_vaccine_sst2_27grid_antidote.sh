#!/usr/bin/env bash
set -Eeuo pipefail

# Antidote-code Vaccine baseline, SST-2 benign utility plus BeaverTails HS.
# This script intentionally calls the original Antidote train/eval entrypoints.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

CONDA_ENV="${CONDA_ENV:-vaccine}"
MODEL_PATH="${MODEL_PATH:-/root/project/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9}"
CACHE_DIR="${CACHE_DIR:-/root/project/cache}"
GPU_LIST="${GPU_LIST:-0,1}"
TRAIN_JOBS_PER_GPU="${TRAIN_JOBS_PER_GPU:-1}"

RHO="${RHO:-2}"
SAMPLE_NUM="${SAMPLE_NUM:-5000}"
HS_TEST_SIZE="${HS_TEST_SIZE:-1000}"
MODEL_TAG="${MODEL_TAG:-Llama-2-7b-hf}"
GRID_INDEX_MOD="${GRID_INDEX_MOD:-}"
GRID_INDEX_REMAINDER="${GRID_INDEX_REMAINDER:-}"

RESULT_DIR="${RESULT_DIR:-results/vaccine_sst2_27grid_antidote}"
LOG_DIR="${RESULT_DIR}/logs"
RECORD_DIR="${RESULT_DIR}/records"
SST2_DIR="${RESULT_DIR}/sst2_outputs"
HS_DIR="${RESULT_DIR}/hs_outputs"

ALIGN_CKPT="${ROOT_DIR}/ckpt/${MODEL_TAG}_vaccine_${RHO}_antidote"

mkdir -p data ckpt/sst2 "${LOG_DIR}" "${RECORD_DIR}" "${SST2_DIR}" "${HS_DIR}"

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}/datasets}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

die() {
  echo "[$(timestamp)] ERROR: $*" >&2
  exit 1
}

has_adapter() {
  local dir="$1"
  [[ -f "${dir}/adapter_config.json" ]] && { [[ -f "${dir}/adapter_model.bin" ]] || [[ -f "${dir}/adapter_model.safetensors" ]]; }
}

ensure_env() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
}

ensure_token() {
  if [[ -r huggingface_token.txt ]]; then
    return
  fi
  die "huggingface_token.txt is missing. Put it at ${ROOT_DIR}/huggingface_token.txt before running."
}

ensure_model() {
  [[ -d "${MODEL_PATH}" ]] || die "MODEL_PATH does not exist: ${MODEL_PATH}"
  [[ -f "${MODEL_PATH}/config.json" ]] || die "Missing config.json in MODEL_PATH: ${MODEL_PATH}"
  [[ -f "${MODEL_PATH}/tokenizer.model" || -f "${MODEL_PATH}/tokenizer.json" ]] || die "Missing tokenizer file in MODEL_PATH: ${MODEL_PATH}"
  find "${MODEL_PATH}" -maxdepth 1 \( -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) | grep -q . || die "Missing model shard files in MODEL_PATH: ${MODEL_PATH}"
}

ensure_sst2_data() {
  if [[ -f data/sst2.json ]]; then
    return
  fi
  python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

out = Path("data/sst2.json")
out.parent.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("glue", "sst2")
records = []
for ex in dataset["train"]:
    records.append({
        "instruction": "Analyze the sentiment of the input, and respond only positive or negative",
        "input": ex["sentence"],
        "output": "positive" if ex["label"] else "negative",
    })
out.write_text(json.dumps(records, indent=4), encoding="utf-8")
print(f"wrote {out} with {len(records)} records")
PY
}

run_alignment() {
  if has_adapter "${ALIGN_CKPT}"; then
    echo "[$(timestamp)] Alignment exists, skip: ${ALIGN_CKPT}"
    return
  fi

  local align_gpu="${GPU_LIST%%,*}"
  echo "[$(timestamp)] Start alignment on GPU ${align_gpu} -> ${ALIGN_CKPT}"
  CUDA_VISIBLE_DEVICES="${align_gpu}" python train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path PKU-Alignment/BeaverTails_safe \
    --bf16 True \
    --output_dir "${ALIGN_CKPT}" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 0 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --cache_dir "${CACHE_DIR}" \
    --optimizer vaccine \
    --sample_num "${SAMPLE_NUM}" \
    --rho "${RHO}" \
    > "${LOG_DIR}/alignment.log" 2>&1

  has_adapter "${ALIGN_CKPT}" || die "Alignment finished but adapter files are missing: ${ALIGN_CKPT}"
  echo "[$(timestamp)] Alignment completed: ${ALIGN_CKPT}"
}

write_record() {
  local idx="$1"
  local lr="$2"
  local epochs="$3"
  local ratio="$4"
  local attack_ckpt="$5"
  local sst2_output="$6"
  local hs_output="$7"
  local hs_eval="$8"
  local status="$9"
  local error_msg="${10:-}"

  GRID_INDEX="${idx}" \
  LEARNING_RATE="${lr}" \
  EPOCHS="${epochs}" \
  POISON_RATIO="${ratio}" \
  ATTACK_CKPT="${attack_ckpt}" \
  SST2_OUTPUT="${sst2_output}" \
  HS_OUTPUT="${hs_output}" \
  HS_EVAL_OUTPUT="${hs_eval}" \
  STATUS="${status}" \
  ERROR_MSG="${error_msg}" \
  ALIGN_CKPT="${ALIGN_CKPT}" \
  RESULT_DIR="${RESULT_DIR}" \
  MODEL_PATH="${MODEL_PATH}" \
  RHO="${RHO}" \
  SAMPLE_NUM="${SAMPLE_NUM}" \
  HS_TEST_SIZE="${HS_TEST_SIZE}" \
  python - <<'PY'
import json
import os
import re
from pathlib import Path

idx = int(os.environ["GRID_INDEX"])

def read_metric(path, pattern):
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = data if isinstance(data, list) else [data]
    for item in reversed(candidates):
        if isinstance(item, str):
            m = re.search(pattern, item)
            if m:
                return float(m.group(1))
    return None

sst2_output = os.environ["SST2_OUTPUT"]
hs_eval = os.environ["HS_EVAL_OUTPUT"]
record = {
    "grid_index": idx,
    "method": "Vaccine",
    "codebase": "Antidote",
    "model": "meta-llama/Llama-2-7b-hf",
    "model_path": os.environ["MODEL_PATH"],
    "lora_config": {
        "r": 256,
        "lora_alpha": 4,
        "lora_dropout": 0,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    },
    "rho": float(os.environ["RHO"]),
    "sample_num": int(float(os.environ["SAMPLE_NUM"])),
    "hs_test_size": int(float(os.environ["HS_TEST_SIZE"])),
    "benign_dataset": "sst2",
    "harmful_dataset": "PKU-Alignment/BeaverTails_dangerous",
    "learning_rate": os.environ["LEARNING_RATE"],
    "epochs": int(float(os.environ["EPOCHS"])),
    "poison_ratio": float(os.environ["POISON_RATIO"]),
    "harmful_ratio": float(os.environ["POISON_RATIO"]),
    "alignment_checkpoint": os.environ["ALIGN_CKPT"],
    "attack_checkpoint": os.environ["ATTACK_CKPT"],
    "sst2_output": sst2_output,
    "hs_output": os.environ["HS_OUTPUT"],
    "hs_eval_output": hs_eval,
    "sst2_accuracy": read_metric(sst2_output, r"score=([0-9.]+)"),
    "harmful_score": read_metric(hs_eval, r"final\s+score:([0-9.]+)"),
    "status": os.environ["STATUS"],
    "error": os.environ["ERROR_MSG"] or None,
}
record_dir = Path(os.environ["RESULT_DIR"]) / "records"
record_dir.mkdir(parents=True, exist_ok=True)
(record_dir / f"grid_{idx:02d}.json").write_text(
    json.dumps(record, indent=2),
    encoding="utf-8",
)
PY
}

run_one_grid() {
  local gpu="$1"
  local idx="$2"
  local lr="$3"
  local epochs="$4"
  local ratio="$5"

  local ratio_tag="${ratio/./p}"
  local ckpt="${ROOT_DIR}/ckpt/sst2/${MODEL_TAG}_vaccine_f_${RHO}_pr${ratio}_n${SAMPLE_NUM}_lr${lr}_ep${epochs}_antidote"
  local sst2_output="${ROOT_DIR}/${SST2_DIR}/grid_${idx}_${MODEL_TAG}_pr${ratio_tag}_lr${lr}_ep${epochs}.json"
  local hs_output="${ROOT_DIR}/${HS_DIR}/grid_${idx}_${MODEL_TAG}_pr${ratio_tag}_lr${lr}_ep${epochs}_beavertails_${HS_TEST_SIZE}.json"
  local hs_eval="${hs_output}_sentiment_eval.json"
  local log="${ROOT_DIR}/${LOG_DIR}/grid_${idx}.log"

  echo "[$(timestamp)] GPU ${gpu} start grid ${idx}: lr=${lr}, epochs=${epochs}, poison_ratio=${ratio}" | tee "${log}"

  if ! has_adapter "${ckpt}"; then
    if ! CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --model_name_or_path "${MODEL_PATH}" \
      --lora_folder "${ALIGN_CKPT}" \
      --data_path PKU-Alignment/BeaverTails_dangerous \
      --bf16 True \
      --output_dir "${ckpt}" \
      --num_train_epochs "${epochs}" \
      --per_device_train_batch_size 5 \
      --per_device_eval_batch_size 5 \
      --gradient_accumulation_steps 1 \
      --save_strategy "steps" \
      --save_steps 100000 \
      --save_total_limit 0 \
      --learning_rate "${lr}" \
      --weight_decay 0.1 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type "constant" \
      --logging_steps 10 \
      --tf32 True \
      --eval_steps 2000 \
      --cache_dir "${CACHE_DIR}" \
      --optimizer normal \
      --evaluation_strategy "steps" \
      --sample_num "${SAMPLE_NUM}" \
      --poison_ratio "${ratio}" \
      --label_smoothing_factor 0 \
      --benign_dataset data/sst2.json \
      >> "${log}" 2>&1; then
      write_record "${idx}" "${lr}" "${epochs}" "${ratio}" "${ckpt}" "${sst2_output}" "${hs_output}" "${hs_eval}" "failed" "attack training failed"
      return 1
    fi
  else
    echo "[$(timestamp)] Grid ${idx} checkpoint exists, skip train: ${ckpt}" >> "${log}"
  fi

  if [[ ! -f "${sst2_output}" ]]; then
    if ! (cd sst2 && CUDA_VISIBLE_DEVICES="${gpu}" python pred_eval.py \
      --lora_folder "${ALIGN_CKPT}" \
      --lora_folder2 "${ckpt}" \
      --model_folder "${MODEL_PATH}" \
      --cache_dir "${CACHE_DIR}" \
      --output_path "${sst2_output}") >> "${log}" 2>&1; then
      write_record "${idx}" "${lr}" "${epochs}" "${ratio}" "${ckpt}" "${sst2_output}" "${hs_output}" "${hs_eval}" "failed" "sst2 eval failed"
      return 1
    fi
  else
    echo "[$(timestamp)] Grid ${idx} SST2 output exists, skip: ${sst2_output}" >> "${log}"
  fi

  if [[ ! -f "${hs_output}" ]]; then
    if ! (cd poison/evaluation && CUDA_VISIBLE_DEVICES="${gpu}" python pred.py \
      --lora_folder "${ALIGN_CKPT}" \
      --lora_folder2 "${ckpt}" \
      --model_folder "${MODEL_PATH}" \
      --instruction_path BeaverTails \
      --cache_dir "${CACHE_DIR}" \
      --num_test_data "${HS_TEST_SIZE}" \
      --output_path "${hs_output}") >> "${log}" 2>&1; then
      write_record "${idx}" "${lr}" "${epochs}" "${ratio}" "${ckpt}" "${sst2_output}" "${hs_output}" "${hs_eval}" "failed" "hs generation failed"
      return 1
    fi
  else
    echo "[$(timestamp)] Grid ${idx} HS output exists, skip generation: ${hs_output}" >> "${log}"
  fi

  if [[ ! -f "${hs_eval}" ]]; then
    if ! (cd poison/evaluation && CUDA_VISIBLE_DEVICES="${gpu}" python eval_sentiment.py \
      --input_path "${hs_output}") >> "${log}" 2>&1; then
      write_record "${idx}" "${lr}" "${epochs}" "${ratio}" "${ckpt}" "${sst2_output}" "${hs_output}" "${hs_eval}" "failed" "hs scoring failed"
      return 1
    fi
  else
    echo "[$(timestamp)] Grid ${idx} HS eval exists, skip scoring: ${hs_eval}" >> "${log}"
  fi

  write_record "${idx}" "${lr}" "${epochs}" "${ratio}" "${ckpt}" "${sst2_output}" "${hs_output}" "${hs_eval}" "completed" ""
  echo "[$(timestamp)] GPU ${gpu} completed grid ${idx}" >> "${log}"
}

merge_summary() {
  RESULT_DIR="${RESULT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
records = []
for path in sorted((root / "records").glob("grid_*.json")):
    records.append(json.loads(path.read_text(encoding="utf-8")))
records.sort(key=lambda x: x["grid_index"])
summary = root / "results_summary_vaccine_sst2_27grid_antidote.json"
summary.write_text(json.dumps(records, indent=2), encoding="utf-8")

total = len(records)
completed = sum(r.get("status") == "completed" for r in records)
with_metrics = sum(r.get("sst2_accuracy") is not None and r.get("harmful_score") is not None for r in records)
print(f"wrote {summary} with {total} records")
print(f"completed {completed}")
print(f"with_metrics {with_metrics}")
PY
}

run_gpu_worker() {
  local gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local idx=0
  local had_failure=0

  for lr in 1e-5 5e-5 1e-4; do
    for epochs in 5 10 20; do
      for ratio in 0.01 0.05 0.10; do
        local take=0
        if [[ -n "${GRID_INDEX_MOD}" ]]; then
          if (( idx % GRID_INDEX_MOD == GRID_INDEX_REMAINDER )); then
            take=1
          fi
        elif (( idx % worker_count == worker_index )); then
          take=1
        fi
        if (( take == 1 )); then
          if ! run_one_grid "${gpu}" "${idx}" "${lr}" "${epochs}" "${ratio}"; then
            had_failure=1
          fi
        fi
        idx=$((idx + 1))
      done
    done
  done

  return "${had_failure}"
}

main() {
  ensure_env
  ensure_token
  ensure_model
  ensure_sst2_data
  run_alignment

  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
  local gpu_count="${#GPUS[@]}"
  [[ "${gpu_count}" -gt 0 ]] || die "GPU_LIST is empty"
  echo "[$(timestamp)] Start 27-grid run on GPUs: ${GPU_LIST}; one fixed worker per GPU; HS_TEST_SIZE=${HS_TEST_SIZE}"

  local had_failure=0

  for worker_index in "${!GPUS[@]}"; do
    run_gpu_worker "${GPUS[worker_index]}" "${worker_index}" "${gpu_count}" &
  done

  while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
    if ! wait -n; then
      had_failure=1
    fi
  done

  merge_summary
  if [[ "${had_failure}" -ne 0 ]]; then
    die "One or more grids failed. Check ${LOG_DIR} and rerun the same script to resume."
  fi
  echo "[$(timestamp)] Done."
}

main "$@"
