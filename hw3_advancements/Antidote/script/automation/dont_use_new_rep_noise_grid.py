#!/usr/bin/env python3
"""Run Rep_Noise attack hyperparameter sweeps with robust logging and JSON summaries.

Pipeline per run:
1) Fine-tune from the aligned phase1 LoRA checkpoint.
2) Safety evaluation on BeaverTails harmful prompts.
3) Safety evaluation on AdvBench prompts.
4) Utility evaluation on selected downstream tasks.
5) Optional cleanup: delete the run LoRA checkpoint to save disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TASKS = ("sst2", "gsm8k", "agnews")
SAFETY_EVALS = ("beavertails", "advbench")

TASK_TO_BENIGN_DATA = {
    "sst2": "data/sst2.json",
    "gsm8k": "data/gsm8k.json",
    "agnews": "data/agnews.json",
}

KEY_LOG_TOKENS = (
    "Loading data...",
    "Formatting inputs...",
    "Tokenizing inputs...",
    "Recover LoRA weights..",
    "Initialize Lora weights..",
    "Estimated total time",
    "final score:",
    "score=",
    "train_runtime",
    "train_samples_per_second",
    "Saving model checkpoint",
)
TRAIN_PROGRESS_PATTERN = re.compile(
    r"(?:\bepoch\b|(?:^|[\\s'\"{])loss(?:[\\s'\":,}]|$)|learning_rate|eval_)",
    re.IGNORECASE,
)

SCORE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
EXPLICIT_SCORE_PATTERN = re.compile(
    r"(?:final\s*score|score)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class StepResult:
    name: str
    status: str
    return_code: int
    duration_sec: float
    log_file: str


@dataclass
class RunResult:
    run_id: str
    index: int
    status: str
    duration_sec: float
    variable_hyperparameters: Dict[str, float]
    resolved_parameters: Dict[str, object]
    datasets: Dict[str, object]
    output_paths: Dict[str, object]
    cleanup: Dict[str, object]
    steps: List[StepResult]
    metrics: Dict[str, object]
    errors: List[str]


def parse_csv_numbers(raw: str, cast_type):
    values: List = []
    for token in raw.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        values.append(cast_type(cleaned))
    if not values:
        raise ValueError(f"Failed to parse numeric list from: {raw}")
    return values


def parse_harmful_ratios(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        raw_token = token.strip()
        if not raw_token:
            continue
        has_percent = "%" in raw_token
        cleaned = raw_token.replace("%", "")
        num = float(cleaned)

        # Parsing rule:
        # - "1,5,10" are treated as percentages.
        # - tokens with '%' are treated as percentages.
        # - decimal/scientific forms like "0.01" or "1e-2" are treated as ratios.
        if has_percent:
            num = num / 100.0
        elif any(ch in cleaned.lower() for ch in (".", "e")):
            num = num
        else:
            num = num / 100.0

        if num < 0.0 or num > 1.0:
            raise ValueError(f"Invalid harmful ratio token: {token}")
        values.append(num)
    if not values:
        raise ValueError(f"Failed to parse harmful ratios from: {raw}")
    return values


def parse_csv_tasks(raw: str) -> List[str]:
    tasks: List[str] = []
    for token in raw.split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned not in TASKS:
            raise ValueError(f"Unsupported task: {cleaned}. Choose from: {TASKS}")
        tasks.append(cleaned)
    if not tasks:
        raise ValueError(f"Failed to parse task list from: {raw}")
    return tasks


def parse_csv_safety_evals(raw: str) -> List[str]:
    evals: List[str] = []
    for token in raw.split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned not in SAFETY_EVALS:
            raise ValueError(f"Unsupported safety eval dataset: {cleaned}. Choose from: {SAFETY_EVALS}")
        evals.append(cleaned)
    if not evals:
        raise ValueError(f"Failed to parse safety eval list from: {raw}")
    return evals


def ratio_to_pct_string(ratio: float) -> str:
    return f"{ratio * 100:.2f}%"


def ratio_to_tag_value(ratio: float) -> str:
    return f"{ratio:.6f}".rstrip("0").rstrip(".")


def should_echo_line(line: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "none":
        return False
    for token in KEY_LOG_TOKENS:
        if token in line:
            return True
    stripped = line.strip()
    if TRAIN_PROGRESS_PATTERN.search(stripped):
        return True
    if stripped and stripped.endswith("it/s"):
        return True
    return False


def run_streamed_command(
    command: Sequence[str],
    cwd: Path,
    log_file: Path,
    step_name: str,
    echo_mode: str,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with log_file.open("w", encoding="utf-8") as fh:
        fh.write(f"# Step: {step_name}\n")
        fh.write(f"# Start: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# CWD: {cwd}\n")
        fh.write("# Command:\n")
        fh.write(" ".join(command) + "\n\n")
        if env_overrides:
            fh.write("# Env overrides:\n")
            for key, value in env_overrides.items():
                fh.write(f"{key}={value}\n")
            fh.write("\n")
        fh.flush()

        process_env = os.environ.copy()
        if env_overrides:
            process_env.update(env_overrides)

        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            fh.write(raw_line)
            if should_echo_line(raw_line.rstrip("\n"), echo_mode):
                print(f"[{step_name}] {raw_line.rstrip()}", flush=True)

        return_code = process.wait()

        fh.write("\n")
        fh.write(f"# End: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# Return code: {return_code}\n")

    return return_code, time.time() - start


def load_json_if_exists(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def extract_score_from_json_blob(blob: object) -> Optional[float]:
    if isinstance(blob, dict):
        for key in ("score", "final_score", "harmful_score"):
            if key in blob:
                try:
                    return float(blob[key])
                except Exception:
                    continue

    if not isinstance(blob, list):
        return None

    for item in reversed(blob):
        if isinstance(item, (int, float)):
            return float(item)
        if not isinstance(item, str):
            continue
        explicit = EXPLICIT_SCORE_PATTERN.search(item)
        if explicit:
            return float(explicit.group(1))
        fallback = SCORE_PATTERN.search(item)
        if fallback:
            return float(fallback.group(1))
    return None


def parse_safety_score_percent(safety_eval_json: Path) -> Optional[float]:
    blob = load_json_if_exists(safety_eval_json)
    if blob is None:
        return None
    score = extract_score_from_json_blob(blob)
    if score is None:
        return None
    return round(score, 4)


def parse_utility_score_percent(utility_json: Path) -> Optional[float]:
    blob = load_json_if_exists(utility_json)
    if blob is None:
        return None
    score = extract_score_from_json_blob(blob)
    if score is None:
        return None
    if score <= 1.0:
        score = score * 100.0
    return round(score, 4)


def write_summary_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def ensure_data_files(
    root: Path,
    python_bin: str,
    required_tasks: Sequence[str],
    build_if_missing: bool,
    log_dir: Path,
    echo_mode: str,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "required_tasks": list(required_tasks),
        "built": [],
        "already_present": [],
        "missing": [],
    }

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for task in required_tasks:
        data_file = root / TASK_TO_BENIGN_DATA[task]
        if data_file.exists():
            summary["already_present"].append(str(data_file))
            continue
        summary["missing"].append(str(data_file))

    if not summary["missing"]:
        return summary

    if not build_if_missing:
        return summary

    for task in required_tasks:
        data_file = root / TASK_TO_BENIGN_DATA[task]
        if data_file.exists():
            continue

        build_script = root / task / "build_dataset.py"
        if not build_script.exists():
            raise FileNotFoundError(f"Missing dataset build script: {build_script}")

        step_name = f"build-data:{task}"
        print(f"[setup] Building missing benign dataset for task={task}", flush=True)
        rc, _ = run_streamed_command(
            command=[python_bin, "build_dataset.py"],
            cwd=build_script.parent,
            log_file=log_dir / f"{step_name}.log",
            step_name=step_name,
            echo_mode=echo_mode,
        )
        if rc != 0:
            raise RuntimeError(f"Data build failed for {task}, return code={rc}")

        summary["built"].append(str(data_file))

    return summary


def read_hf_token_from_file(root: Path, token_file_rel_path: str) -> Tuple[Optional[str], Dict[str, object]]:
    token_path = (root / token_file_rel_path).resolve()
    info: Dict[str, object] = {
        "token_file": str(token_path),
        "token_source": "file",
        "token_loaded": False,
        "status": "missing",
    }

    if not token_path.exists():
        return None, info

    try:
        with token_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                token = line.strip()
                if token:
                    info["token_loaded"] = True
                    info["status"] = "loaded"
                    return token, info
    except Exception as exc:
        info["status"] = "read_error"
        info["error"] = str(exc)
        return None, info

    info["status"] = "empty"
    return None, info


def pick_advbench_prompt_field(sample: Dict[str, object], preferred_field: str) -> str:
    if preferred_field and preferred_field in sample:
        value = sample.get(preferred_field)
        if isinstance(value, str) and value.strip():
            return preferred_field

    candidates = ("instruction", "prompt", "goal", "question", "query", "input")
    for field in candidates:
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return field
    raise ValueError(
        "Could not infer prompt field for AdvBench sample. "
        "Please set --advbench-prompt-field explicitly."
    )


def ensure_advbench_instruction_file(
    root: Path,
    target_rel_path: str,
    num_test_data: int,
    hf_dataset: str,
    hf_split: str,
    prompt_field: str,
    hf_token: Optional[str],
    dry_run: bool,
) -> Dict[str, object]:
    target_path = (root / target_rel_path).resolve()
    if target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as fh:
                rows = json.load(fh)
            count = len(rows) if isinstance(rows, list) else None
        except Exception:
            count = None
        return {
            "status": "ready",
            "source": "existing_file",
            "path": str(target_path),
            "num_prompts": count,
        }

    if dry_run:
        return {
            "status": "missing_dry_run",
            "source": "not_created_in_dry_run",
            "path": str(target_path),
            "num_prompts": None,
        }

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "AdvBench instruction file is missing and `datasets` is not available. "
            "Install `datasets`, or provide --advbench-instruction-path with a JSON file."
        ) from exc

    print(f"[setup] Building AdvBench instruction file from {hf_dataset}:{hf_split}", flush=True)

    dataset = None
    auth_mode = "anonymous"
    auth_errors: List[str] = []
    if hf_token:
        try:
            dataset = load_dataset(hf_dataset, split=hf_split, token=hf_token)
            auth_mode = "token"
        except TypeError:
            # Backward compatibility for older datasets versions.
            try:
                dataset = load_dataset(hf_dataset, split=hf_split, use_auth_token=hf_token)
                auth_mode = "use_auth_token"
            except Exception as exc:
                auth_errors.append(f"use_auth_token load failed: {exc}")
        except Exception as exc:
            auth_errors.append(f"token load failed: {exc}")

    if dataset is None:
        try:
            dataset = load_dataset(hf_dataset, split=hf_split)
            auth_mode = "anonymous"
        except Exception as exc:
            auth_msg = "; ".join(auth_errors) if auth_errors else "none"
            raise RuntimeError(
                f"Failed to load AdvBench dataset `{hf_dataset}` split `{hf_split}`. "
                f"Anonymous load error: {exc}. Token attempts: {auth_msg}"
            ) from exc

    if len(dataset) == 0:
        raise RuntimeError(f"AdvBench dataset has no rows: {hf_dataset}:{hf_split}")

    sample = dataset[0]
    if not isinstance(sample, dict):
        raise RuntimeError("Unexpected AdvBench row type: expected dict")

    use_field = pick_advbench_prompt_field(sample, prompt_field)
    limit = len(dataset) if num_test_data <= 0 else min(num_test_data, len(dataset))

    rows: List[Dict[str, str]] = []
    for idx in range(limit):
        row = dataset[idx]
        text = str(row.get(use_field, "")).strip() if isinstance(row, dict) else ""
        if not text:
            continue
        rows.append({"instruction": text})

    if not rows:
        raise RuntimeError(
            f"Failed to extract prompt text from AdvBench using field `{use_field}`."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    return {
        "status": "ready",
        "source": "generated_from_hf",
        "path": str(target_path),
        "hf_dataset": hf_dataset,
        "hf_split": hf_split,
        "prompt_field": use_field,
        "auth_mode": auth_mode,
        "num_prompts": len(rows),
    }


def merge_env(base: Optional[Dict[str, str]], updates: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    merged: Dict[str, str] = {}
    if base:
        merged.update(base)
    if updates:
        merged.update(updates)
    return merged if merged else None


def compute_dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except FileNotFoundError:
                continue
    return total


def cleanup_checkpoint_dir(
    checkpoint_dir: Path,
    allowed_parent: Path,
    dry_run: bool,
    should_delete: bool,
) -> Dict[str, object]:
    checkpoint_dir = checkpoint_dir.resolve()
    allowed_parent = allowed_parent.resolve()
    exists = checkpoint_dir.exists()
    bytes_before = compute_dir_size_bytes(checkpoint_dir) if exists else 0

    result: Dict[str, object] = {
        "delete_run_checkpoint": should_delete,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_exists_before_cleanup": exists,
        "checkpoint_size_bytes_before_delete": bytes_before,
        "checkpoint_size_gb_before_delete": round(bytes_before / (1024 ** 3), 6),
        "checkpoint_deleted": False,
        "checkpoint_cleanup_status": "skipped_not_requested" if not should_delete else "pending",
        "checkpoint_cleanup_error": None,
    }

    if not should_delete:
        return result

    if not exists:
        result["checkpoint_cleanup_status"] = "skipped_missing"
        return result

    try:
        checkpoint_dir.relative_to(allowed_parent)
    except ValueError:
        result["checkpoint_cleanup_status"] = "skipped_outside_allowed_parent"
        result["checkpoint_cleanup_error"] = (
            f"Refused to delete path outside allowed parent: {checkpoint_dir} (allowed: {allowed_parent})"
        )
        return result

    if dry_run:
        result["checkpoint_cleanup_status"] = "dry_run_not_deleted"
        return result

    try:
        shutil.rmtree(checkpoint_dir)
        result["checkpoint_deleted"] = True
        result["checkpoint_cleanup_status"] = "deleted"
    except Exception as exc:
        result["checkpoint_cleanup_status"] = "delete_failed"
        result["checkpoint_cleanup_error"] = str(exc)
    return result


def build_train_command(
    python_bin: str,
    model_path: str,
    base_lora: str,
    attack_dataset: str,
    benign_dataset: str,
    output_dir: str,
    lr: float,
    epochs: int,
    harmful_ratio: float,
    sample_num: int,
    batch_size: int,
    eval_batch_size: int,
    grad_acc_steps: int,
    save_steps: int,
    eval_steps: int,
    weight_decay: float,
    warmup_ratio: float,
    scheduler: str,
    logging_steps: int,
    cache_dir: str,
    max_memory_per_gpu: str,
    cpu_offload_gib: int,
    use_gradient_checkpointing: bool,
    optimizer: str,
    alpha : float,
    beta : float,
    bf16 : bool,
    tf32 : bool,
) -> List[str]:
    command = [
        python_bin,
        "train.py",
        "--model_name_or_path",
        model_path,
        "--lora_folder",
        base_lora,
        "--data_path",
        attack_dataset,
        "--output_dir",
        output_dir,
        "--num_train_epochs",
        str(epochs),
        "--per_device_train_batch_size",
        str(batch_size),
        "--per_device_eval_batch_size",
        str(eval_batch_size),
        "--gradient_accumulation_steps",
        str(grad_acc_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(save_steps),
        "--save_total_limit",
        "1",
        "--learning_rate",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--warmup_ratio",
        str(warmup_ratio),
        "--lr_scheduler_type",
        scheduler,
        "--logging_steps",
        str(logging_steps),
        "--eval_steps",
        str(eval_steps),
        "--cache_dir",
        cache_dir,
        "--optimizer",
        "normal",
        "--evaluation_strategy",
        "no",
        "--sample_num",
        str(sample_num),
        "--poison_ratio",
        str(harmful_ratio),
        "--label_smoothing_factor",
        "0",
        "--benign_dataset",
        benign_dataset,
        "--rho",
        str(alpha),
        "--lamb",
        str(beta),
    ]

    if max_memory_per_gpu:
        command.extend(["--max_memory_per_gpu", max_memory_per_gpu])
    if cpu_offload_gib > 0:
        command.extend(["--cpu_offload_gib", str(cpu_offload_gib)])
    if use_gradient_checkpointing:
        command.extend(["--use_gradient_checkpointing", "True"])
    if not bf16:
        command.extend(["--bf16", "False"])
    if not tf32:
        command.extend(["--tf32", "False"])
    

    return command


def build_safety_pred_command(
    python_bin: str,
    model_path: str,
    base_lora: str,
    lora_after: str,
    output_path: str,
    num_test_data: int,
    instruction_path: Optional[str] = None,
) -> List[str]:
    command = [
        python_bin,
        "pred.py",
        "--lora_folder",
        base_lora,
        "--lora_folder2",
        lora_after,
        "--model_folder",
        model_path,
        "--output_path",
        output_path,
        "--num_test_data",
        str(num_test_data),
    ]
    if instruction_path:
        command.extend(["--instruction_path", instruction_path])
    return command


def build_safety_eval_command(python_bin: str, input_path: str) -> List[str]:
    return [
        python_bin,
        "eval_sentiment.py",
        "--input_path",
        input_path,
    ]


def build_utility_eval_command(
    python_bin: str,
    task: str,
    model_path: str,
    base_lora: str,
    lora_after: str,
    output_path: str,
    num_test_data: int,
) -> List[str]:
    command = [
        python_bin,
        "pred_eval.py",
        "--lora_folder",
        base_lora,
        "--lora_folder2",
        lora_after,
        "--model_folder",
        model_path,
        "--output_path",
        output_path,
    ]
    if task == "gsm8k":
        command.extend(["--num_test_data", str(num_test_data)])
    return command


def make_run_tag(
    model_short_name: str,
    harmful_ratio: float,
    sample_num: int,
    lr: float,
    epochs: int,
    benign_task: str,
) -> str:
    ratio_tag = ratio_to_tag_value(harmful_ratio)
    return (
        f"{model_short_name}_rep_noise_f_"
        f"{ratio_tag}_{sample_num}_{lr}_{epochs}_{benign_task}"
    )


def summarize_runs(runs: Sequence[RunResult]) -> Dict[str, object]:
    total = len(runs)
    success_runs = [r for r in runs if r.status == "success"]
    failed_runs = [r for r in runs if r.status != "success"]

    best_beavertails = None
    comparable_beavertails = []
    for run in success_runs:
        score = run.metrics.get("harmful_scores_percent_by_dataset", {}).get("beavertails")
        if score is None:
            continue
        comparable_beavertails.append((run, float(score)))
    if comparable_beavertails:
        best_run, score = min(comparable_beavertails, key=lambda item: item[1])
        best_beavertails = {
            "run_id": best_run.run_id,
            "harmful_score_percent_beavertails": score,
            "variable_hyperparameters": best_run.variable_hyperparameters,
        }

    return {
        "total_runs": total,
        "successful_runs": len(success_runs),
        "failed_runs": len(failed_runs),
        "best_beavertails_run": best_beavertails,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rep_noise attack grid experiment orchestrator")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to Antidote project root",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--base-lora-folder",
        type=str,
        default="",
        help="Default: ckpt/<model_basename>_rep_noise",
    )

    parser.add_argument("--benign-task", type=str, default="sst2", choices=TASKS)
    parser.add_argument(
        "--attack-dataset",
        type=str,
        default="PKU-Alignment/BeaverTails_dangerous",
        help="Training dataset used for harmful fine-tuning stage",
    )
    parser.add_argument(
        "--utility-evals",
        type=str,
        default="sst2,gsm8k,agnews",
        help="Comma-separated tasks in {sst2,gsm8k,agnews}",
    )
    parser.add_argument(
        "--safety-evals",
        type=str,
        default="beavertails,advbench",
        help="Comma-separated safety datasets in {beavertails,advbench}",
    )

    parser.add_argument(
        "--advbench-instruction-path",
        type=str,
        default="data/advbench_eval_instructions.json",
        help="Instruction JSON for AdvBench safety eval (list of {instruction: ...})",
    )
    parser.add_argument(
        "--advbench-hf-dataset",
        type=str,
        default="walledai/AdvBench",
        help="HF dataset id used when --advbench-instruction-path is missing",
    )
    parser.add_argument(
        "--advbench-hf-split",
        type=str,
        default="train",
        help="HF split used when --advbench-instruction-path is missing",
    )
    parser.add_argument(
        "--advbench-prompt-field",
        type=str,
        default="",
        help="Optional field name for AdvBench prompt text, auto-detected when empty",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional HuggingFace token override for loading private/gated datasets",
    )
    parser.add_argument(
        "--hf-token-file",
        type=str,
        default="huggingface_token.txt",
        help="Path (relative to project root) to HuggingFace token file",
    )

    parser.add_argument("--lrs", type=str, default="1e-5,5e-5,1e-4")
    parser.add_argument("--epochs", type=str, default="5,10,20")
    parser.add_argument(
        "--harmful-ratios",
        type=str,
        default="1,5,10",
        help="Comma-separated values in percent (1,5,10) or ratio (0.01,0.05,0.10)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="1-based grid iteration index to start from",
    )

    parser.add_argument("--sample-num", type=int, default=5000)
    parser.add_argument("--train-batch-size", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="constant")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--num-test-data", type=int, default=1000)
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--max-memory-per-gpu",
        type=str,
        default="",
        help="Optional per-GPU memory cap passed to train.py, e.g. 38GiB",
    )
    parser.add_argument(
        "--cpu-offload-gib",
        type=int,
        default=0,
        help="Optional CPU offload memory cap passed to train.py",
    )
    parser.add_argument(
        "--use-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing in train.py",
    )

    parser.add_argument(
        "--build-data-if-missing",
        dest="build_data_if_missing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-build-data-if-missing",
        dest="build_data_if_missing",
        action="store_false",
    )

    parser.add_argument(
        "--delete-run-checkpoint",
        dest="delete_run_checkpoint",
        action="store_true",
        default=False,
        help="Delete each run LoRA checkpoint after evaluations to save disk",
    )
    parser.add_argument(
        "--keep-run-checkpoint",
        dest="delete_run_checkpoint",
        action="store_false",
        help="Keep each run LoRA checkpoint",
    )

    parser.add_argument("--train-only", action="store_true", default=False) 

    parser.add_argument(
        "--force-no-weights-only-load",
        dest="force_no_weights_only_load",
        action="store_true",
        default=True,
        help=(
            "Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for subprocesses to avoid "
            "legacy checkpoint deserialization issues on newer torch versions"
        ),
    )
    parser.add_argument(
        "--no-force-no-weights-only-load",
        dest="force_no_weights_only_load",
        action="store_false",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue remaining runs when one run fails",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument(
        "--echo-mode",
        type=str,
        default="key",
        choices=("key", "all", "none"),
        help="Console streaming mode for subprocess logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rep_noise_grid",
        help="Used to name output directory under experiments/",
    )

    parser.add_argument(
        "--optimizer",
        type = str,
        default = "",
        help="Isn't used because rep_noise only matters at alignment stage",
    )

    parser.add_argument(
        "--alpha", # Used as rho in the repo 
        type = float,
        default = 0.1,
    )
    parser.add_argument(
        "--beta",
        type = float,
        default =  0.001,
    )

    parser.add_argument(
        "--bf16",
        type = bool,
        default = True,
    )

    parser.add_argument(
        "--tf32",
        type = bool,
        default = True,
    )



    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.project_root).resolve()
    if not (root / "train.py").exists():
        print(f"[error] Could not find train.py under project root: {root}")
        return 2

    utility_tasks = parse_csv_tasks(args.utility_evals)
    safety_evals = parse_csv_safety_evals(args.safety_evals)
    lrs = parse_csv_numbers(args.lrs, float)
    epochs = parse_csv_numbers(args.epochs, int)
    harmful_ratios = parse_harmful_ratios(args.harmful_ratios)

    model_short = Path(args.model_path).name
    base_lora = args.base_lora_folder if args.base_lora_folder else f"ckpt/{model_short}_rep_noise"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / "experiments" / args.experiment_name / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    setup_log_dir = run_root / "setup_logs"
    summary_path = run_root / "results_summary.json"

    required_data_tasks = sorted(set([args.benign_task] + utility_tasks))

    base_env: Dict[str, str] = {}
    if args.force_no_weights_only_load:
        base_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    train_env = merge_env(base_env, {"CUDA_VISIBLE_DEVICES": args.gpu_id})
    eval_env = merge_env(base_env, {"CUDA_VISIBLE_DEVICES": args.gpu_id})

    print("=" * 88)
    print("rep_noise attack grid orchestrator")
    print(f"Project root                 : {root}")
    print(f"Experiment folder            : {run_root}")
    print(f"Model path                   : {args.model_path}")
    print(f"Base rep_noise LoRA            : {base_lora}")
    print(f"Attack dataset               : {args.attack_dataset}")
    print(f"Benign task (train mixture)  : {args.benign_task}")
    print(f"Safety eval datasets         : {safety_evals}")
    print(f"Utility eval tasks           : {utility_tasks}")
    print(f"LR list                      : {lrs}")
    print(f"Epoch list                   : {epochs}")
    print(f"Harmful ratios               : {[ratio_to_pct_string(r) for r in harmful_ratios]}")
    print(f"Optimizer                    : normal")
    print(f"Alpha                        : {args.alpha}")
    print(f"Beta                         : {args.beta}")
    print(f"Train Only                     : {args.train_only}")

    print(f"")


    
    print(f"GPU ID                       : {args.gpu_id}")
    if args.max_memory_per_gpu:
        print(f"Max mem per GPU          : {args.max_memory_per_gpu}")
    if args.cpu_offload_gib > 0:
        print(f"CPU offload GiB          : {args.cpu_offload_gib}")
    print(f"Gradient checkpoint          : {args.use_gradient_checkpointing}")
    print(f"Delete run checkpoint        : {args.delete_run_checkpoint}")
    print(f"Force no weights-only load   : {args.force_no_weights_only_load}")
    print(
        "Train defaults              : "
        f"batch={args.train_batch_size}, eval_batch={args.eval_batch_size}, "
        f"grad_acc={args.grad_acc_steps}, sample_num={args.sample_num}, "
        f"eval_steps={args.eval_steps}"
    )

    if args.hf_token.strip():
        hf_token: Optional[str] = args.hf_token.strip()
        hf_token_info: Dict[str, object] = {
            "token_source": "arg",
            "token_loaded": True,
            "status": "loaded",
        }
    else:
        hf_token, hf_token_info = read_hf_token_from_file(root, args.hf_token_file)
    print(
        "HF token                   : "
        f"source={hf_token_info.get('token_source')}, "
        f"loaded={hf_token_info.get('token_loaded')}, "
        f"status={hf_token_info.get('status')}"
    )
    print("=" * 88)

    if args.dry_run:
        data_summary = {
            "required_tasks": required_data_tasks,
            "built": [],
            "already_present": [],
            "missing": [],
            "note": "dry-run mode: data preparation skipped",
        }
    else:
        data_summary = ensure_data_files(
            root=root,
            python_bin=args.python_bin,
            required_tasks=required_data_tasks,
            build_if_missing=args.build_data_if_missing,
            log_dir=setup_log_dir,
            echo_mode=args.echo_mode,
        )

        if data_summary["missing"] and not args.build_data_if_missing:
            print("[error] Missing data files and auto-build disabled:")
            for item in data_summary["missing"]:
                print(f"  - {item}")
            print("Tip: rerun with --build-data-if-missing")
            return 3

    advbench_instruction_info: Dict[str, object] = {
        "status": "not_requested",
        "source": None,
        "path": None,
        "num_prompts": None,
    }
    advbench_instruction_path: Optional[Path] = None
    if "advbench" in safety_evals:
        try:
            advbench_instruction_info = ensure_advbench_instruction_file(
                root=root,
                target_rel_path=args.advbench_instruction_path,
                num_test_data=args.num_test_data,
                hf_dataset=args.advbench_hf_dataset,
                hf_split=args.advbench_hf_split,
                prompt_field=args.advbench_prompt_field,
                hf_token=hf_token,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"[error] Failed to prepare AdvBench instruction file: {exc}")
            return 4
        if advbench_instruction_info.get("path"):
            advbench_instruction_path = Path(str(advbench_instruction_info["path"]))

    grid: List[Tuple[float, int, float]] = []
    for lr in lrs:
        for ep in epochs:
            for ratio in harmful_ratios:
                grid.append((lr, ep, ratio))

    total_runs = len(grid)
    if args.start_iteration < 1:
        print(f"[error] --start-iteration must be >= 1, got {args.start_iteration}")
        return 5
    if args.start_iteration > total_runs:
        print(
            f"[error] --start-iteration={args.start_iteration} exceeds total grid size={total_runs}"
        )
        return 5

    execution_grid: List[Tuple[int, Tuple[float, int, float]]] = list(
        enumerate(grid, start=1)
    )[args.start_iteration - 1 :]
    planned_runs_from_start = len(execution_grid)
    run_results: List[RunResult] = []

    summary_header: Dict[str, object] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(root),
            "experiment_root": str(run_root),
            "model_path": args.model_path,
            "python_bin": args.python_bin,
            "json_schema_version": "rep_noise_grid_v2",
        },
        "attack_config": {
            "attack_dataset": args.attack_dataset,
            "base_lora_folder": base_lora,
            "benign_task": args.benign_task,
            "benign_dataset_path": TASK_TO_BENIGN_DATA[args.benign_task],
        },
        "evaluation_config": {
            "safety_eval_datasets": safety_evals,
            "advbench_instruction": advbench_instruction_info,
            "hf_token_config": hf_token_info,
            "utility_eval_tasks": utility_tasks,
        },
        "defaults": {
            "sample_num": args.sample_num,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_acc_steps": args.grad_acc_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": args.scheduler,
            "logging_steps": args.logging_steps,
            "cache_dir": args.cache_dir,
            "num_test_data": args.num_test_data,
            "gpu_id": args.gpu_id,
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "cpu_offload_gib": args.cpu_offload_gib,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
            "build_data_if_missing": args.build_data_if_missing,
            "delete_run_checkpoint": args.delete_run_checkpoint,
            "force_no_weights_only_load": args.force_no_weights_only_load,
            "hf_token_file": args.hf_token_file,
            "continue_on_error": args.continue_on_error,
            "echo_mode": args.echo_mode,
            "dry_run": args.dry_run,
            "start_iteration": args.start_iteration,
            "planned_runs_from_start": planned_runs_from_start,
            "optimizer" : "normal",
            "alpha" : args.alpha,
            "beta" : args.beta,
            "bf16" : args.bf16,
            "tf32" : args.tf32,
            "train_only" :args.train_only,
        },
        "data_prep": data_summary,
        "runs": [],
        "aggregate": {},
    }

    if args.dry_run:
        print("[dry-run] Commands will not be executed.")

    print(
        f"[grid] start_iteration={args.start_iteration}, "
        f"planned_runs_from_start={planned_runs_from_start}, total_grid_size={total_runs}"
    )

    for idx, (lr, ep, ratio) in execution_grid:
        ratio_tag = ratio_to_tag_value(ratio)
        run_id = f"run_{idx:03d}_lr{lr}_ep{ep}_ratio{ratio_tag}"
        run_dir = run_root / run_id
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        train_tag = make_run_tag(
            model_short_name=model_short,
            harmful_ratio=ratio,
            sample_num=args.sample_num,
            lr=lr,
            epochs=ep,
            benign_task=args.benign_task,
        )

        train_output_dir = root / "ckpt" / args.benign_task / train_tag

        safety_pred_outputs: Dict[str, Path] = {}
        safety_eval_jsons: Dict[str, Path] = {}
        for dataset_name in safety_evals:
            pred_output = root / "data" / "poison" / args.benign_task / dataset_name / train_tag
            safety_pred_outputs[dataset_name] = pred_output
            safety_eval_jsons[dataset_name] = Path(str(pred_output) + "_sentiment_eval.json")

        utility_output_paths: Dict[str, Path] = {
            task: root / "data" / task / train_tag for task in utility_tasks
        }

        print("-" * 88)
        print(
            f"[{idx}/{total_runs}] {run_id}: "
            f"lr={lr}, epochs={ep}, harmful_ratio={ratio_to_pct_string(ratio)}"
        )
        print(f"Output tag: {train_tag}")

        run_start = time.time()
        steps: List[StepResult] = []
        errors: List[str] = []
        run_failed = False

        step_plan: List[Tuple[str, Sequence[str], Path, Path, Optional[Dict[str, str]]]] = []

        train_cmd = build_train_command(
            python_bin=args.python_bin,
            model_path=args.model_path,
            base_lora=base_lora,
            attack_dataset=args.attack_dataset,
            benign_dataset=TASK_TO_BENIGN_DATA[args.benign_task],
            output_dir=str(train_output_dir),
            lr=lr,
            epochs=ep,
            harmful_ratio=ratio,
            sample_num=args.sample_num,
            batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            grad_acc_steps=args.grad_acc_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            scheduler=args.scheduler,
            logging_steps=args.logging_steps,
            cache_dir=args.cache_dir,
            max_memory_per_gpu=args.max_memory_per_gpu,
            cpu_offload_gib=args.cpu_offload_gib,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            optimizer=args.optimizer,
            alpha = args.alpha, 
            beta = args.beta, 
            bf16 = args.bf16,
            tf32 = args.tf32
        )
        step_plan.append(("train", train_cmd, root, log_dir / "train.log", train_env))

        if not args.train_only :

            for dataset_name in safety_evals:
                instruction_path: Optional[str]
                if dataset_name == "beavertails":
                    instruction_path = "BeaverTails"
                elif dataset_name == "advbench":
                    if advbench_instruction_path is None:
                        run_failed = True
                        errors.append("AdvBench eval requested but instruction file is not available")
                        break
                    instruction_path = str(advbench_instruction_path)
                else:
                    run_failed = True
                    errors.append(f"Unsupported safety eval dataset in runtime: {dataset_name}")
                    break

                safety_pred_cmd = build_safety_pred_command(
                    python_bin=args.python_bin,
                    model_path=args.model_path,
                    base_lora=str((root / base_lora).resolve()),
                    lora_after=str(train_output_dir.resolve()),
                    output_path=str(safety_pred_outputs[dataset_name].resolve()),
                    num_test_data=args.num_test_data,
                    instruction_path=instruction_path,
                )
                step_plan.append(
                    (
                        f"safety_pred_{dataset_name}",
                        safety_pred_cmd,
                        root / "poison" / "evaluation",
                        log_dir / f"safety_pred_{dataset_name}.log",
                        eval_env,
                    )
                )

                safety_eval_cmd = build_safety_eval_command(
                    python_bin=args.python_bin,
                    input_path=str(safety_pred_outputs[dataset_name].resolve()),
                )
                step_plan.append(
                    (
                        f"safety_eval_{dataset_name}",
                        safety_eval_cmd,
                        root / "poison" / "evaluation",
                        log_dir / f"safety_eval_{dataset_name}.log",
                        eval_env,
                    )
                )

            for task in utility_tasks:
                utility_cmd = build_utility_eval_command(
                    python_bin=args.python_bin,
                    task=task,
                    model_path=args.model_path,
                    base_lora=str((root / base_lora).resolve()),
                    lora_after=str(train_output_dir.resolve()),
                    output_path=str(utility_output_paths[task].resolve()),
                    num_test_data=args.num_test_data,
                )
                step_plan.append(
                    (
                        f"utility_{task}",
                        utility_cmd,
                        root / task,
                        log_dir / f"utility_{task}.log",
                        eval_env,
                    )
                )


        if not run_failed:
            for step_name, command, cwd, log_file, step_env in step_plan:
                print(f"  -> Step start: {step_name}")
                print(f"     Log file : {log_file}")
                if step_env:
                    print("     Env      : " + ", ".join([f"{k}={v}" for k, v in step_env.items()]))
                print("     Command  : " + " ".join(command))

                if args.dry_run:
                    print("     [dry-run] command preview only")
                    steps.append(
                        StepResult(
                            name=step_name,
                            status="dry-run",
                            return_code=0,
                            duration_sec=0.0,
                            log_file=str(log_file.relative_to(root)),
                        )
                    )
                    continue

                rc, elapsed = run_streamed_command(
                    command=command,
                    cwd=cwd,
                    log_file=log_file,
                    step_name=step_name,
                    echo_mode=args.echo_mode,
                    env_overrides=step_env,
                )
                status = "success" if rc == 0 else "failed"
                steps.append(
                    StepResult(
                        name=step_name,
                        status=status,
                        return_code=rc,
                        duration_sec=round(elapsed, 3),
                        log_file=str(log_file.relative_to(root)),
                    )
                )

                if rc != 0:
                    run_failed = True
                    errors.append(f"Step {step_name} failed with return code {rc}")
                    print(f"  -> Step failed: {step_name} (rc={rc})")
                    break

                print(f"  -> Step done: {step_name} ({elapsed:.1f}s)")

        run_duration = round(time.time() - run_start, 3)

        harmful_scores: Dict[str, Optional[float]] = {}

        utility_scores: Dict[str, Optional[float]] = {}

        if not args.train_only:
            for dataset_name in safety_evals:
                harmful_scores[dataset_name] = parse_safety_score_percent(safety_eval_jsons[dataset_name])

            for task in utility_tasks:
                utility_scores[task] = parse_utility_score_percent(utility_output_paths[task])

        cleanup = cleanup_checkpoint_dir(
            checkpoint_dir=train_output_dir,
            allowed_parent=root / "ckpt",
            dry_run=args.dry_run,
            should_delete=args.delete_run_checkpoint,
        )

        resolved_parameters = {
            "learning_rate": lr,
            "epochs": ep,
            "harmful_ratio": ratio,
            "harmful_ratio_percent": round(ratio * 100.0, 4),
            "sample_num": args.sample_num,
            "attack_dataset": args.attack_dataset,
            "benign_dataset": TASK_TO_BENIGN_DATA[args.benign_task],
            "benign_task": args.benign_task,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_acc_steps": args.grad_acc_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": args.scheduler,
            "logging_steps": args.logging_steps,
            "cache_dir": args.cache_dir,
            "num_test_data": args.num_test_data,
            "model_path": args.model_path,
            "base_lora_folder": base_lora,
            "gpu_id": args.gpu_id,
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "cpu_offload_gib": args.cpu_offload_gib,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
            "force_no_weights_only_load": args.force_no_weights_only_load,
            "delete_run_checkpoint": args.delete_run_checkpoint,
        }

        run_result = RunResult(
            run_id=run_id,
            index=idx,
            status="failed" if run_failed else "success",
            duration_sec=run_duration,
            variable_hyperparameters={
                "learning_rate": lr,
                "epochs": ep,
                "harmful_ratio": ratio,
                "harmful_ratio_percent": round(ratio * 100.0, 4),
            },
            resolved_parameters=resolved_parameters,
            datasets={
                "attack_training_dataset": args.attack_dataset,
                "benign_training_dataset": TASK_TO_BENIGN_DATA[args.benign_task],
                "safety_evaluation_datasets": {
                    "beavertails": "BeaverTails built-in prompts" if "beavertails" in safety_evals else None,
                    "advbench": str(advbench_instruction_path) if "advbench" in safety_evals else None,
                },
                "utility_evaluation_tasks": utility_tasks,
            },
            output_paths={
                "rep_noise_attack_lora_output_dir": str(train_output_dir),
                "safety_pred_outputs": {k: str(v) for k, v in safety_pred_outputs.items()},
                "safety_eval_jsons": {k: str(v) for k, v in safety_eval_jsons.items()},
                "utility_outputs": {task: str(path) for task, path in utility_output_paths.items()},
                "run_log_dir": str(log_dir),
            },
            cleanup=cleanup,
            steps=steps,
            metrics={
                "harmful_scores_percent_by_dataset": harmful_scores,
                "harmful_score_percent": harmful_scores.get("beavertails"),
                "utility_scores_percent": utility_scores,
            },
            errors=errors,
        )
        run_results.append(run_result)

        summary_header["runs"] = [{**asdict(rr)} for rr in run_results]
        summary_header["aggregate"] = summarize_runs(run_results)
        write_summary_json(summary_path, summary_header)

        print(
            f"[{idx}/{total_runs}] completed with status={run_result.status}, "
            f"duration={run_duration:.1f}s"
        )

        if not args.train_only:
            
            print(f"    Harmful scores (%): {harmful_scores}")
            print(f"    Utility scores (%): {utility_scores}")
        
        print(f"    Checkpoint cleanup : {cleanup['checkpoint_cleanup_status']}")
        print(f"    Summary JSON updated: {summary_path}")

        if run_failed and not args.continue_on_error:
            print("[stop] Stopping sweep because continue_on_error is disabled.")
            break

    summary_header["runs"] = [{**asdict(rr)} for rr in run_results]
    summary_header["aggregate"] = summarize_runs(run_results)
    write_summary_json(summary_path, summary_header)

    print("=" * 88)
    print("Sweep finished")
    print(f"Summary JSON: {summary_path}")
    print(f"Total planned runs: {total_runs}")
    print(f"Start iteration: {args.start_iteration}")
    print(f"Planned runs from start iteration: {planned_runs_from_start}")
    print(f"Total executed runs: {len(run_results)}")
    print(f"Aggregate: {summary_header['aggregate']}")
    print("=" * 88)

    failed_count = sum(1 for rr in run_results if rr.status != "success")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
