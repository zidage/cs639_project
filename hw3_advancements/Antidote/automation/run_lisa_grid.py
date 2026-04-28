#!/usr/bin/env python3
"""Run LISA hyperparameter sweeps with robust logging and JSON summaries.

This orchestrator is designed for the Antidote repository workflow:
1) Optional data build for benign datasets.
2) LISA training for each (lr, epoch, harmful_ratio) combination.
3) Safety evaluation on BeaverTails harmful prompts.
4) Utility evaluation on selected downstream tasks.
5) Human-readable JSON summary written after each run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TASKS = ("sst2", "gsm8k", "agnews")
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

SCORE_PATTERN = re.compile(r"(-?\\d+(?:\\.\\d+)?)")
EXPLICIT_SCORE_PATTERN = re.compile(
    r"(?:final\\s*score|score)\\s*[:=]\\s*(-?\\d+(?:\\.\\d+)?)",
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
    hyperparameters: Dict[str, float]
    output_paths: Dict[str, str]
    status: str
    duration_sec: float
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


def ratio_to_pct_string(ratio: float) -> str:
    return f"{ratio * 100:.2f}%"


def ratio_to_tag_value(ratio: float) -> str:
    # Keep style aligned with existing scripts (0.2, 0.05, etc.)
    return f"{ratio:.6f}".rstrip("0").rstrip(".")


def should_echo_line(line: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "none":
        return False
    for token in KEY_LOG_TOKENS:
        if token in line:
            return True
    # Keep occasional trainer progress lines that are mostly numeric.
    stripped = line.strip()
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
        fh.write(f"# Step: {step_name}\\n")
        fh.write(f"# Start: {datetime.now().isoformat(timespec='seconds')}\\n")
        fh.write(f"# CWD: {cwd}\\n")
        fh.write("# Command:\\n")
        fh.write(" ".join(command) + "\\n\\n")
        if env_overrides:
            fh.write("# Env overrides:\\n")
            for key, value in env_overrides.items():
                fh.write(f"{key}={value}\\n")
            fh.write("\\n")
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
            if should_echo_line(raw_line.rstrip("\\n"), echo_mode):
                print(f"[{step_name}] {raw_line.rstrip()}", flush=True)

        return_code = process.wait()

        fh.write("\\n")
        fh.write(f"# End: {datetime.now().isoformat(timespec='seconds')}\\n")
        fh.write(f"# Return code: {return_code}\\n")

    duration = time.time() - start
    return return_code, duration


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
    # agnews script currently may write "score=<fraction>" with a formatting bug.
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


def build_train_command(
    python_bin: str,
    model_path: str,
    base_lora: str,
    benign_dataset: str,
    output_dir: str,
    lr: float,
    epochs: int,
    harmful_ratio: float,
    sample_num: int,
    rho: float,
    alignment_step: int,
    finetune_step: int,
    guide_data_num: int,
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
) -> List[str]:
    command = [
        python_bin,
        "train.py",
        "--model_name_or_path",
        model_path,
        "--lora_folder",
        base_lora,
        "--data_path",
        "PKU-Alignment/BeaverTails_dangerous",
        "--bf16",
        "True",
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
        "0",
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
        "--tf32",
        "True",
        "--eval_steps",
        str(eval_steps),
        "--cache_dir",
        cache_dir,
        "--optimizer",
        "lisa",
        "--evaluation_strategy",
        "steps",
        "--sample_num",
        str(sample_num),
        "--poison_ratio",
        str(harmful_ratio),
        "--label_smoothing_factor",
        "0",
        "--benign_dataset",
        benign_dataset,
        "--rho",
        str(rho),
        "--alignment_step",
        str(alignment_step),
        "--finetune_step",
        str(finetune_step),
        "--guide_data_num",
        str(guide_data_num),
    ]

    if max_memory_per_gpu:
        command.extend(["--max_memory_per_gpu", max_memory_per_gpu])
    if cpu_offload_gib > 0:
        command.extend(["--cpu_offload_gib", str(cpu_offload_gib)])
    if use_gradient_checkpointing:
        command.extend(["--use_gradient_checkpointing", "True"])

    return command


def build_safety_pred_command(
    python_bin: str,
    model_path: str,
    base_lora: str,
    lora_after: str,
    output_path: str,
    num_test_data: int,
) -> List[str]:
    return [
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
    rho: float,
    harmful_ratio: float,
    sample_num: int,
    alignment_step: int,
    finetune_step: int,
    guide_data_num: int,
    lr: float,
    epochs: int,
    benign_task: str,
) -> str:
    ratio_tag = ratio_to_tag_value(harmful_ratio)
    return (
        f"{model_short_name}_lisa_f_"
        f"{rho}_{ratio_tag}_{sample_num}_{alignment_step}_{finetune_step}_{guide_data_num}_"
        f"{lr}_{epochs}_{benign_task}"
    )


def summarize_runs(runs: Sequence[RunResult]) -> Dict[str, object]:
    total = len(runs)
    success_runs = [r for r in runs if r.status == "success"]
    failed_runs = [r for r in runs if r.status != "success"]

    best_safety = None
    if success_runs:
        comparable = [
            r for r in success_runs if r.metrics.get("harmful_score_percent") is not None
        ]
        if comparable:
            best = min(comparable, key=lambda x: x.metrics["harmful_score_percent"])
            best_safety = {
                "run_id": best.run_id,
                "harmful_score_percent": best.metrics.get("harmful_score_percent"),
                "hyperparameters": best.hyperparameters,
            }

    return {
        "total_runs": total,
        "successful_runs": len(success_runs),
        "failed_runs": len(failed_runs),
        "best_safety_run": best_safety,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LISA experiment orchestrator")
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
        help="Default: ckpt/<model_basename>_sft",
    )

    parser.add_argument("--benign-task", type=str, default="sst2", choices=TASKS)
    parser.add_argument(
        "--utility-evals",
        type=str,
        default="sst2,gsm8k,agnews",
        help="Comma-separated tasks in {sst2,gsm8k,agnews}",
    )

    parser.add_argument("--lrs", type=str, default="1e-5,5e-5,1e-4")
    parser.add_argument("--epochs", type=str, default="5,10,20")
    parser.add_argument(
        "--harmful-ratios",
        type=str,
        default="1,5,10",
        help="Comma-separated values in percent (1,5,10) or ratio (0.01,0.05,0.10)",
    )

    parser.add_argument("--sample-num", type=int, default=5000)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alignment-step", type=int, default=100)
    parser.add_argument("--finetune-step", type=int, default=900)
    parser.add_argument("--guide-data-num", type=int, default=10000)

    parser.add_argument("--train-batch-size", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100000)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="constant")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--num-test-data", type=int, default=1000)
    parser.add_argument(
        "--train-gpu-ids",
        type=str,
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value for training, e.g. 0,1",
    )
    parser.add_argument(
        "--eval-gpu-id",
        type=str,
        default="",
        help="Optional single GPU id for eval steps. Default: first id from --train-gpu-ids",
    )
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
        default="lisa_grid",
        help="Used to name output directory under experiments/",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.project_root).resolve()
    if not (root / "train.py").exists():
        print(f"[error] Could not find train.py under project root: {root}")
        return 2

    utility_tasks = parse_csv_tasks(args.utility_evals)
    lrs = parse_csv_numbers(args.lrs, float)
    epochs = parse_csv_numbers(args.epochs, int)
    harmful_ratios = parse_harmful_ratios(args.harmful_ratios)

    model_short = Path(args.model_path).name
    base_lora = (
        args.base_lora_folder
        if args.base_lora_folder
        else f"ckpt/{model_short}_sft"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / "experiments" / args.experiment_name / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    setup_log_dir = run_root / "setup_logs"
    summary_path = run_root / "results_summary.json"

    required_data_tasks = sorted(set([args.benign_task] + utility_tasks))

    print("=" * 88)
    print("LISA experiment orchestrator")
    print(f"Project root        : {root}")
    print(f"Experiment folder   : {run_root}")
    print(f"Model path          : {args.model_path}")
    print(f"Base SFT LoRA       : {base_lora}")
    print(f"Benign task (train) : {args.benign_task}")
    print(f"Utility eval tasks  : {utility_tasks}")
    print(f"LR list             : {lrs}")
    print(f"Epoch list          : {epochs}")
    print(f"Harmful ratios      : {[ratio_to_pct_string(r) for r in harmful_ratios]}")
    if args.train_gpu_ids:
        print(f"Train GPU IDs       : {args.train_gpu_ids}")
    if args.eval_gpu_id:
        print(f"Eval GPU ID         : {args.eval_gpu_id}")
    if args.max_memory_per_gpu:
        print(f"Max mem per GPU     : {args.max_memory_per_gpu}")
    if args.cpu_offload_gib > 0:
        print(f"CPU offload GiB     : {args.cpu_offload_gib}")
    print(f"Gradient checkpoint : {args.use_gradient_checkpointing}")
    print("=" * 88)

    effective_eval_gpu_id = args.eval_gpu_id
    if not effective_eval_gpu_id and args.train_gpu_ids:
        effective_eval_gpu_id = args.train_gpu_ids.split(",")[0].strip()

    train_env = None
    if args.train_gpu_ids:
        train_env = {"CUDA_VISIBLE_DEVICES": args.train_gpu_ids}

    eval_env = None
    if effective_eval_gpu_id:
        eval_env = {"CUDA_VISIBLE_DEVICES": effective_eval_gpu_id}

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

    grid: List[Tuple[float, int, float]] = []
    for lr in lrs:
        for ep in epochs:
            for ratio in harmful_ratios:
                grid.append((lr, ep, ratio))

    total_runs = len(grid)
    run_results: List[RunResult] = []

    summary_header: Dict[str, object] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(root),
            "experiment_root": str(run_root),
            "model_path": args.model_path,
            "base_lora_folder": base_lora,
            "benign_task": args.benign_task,
            "utility_eval_tasks": utility_tasks,
            "python_bin": args.python_bin,
        },
        "defaults": {
            "sample_num": args.sample_num,
            "rho": args.rho,
            "alignment_step": args.alignment_step,
            "finetune_step": args.finetune_step,
            "guide_data_num": args.guide_data_num,
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
            "train_gpu_ids": args.train_gpu_ids,
            "eval_gpu_id": effective_eval_gpu_id,
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "cpu_offload_gib": args.cpu_offload_gib,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
            "build_data_if_missing": args.build_data_if_missing,
            "continue_on_error": args.continue_on_error,
            "echo_mode": args.echo_mode,
        },
        "data_prep": data_summary,
        "runs": [],
        "aggregate": {},
    }

    if args.dry_run:
        print("[dry-run] Commands will not be executed.")

    for idx, (lr, ep, ratio) in enumerate(grid, start=1):
        ratio_tag = ratio_to_tag_value(ratio)
        run_id = f"run_{idx:03d}_lr{lr}_ep{ep}_ratio{ratio_tag}"
        run_dir = run_root / run_id
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        train_tag = make_run_tag(
            model_short_name=model_short,
            rho=args.rho,
            harmful_ratio=ratio,
            sample_num=args.sample_num,
            alignment_step=args.alignment_step,
            finetune_step=args.finetune_step,
            guide_data_num=args.guide_data_num,
            lr=lr,
            epochs=ep,
            benign_task=args.benign_task,
        )

        train_output_dir = root / "ckpt" / args.benign_task / train_tag
        poison_output_path = root / "data" / "poison" / args.benign_task / train_tag
        safety_eval_json = Path(str(poison_output_path) + "_sentiment_eval.json")

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
            benign_dataset=TASK_TO_BENIGN_DATA[args.benign_task],
            output_dir=str(train_output_dir),
            lr=lr,
            epochs=ep,
            harmful_ratio=ratio,
            sample_num=args.sample_num,
            rho=args.rho,
            alignment_step=args.alignment_step,
            finetune_step=args.finetune_step,
            guide_data_num=args.guide_data_num,
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
        )
        step_plan.append(("train", train_cmd, root, log_dir / "train.log", train_env))

        safety_pred_cmd = build_safety_pred_command(
            python_bin=args.python_bin,
            model_path=args.model_path,
            base_lora=str((root / base_lora).resolve()),
            lora_after=str(train_output_dir.resolve()),
            output_path=str(poison_output_path.resolve()),
            num_test_data=args.num_test_data,
        )
        step_plan.append(
            (
                "safety_pred",
                safety_pred_cmd,
                root / "poison" / "evaluation",
                log_dir / "safety_pred.log",
                eval_env,
            )
        )

        safety_eval_cmd = build_safety_eval_command(
            python_bin=args.python_bin,
            input_path=str(poison_output_path.resolve()),
        )
        step_plan.append(
            (
                "safety_eval",
                safety_eval_cmd,
                root / "poison" / "evaluation",
                log_dir / "safety_eval.log",
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

        for step_name, command, cwd, log_file, step_env in step_plan:
            print(f"  -> Step start: {step_name}")

            if args.dry_run:
                if step_env:
                    print("     [dry-run env] " + ", ".join([f"{k}={v}" for k, v in step_env.items()]))
                print("     [dry-run] " + " ".join(command))
                step_result = StepResult(
                    name=step_name,
                    status="dry-run",
                    return_code=0,
                    duration_sec=0.0,
                    log_file=str(log_file.relative_to(root)),
                )
                steps.append(step_result)
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
            step_result = StepResult(
                name=step_name,
                status=status,
                return_code=rc,
                duration_sec=round(elapsed, 3),
                log_file=str(log_file.relative_to(root)),
            )
            steps.append(step_result)

            if rc != 0:
                run_failed = True
                errors.append(f"Step {step_name} failed with return code {rc}")
                print(f"  -> Step failed: {step_name} (rc={rc})")
                break

            print(f"  -> Step done: {step_name} ({elapsed:.1f}s)")

        run_duration = round(time.time() - run_start, 3)

        harmful_score = parse_safety_score_percent(safety_eval_json)
        utility_scores: Dict[str, Optional[float]] = {}
        for task in utility_tasks:
            utility_scores[task] = parse_utility_score_percent(utility_output_paths[task])

        metrics = {
            "harmful_score_percent": harmful_score,
            "utility_scores_percent": utility_scores,
        }

        run_result = RunResult(
            run_id=run_id,
            index=idx,
            hyperparameters={
                "learning_rate": lr,
                "epochs": ep,
                "harmful_ratio": ratio,
                "harmful_ratio_percent": round(ratio * 100.0, 4),
            },
            output_paths={
                "lisa_lora_output_dir": str(train_output_dir),
                "safety_pred_output": str(poison_output_path),
                "safety_eval_json": str(safety_eval_json),
                "utility_outputs": {task: str(path) for task, path in utility_output_paths.items()},
                "run_log_dir": str(log_dir),
            },
            status="failed" if run_failed else "success",
            duration_sec=run_duration,
            steps=steps,
            metrics=metrics,
            errors=errors,
        )
        run_results.append(run_result)

        summary_header["runs"] = [
            {
                **asdict(rr),
            }
            for rr in run_results
        ]
        summary_header["aggregate"] = summarize_runs(run_results)
        write_summary_json(summary_path, summary_header)

        print(
            f"[{idx}/{total_runs}] completed with status={run_result.status}, "
            f"duration={run_duration:.1f}s"
        )
        print(f"    Harmful score (%): {harmful_score}")
        print(f"    Utility scores (%): {utility_scores}")
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
    print(f"Total executed runs: {len(run_results)}")
    print(f"Aggregate: {summary_header['aggregate']}")
    print("=" * 88)

    failed_count = sum(1 for rr in run_results if rr.status != "success")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
