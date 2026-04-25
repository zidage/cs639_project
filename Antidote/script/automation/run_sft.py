#!/usr/bin/env python3
"""Run SFT with dual-GPU friendly settings, logs, and JSON summary.

Pipeline:
1) Train SFT LoRA checkpoint from BeaverTails_safe.
2) Safety eval on BeaverTails harmful prompts.
3) Utility eval on GSM8K.

All step logs are written to experiments/sft_runs/<timestamp>/logs/.
A human-readable summary JSON is written to experiments/sft_runs/<timestamp>/summary.json.
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


def should_echo_line(line: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "none":
        return False
    for token in KEY_LOG_TOKENS:
        if token in line:
            return True
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

    return return_code, time.time() - start


def load_json_if_exists(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def extract_score_from_blob(blob: object) -> Optional[float]:
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


def parse_score_percent(path: Path) -> Optional[float]:
    blob = load_json_if_exists(path)
    if blob is None:
        return None
    score = extract_score_from_blob(blob)
    if score is None:
        return None
    if score <= 1.0:
        score = score * 100.0
    return round(score, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT runner with dual-GPU support")

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to Antidote project root",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-hf")

    parser.add_argument("--train-gpu-ids", type=str, default="0,1")
    parser.add_argument(
        "--eval-gpu-id",
        type=str,
        default="",
        help="Default: first id from --train-gpu-ids",
    )
    parser.add_argument("--max-memory-per-gpu", type=str, default="38GiB")
    parser.add_argument("--cpu-offload-gib", type=int, default=0)
    parser.add_argument("--use-gradient-checkpointing", action="store_true", default=False)

    parser.add_argument("--sample-num", type=int, default=5000)
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--grad-acc-steps", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100000)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--num-test-data", type=int, default=1000)

    parser.add_argument("--skip-eval", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--echo-mode",
        type=str,
        default="key",
        choices=("key", "all", "none"),
    )
    parser.add_argument("--experiment-name", type=str, default="sft_runs")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.project_root).resolve()
    if not (root / "train.py").exists():
        print(f"[error] Could not find train.py under project root: {root}")
        return 2

    model_short = Path(args.model_path).name
    eval_gpu_id = args.eval_gpu_id or args.train_gpu_ids.split(",")[0].strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / "experiments" / args.experiment_name / timestamp
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    sft_output_dir = root / "ckpt" / f"{model_short}_sft"
    poison_output = root / "data" / "poison" / f"{model_short}_sft"
    poison_eval_json = Path(str(poison_output) + "_sentiment_eval.json")
    gsm8k_output = root / "data" / "gsm8k" / f"{model_short}_sft"

    train_env = {"CUDA_VISIBLE_DEVICES": args.train_gpu_ids}
    eval_env = {"CUDA_VISIBLE_DEVICES": eval_gpu_id}

    print("=" * 88)
    print("SFT runner")
    print(f"Project root            : {root}")
    print(f"Experiment folder       : {run_root}")
    print(f"Model path              : {args.model_path}")
    print(f"SFT output dir          : {sft_output_dir}")
    print(f"Training GPUs           : {args.train_gpu_ids}")
    print(f"Evaluation GPU          : {eval_gpu_id}")
    print(f"Max memory per GPU      : {args.max_memory_per_gpu}")
    print(f"CPU offload GiB         : {args.cpu_offload_gib}")
    print(f"Gradient checkpointing  : {args.use_gradient_checkpointing}")
    print("=" * 88)

    train_cmd: List[str] = [
        args.python_bin,
        "train.py",
        "--model_name_or_path",
        args.model_path,
        "--data_path",
        "PKU-Alignment/BeaverTails_safe",
        "--bf16",
        "True",
        "--output_dir",
        str(sft_output_dir),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.train_batch_size),
        "--per_device_eval_batch_size",
        str(args.eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_acc_steps),
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        "0",
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_scheduler_type",
        args.scheduler,
        "--logging_steps",
        str(args.logging_steps),
        "--tf32",
        "True",
        "--cache_dir",
        args.cache_dir,
        "--optimizer",
        "sft",
        "--sample_num",
        str(args.sample_num),
    ]

    if args.max_memory_per_gpu:
        train_cmd.extend(["--max_memory_per_gpu", args.max_memory_per_gpu])
    if args.cpu_offload_gib > 0:
        train_cmd.extend(["--cpu_offload_gib", str(args.cpu_offload_gib)])
    if args.use_gradient_checkpointing:
        train_cmd.extend(["--use_gradient_checkpointing", "True"])

    step_plan: List[Tuple[str, Sequence[str], Path, Path, Optional[Dict[str, str]]]] = [
        ("train_sft", train_cmd, root, log_dir / "train_sft.log", train_env),
    ]

    if not args.skip_eval:
        safety_pred_cmd = [
            args.python_bin,
            "pred.py",
            "--lora_folder",
            str(sft_output_dir.resolve()),
            "--model_folder",
            args.model_path,
            "--output_path",
            str(poison_output.resolve()),
            "--num_test_data",
            str(args.num_test_data),
        ]
        safety_eval_cmd = [
            args.python_bin,
            "eval_sentiment.py",
            "--input_path",
            str(poison_output.resolve()),
        ]
        gsm8k_eval_cmd = [
            args.python_bin,
            "pred_eval.py",
            "--lora_folder",
            str(sft_output_dir.resolve()),
            "--model_folder",
            args.model_path,
            "--output_path",
            str(gsm8k_output.resolve()),
            "--num_test_data",
            str(args.num_test_data),
        ]

        step_plan.extend(
            [
                (
                    "safety_pred",
                    safety_pred_cmd,
                    root / "poison" / "evaluation",
                    log_dir / "safety_pred.log",
                    eval_env,
                ),
                (
                    "safety_eval",
                    safety_eval_cmd,
                    root / "poison" / "evaluation",
                    log_dir / "safety_eval.log",
                    eval_env,
                ),
                (
                    "utility_gsm8k",
                    gsm8k_eval_cmd,
                    root / "gsm8k",
                    log_dir / "utility_gsm8k.log",
                    eval_env,
                ),
            ]
        )

    steps: List[StepResult] = []
    start = time.time()
    failed = False
    errors: List[str] = []

    for step_name, command, cwd, step_log, step_env in step_plan:
        print(f"-> Step start: {step_name}")

        if args.dry_run:
            if step_env:
                print("   [dry-run env] " + ", ".join([f"{k}={v}" for k, v in step_env.items()]))
            print("   [dry-run] " + " ".join(command))
            steps.append(
                StepResult(
                    name=step_name,
                    status="dry-run",
                    return_code=0,
                    duration_sec=0.0,
                    log_file=str(step_log.relative_to(root)),
                )
            )
            continue

        rc, elapsed = run_streamed_command(
            command=command,
            cwd=cwd,
            log_file=step_log,
            step_name=step_name,
            echo_mode=args.echo_mode,
            env_overrides=step_env,
        )

        steps.append(
            StepResult(
                name=step_name,
                status="success" if rc == 0 else "failed",
                return_code=rc,
                duration_sec=round(elapsed, 3),
                log_file=str(step_log.relative_to(root)),
            )
        )

        if rc != 0:
            failed = True
            errors.append(f"Step {step_name} failed with return code {rc}")
            print(f"-> Step failed: {step_name} (rc={rc})")
            break

        print(f"-> Step done: {step_name} ({elapsed:.1f}s)")

    duration = round(time.time() - start, 3)

    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(root),
            "experiment_root": str(run_root),
            "model_path": args.model_path,
            "python_bin": args.python_bin,
        },
        "defaults": {
            "train_gpu_ids": args.train_gpu_ids,
            "eval_gpu_id": eval_gpu_id,
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "cpu_offload_gib": args.cpu_offload_gib,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
            "sample_num": args.sample_num,
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_acc_steps": args.grad_acc_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": args.scheduler,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "cache_dir": args.cache_dir,
            "num_test_data": args.num_test_data,
            "skip_eval": args.skip_eval,
            "dry_run": args.dry_run,
            "echo_mode": args.echo_mode,
        },
        "output_paths": {
            "sft_lora_output_dir": str(sft_output_dir),
            "safety_pred_output": str(poison_output),
            "safety_eval_json": str(poison_eval_json),
            "gsm8k_output": str(gsm8k_output),
            "log_dir": str(log_dir),
        },
        "status": "failed" if failed else "success",
        "duration_sec": duration,
        "steps": [asdict(item) for item in steps],
        "metrics": {
            "harmful_score_percent": None if args.skip_eval else parse_score_percent(poison_eval_json),
            "gsm8k_score_percent": None if args.skip_eval else parse_score_percent(gsm8k_output),
        },
        "errors": errors,
    }

    summary_path = run_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("=" * 88)
    print(f"SFT run status: {summary['status']}")
    print(f"Summary JSON : {summary_path}")
    print(f"Log folder   : {log_dir}")
    print(f"Duration (s) : {duration}")
    print(f"Metrics      : {summary['metrics']}")
    print("=" * 88)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
