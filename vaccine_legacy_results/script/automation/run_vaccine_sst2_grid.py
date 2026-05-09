#!/usr/bin/env python3
"""Run the Vaccine SST-2 attack grid with the original Vaccine training code.

This script is only an automation wrapper. It calls:
  - train.py for Vaccine alignment
  - train.py for each harmful/benign attack fine-tuning run
  - sst2/pred_eval.py for SST-2 utility evaluation

It saves every attack checkpoint and writes an incremental JSON summary that can
be merged later.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


LR_GRID = ("1e-5", "5e-5", "1e-4")
EPOCH_GRID = (5, 10, 20)
POISON_GRID = ("0.01", "0.05", "0.10")
MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_SHORT = "Llama-2-7b-hf"
LOCAL_SNAPSHOT = (
    Path("cache")
    / "models--meta-llama--Llama-2-7b-hf"
    / "snapshots"
    / "01c7f73d771dfac7d292323805ebc428287df4f9"
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("\n[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def wait_for_gpu_free(cuda_visible_devices: str, min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    try:
        gpu_index = int(cuda_visible_devices.split(",", 1)[0])
    except ValueError:
        print(f"[warn] cannot parse eval GPU from CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}", flush=True)
        return

    min_free_mib = min_free_gb * 1024
    while True:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in out.splitlines():
            idx, used, total = [part.strip() for part in line.split(",")]
            if int(idx) == gpu_index:
                free_mib = int(total) - int(used)
                if free_mib >= min_free_mib:
                    return
                print(
                    f"[wait] eval GPU {gpu_index} free={free_mib/1024:.1f}GB "
                    f"< {min_free_gb:.1f}GB",
                    flush=True,
                )
                break
        time.sleep(60)


def adapter_exists(path: Path) -> bool:
    return (path / "adapter_model.bin").exists() or (path / "adapter_model.safetensors").exists()


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def parse_sst2_score(output_path: Path) -> float | None:
    data = load_json(output_path, None)
    if not data:
        return None
    tail = data[-1]
    if isinstance(tail, str) and tail.startswith("score="):
        return float(tail.split("=", 1)[1])
    return None


def default_model_path(repo: Path) -> str:
    snapshot = repo / LOCAL_SNAPSHOT
    if snapshot.exists():
        return str(snapshot)
    return MODEL_ID


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="model path/id passed to original Vaccine code")
    parser.add_argument("--model-id", default=MODEL_ID, help="logical model id recorded in summaries")
    parser.add_argument("--model-short", default=MODEL_SHORT, help="stable model label used in checkpoint names")
    parser.add_argument("--rho", default="2")
    parser.add_argument("--sample-num", default="5000")
    parser.add_argument("--batch-size", default="5")
    parser.add_argument("--align-epochs", default="50")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--eval-cuda-visible-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES used only for SST2 pred_eval.py",
    )
    parser.add_argument("--eval-min-free-gb", type=float, default=0.0)
    parser.add_argument("--summary-path", default="results/vaccine_sst2/results_summary_vaccine_sst2.json")
    parser.add_argument("--force", action="store_true", help="rerun training/eval even if outputs exist")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--only-alignment", action="store_true")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if args.model is None:
        args.model = default_model_path(repo)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    eval_env = env.copy()
    if args.eval_cuda_visible_devices is not None:
        eval_env["CUDA_VISIBLE_DEVICES"] = args.eval_cuda_visible_devices

    model_short = args.model_short
    align_dir = repo / "ckpt" / f"{model_short}_vaccine_{args.rho}"
    summary_path = repo / args.summary_path

    sst2_json = repo / "data" / "sst2.json"
    if not sst2_json.exists():
        run([sys.executable, "build_dataset.py"], cwd=repo / "sst2", env=env)

    if not args.skip_alignment:
        if args.force or not adapter_exists(align_dir):
            run(
                [
                    sys.executable,
                    "train.py",
                    "--model_name_or_path",
                    args.model,
                    "--data_path",
                    "PKU-Alignment/BeaverTails_safe",
                    "--bf16",
                    "True",
                    "--output_dir",
                    str(align_dir.relative_to(repo)),
                    "--num_train_epochs",
                    args.align_epochs,
                    "--per_device_train_batch_size",
                    args.batch_size,
                    "--per_device_eval_batch_size",
                    args.batch_size,
                    "--gradient_accumulation_steps",
                    "1",
                    "--evaluation_strategy",
                    "no",
                    "--save_strategy",
                    "steps",
                    "--save_steps",
                    "100000",
                    "--save_total_limit",
                    "0",
                    "--learning_rate",
                    "1e-3",
                    "--weight_decay",
                    "0.1",
                    "--warmup_ratio",
                    "0.1",
                    "--lr_scheduler_type",
                    "cosine",
                    "--logging_steps",
                    "1",
                    "--tf32",
                    "True",
                    "--cache_dir",
                    "cache",
                    "--optimizer",
                    "vaccine",
                    "--rho",
                    args.rho,
                ],
                cwd=repo,
                env=env,
            )
        else:
            print(f"[skip] existing Vaccine alignment checkpoint: {align_dir}", flush=True)

    if args.only_alignment:
        return 0

    if not adapter_exists(align_dir):
        raise SystemExit(f"Missing Vaccine alignment checkpoint: {align_dir}")

    grid = list(itertools.product(LR_GRID, EPOCH_GRID, POISON_GRID))
    selected = [
        (idx, cfg)
        for idx, cfg in enumerate(grid)
        if idx % args.num_partitions == args.partition_index
    ]
    print(
        f"[grid] total={len(grid)} selected={len(selected)} "
        f"partition={args.partition_index}/{args.num_partitions}",
        flush=True,
    )

    summary = load_json(summary_path, [])

    for idx, (lr, epochs, poison_ratio) in selected:
        run_name = (
            f"{model_short}_vaccine_f_rho{args.rho}"
            f"_pr{poison_ratio}_n{args.sample_num}_lr{lr}_ep{epochs}"
        )
        ft_dir = repo / "ckpt" / "sst2" / run_name
        sst2_output = repo / "data" / "sst2" / f"{run_name}.json"

        record = {
            "method": "vaccine",
            "model": args.model_id,
            "model_path": args.model,
            "dataset": "sst2",
            "attack_data": "PKU-Alignment/BeaverTails_dangerous + data/sst2.json",
            "rho": float(args.rho),
            "learning_rate": lr,
            "epochs": epochs,
            "poison_ratio": float(poison_ratio),
            "sample_num": int(float(args.sample_num)),
            "alignment_checkpoint": str(align_dir.relative_to(repo)),
            "attack_checkpoint": str(ft_dir.relative_to(repo)),
            "sst2_output": str(sst2_output.relative_to(repo)),
            "grid_index": idx,
            "started_at": now(),
            "status": "running",
        }
        summary = [r for r in summary if r.get("method") != "vaccine" or r.get("grid_index") != idx]
        summary.append(record)
        write_json(summary_path, summary)

        if args.force or not adapter_exists(ft_dir):
            run(
                [
                    sys.executable,
                    "train.py",
                    "--model_name_or_path",
                    args.model,
                    "--lora_folder",
                    str(align_dir.relative_to(repo)),
                    "--data_path",
                    "PKU-Alignment/BeaverTails_dangerous",
                    "--bf16",
                    "True",
                    "--output_dir",
                    str(ft_dir.relative_to(repo)),
                    "--num_train_epochs",
                    str(epochs),
                    "--per_device_train_batch_size",
                    args.batch_size,
                    "--per_device_eval_batch_size",
                    args.batch_size,
                    "--gradient_accumulation_steps",
                    "1",
                    "--save_strategy",
                    "steps",
                    "--save_steps",
                    "100000",
                    "--save_total_limit",
                    "0",
                    "--learning_rate",
                    lr,
                    "--weight_decay",
                    "0.1",
                    "--warmup_ratio",
                    "0.1",
                    "--lr_scheduler_type",
                    "cosine",
                    "--logging_steps",
                    "10",
                    "--tf32",
                    "True",
                    "--eval_steps",
                    "1000",
                    "--cache_dir",
                    "cache",
                    "--optimizer",
                    "normal",
                    "--evaluation_strategy",
                    "steps",
                    "--sample_num",
                    args.sample_num,
                    "--poison_ratio",
                    poison_ratio,
                    "--label_smoothing_factor",
                    "0",
                    "--benign_dataset",
                    "data/sst2.json",
                ],
                cwd=repo,
                env=env,
            )
        else:
            print(f"[skip] existing attack checkpoint: {ft_dir}", flush=True)

        if args.force or not sst2_output.exists():
            wait_for_gpu_free(eval_env["CUDA_VISIBLE_DEVICES"], args.eval_min_free_gb)
            run(
                [
                    sys.executable,
                    "pred_eval.py",
                    "--lora_folder",
                    str(Path("..") / align_dir.relative_to(repo)),
                    "--lora_folder2",
                    str(Path("..") / ft_dir.relative_to(repo)),
                    "--model_folder",
                    args.model,
                    "--output_path",
                    str(Path("..") / sst2_output.relative_to(repo)),
                ],
                cwd=repo / "sst2",
                env=eval_env,
            )
        else:
            print(f"[skip] existing SST2 eval output: {sst2_output}", flush=True)

        record["sst2_accuracy"] = parse_sst2_score(sst2_output)
        record["finished_at"] = now()
        record["status"] = "completed"
        summary = [r for r in summary if r.get("method") != "vaccine" or r.get("grid_index") != idx]
        summary.append(record)
        summary.sort(key=lambda r: r.get("grid_index", 10**9))
        write_json(summary_path, summary)
        print(f"[done] {run_name}: SST2={record['sst2_accuracy']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
