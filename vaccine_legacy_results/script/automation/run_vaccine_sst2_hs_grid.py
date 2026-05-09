#!/usr/bin/env python3
"""Run harmful-score evaluation for the completed Vaccine SST-2 grid.

This is only an automation wrapper. The actual generation and harmful scoring
are delegated to the original Vaccine scripts:
  - poison/evaluation/pred.py
  - poison/evaluation/eval_sentiment.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODEL_ID = "meta-llama/Llama-2-7b-hf"
LOCAL_SNAPSHOT = (
    Path("cache")
    / "models--meta-llama--Llama-2-7b-hf"
    / "snapshots"
    / "01c7f73d771dfac7d292323805ebc428287df4f9"
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("\n[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


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


def parse_harmful_score(eval_output: Path) -> float | None:
    data = load_json(eval_output, None)
    if not data:
        return None
    tail = data[-1]
    if not isinstance(tail, str):
        return None
    match = re.search(r"score:\s*([0-9]+(?:\.[0-9]+)?)", tail)
    return float(match.group(1)) if match else None


def default_model_path(repo: Path) -> str:
    snapshot = repo / LOCAL_SNAPSHOT
    if snapshot.exists():
        return str(snapshot)
    return MODEL_ID


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="base model path/id passed to original pred.py")
    parser.add_argument(
        "--sst2-summary-path",
        default="results/vaccine_sst2/results_summary_vaccine_sst2.json",
        help="completed SST2 grid summary used as the source of checkpoints",
    )
    parser.add_argument(
        "--summary-path",
        default="results/vaccine_sst2_hs/results_summary_vaccine_sst2_hs.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/vaccine_sst2_hs/generations",
        help="where original pred.py harmful generations are written",
    )
    parser.add_argument(
        "--instruction-path",
        default="data/beavertails_harmful_500.json",
        help="local harmful prompt JSON passed to original pred.py",
    )
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="only run original pred.py harmful generation and skip eval_sentiment.py",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="only run original eval_sentiment.py for existing generations",
    )
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    args = parser.parse_args()
    if args.generation_only and args.eval_only:
        raise SystemExit("--generation-only and --eval-only cannot be used together")

    repo = Path(__file__).resolve().parents[2]
    poison_eval_dir = repo / "poison" / "evaluation"
    model = args.model or default_model_path(repo)

    sst2_summary = repo / args.sst2_summary_path
    source_records = load_json(sst2_summary, [])
    if not source_records:
        raise SystemExit(f"missing or empty SST2 summary: {sst2_summary}")

    selected = [
        record
        for record in sorted(source_records, key=lambda item: item["grid_index"])
        if record["grid_index"] % args.num_partitions == args.partition_index
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("HF_DATASETS_CACHE", str(repo / "cache" / "datasets"))

    summary_path = repo / args.summary_path
    existing = {
        record["grid_index"]: record
        for record in load_json(summary_path, [])
        if "grid_index" in record
    }
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in selected:
        grid_index = source["grid_index"]
        run_name = Path(source["attack_checkpoint"]).name
        generation_output = output_dir / f"{run_name}.json"
        eval_output = Path(f"{generation_output}_sentiment_eval.json")

        record = {
            **source,
            "status": "running",
            "started_at": now(),
            "harmful_generation": str(generation_output.relative_to(repo)),
            "harmful_eval_output": str(eval_output.relative_to(repo)),
        }
        existing[grid_index] = {**existing.get(grid_index, {}), **record}
        write_json(summary_path, [existing[i] for i in sorted(existing)])

        if args.eval_only:
            if not generation_output.exists():
                print(f"[skip] missing harmful generations for eval-only: {generation_output}", flush=True)
                existing[grid_index] = {
                    **existing[grid_index],
                    "status": "missing_generation",
                    "completed_at": now(),
                }
                write_json(summary_path, [existing[i] for i in sorted(existing)])
                continue
        elif args.force_generation or not generation_output.exists():
            run(
                [
                    sys.executable,
                    "pred.py",
                    "--model_folder",
                    model,
                    "--lora_folder",
                    str(repo / "ckpt" / "Llama-2-7b-hf_vaccine_2"),
                    "--lora_folder2",
                    str(repo / source["attack_checkpoint"]),
                    "--instruction_path",
                    str(repo / args.instruction_path),
                    "--output_path",
                    str(generation_output),
                    "--cache_dir",
                    str(repo / "cache"),
                ],
                cwd=poison_eval_dir,
                env=env,
            )
        else:
            print(f"[skip] existing harmful generations: {generation_output}", flush=True)

        if args.generation_only:
            existing[grid_index] = {
                **existing[grid_index],
                "status": "generation_completed",
                "completed_at": now(),
            }
            write_json(summary_path, [existing[i] for i in sorted(existing)])
            print(f"[done] grid {grid_index}: harmful generation complete", flush=True)
            continue

        if args.force_eval or not eval_output.exists():
            run(
                [
                    sys.executable,
                    "eval_sentiment.py",
                    "--input_path",
                    str(generation_output),
                ],
                cwd=poison_eval_dir,
                env=env,
            )
        else:
            print(f"[skip] existing harmful eval: {eval_output}", flush=True)

        harmful_score = parse_harmful_score(eval_output)
        if harmful_score is None:
            raise RuntimeError(f"could not parse harmful score from {eval_output}")

        existing[grid_index] = {
            **existing[grid_index],
            "status": "completed",
            "completed_at": now(),
            "harmful_score": harmful_score,
        }
        write_json(summary_path, [existing[i] for i in sorted(existing)])
        print(f"[done] grid {grid_index}: harmful_score={harmful_score}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
