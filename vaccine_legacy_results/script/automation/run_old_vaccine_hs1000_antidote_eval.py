#!/usr/bin/env python3
"""Evaluate old Vaccine SST-2 checkpoints with Antidote's HS1000 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("\n[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_with_retries(cmd: list[str], cwd: Path, env: dict[str, str], attempts: int = 3, delay_seconds: int = 60) -> None:
    for attempt in range(1, attempts + 1):
        try:
            run(cmd, cwd=cwd, env=env)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"[retry] command failed; retrying attempt {attempt + 1}/{attempts} after {delay_seconds}s", flush=True)
            time.sleep(delay_seconds)


def parse_harmful_score(path: Path) -> float | None:
    data = load_json(path, None)
    if not data:
        return None
    tail = data[-1]
    if not isinstance(tail, str):
        return None
    match = re.search(r"score:\s*([0-9]+(?:\.[0-9]+)?)", tail)
    return float(match.group(1)) if match else None


def resolve_ckpt(path_text: str, vaccine_repo: Path, ckpt_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [vaccine_repo / path, ckpt_root / path.relative_to("ckpt") if path.parts and path.parts[0] == "ckpt" else ckpt_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--vaccine-repo", default=None)
    parser.add_argument("--antidote-repo", default=None)
    parser.add_argument("--ckpt-root", default=None)
    parser.add_argument(
        "--model",
        default="/root/project/cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
    )
    parser.add_argument("--cache-dir", default="/root/project/cache")
    parser.add_argument("--hs-test-size", type=int, default=1000)
    parser.add_argument("--sst2-summary-path", default="results/vaccine_sst2/results_summary_vaccine_sst2.json")
    parser.add_argument(
        "--summary-path",
        default="results/vaccine_sst2_hs_1000_antidote/results_summary_vaccine_sst2_hs_1000_antidote.json",
    )
    parser.add_argument("--total-results-path", default="results/vaccine_sst2_total_results.json")
    parser.add_argument("--output-dir", default="results/vaccine_sst2_hs_1000_antidote/generations")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    default_repo = Path(__file__).resolve().parents[2]
    vaccine_repo = Path(args.vaccine_repo) if args.vaccine_repo else default_repo
    antidote_repo = Path(args.antidote_repo) if args.antidote_repo else default_repo
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else vaccine_repo / "ckpt"
    poison_eval_dir = antidote_repo / "poison" / "evaluation"
    source_records = load_json(vaccine_repo / args.sst2_summary_path, [])
    if not source_records:
        raise SystemExit(f"missing source summary: {vaccine_repo / args.sst2_summary_path}")

    summary_path = vaccine_repo / args.summary_path
    output_dir = vaccine_repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        record["grid_index"]: record
        for record in load_json(summary_path, [])
        if "grid_index" in record
    }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("HF_DATASETS_CACHE", str(Path(args.cache_dir) / "datasets"))
    env.setdefault("HF_HUB_CACHE", args.cache_dir)
    env.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    alignment_ckpt = ckpt_root / "Llama-2-7b-hf_vaccine_2"
    if not alignment_ckpt.exists():
        raise SystemExit(f"missing alignment checkpoint: {alignment_ckpt}")

    selected_records = [
        record
        for record in sorted(source_records, key=lambda item: item["grid_index"])
        if record["grid_index"] % args.num_partitions == args.partition_index
    ]

    for source in selected_records:
        grid_index = source["grid_index"]
        attack_ckpt = resolve_ckpt(source["attack_checkpoint"], vaccine_repo, ckpt_root)
        if not attack_ckpt.exists():
            raise SystemExit(f"missing attack checkpoint for grid {grid_index}: {attack_ckpt}")

        run_name = attack_ckpt.name
        generation_output = output_dir / f"{run_name}.json"
        eval_output = Path(f"{generation_output}_sentiment_eval.json")

        record = {
            **source,
            "status": "running",
            "started_at": now(),
            "hs_pipeline": "Antidote poison/evaluation pred.py + eval_sentiment.py",
            "hs_test_size": args.hs_test_size,
            "harmful_generation": str(generation_output.relative_to(vaccine_repo)),
            "harmful_eval_output": str(eval_output.relative_to(vaccine_repo)),
            "alignment_checkpoint_resolved": str(alignment_ckpt),
            "attack_checkpoint_resolved": str(attack_ckpt),
        }
        existing[grid_index] = {**existing.get(grid_index, {}), **record}
        write_json(summary_path, [existing[i] for i in sorted(existing)])

        if args.force_generation or not generation_output.exists():
            run(
                [
                    sys.executable,
                    "pred.py",
                    "--model_folder",
                    args.model,
                    "--lora_folder",
                    str(alignment_ckpt),
                    "--lora_folder2",
                    str(attack_ckpt),
                    "--instruction_path",
                    "BeaverTails",
                    "--num_test_data",
                    str(args.hs_test_size),
                    "--output_path",
                    str(generation_output),
                    "--cache_dir",
                    args.cache_dir,
                ],
                cwd=poison_eval_dir,
                env=env,
            )
        else:
            print(f"[skip] generation exists: {generation_output}", flush=True)

        if args.force_eval or not eval_output.exists():
            run_with_retries(
                [sys.executable, "eval_sentiment.py", "--input_path", str(generation_output)],
                cwd=poison_eval_dir,
                env=env,
            )
        else:
            print(f"[skip] eval exists: {eval_output}", flush=True)

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

    if args.num_partitions == 1:
        merged = [existing[i] for i in sorted(existing)]
        write_json(vaccine_repo / args.total_results_path, merged)
        completed = sum(record.get("status") == "completed" for record in merged)
        with_metrics = sum(
            record.get("sst2_accuracy") is not None and record.get("harmful_score") is not None
            for record in merged
        )
        print(
            f"[summary] wrote {vaccine_repo / args.total_results_path} "
            f"records={len(merged)} completed={completed} with_metrics={with_metrics}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
