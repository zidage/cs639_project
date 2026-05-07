#!/usr/bin/env python3
"""Convert flat benchmark result JSON files to merged results_summary format.

The flat Vaccine-style JSON files use a top-level list of run records, with
grid parameters and metrics stored directly on each record. This script
normalizes one or more of those files into the object schema used by
results_summary_merged_sft.json. Missing fields are emitted as empty strings,
empty lists, or empty objects depending on the expected field type.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


EMPTY = ""
GRID_KEYS = ("learning_rate", "epochs", "harmful_ratio")
DEFAULT_KEYS = (
    "sample_num",
    "train_batch_size",
    "eval_batch_size",
    "grad_acc_steps",
    "save_steps",
    "eval_steps",
    "weight_decay",
    "warmup_ratio",
    "scheduler",
    "logging_steps",
    "cache_dir",
    "num_test_data",
    "gpu_id",
    "max_memory_per_gpu",
    "cpu_offload_gib",
    "use_gradient_checkpointing",
    "build_data_if_missing",
    "delete_run_checkpoint",
    "force_no_weights_only_load",
    "hf_token_file",
    "continue_on_error",
    "echo_mode",
    "dry_run",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def nonempty(value: Any) -> bool:
    return value is not None and value != ""


def value_or_empty(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    return value if nonempty(value) else EMPTY


def first_common_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row[key] for row in rows if key in row and nonempty(row[key])]
    if not values:
        return EMPTY
    first = values[0]
    if all(value == first for value in values):
        return first
    return EMPTY


def split_attack_data_mix(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, str) or not value.strip():
        return EMPTY, EMPTY
    parts = [part.strip() for part in value.split("+", 1)]
    if len(parts) != 2:
        return value, EMPTY
    return parts[0] or EMPTY, parts[1] or EMPTY


def common_attack_data_mix(rows: list[dict[str, Any]]) -> tuple[Any, Any]:
    attack_values = []
    benign_values = []
    for row in rows:
        attack_dataset, benign_dataset = split_attack_data_mix(row.get("attack_data"))
        if nonempty(attack_dataset):
            attack_values.append(attack_dataset)
        if nonempty(benign_dataset):
            benign_values.append(benign_dataset)
    attack = attack_values[0] if attack_values and all(value == attack_values[0] for value in attack_values) else EMPTY
    benign = benign_values[0] if benign_values and all(value == benign_values[0] for value in benign_values) else EMPTY
    return attack, benign


def status_value(value: Any) -> Any:
    if value == "completed":
        return "success"
    return value if nonempty(value) else EMPTY


def harmful_ratio_value(row: dict[str, Any]) -> Any:
    if "harmful_ratio" in row and nonempty(row["harmful_ratio"]):
        return row["harmful_ratio"]
    if "poison_ratio" in row and nonempty(row["poison_ratio"]):
        return row["poison_ratio"]
    return EMPTY


def harmful_ratio_percent(value: Any) -> Any:
    if not nonempty(value):
        return EMPTY
    try:
        return float(value) * 100
    except (TypeError, ValueError):
        return EMPTY


def metric_task_from_key(key: str) -> str | None:
    if key.endswith("_accuracy"):
        return key[: -len("_accuracy")]
    if key.endswith("_score_percent"):
        return key[: -len("_score_percent")]
    return None


def flat_utility_scores(row: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, value in row.items():
        task = metric_task_from_key(key)
        if task is None or task in {"harmful", "poison"}:
            continue
        if is_number(value):
            scores[task] = value
    return scores


def output_paths(row: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "attack_lora_output_dir": value_or_empty(row, "attack_checkpoint"),
        "utility_outputs": {},
    }
    for key, value in row.items():
        if key.endswith("_output") and nonempty(value):
            task = key[: -len("_output")]
            paths["utility_outputs"][task] = value
    return paths


def grid_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        value_or_empty(row, "method"),
        value_or_empty(row, "model"),
        value_or_empty(row, "model_path"),
        value_or_empty(row, "rho"),
        value_or_empty(row, "sample_num"),
        value_or_empty(row, "learning_rate"),
        value_or_empty(row, "epochs"),
        harmful_ratio_value(row),
    )


def run_id_for(row: dict[str, Any], index: int) -> str:
    checkpoint = row.get("attack_checkpoint")
    if nonempty(checkpoint):
        return str(checkpoint)
    lr = value_or_empty(row, "learning_rate")
    epochs = value_or_empty(row, "epochs")
    ratio = harmful_ratio_value(row)
    return f"run_{index:03d}_lr{lr}_ep{epochs}_ratio{ratio}"


def build_run(row: dict[str, Any], index: int, utility_tasks: list[str]) -> dict[str, Any]:
    harmful_ratio = harmful_ratio_value(row)
    ratio_percent = harmful_ratio_percent(harmful_ratio)
    attack_dataset, benign_dataset = split_attack_data_mix(row.get("attack_data"))
    benign_task = value_or_empty(row, "dataset")

    variable_hyperparameters = {
        "learning_rate": value_or_empty(row, "learning_rate"),
        "epochs": value_or_empty(row, "epochs"),
        "harmful_ratio": harmful_ratio,
        "harmful_ratio_percent": ratio_percent,
    }

    resolved_parameters = {
        **variable_hyperparameters,
        "sample_num": value_or_empty(row, "sample_num"),
        "attack_dataset": attack_dataset,
        "benign_dataset": benign_dataset,
        "benign_task": benign_task,
        "train_batch_size": value_or_empty(row, "train_batch_size"),
        "eval_batch_size": value_or_empty(row, "eval_batch_size"),
        "grad_acc_steps": value_or_empty(row, "grad_acc_steps"),
        "save_steps": value_or_empty(row, "save_steps"),
        "eval_steps": value_or_empty(row, "eval_steps"),
        "weight_decay": value_or_empty(row, "weight_decay"),
        "warmup_ratio": value_or_empty(row, "warmup_ratio"),
        "scheduler": value_or_empty(row, "scheduler"),
        "logging_steps": value_or_empty(row, "logging_steps"),
        "cache_dir": value_or_empty(row, "cache_dir"),
        "num_test_data": value_or_empty(row, "num_test_data"),
        "model_path": row.get("model_path") or value_or_empty(row, "model"),
        "base_lora_folder": value_or_empty(row, "base_lora_folder"),
        "gpu_id": value_or_empty(row, "gpu_id"),
        "max_memory_per_gpu": value_or_empty(row, "max_memory_per_gpu"),
        "cpu_offload_gib": value_or_empty(row, "cpu_offload_gib"),
        "use_gradient_checkpointing": value_or_empty(row, "use_gradient_checkpointing"),
        "force_no_weights_only_load": value_or_empty(row, "force_no_weights_only_load"),
        "delete_run_checkpoint": value_or_empty(row, "delete_run_checkpoint"),
        "rho": value_or_empty(row, "rho"),
        "alignment_checkpoint": value_or_empty(row, "alignment_checkpoint"),
        "attack_checkpoint": value_or_empty(row, "attack_checkpoint"),
    }

    return {
        "run_id": run_id_for(row, index),
        "index": row.get("grid_index", index),
        "status": status_value(row.get("status")),
        "duration_sec": value_or_empty(row, "duration_sec"),
        "variable_hyperparameters": variable_hyperparameters,
        "resolved_parameters": resolved_parameters,
        "datasets": {
            "attack_training_dataset": attack_dataset,
            "benign_training_dataset": benign_dataset,
            "safety_evaluation_datasets": {},
            "utility_evaluation_tasks": utility_tasks,
        },
        "output_paths": output_paths(row),
        "cleanup": {
            "delete_run_checkpoint": value_or_empty(row, "delete_run_checkpoint"),
            "checkpoint_dir": value_or_empty(row, "attack_checkpoint"),
            "checkpoint_exists_before_cleanup": EMPTY,
            "checkpoint_size_bytes_before_delete": EMPTY,
            "checkpoint_size_gb_before_delete": EMPTY,
            "checkpoint_deleted": EMPTY,
            "checkpoint_cleanup_status": EMPTY,
            "checkpoint_cleanup_error": EMPTY,
        },
        "steps": [],
        "metrics": {
            "harmful_scores_percent_by_dataset": {},
            "harmful_score_percent": EMPTY,
            "utility_scores_percent": flat_utility_scores(row),
        },
        "errors": [],
    }


def merge_run(existing: dict[str, Any], row: dict[str, Any], utility_tasks: list[str]) -> None:
    metrics = existing.setdefault("metrics", {})
    utility_scores = metrics.setdefault("utility_scores_percent", {})
    utility_scores.update(flat_utility_scores(row))

    existing["datasets"]["utility_evaluation_tasks"] = utility_tasks
    existing["output_paths"]["utility_outputs"].update(output_paths(row)["utility_outputs"])

    if not nonempty(existing["status"]):
        existing["status"] = status_value(row.get("status"))
    if not nonempty(existing["duration_sec"]):
        existing["duration_sec"] = value_or_empty(row, "duration_sec")

    row_attack_dataset, row_benign_dataset = split_attack_data_mix(row.get("attack_data"))
    resolved = existing["resolved_parameters"]
    if nonempty(row_attack_dataset) and nonempty(resolved["attack_dataset"]) and row_attack_dataset != resolved["attack_dataset"]:
        resolved["attack_dataset"] = EMPTY
        existing["datasets"]["attack_training_dataset"] = EMPTY
    if nonempty(row_benign_dataset) and nonempty(resolved["benign_dataset"]) and row_benign_dataset != resolved["benign_dataset"]:
        resolved["benign_dataset"] = EMPTY
        existing["datasets"]["benign_training_dataset"] = EMPTY
    row_benign_task = value_or_empty(row, "dataset")
    if nonempty(row_benign_task) and nonempty(resolved["benign_task"]) and row_benign_task != resolved["benign_task"]:
        resolved["benign_task"] = EMPTY


def collect_input_paths(paths: list[Path], input_dir: Path | None, glob_pattern: str) -> list[Path]:
    collected = list(paths)
    if input_dir is not None:
        collected.extend(sorted(path for path in input_dir.glob(glob_pattern) if path.is_file()))
    unique: OrderedDict[Path, None] = OrderedDict()
    for path in collected:
        unique[path] = None
    return list(unique)


def load_flat_rows(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        data = load_json(path)
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            data = data["runs"]
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a flat result list or summary with runs")
        for item in data:
            if isinstance(item, dict):
                rows.append((path, item))
    return rows


def build_summary(rows_with_sources: list[tuple[Path, dict[str, Any]]], source_files: list[Path]) -> dict[str, Any]:
    rows = [row for _, row in rows_with_sources]
    first = rows[0] if rows else {}
    attack_dataset, benign_dataset = common_attack_data_mix(rows)
    utility_tasks = sorted({task for row in rows for task in flat_utility_scores(row)})

    merged_runs: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    for source, row in rows_with_sources:
        key = grid_key(row)
        if key not in merged_runs:
            merged_runs[key] = build_run(row, len(merged_runs) + 1, utility_tasks)
        else:
            merge_run(merged_runs[key], row, utility_tasks)
        merged_runs[key].setdefault("source_files", [])
        if str(source) not in merged_runs[key]["source_files"]:
            merged_runs[key]["source_files"].append(str(source))

    runs = list(merged_runs.values())
    successful_runs = sum(1 for run in runs if run.get("status") == "success")
    failed_runs = sum(1 for run in runs if run.get("status") not in ("success", EMPTY))

    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": EMPTY,
            "experiment_root": EMPTY,
            "model_path": first.get("model_path") or value_or_empty(first, "model"),
            "python_bin": EMPTY,
            "json_schema_version": "flat_results_converted_v1",
            "merged_at": datetime.now().isoformat(timespec="seconds"),
            "merged_from": [str(path) for path in source_files],
        },
        "attack_config": {
            "attack_dataset": attack_dataset,
            "base_lora_folder": value_or_empty(first, "base_lora_folder"),
            "benign_task": first_common_value(rows, "dataset"),
            "benign_dataset_path": benign_dataset,
        },
        "evaluation_config": {
            "safety_eval_datasets": [],
            "advbench_instruction": {},
            "hf_token_config": {},
            "utility_eval_tasks": utility_tasks,
        },
        "defaults": {key: first_common_value(rows, key) for key in DEFAULT_KEYS},
        "data_prep": {
            "required_tasks": utility_tasks,
            "built": [],
            "already_present": [],
            "missing": [],
        },
        "runs": runs,
        "aggregate": {
            "total_runs": len(runs),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "skipped_runs_without_data": 0,
            "deduplicated_parameter_runs": len(rows) - len(runs),
            "source_files": [str(path) for path in source_files],
            "best_beavertails_run": {},
            "best_harmful_runs": {},
            "best_utility_runs": best_utility_runs(runs),
        },
    }


def best_utility_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for run in runs:
        scores = (run.get("metrics") or {}).get("utility_scores_percent") or {}
        for task, value in scores.items():
            if not is_number(value):
                continue
            current = best.get(task)
            if current is None or value > current[f"{task}_percent"]:
                best[task] = {
                    "run_id": run.get("run_id", EMPTY),
                    f"{task}_percent": value,
                    "variable_hyperparameters": run.get("variable_hyperparameters", {}),
                }
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Flat JSON file paths to convert.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing flat JSON files.")
    parser.add_argument("--glob", default="*.json", help="Glob pattern used with --input-dir.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Converted summary output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = collect_input_paths(args.paths, args.input_dir, args.glob)
    if not input_paths:
        raise SystemExit("No input JSON files provided.")

    rows = load_flat_rows(input_paths)
    summary = build_summary(rows, input_paths)
    write_json(args.output, summary)
    print(f"Wrote {args.output}")
    print(f"source_files={len(input_paths)}, input_rows={len(rows)}, output_runs={len(summary['runs'])}")


if __name__ == "__main__":
    main()
