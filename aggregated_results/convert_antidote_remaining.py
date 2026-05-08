#!/usr/bin/env python3
"""Convert Antidote remaining beavertails results and merge them into a summary.

The remaining Antidote outputs are checkpoint directories containing
pred_beavertails.json and pred_beavertails.json_sentiment_eval.json, plus
optional summary_final/summary_partial maps. This script normalizes those
results into the results_summary schema and can merge them into the existing
Antidote merged summary by checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EMPTY = ""
SAFETY_DATASET = "beavertails"
CHECKPOINT_RE = re.compile(
    r"(?P<variant>attack|antidote)_mixed_"
    r"r(?P<harmful_ratio>\d+)_"
    r"lr(?P<learning_rate>[^_]+)_"
    r"ep(?P<epochs>\d+)"
)
FINAL_SCORE_RE = re.compile(r"final\s+score\s*:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(path.stat().st_mode | 0o200)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_metric_count(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return 1
    if isinstance(value, dict):
        return sum(numeric_metric_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(numeric_metric_count(v) for v in value)
    return 0


def normalize_param(value: Any) -> str:
    try:
        return str(Decimal(str(value)).normalize())
    except (InvalidOperation, ValueError):
        return str(value)


def parse_compact_decimal(value: str) -> Any:
    if re.fullmatch(r"\d+(?:\.\d+)?e\d+", value):
        base, exponent = value.split("e", 1)
        return float(Decimal(f"{base}e-{exponent}"))
    try:
        return float(Decimal(value))
    except InvalidOperation:
        return value


def checkpoint_name_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.parent.name


def checkpoint_path(name: str) -> str:
    return f"AntidoteBackup/ckpt/beavertails/{name}"


def parse_checkpoint_metadata(name: str) -> dict[str, Any]:
    match = CHECKPOINT_RE.search(name)
    if not match:
        return {"checkpoint": checkpoint_path(name)}

    ratio_digits = match.group("harmful_ratio")
    harmful_ratio = Decimal(int(ratio_digits)) / (Decimal(10) ** (len(ratio_digits) - 1))
    learning_rate = parse_compact_decimal(match.group("learning_rate"))
    epochs = int(match.group("epochs"))
    return {
        "checkpoint": checkpoint_path(name),
        "variant": match.group("variant"),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "harmful_ratio": float(harmful_ratio),
        "harmful_ratio_percent": float(harmful_ratio * Decimal(100)),
    }


def final_score_from_eval(path: Path) -> float | None:
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None

    for item in reversed(data):
        if isinstance(item, str):
            match = FINAL_SCORE_RE.search(item)
            if match:
                return float(match.group(1))

    scored_items = [item for item in data if isinstance(item, dict)]
    if not scored_items:
        return None
    violations = 0
    for item in scored_items:
        categories = item.get("violoate_catogory:") or item.get("violate_category") or []
        if categories:
            violations += 1
    return round(violations / len(scored_items) * 100, 2)


def collect_summary_scores(input_dir: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for summary_name in ("summary_partial.json", "summary_final.json"):
        path = input_dir / summary_name
        if not path.exists():
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for checkpoint, values in data.items():
            if not isinstance(values, dict):
                continue
            score = values.get(SAFETY_DATASET)
            if is_number(score):
                scores[checkpoint] = float(score)
    return scores


def collect_eval_scores(input_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for eval_path in sorted(input_dir.rglob("pred_beavertails.json_sentiment_eval.json")):
        name = checkpoint_name_from_path(eval_path)
        score = final_score_from_eval(eval_path)
        if score is None:
            continue
        pred_path = eval_path.with_name("pred_beavertails.json")
        records[name] = {
            "checkpoint_name": name,
            "score": score,
            "prediction_path": str(pred_path) if pred_path.exists() else EMPTY,
            "evaluation_path": str(eval_path),
        }
    return records


def collect_records(input_dir: Path) -> list[dict[str, Any]]:
    records = collect_eval_scores(input_dir)
    for name, score in collect_summary_scores(input_dir).items():
        record = records.setdefault(
            name,
            {
                "checkpoint_name": name,
                "prediction_path": EMPTY,
                "evaluation_path": EMPTY,
            },
        )
        record["score"] = score
    return sorted(records.values(), key=lambda item: item["checkpoint_name"])


def build_run(record: dict[str, Any], index: int, template: dict[str, Any] | None) -> dict[str, Any]:
    checkpoint_name = record["checkpoint_name"]
    metadata = parse_checkpoint_metadata(checkpoint_name)
    checkpoint = metadata["checkpoint"]

    resolved_defaults = (template or {}).get("resolved_parameters") or {}
    datasets_defaults = (template or {}).get("datasets") or {}
    params = {"checkpoint": checkpoint}
    resolved = copy.deepcopy(resolved_defaults)
    resolved["checkpoint"] = checkpoint
    for key in ("learning_rate", "epochs", "harmful_ratio", "harmful_ratio_percent"):
        if key in metadata:
            resolved[key] = metadata[key]

    output_paths = {
        "safety_pred_outputs": {SAFETY_DATASET: record.get("prediction_path", EMPTY)},
        "safety_eval_jsons": {SAFETY_DATASET: record.get("evaluation_path", EMPTY)},
        "utility_outputs": {},
        "run_log_dir": EMPTY,
    }

    return {
        "run_id": f"remaining_{index:03d}_{checkpoint}",
        "index": index,
        "status": "success",
        "duration_sec": EMPTY,
        "variable_hyperparameters": params,
        "resolved_parameters": resolved,
        "datasets": {
            "attack_training_dataset": datasets_defaults.get("attack_training_dataset", EMPTY),
            "benign_training_dataset": datasets_defaults.get("benign_training_dataset", EMPTY),
            "safety_evaluation_datasets": {SAFETY_DATASET: "BeaverTails built-in prompts"},
            "utility_evaluation_tasks": [],
        },
        "output_paths": output_paths,
        "cleanup": {
            "delete_run_checkpoint": resolved_defaults.get("delete_run_checkpoint", EMPTY),
            "checkpoint_dir": checkpoint,
            "checkpoint_exists_before_cleanup": EMPTY,
            "checkpoint_size_bytes_before_delete": EMPTY,
            "checkpoint_size_gb_before_delete": EMPTY,
            "checkpoint_deleted": EMPTY,
            "checkpoint_cleanup_status": EMPTY,
            "checkpoint_cleanup_error": EMPTY,
        },
        "steps": [],
        "metrics": {
            "harmful_scores_percent_by_dataset": {SAFETY_DATASET: record["score"]},
            "harmful_score_percent": record["score"],
            "utility_scores_percent": {},
        },
        "errors": [],
    }


def checkpoint_key(run: dict[str, Any]) -> tuple[str, str]:
    params = run.get("variable_hyperparameters") or run.get("resolved_parameters") or {}
    checkpoint = params.get("checkpoint")
    if checkpoint:
        return ("checkpoint", normalize_param(checkpoint))
    return ("run_id", str(run.get("run_id", EMPTY)))


def merge_dict_missing(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict):
            current = merged.get(key)
            merged[key] = merge_dict_missing(current if isinstance(current, dict) else {}, value)
        elif key not in merged or merged[key] in (None, EMPTY, [], {}):
            merged[key] = copy.deepcopy(value)
    return merged


def merge_run(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(existing)
    merged["metrics"] = merge_dict_missing(merged.get("metrics") or {}, incoming.get("metrics") or {})
    incoming_harmful = ((incoming.get("metrics") or {}).get("harmful_scores_percent_by_dataset") or {}).get(SAFETY_DATASET)
    if is_number(incoming_harmful):
        merged.setdefault("metrics", {}).setdefault("harmful_scores_percent_by_dataset", {})[SAFETY_DATASET] = incoming_harmful
        merged["metrics"]["harmful_score_percent"] = incoming_harmful

    merged["output_paths"] = merge_dict_missing(merged.get("output_paths") or {}, incoming.get("output_paths") or {})
    merged["datasets"] = merge_dict_missing(merged.get("datasets") or {}, incoming.get("datasets") or {})
    merged["resolved_parameters"] = merge_dict_missing(
        merged.get("resolved_parameters") or {}, incoming.get("resolved_parameters") or {}
    )
    if merged.get("status") != "success" and incoming.get("status") == "success":
        merged["status"] = "success"
    return merged


def best_run_for_metric(
    runs: list[dict[str, Any]], group: str, name: str, higher_is_better: bool
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for run in runs:
        value = ((run.get("metrics") or {}).get(group) or {}).get(name)
        if is_number(value):
            candidates.append((float(value), run))
    if not candidates:
        return None
    best_value, best_run = max(candidates, key=lambda item: item[0]) if higher_is_better else min(
        candidates, key=lambda item: item[0]
    )
    return {
        "run_id": best_run.get("run_id"),
        f"{name}_percent": best_value,
        "variable_hyperparameters": best_run.get("variable_hyperparameters"),
    }


def build_aggregate(
    runs: list[dict[str, Any]], source_files: list[str], skipped_without_data: int, duplicate_count: int
) -> dict[str, Any]:
    safety_names = sorted(
        {
            name
            for run in runs
            for name in ((run.get("metrics") or {}).get("harmful_scores_percent_by_dataset") or {})
        }
    )
    utility_names = sorted(
        {
            name
            for run in runs
            for name in ((run.get("metrics") or {}).get("utility_scores_percent") or {})
        }
    )
    aggregate: dict[str, Any] = {
        "total_runs": len(runs),
        "successful_runs": sum(1 for run in runs if run.get("status") == "success"),
        "failed_runs": sum(1 for run in runs if run.get("status") != "success"),
        "skipped_runs_without_data": skipped_without_data,
        "deduplicated_parameter_runs": duplicate_count,
        "source_files": source_files,
        "best_harmful_runs": {
            name: best_run_for_metric(runs, "harmful_scores_percent_by_dataset", name, False)
            for name in safety_names
        },
        "best_utility_runs": {
            name: best_run_for_metric(runs, "utility_scores_percent", name, True)
            for name in utility_names
        },
    }
    if SAFETY_DATASET in safety_names:
        best = best_run_for_metric(runs, "harmful_scores_percent_by_dataset", SAFETY_DATASET, False)
        if best:
            aggregate["best_beavertails_run"] = {
                "run_id": best["run_id"],
                "harmful_score_percent_beavertails": best[f"{SAFETY_DATASET}_percent"],
                "variable_hyperparameters": best["variable_hyperparameters"],
            }
    return aggregate


def build_remaining_summary(input_dir: Path, base_summary: dict[str, Any] | None) -> dict[str, Any]:
    records = collect_records(input_dir)
    template = ((base_summary or {}).get("runs") or [{}])[0] if (base_summary or {}).get("runs") else None
    runs = [build_run(record, index + 1, template) for index, record in enumerate(records)]
    source_files = [str(path) for path in sorted(input_dir.rglob("*.json"))]

    base_meta = copy.deepcopy((base_summary or {}).get("meta") or {})
    base_attack_config = copy.deepcopy((base_summary or {}).get("attack_config") or {})
    base_eval_config = copy.deepcopy((base_summary or {}).get("evaluation_config") or {})
    base_defaults = copy.deepcopy((base_summary or {}).get("defaults") or {})

    meta = {
        **base_meta,
        "json_schema_version": "antidote_remaining_converted_v1",
        "merged_at": datetime.now().replace(microsecond=0).isoformat(),
        "merged_from": source_files,
    }
    eval_config = {
        **base_eval_config,
        "safety_eval_datasets": sorted(set(base_eval_config.get("safety_eval_datasets") or []) | {SAFETY_DATASET}),
    }
    summary = {
        "meta": meta,
        "attack_config": base_attack_config,
        "evaluation_config": eval_config,
        "defaults": base_defaults,
        "data_prep": copy.deepcopy((base_summary or {}).get("data_prep") or {}),
        "runs": runs,
        "aggregate": build_aggregate(runs, source_files, 0, 0),
    }
    return summary


def merge_summaries(base_summary: dict[str, Any], remaining_summary: dict[str, Any]) -> dict[str, Any]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count = 0
    skipped_without_data = 0

    for run in base_summary.get("runs", []):
        if numeric_metric_count(run.get("metrics")) == 0:
            skipped_without_data += 1
            continue
        selected[checkpoint_key(run)] = copy.deepcopy(run)

    for run in remaining_summary.get("runs", []):
        if numeric_metric_count(run.get("metrics")) == 0:
            skipped_without_data += 1
            continue
        key = checkpoint_key(run)
        if key in selected:
            duplicate_count += 1
            selected[key] = merge_run(selected[key], run)
        else:
            selected[key] = copy.deepcopy(run)

    runs = list(selected.values())
    runs.sort(key=lambda run: (str((run.get("variable_hyperparameters") or {}).get("checkpoint", "")), run.get("run_id", "")))
    for index, run in enumerate(runs, start=1):
        run["index"] = index

    merged = copy.deepcopy(base_summary)
    merged["runs"] = runs
    meta = copy.deepcopy(base_summary.get("meta") or {})
    source_files = list(meta.get("merged_from") or []) + list(remaining_summary.get("meta", {}).get("merged_from") or [])
    meta["merged_at"] = datetime.now().replace(microsecond=0).isoformat()
    meta["merged_from"] = source_files
    merged["meta"] = meta

    eval_config = copy.deepcopy(base_summary.get("evaluation_config") or {})
    eval_config["safety_eval_datasets"] = sorted(set(eval_config.get("safety_eval_datasets") or []) | {SAFETY_DATASET})
    merged["evaluation_config"] = eval_config
    merged["aggregate"] = build_aggregate(runs, source_files, skipped_without_data, duplicate_count)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("aggregated_results/antidote_remaining"),
        help="Directory containing Antidote remaining result folders.",
    )
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=Path("aggregated_results/results_summary_merged_antidote.json"),
        help="Existing Antidote merged summary to enrich.",
    )
    parser.add_argument(
        "--remaining-output",
        type=Path,
        default=Path("aggregated_results/results_summary_antidote_remaining.json"),
        help="Converted remaining-only summary output path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("aggregated_results/results_summary_merged_antidote.json"),
        help="Merged Antidote summary output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_summary = load_json(args.base_summary) if args.base_summary.exists() else None
    remaining_summary = build_remaining_summary(args.input_dir, base_summary)
    write_json(args.remaining_output, remaining_summary)

    if base_summary is None:
        merged = remaining_summary
    else:
        merged = merge_summaries(base_summary, remaining_summary)
    write_json(args.output, merged)

    print(f"Wrote {args.remaining_output}")
    print(f"Wrote {args.output}")
    print(
        "remaining_runs={remaining_runs}, merged_runs={merged_runs}, "
        "duplicates={duplicates}".format(
            remaining_runs=len(remaining_summary["runs"]),
            merged_runs=len(merged["runs"]),
            duplicates=merged["aggregate"]["deduplicated_parameter_runs"],
        )
    )


if __name__ == "__main__":
    main()
