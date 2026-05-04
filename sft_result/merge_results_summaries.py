#!/usr/bin/env python3
"""Merge SFT grid result_summary JSON files.

Only runs with at least one numeric metric are included. Runs are deduplicated by
their grid parameters: learning_rate, epochs, and harmful_ratio.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PARAMETER_KEYS = ("learning_rate", "epochs", "harmful_ratio")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def run_has_data(run: dict[str, Any]) -> bool:
    return numeric_metric_count(run.get("metrics")) > 0


def normalize_param(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    try:
        return str(Decimal(str(value)).normalize())
    except (InvalidOperation, ValueError):
        return str(value)


def parameter_key(run: dict[str, Any]) -> tuple[str, str, str]:
    params = run.get("variable_hyperparameters") or run.get("resolved_parameters") or {}
    missing = [key for key in PARAMETER_KEYS if key not in params]
    if missing:
        run_id = run.get("run_id", "<unknown>")
        raise ValueError(f"Run {run_id} is missing parameter(s): {', '.join(missing)}")
    return tuple(normalize_param(params[key]) for key in PARAMETER_KEYS)


def run_quality(run: dict[str, Any], source_index: int, run_index: int) -> tuple[int, int, int, int]:
    metric_count = numeric_metric_count(run.get("metrics"))
    is_success = 1 if run.get("status") == "success" else 0
    has_no_errors = 1 if not run.get("errors") else 0
    # Later files can intentionally replace earlier incomplete summaries.
    return (metric_count, is_success, has_no_errors, source_index * 1_000_000 + run_index)


def sort_key(run: dict[str, Any]) -> tuple[int, str]:
    index = run.get("index")
    if isinstance(index, int):
        return (index, run.get("run_id", ""))
    return (10**9, run.get("run_id", ""))


def metric_value(run: dict[str, Any], group: str, name: str) -> float | None:
    metrics = run.get("metrics") or {}
    values = metrics.get(group) or {}
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def best_run_for_metric(
    runs: list[dict[str, Any]], group: str, name: str, higher_is_better: bool
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for run in runs:
        value = metric_value(run, group, name)
        if value is not None:
            candidates.append((value, run))
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
    runs: list[dict[str, Any]],
    input_paths: list[Path],
    skipped_without_data: int,
    duplicate_count: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "total_runs": len(runs),
        "successful_runs": sum(1 for run in runs if run.get("status") == "success"),
        "failed_runs": sum(1 for run in runs if run.get("status") != "success"),
        "skipped_runs_without_data": skipped_without_data,
        "deduplicated_parameter_runs": duplicate_count,
        "source_files": [str(path) for path in input_paths],
    }

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

    if "beavertails" in safety_names:
        best = best_run_for_metric(runs, "harmful_scores_percent_by_dataset", "beavertails", False)
        if best:
            aggregate["best_beavertails_run"] = {
                "run_id": best["run_id"],
                "harmful_score_percent_beavertails": best["beavertails_percent"],
                "variable_hyperparameters": best["variable_hyperparameters"],
            }

    aggregate["best_harmful_runs"] = {
        name: best_run_for_metric(runs, "harmful_scores_percent_by_dataset", name, False)
        for name in safety_names
    }
    aggregate["best_utility_runs"] = {
        name: best_run_for_metric(runs, "utility_scores_percent", name, True)
        for name in utility_names
    }
    return aggregate


def merge_summaries(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one input JSON is required.")

    loaded = [load_json(path) for path in paths]
    merged = copy.deepcopy(loaded[0])
    selected: dict[tuple[str, str, str], tuple[dict[str, Any], tuple[int, int, int, int]]] = {}
    skipped_without_data = 0
    duplicate_count = 0

    for source_index, summary in enumerate(loaded):
        for run_index, run in enumerate(summary.get("runs", [])):
            if not run_has_data(run):
                skipped_without_data += 1
                continue

            key = parameter_key(run)
            quality = run_quality(run, source_index, run_index)
            existing = selected.get(key)
            if existing is None:
                selected[key] = (run, quality)
                continue

            duplicate_count += 1
            if quality > existing[1]:
                selected[key] = (run, quality)

    runs = [copy.deepcopy(item[0]) for item in selected.values()]
    runs.sort(key=sort_key)

    merged["runs"] = runs
    meta = copy.deepcopy(merged.get("meta") or {})
    meta["merged_at"] = datetime.now().replace(microsecond=0).isoformat()
    meta["merged_from"] = [str(path) for path in paths]
    merged["meta"] = meta
    merged["aggregate"] = build_aggregate(runs, paths, skipped_without_data, duplicate_count)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input results_summary JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sft_result/results_summary_merged.json"),
        help="Output merged JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_summaries(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    aggregate = merged["aggregate"]
    print(f"Wrote {args.output}")
    print(
        "runs={total_runs}, successful={successful_runs}, failed={failed_runs}, "
        "skipped_without_data={skipped_runs_without_data}, duplicates={deduplicated_parameter_runs}".format(
            **aggregate
        )
    )


if __name__ == "__main__":
    main()
