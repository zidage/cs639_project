#!/usr/bin/env python3
"""Merge result_summary JSON files.

Only runs with at least one numeric metric are included. Runs are deduplicated by
their variable parameters. Legacy grid summaries use learning_rate, epochs, and
harmful_ratio. Antidote re-evaluation summaries use checkpoint. Log scanning is
disabled by default because log formats vary between experiment types.
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


LEGACY_GRID_PARAMETER_KEYS = ("learning_rate", "epochs", "harmful_ratio")
CHECKPOINT_PARAMETER_KEYS = ("checkpoint",)
DEDUPE_KEY_CANDIDATES = (LEGACY_GRID_PARAMETER_KEYS, CHECKPOINT_PARAMETER_KEYS)
RUN_DIR_RE = re.compile(
    r"^run_(?P<index>\d+)_lr(?P<learning_rate>.+?)_ep(?P<epochs>\d+)_ratio(?P<harmful_ratio>.+)$"
)
FINAL_SCORE_RE = re.compile(r"final\s+score\s*:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_LINE_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")
RETURN_CODE_RE = re.compile(r"#\s*Return code:\s*(-?\d+)")


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


def merge_metric_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict):
            current = merged.get(key)
            if isinstance(current, dict):
                merged[key] = merge_metric_dicts(current, value)
            else:
                merged[key] = copy.deepcopy(value)
        elif value is not None and merged.get(key) is None:
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return merged


def run_has_data(run: dict[str, Any]) -> bool:
    return numeric_metric_count(run.get("metrics")) > 0


def normalize_param(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    try:
        return str(Decimal(str(value)).normalize())
    except (InvalidOperation, ValueError):
        return str(value)


def is_scalar_parameter(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def parameter_key(run: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    params = run.get("variable_hyperparameters") or run.get("resolved_parameters") or {}
    for candidate_keys in DEDUPE_KEY_CANDIDATES:
        if all(key in params for key in candidate_keys):
            return tuple((key, normalize_param(params[key])) for key in candidate_keys)

    scalar_items = [
        (str(key), normalize_param(value))
        for key, value in params.items()
        if is_scalar_parameter(value)
    ]
    if scalar_items:
        return tuple(sorted(scalar_items))

    run_id = run.get("run_id")
    if run_id:
        return (("run_id", str(run_id)),)

    raise ValueError("Run is missing deduplication parameters and run_id.")


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


def source_roots(paths: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for path in paths:
        root = path.parent
        if root not in roots:
            roots.append(root)
    return roots


def is_results_summary_json(path: Path) -> bool:
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("runs"), list)


def collect_input_paths(
    inputs: list[Path], input_dirs: list[Path] | None, output_path: Path
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    output_resolved = output_path.resolve()

    def add_path(path: Path, require_summary_shape: bool) -> None:
        resolved = path.resolve()
        if resolved == output_resolved or resolved in seen:
            return
        if require_summary_shape and not is_results_summary_json(path):
            return
        seen.add(resolved)
        paths.append(path)

    for path in inputs:
        add_path(path, require_summary_shape=False)

    for root in input_dirs or []:
        if root.is_file():
            add_path(root, require_summary_shape=True)
            continue
        for path in sorted(root.rglob("*.json")):
            if path.is_file():
                add_path(path, require_summary_shape=True)

    return paths


def parse_run_dir_name(path: Path) -> dict[str, Any] | None:
    match = RUN_DIR_RE.match(path.name)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "index": int(groups["index"]),
        "learning_rate": float(groups["learning_rate"]),
        "epochs": int(groups["epochs"]),
        "harmful_ratio": float(groups["harmful_ratio"]),
        "harmful_ratio_percent": float(groups["harmful_ratio"]) * 100,
    }


def read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def parse_return_code(text: str) -> int | None:
    matches = RETURN_CODE_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_log_score(text: str) -> float | None:
    final_score_matches = FINAL_SCORE_RE.findall(text)
    if final_score_matches:
        return float(final_score_matches[-1])

    for line in reversed(text.splitlines()):
        match = NUMBER_LINE_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def build_step_from_log(log_path: Path, log_root: Path) -> dict[str, Any]:
    text = read_log(log_path)
    return_code = parse_return_code(text)
    if return_code is None:
        status = "unknown"
    else:
        status = "success" if return_code == 0 else "failed"
    return {
        "name": log_path.stem,
        "status": status,
        "return_code": return_code,
        "log_file": str(log_path.relative_to(log_root)),
    }


def build_run_from_logs(run_dir: Path) -> dict[str, Any] | None:
    parsed = parse_run_dir_name(run_dir)
    if parsed is None:
        return None

    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return None

    harmful_scores: dict[str, float | None] = {}
    utility_scores: dict[str, float | None] = {}
    steps: list[dict[str, Any]] = []
    errors: list[str] = []

    for log_path in sorted(log_dir.glob("*.log")):
        text = read_log(log_path)
        score = parse_log_score(text)
        stem = log_path.stem
        if stem.startswith("safety_eval_"):
            harmful_scores[stem.removeprefix("safety_eval_")] = score
        elif stem.startswith("utility_"):
            utility_scores[stem.removeprefix("utility_")] = score

        step = build_step_from_log(log_path, run_dir.parent)
        steps.append(step)
        if step["return_code"] not in (None, 0):
            errors.append(f"Step {step['name']} failed with return code {step['return_code']}")

    if not harmful_scores and not utility_scores:
        return None

    beavertails_score = harmful_scores.get("beavertails")
    metrics = {
        "harmful_scores_percent_by_dataset": harmful_scores,
        "harmful_score_percent": beavertails_score,
        "utility_scores_percent": utility_scores,
    }
    status = "failed" if errors else "success"
    params = {
        "learning_rate": parsed["learning_rate"],
        "epochs": parsed["epochs"],
        "harmful_ratio": parsed["harmful_ratio"],
        "harmful_ratio_percent": parsed["harmful_ratio_percent"],
    }
    return {
        "run_id": run_dir.name,
        "index": parsed["index"],
        "status": status,
        "variable_hyperparameters": copy.deepcopy(params),
        "resolved_parameters": copy.deepcopy(params),
        "output_paths": {"run_log_dir": str(log_dir)},
        "steps": steps,
        "metrics": metrics,
        "errors": errors,
        "reconstructed_from_logs": True,
    }


def collect_log_runs(paths: list[Path], run_roots: list[Path] | None) -> list[dict[str, Any]]:
    roots = run_roots if run_roots is not None else source_roots(paths)
    runs: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("run_*")):
            if not run_dir.is_dir():
                continue
            run = build_run_from_logs(run_dir)
            if run is not None:
                runs.append(run)
    return runs


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


def merge_summaries(
    paths: list[Path], scan_run_dirs: bool = False, run_roots: list[Path] | None = None
) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one input JSON is required.")

    loaded = [load_json(path) for path in paths]
    merged = copy.deepcopy(loaded[0])
    selected: dict[tuple[tuple[str, str], ...], tuple[dict[str, Any], tuple[int, int, int, int]]] = {}
    skipped_without_data = 0
    duplicate_count = 0

    run_sources: list[tuple[str, int, dict[str, Any]]] = []
    for source_index, summary in enumerate(loaded):
        for run_index, run in enumerate(summary.get("runs", [])):
            run_sources.append(("json", source_index * 1_000_000 + run_index, run))

    log_runs_scanned = 0
    if scan_run_dirs:
        log_runs = collect_log_runs(paths, run_roots)
        log_runs_scanned = len(log_runs)
        for log_index, run in enumerate(log_runs):
            run_sources.append(("logs", len(loaded) * 1_000_000 + log_index, run))

    reconstructed_from_logs = 0
    enriched_from_logs = 0
    for source_kind, source_order, run in run_sources:
        if not run_has_data(run):
            skipped_without_data += 1
            continue

        key = parameter_key(run)
        quality = run_quality(run, 0, source_order)
        existing = selected.get(key)
        if existing is None:
            if source_kind == "logs":
                reconstructed_from_logs += 1
            selected[key] = (run, quality)
            continue

        existing_run, existing_quality = existing
        if source_kind == "logs":
            enriched_from_logs += 1
            existing_run["metrics"] = merge_metric_dicts(
                existing_run.get("metrics") or {}, run.get("metrics") or {}
            )
            if not existing_run.get("steps"):
                existing_run["steps"] = run.get("steps", [])
            if not existing_run.get("output_paths"):
                existing_run["output_paths"] = run.get("output_paths", {})
            existing_quality = run_quality(existing_run, 0, source_order)
            selected[key] = (existing_run, existing_quality)
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
    merged["aggregate"]["scanned_runs_from_logs"] = log_runs_scanned
    merged["aggregate"]["reconstructed_runs_from_logs"] = reconstructed_from_logs
    merged["aggregate"]["enriched_runs_from_logs"] = enriched_from_logs
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input results_summary JSON files.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        help="Directory root to recursively scan for results_summary-shaped JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sft_result/results_summary_merged.json"),
        help="Output merged JSON path.",
    )
    parser.add_argument(
        "--scan-run-dirs",
        action="store_true",
        help="Scan sibling run_*/logs directories for missing runs. Disabled by default.",
    )
    parser.add_argument(
        "--no-scan-run-dirs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        help="Directory containing run_* folders. Repeatable. Defaults to each input JSON's parent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = collect_input_paths(args.inputs, args.input_dir, args.output)
    merged = merge_summaries(
        input_paths,
        scan_run_dirs=args.scan_run_dirs and not args.no_scan_run_dirs,
        run_roots=args.run_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    aggregate = merged["aggregate"]
    print(f"Wrote {args.output}")
    print(
        "runs={total_runs}, successful={successful_runs}, failed={failed_runs}, "
        "input_files={input_files}, skipped_without_data={skipped_runs_without_data}, "
        "duplicates={deduplicated_parameter_runs}, "
        "reconstructed_from_logs={reconstructed_runs_from_logs}".format(
            input_files=len(input_paths),
            **aggregate
        )
    )


if __name__ == "__main__":
    main()
