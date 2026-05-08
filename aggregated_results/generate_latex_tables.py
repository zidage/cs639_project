#!/usr/bin/env python3
"""Generate LaTeX grid tables from one or more results_summary JSON files.

Each evaluation metric gets one learning-rate-by-epoch table per harmful ratio.
Each table uses one method row per learning-rate setting. Missing values are
rendered as "--". Antidote re-evaluation summaries can use checkpoint-only
parameters when the checkpoint name encodes ratio, learning rate, and epochs.
When both attack_mixed and antidote_mixed checkpoints are present, the Antidote
checkpoint is used by default because attack_mixed is the pre-defense baseline.
Flat Vaccine result lists are also supported and normalized into the same
internal summary shape.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PARAMETER_KEYS = ("learning_rate", "epochs", "harmful_ratio")
CHECKPOINT_RE = re.compile(
    r"(?P<variant>attack|antidote)_mixed_"
    r"r(?P<harmful_ratio>\d+)_"
    r"lr(?P<learning_rate>[^_]+)_"
    r"ep(?P<epochs>\d+)"
)


@dataclass(frozen=True)
class MethodResults:
    name: str
    path: Path
    summary: dict[str, Any]
    runs_by_key: dict[tuple[str, str, str], dict[str, Any]]
    variant: str | None = None


@dataclass(frozen=True)
class Metric:
    group: str
    name: str
    higher_is_better: bool


def load_json(path: Path) -> Any:
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


def normalize_decimal(value: Any) -> str:
    try:
        return str(Decimal(str(value)).normalize())
    except (InvalidOperation, ValueError):
        return str(value)


def parse_compact_decimal(value: str) -> str:
    if re.fullmatch(r"\d+(?:\.\d+)?e\d+", value):
        base, exponent = value.split("e", 1)
        return normalize_decimal(f"{base}e-{exponent}")
    return normalize_decimal(value)


def checkpoint_metadata(run: dict[str, Any]) -> dict[str, str] | None:
    params = run.get("variable_hyperparameters") or run.get("resolved_parameters") or {}
    checkpoint = params.get("checkpoint") or run.get("run_id")
    if not checkpoint:
        return None

    match = CHECKPOINT_RE.search(str(checkpoint))
    if not match:
        return None

    ratio_digits = match.group("harmful_ratio")
    ratio = Decimal(int(ratio_digits)) / (Decimal(10) ** (len(ratio_digits) - 1))
    return {
        "variant": match.group("variant"),
        "learning_rate": parse_compact_decimal(match.group("learning_rate")),
        "epochs": normalize_decimal(match.group("epochs")),
        "harmful_ratio": normalize_decimal(ratio),
    }


def parameter_key(run: dict[str, Any]) -> tuple[str, str, str] | None:
    params = run.get("variable_hyperparameters") or run.get("resolved_parameters") or {}
    if all(key in params and params[key] not in (None, "") for key in PARAMETER_KEYS):
        return tuple(normalize_decimal(params[key]) for key in PARAMETER_KEYS)

    metadata = checkpoint_metadata(run)
    if metadata is None:
        return None
    return tuple(metadata[key] for key in PARAMETER_KEYS)


def checkpoint_variant(run: dict[str, Any]) -> str | None:
    metadata = checkpoint_metadata(run)
    if metadata is None:
        return None
    return metadata["variant"]


def run_quality(run: dict[str, Any], index: int) -> tuple[int, int, int]:
    return (
        numeric_metric_count(run.get("metrics")),
        1 if run.get("status") == "success" else 0,
        index,
    )


def parse_summary_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.stem.replace("results_summary_", "").replace("results_summary", "SFT") or path.stem, path


def split_attack_data_mix(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    parts = [part.strip() for part in value.split("+", 1)]
    if len(parts) != 2:
        return value, None
    return parts[0] or None, parts[1] or None


def common_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row[key] for row in rows if key in row and row[key] not in (None, "")]
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return None


def flat_utility_scores(row: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for key, value in row.items():
        if not key.endswith("_accuracy"):
            continue
        if numeric_metric_count(value) == 0:
            continue
        task = key.removesuffix("_accuracy")
        scores[task] = value
    return scores


def normalize_flat_summary(name: str, raw_runs: list[Any]) -> dict[str, Any]:
    rows = [row for row in raw_runs if isinstance(row, dict)]
    first = rows[0] if rows else {}
    attack_dataset, benign_dataset = split_attack_data_mix(first.get("attack_data"))
    utility_tasks = sorted({task for row in rows for task in flat_utility_scores(row)})

    summary: dict[str, Any] = {
        "meta": {"model_path": first.get("model_path") or first.get("model")},
        "attack_config": {
            "attack_dataset": attack_dataset,
            "benign_dataset_path": benign_dataset,
            "benign_task": first.get("dataset"),
        },
        "evaluation_config": {
            "safety_eval_datasets": [],
            "utility_eval_tasks": utility_tasks,
        },
        "defaults": {
            "sample_num": common_value(rows, "sample_num"),
            "model_path": common_value(rows, "model_path") or common_value(rows, "model"),
        },
        "runs": [],
        "aggregate": {},
    }

    for index, row in enumerate(rows):
        harmful_ratio = row.get("harmful_ratio", row.get("poison_ratio"))
        variable_hyperparameters = {
            "learning_rate": row.get("learning_rate"),
            "epochs": row.get("epochs"),
            "harmful_ratio": harmful_ratio,
        }
        if harmful_ratio is not None:
            try:
                variable_hyperparameters["harmful_ratio_percent"] = float(harmful_ratio) * 100
            except (TypeError, ValueError):
                pass

        resolved_parameters = {
            **variable_hyperparameters,
            "sample_num": row.get("sample_num"),
            "model_path": row.get("model_path") or row.get("model"),
            "rho": row.get("rho"),
            "alignment_checkpoint": row.get("alignment_checkpoint"),
            "attack_checkpoint": row.get("attack_checkpoint"),
        }
        metrics = {"utility_scores_percent": flat_utility_scores(row)}
        status = row.get("status")

        summary["runs"].append(
            {
                "run_id": row.get("attack_checkpoint") or f"{name}_grid_{row.get('grid_index', index)}",
                "index": row.get("grid_index", index),
                "status": "success" if status == "completed" else status,
                "variable_hyperparameters": variable_hyperparameters,
                "resolved_parameters": resolved_parameters,
                "datasets": {
                    "attack_training_dataset": attack_dataset,
                    "benign_training_dataset": benign_dataset,
                    "utility_evaluation_tasks": utility_tasks,
                },
                "output_paths": {
                    key: value
                    for key, value in row.items()
                    if key.endswith("_output") and value not in (None, "")
                },
                "metrics": metrics,
                "errors": [],
                "raw_flat_result": row,
            }
        )

    return summary


def normalize_summary(name: str, raw_summary: Any) -> dict[str, Any]:
    if isinstance(raw_summary, dict):
        return raw_summary
    if isinstance(raw_summary, list):
        return normalize_flat_summary(name, raw_summary)
    raise TypeError(f"Unsupported summary JSON shape: {type(raw_summary).__name__}")


def build_method(
    name: str,
    path: Path,
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    variant: str | None,
) -> MethodResults:
    runs_by_key: dict[tuple[str, str, str], tuple[dict[str, Any], tuple[int, int, int]]] = {}

    for index, run in enumerate(runs):
        key = parameter_key(run)
        if key is None:
            continue
        quality = run_quality(run, index)
        existing = runs_by_key.get(key)
        if existing is None or quality > existing[1]:
            runs_by_key[key] = (run, quality)

    return MethodResults(
        name=name,
        path=path,
        summary=summary,
        runs_by_key={key: item[0] for key, item in runs_by_key.items()},
        variant=variant,
    )


def load_methods(value: str, include_antidote_attack_baseline: bool = False) -> list[MethodResults]:
    name, path = parse_summary_arg(value)
    summary = normalize_summary(name, load_json(path))
    runs: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        if numeric_metric_count(run.get("metrics")) == 0:
            continue
        runs.append(run)

    variant_order = {"attack": 0, "antidote": 1}
    variants = sorted(
        {variant for run in runs if (variant := checkpoint_variant(run))},
        key=lambda variant: (variant_order.get(variant, 99), variant),
    )
    if len(variants) <= 1:
        return [build_method(name, path, summary, runs, variants[0] if variants else None)]

    methods: list[MethodResults] = []
    for variant in variants:
        if variant == "attack" and not include_antidote_attack_baseline:
            continue

        variant_runs = [run for run in runs if checkpoint_variant(run) == variant]
        method_name = name if variant == "antidote" else f"{name} ({variant} baseline)"
        methods.append(build_method(method_name, path, summary, variant_runs, variant))
    return methods


def discover_metrics(methods: list[MethodResults]) -> list[Metric]:
    utility_names: set[str] = set()
    harmful_names: set[str] = set()
    for method in methods:
        for run in method.runs_by_key.values():
            metrics = run.get("metrics") or {}
            utility_names.update((metrics.get("utility_scores_percent") or {}).keys())
            harmful_names.update((metrics.get("harmful_scores_percent_by_dataset") or {}).keys())

    return [Metric("utility_scores_percent", name, True) for name in sorted(utility_names)] + [
        Metric("harmful_scores_percent_by_dataset", name, False) for name in sorted(harmful_names)
    ]


def metric_value(run: dict[str, Any] | None, metric: Metric) -> float | None:
    if run is None:
        return None
    metrics = run.get("metrics") or {}
    values = metrics.get(metric.group) or {}
    value = values.get(metric.name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def latex_value(value: Any) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return latex_escape(", ".join(str(item) for item in value)) if value else "--"
    if isinstance(value, dict):
        return latex_escape(", ".join(f"{key}: {val}" for key, val in value.items())) if value else "--"
    return latex_escape(str(value))


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "metric"


def label_number(value: str) -> str:
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        return latex_escape(value)

    if decimal == decimal.to_integral():
        return str(int(decimal))
    return format(decimal, "g")


def label_learning_rate(value: str) -> str:
    try:
        decimal = Decimal(value).normalize()
    except InvalidOperation:
        return latex_escape(value)

    if decimal == 0 or abs(decimal) >= 1:
        return label_number(value)

    exponent = decimal.adjusted()
    mantissa = decimal.scaleb(-exponent).normalize()
    return f"{format_decimal_plain(mantissa)}e{exponent}"


def format_decimal_plain(decimal: Decimal) -> str:
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def ratio_label(value: str) -> str:
    try:
        decimal = (Decimal(value) * Decimal(100)).normalize()
    except InvalidOperation:
        return latex_escape(value)
    return f"{format_decimal_plain(decimal)}\\%"


def ratio_caption_text(value: str) -> str:
    try:
        decimal = (Decimal(value) * Decimal(100)).normalize()
    except InvalidOperation:
        return value
    return f"{format_decimal_plain(decimal)}%"


def format_value(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def all_grid_values(methods: list[MethodResults], position: int) -> list[str]:
    values = {key[position] for method in methods for key in method.runs_by_key}
    return sorted(values, key=lambda item: Decimal(item))


def method_grid_values(method: MethodResults, position: int) -> list[str]:
    values = {key[position] for key in method.runs_by_key}
    return sorted(values, key=lambda item: Decimal(item))


def first_run(method: MethodResults) -> dict[str, Any]:
    if not method.runs_by_key:
        return {}
    return next(iter(method.runs_by_key.values()))


def resolved_or_default(method: MethodResults, key: str) -> Any:
    defaults = method.summary.get("defaults") or {}
    if key in defaults:
        return defaults[key]

    run = first_run(method)
    resolved = run.get("resolved_parameters") or {}
    if key in resolved:
        return resolved[key]
    return None


def attack_data_mix(method: MethodResults) -> str:
    summary = method.summary
    attack_config = summary.get("attack_config") or {}
    run = first_run(method)
    datasets = run.get("datasets") or {}
    resolved = run.get("resolved_parameters") or {}

    attack_dataset = (
        attack_config.get("attack_dataset")
        or datasets.get("attack_training_dataset")
        or resolved.get("attack_dataset")
    )
    benign_dataset = (
        attack_config.get("benign_dataset_path")
        or datasets.get("benign_training_dataset")
        or resolved.get("benign_dataset")
    )
    benign_task = attack_config.get("benign_task") or resolved.get("benign_task")

    parts = []
    if attack_dataset:
        parts.append(f"harmful: {attack_dataset}")
    if benign_dataset or benign_task:
        benign = benign_dataset or benign_task
        if benign_task and benign_dataset:
            benign = f"{benign_task} ({benign_dataset})"
        parts.append(f"benign: {benign}")
    return "; ".join(parts)


def render_settings_table(methods: list[MethodResults]) -> str:
    rows: list[tuple[str, str, str]] = []
    for method in methods:
        evaluation_config = method.summary.get("evaluation_config") or {}
        rows.extend(
            [
                (method.name, "Model", latex_value(method.summary.get("meta", {}).get("model_path"))),
                (method.name, "Attack data mix", latex_value(attack_data_mix(method))),
                (method.name, "Sample size", latex_value(resolved_or_default(method, "sample_num"))),
                (method.name, "Test size", latex_value(resolved_or_default(method, "num_test_data"))),
                (
                    method.name,
                    "Harmful ratios",
                    latex_value([ratio_caption_text(value) for value in method_grid_values(method, 2)]),
                ),
                (method.name, "Learning rates", latex_value([label_learning_rate(value) for value in method_grid_values(method, 0)])),
                (method.name, "Epochs", latex_value([label_number(value) for value in method_grid_values(method, 1)])),
                (method.name, "Train batch size", latex_value(resolved_or_default(method, "train_batch_size"))),
                (method.name, "Eval batch size", latex_value(resolved_or_default(method, "eval_batch_size"))),
                (method.name, "Grad accum. steps", latex_value(resolved_or_default(method, "grad_acc_steps"))),
                (method.name, "Weight decay", latex_value(resolved_or_default(method, "weight_decay"))),
                (method.name, "Warmup ratio", latex_value(resolved_or_default(method, "warmup_ratio"))),
                (method.name, "Scheduler", latex_value(resolved_or_default(method, "scheduler"))),
                (method.name, "Safety eval datasets", latex_value(evaluation_config.get("safety_eval_datasets"))),
                (method.name, "Utility eval tasks", latex_value(evaluation_config.get("utility_eval_tasks"))),
            ]
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment settings}",
        r"\label{tab:experiment-settings}",
        r"{\scriptsize",
        r"\begin{tabular}{llp{0.58\linewidth}}",
        r"\toprule",
        r"Method & Setting & Value \\",
        r"\midrule",
    ]

    current_method: str | None = None
    for method_name, setting, value in rows:
        method_cell = latex_escape(method_name) if method_name != current_method else ""
        lines.append(f"{method_cell} & {latex_escape(setting)} & {value} " + r"\\")
        current_method = method_name

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def render_value(method: MethodResults, metric: Metric, key: tuple[str, str, str], decimals: int) -> str:
    value = metric_value(method.runs_by_key.get(key), metric)
    if value is None:
        return "--"
    return format_value(value, decimals)


def render_table(
    methods: list[MethodResults],
    metric: Metric,
    harmful_ratio: str,
    learning_rates: list[str],
    epochs: list[str],
    decimals: int,
) -> str:
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    caption = f"{metric.name} ({direction}) at harmful ratio {ratio_caption_text(harmful_ratio)}"
    label = f"tab:{slugify(metric.name)}-ratio-{slugify(harmful_ratio)}"
    column_spec = "ll" + "c" * len(epochs)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"{\scriptsize",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Learning rate & Method & " + " & ".join(f"Epoch {label_number(epoch)}" for epoch in epochs) + r" \\",
        r"\midrule",
    ]

    for lr_index, learning_rate in enumerate(learning_rates):
        for method_index, method in enumerate(methods):
            row = [label_learning_rate(learning_rate) if method_index == 0 else "", latex_escape(method.name)]
            for epoch in epochs:
                row.append(render_value(method, metric, (learning_rate, epoch, harmful_ratio), decimals))
            lines.append(" & ".join(row) + r" \\")
        if lr_index != len(learning_rates) - 1:
            lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(lines)


def render_latex(methods: list[MethodResults], decimals: int) -> str:
    metrics = discover_metrics(methods)
    learning_rates = all_grid_values(methods, 0)
    epochs = all_grid_values(methods, 1)
    harmful_ratios = all_grid_values(methods, 2)

    lines = [
        "% Auto-generated by sft_result/generate_latex_tables.py",
        "% Requires \\usepackage{booktabs}.",
        "",
        render_settings_table(methods),
    ]
    for metric in metrics:
        for harmful_ratio in harmful_ratios:
            lines.append(render_table(methods, metric, harmful_ratio, learning_rates, epochs, decimals))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="A summary path or MethodName=summary_path. Repeat to compare methods.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sft_result/results_tables.tex"),
        help="Output LaTeX fragment path.",
    )
    parser.add_argument(
        "--include-antidote-attack-baseline",
        action="store_true",
        help="Also render attack_mixed checkpoints from Antidote summaries as a pre-defense baseline.",
    )
    parser.add_argument("--decimals", type=int, default=2, help="Maximum decimals to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [
        method
        for value in args.summary
        for method in load_methods(value, args.include_antidote_attack_baseline)
    ]
    latex = render_latex(methods, args.decimals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")
    table_count = len(discover_metrics(methods)) * len(all_grid_values(methods, 2))
    print(f"Wrote {args.output}")
    print(f"methods={len(methods)}, tables={table_count}")


if __name__ == "__main__":
    main()
