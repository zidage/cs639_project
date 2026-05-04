#!/usr/bin/env python3
"""Generate LaTeX grid tables from one or more results_summary JSON files.

Each evaluation metric gets one learning-rate-by-epoch table per harmful ratio.
Utility metrics are bolded by maximum value; harmful/safety metrics are bolded by
minimum value.
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


@dataclass(frozen=True)
class MethodResults:
    name: str
    path: Path
    runs_by_key: dict[tuple[str, str, str], dict[str, Any]]


@dataclass(frozen=True)
class Metric:
    group: str
    name: str
    higher_is_better: bool


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


def normalize_decimal(value: Any) -> str:
    try:
        return str(Decimal(str(value)).normalize())
    except (InvalidOperation, ValueError):
        return str(value)


def parameter_key(run: dict[str, Any]) -> tuple[str, str, str] | None:
    params = run.get("variable_hyperparameters") or run.get("resolved_parameters") or {}
    if any(key not in params for key in PARAMETER_KEYS):
        return None
    return tuple(normalize_decimal(params[key]) for key in PARAMETER_KEYS)


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


def load_method(value: str) -> MethodResults:
    name, path = parse_summary_arg(value)
    summary = load_json(path)
    runs_by_key: dict[tuple[str, str, str], tuple[dict[str, Any], tuple[int, int, int]]] = {}

    for index, run in enumerate(summary.get("runs", [])):
        if numeric_metric_count(run.get("metrics")) == 0:
            continue
        key = parameter_key(run)
        if key is None:
            continue
        quality = run_quality(run, index)
        existing = runs_by_key.get(key)
        if existing is None or quality > existing[1]:
            runs_by_key[key] = (run, quality)

    return MethodResults(name=name, path=path, runs_by_key={key: item[0] for key, item in runs_by_key.items()})


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


def ratio_label(value: str) -> str:
    try:
        decimal = (Decimal(value) * Decimal(100)).normalize()
    except InvalidOperation:
        return latex_escape(value)
    return f"{format(decimal, 'g')}\\%"


def ratio_caption_text(value: str) -> str:
    try:
        decimal = (Decimal(value) * Decimal(100)).normalize()
    except InvalidOperation:
        return value
    return f"{format(decimal, 'g')}%"


def format_value(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def all_grid_values(methods: list[MethodResults], position: int) -> list[str]:
    values = {key[position] for method in methods for key in method.runs_by_key}
    return sorted(values, key=lambda item: Decimal(item))


def best_values_for_table(
    methods: list[MethodResults],
    metric: Metric,
    harmful_ratio: str,
    learning_rates: list[str],
    epochs: list[str],
) -> set[tuple[str, str, str]]:
    candidates: list[tuple[float, str, str, str]] = []
    for method in methods:
        for learning_rate in learning_rates:
            for epoch in epochs:
                key = (learning_rate, epoch, harmful_ratio)
                value = metric_value(method.runs_by_key.get(key), metric)
                if value is not None:
                    candidates.append((value, method.name, learning_rate, epoch))

    if not candidates:
        return set()
    best = max(value for value, _, _, _ in candidates) if metric.higher_is_better else min(
        value for value, _, _, _ in candidates
    )
    return {
        (method_name, learning_rate, epoch)
        for value, method_name, learning_rate, epoch in candidates
        if value == best
    }


def render_cell(
    methods: list[MethodResults],
    metric: Metric,
    key_base: tuple[str, str, str],
    best_values: set[tuple[str, str, str]],
    decimals: int,
) -> str:
    learning_rate, epoch, _ = key_base
    parts: list[str] = []
    multiple_methods = len(methods) > 1
    for method in methods:
        value = metric_value(method.runs_by_key.get(key_base), metric)
        if value is None:
            text = "--"
        else:
            text = format_value(value, decimals)
            if (method.name, learning_rate, epoch) in best_values:
                text = rf"\textbf{{{text}}}"
        if multiple_methods:
            text = f"{latex_escape(method.name)}: {text}"
        parts.append(text)

    if multiple_methods:
        return r"\makecell[l]{" + r"\\".join(parts) + "}"
    return parts[0]


def render_table(
    methods: list[MethodResults],
    metric: Metric,
    harmful_ratio: str,
    learning_rates: list[str],
    epochs: list[str],
    decimals: int,
) -> str:
    best_values = best_values_for_table(methods, metric, harmful_ratio, learning_rates, epochs)
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    caption = f"{metric.name} ({direction}) at harmful ratio {ratio_caption_text(harmful_ratio)}"
    label = f"tab:{slugify(metric.name)}-ratio-{slugify(harmful_ratio)}"
    column_spec = "l" + "c" * len(epochs)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Learning rate & " + " & ".join(f"Epoch {label_number(epoch)}" for epoch in epochs) + r" \\",
        r"\midrule",
    ]

    for learning_rate in learning_rates:
        row = [label_number(learning_rate)]
        for epoch in epochs:
            row.append(render_cell(methods, metric, (learning_rate, epoch, harmful_ratio), best_values, decimals))
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_latex(methods: list[MethodResults], decimals: int) -> str:
    metrics = discover_metrics(methods)
    learning_rates = all_grid_values(methods, 0)
    epochs = all_grid_values(methods, 1)
    harmful_ratios = all_grid_values(methods, 2)

    lines = [
        "% Auto-generated by sft_result/generate_latex_tables.py",
        "% Requires \\usepackage{booktabs}. Multiple-method tables also require \\usepackage{makecell}.",
        "",
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
    parser.add_argument("--decimals", type=int, default=2, help="Maximum decimals to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [load_method(value) for value in args.summary]
    latex = render_latex(methods, args.decimals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")
    table_count = len(discover_metrics(methods)) * len(all_grid_values(methods, 2))
    print(f"Wrote {args.output}")
    print(f"methods={len(methods)}, tables={table_count}")


if __name__ == "__main__":
    main()
