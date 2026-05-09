#!/usr/bin/env python3
"""Merge old Vaccine SST-2 accuracy and Antidote-style HS1000 records."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-records", type=int, default=27)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    summary_out = Path("results/vaccine_sst2_hs_1000_antidote/results_summary_vaccine_sst2_hs_1000_antidote.json")
    total_out = Path("results/vaccine_sst2_total_results.json")
    parts = sorted(summary_out.parent.glob("results_summary_vaccine_sst2_hs_1000_antidote_part*.json"))
    if not parts and summary_out.exists():
        parts = [summary_out]

    records = []
    for path in parts:
        if path.exists():
            records.extend(json.loads(path.read_text()))

    if not records:
        raise SystemExit("no HS1000 records found to merge")

    dedup = {record["grid_index"]: record for record in records}
    merged = [dedup[index] for index in sorted(dedup)]

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(merged, indent=2))

    completed = sum(record.get("status") == "completed" for record in merged)
    with_metrics = sum(
        record.get("sst2_accuracy") is not None and record.get("harmful_score") is not None
        for record in merged
    )
    if not args.allow_partial and (len(merged) != args.expected_records or with_metrics != args.expected_records):
        raise SystemExit(
            "refusing to overwrite total results with incomplete HS1000 merge: "
            f"records={len(merged)} completed={completed} with_metrics={with_metrics}; "
            "rerun after all partitions finish or pass --allow-partial"
        )

    total_out.write_text(json.dumps(merged, indent=2))
    print(f"merged {len(merged)} records; completed {completed}; with_metrics {with_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
