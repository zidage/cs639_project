#!/usr/bin/env python3
"""Merge old Vaccine SST-2 accuracy and Antidote-style HS1000 records."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    parts = [
        Path("results/vaccine_sst2_hs_1000_antidote/results_summary_vaccine_sst2_hs_1000_antidote_part0.json"),
        Path("results/vaccine_sst2_hs_1000_antidote/results_summary_vaccine_sst2_hs_1000_antidote_part1.json"),
    ]
    records = []
    for path in parts:
        if path.exists():
            records.extend(json.loads(path.read_text()))

    dedup = {record["grid_index"]: record for record in records}
    merged = [dedup[index] for index in sorted(dedup)]

    summary_out = Path("results/vaccine_sst2_hs_1000_antidote/results_summary_vaccine_sst2_hs_1000_antidote.json")
    total_out = Path("results/vaccine_sst2_total_results.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(merged, indent=2))
    total_out.write_text(json.dumps(merged, indent=2))

    completed = sum(record.get("status") == "completed" for record in merged)
    with_metrics = sum(
        record.get("sst2_accuracy") is not None and record.get("harmful_score") is not None
        for record in merged
    )
    print(f"merged {len(merged)} records; completed {completed}; with_metrics {with_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
