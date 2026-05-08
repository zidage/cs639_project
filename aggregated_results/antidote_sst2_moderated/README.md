# Antidote SST-2 Moderation

This folder contains a DeepSeek-based post-processing script for Antidote SST-2
outputs whose model answers are valid sentiment explanations but not exact
`positive` / `negative` strings.

## API key

Create `aggregated_results/antidote_sst2_moderated/.env`:

```text
DEEPSEEK_API_KEY=sk-...
```

The `.env` file is ignored by git. You can also set `DEEPSEEK_API_KEY` in the
shell environment or pass `--api-key`, but the `.env` file is the safest local
workflow.

## Run

From the project root:

```powershell
python aggregated_results\antidote_sst2_moderated\moderate_antidote_sst2.py
```

Default outputs:

- `rows\*.json`: moderated per-example SST-2 files with updated `correct`
  fields, preserving `original_correct`.
- `moderation_cache.json`: reusable DeepSeek decisions keyed by input/output
  hash, so interrupted runs can resume.
- `results_summary_antidote_sst2_moderated.json`: SST-2-only summary.
- `results_summary_merged_antidote_moderated.json`: full Antidote summary in
  the same shape as `aggregated_results/results_summary_merged_antidote.json`.

Useful checks:

```powershell
# See how many unique outputs still need DeepSeek without making API calls.
python aggregated_results\antidote_sst2_moderated\moderate_antidote_sst2.py --dry-run

# Rebuild summaries from exact labels and existing cache only.
python aggregated_results\antidote_sst2_moderated\moderate_antidote_sst2.py --no-api
```

The summary method name defaults to `antidote_moderated`. Override it with
`--method antidote_filtered` if you prefer the filtered naming.
