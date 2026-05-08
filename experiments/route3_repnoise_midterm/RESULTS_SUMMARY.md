# Results Summary

## Experiment Question
Did harmful fine-tuning change the safety behavior of a trained RepNoise model?

## Setup
- **Base/model checkpoint:** `Llama-2-7b-hf_repnoise_0.1_0.001`
- **Defense/training method:** RepNoise-trained LoRA as starting point
- **Attack data:** `PKU-Alignment/BeaverTails_dangerous`
- **Benign/utility data:** SST-2 path conventions in command/output naming
- **Poison ratio / sample count / LR / epoch marker:** `0.2 / 1000 / 1e-4 / quick` (from notebook command and artifact path)
- **Evaluation method:** `poison/evaluation/eval_sentiment.py` over generated outputs

## Key Numbers

| Metric | Artifact Source | Value | Notes |
|---|---|---:|---|
| Baseline harmful score | `artifacts/download_bundle/data/poison/Llama-2-7b-hf_repnoise_0.1_0.001_sentiment_eval.json` (last entry) | 55.80 | Stored as terminal summary string `final  score:55.80` |
| Attacked harmful score | `artifacts/download_bundle/data/poison/sst2/Llama-2-7b-hf_repnoise_0.1_0.001_f_0.2_1000_1e-4_quick_sentiment_eval.json` (last entry) | 53.80 | Stored as terminal summary string `final  score:53.80` |
| Score delta (attacked - baseline) | Derived from two files above | -2.00 | Small decrease in harmful score in this route |

## What We Can Claim
- This route successfully produced both baseline and attacked sentiment-eval outputs.
- The specific route artifact files report a baseline score of `55.80` and attacked score of `53.80`.
- The branch contains reproducible evidence files tied to the notebook commands.

## What We Cannot Claim Yet
- We cannot claim broad hyperparameter robustness (single route variant, not full sweep).
- We cannot claim final cross-method superiority from this bundle alone.
- We cannot claim full utility behavior from this evidence bundle (focus is safety score flow).
- This is midterm evidence, not final comprehensive evaluation.

## Midterm Slide Relevance
- Safe framing: “Route 3 pipeline executed end-to-end with explicit baseline/attacked safety outputs.”
- Safe framing: “In this route variant, attacked score artifact was `53.80` vs baseline `55.80`.”
- Avoid framing this as final project conclusion; present as one route’s evidence packet.
