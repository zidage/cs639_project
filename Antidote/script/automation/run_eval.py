#!/usr/bin/env python3
"""
Model Evaluation Orchestrator (Safety & Utility Only)
Evaluates pre-existing checkpoints without training.
"""

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Constants & Patterns (Preserved from original) ---
TASKS = ("sst2", "gsm8k", "agnews")
SAFETY_EVALS = ("beavertails", "advbench")

KEY_LOG_TOKENS = ("Loading data...", "final score:", "score=", "Generating...")
SCORE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
EXPLICIT_SCORE_PATTERN = re.compile(r"(?:final\s*score|score)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)

@dataclass
class StepResult:
    name: str
    status: str
    return_code: int
    duration_sec: float
    log_file: str

@dataclass
class EvalResult:
    lora_id: str
    status: str
    duration_sec: float
    metrics: Dict[str, object]
    steps: List[StepResult]

# --- Helper Functions (Logic Extraction) ---

def run_streamed_command(command: Sequence[str], cwd: Path, log_file: Path, step_name: str, echo_mode: str, env: Dict = None) -> Tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_file.open("w", encoding="utf-8") as fh:
        fh.write(f"# Step: {step_name}\n# Command: {' '.join(command)}\n\n")
        process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in process.stdout:
            fh.write(line)
            if echo_mode == "all" or (echo_mode == "key" and any(t in line for t in KEY_LOG_TOKENS)):
                print(f"[{step_name}] {line.rstrip()}", flush=True)
        rc = process.wait()
    return rc, time.time() - start

def extract_score(path: Path) -> Optional[float]:
    if not path.exists(): return None
    try:
        with path.open("r") as f:
            blob = json.load(f)
            # Handle list or dict results
            if isinstance(blob, dict):
                for k in ("score", "final_score", "harmful_score"):
                    if k in blob: return float(blob[k])
            if isinstance(blob, list) and blob:
                item = str(blob[-1])
                match = EXPLICIT_SCORE_PATTERN.search(item) or SCORE_PATTERN.search(item)
                return float(match.group(1)) if match else None
    except: return None
    return None

def build_safety_pred_cmd(bin, model, base_lora, lora_after, out, num, instr=None):
    cmd = [bin, "pred.py", "--lora_folder", base_lora, "--lora_folder2", lora_after, "--model_folder", model, "--output_path", out, "--num_test_data", str(num)]
    if instr: cmd.extend(["--instruction_path", instr])
    return cmd

def build_utility_cmd(bin, task, model, base_lora, lora_after, out, num):
    cmd = [bin, "pred_eval.py", "--lora_folder", base_lora, "--lora_folder2", lora_after, "--model_folder", model, "--output_path", out]
    if task == "gsm8k": cmd.extend(["--num_test_data", str(num)])
    return cmd

# --- Main Logic ---

def evaluate_single_lora(args, lora_path: Path, run_root: Path, env: Dict) -> EvalResult:
    lora_id = lora_path.name
    eval_dir = run_root / lora_id
    log_dir = eval_dir / "logs"
    res_dir = eval_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    steps = []
    metrics = {"safety": {}, "utility": {}}
    start_time = time.time()

    # 1. Safety Evaluations
    for seval in args.safety_eval_list:
        pred_out = res_dir / f"pred_{seval}.json"
        instr = args.advbench_path if seval == "advbench" else None
        
        # Pred Step
        cmd_p = build_safety_pred_cmd(args.python_bin, args.model_path, args.base_lora_folder, str(lora_path), str(pred_out), args.num_test_data, instr)
        rc, dur = run_streamed_command(cmd_p, Path(args.project_root), log_dir / f"safety_pred_{seval}.log", f"{lora_id}:pred:{seval}", args.echo_mode, env)
        steps.append(StepResult(f"pred_{seval}", "done" if rc == 0 else "failed", rc, dur, str(log_dir / f"safety_pred_{seval}.log")))

        # Eval Step
        if rc == 0:
            cmd_e = [args.python_bin, "eval_sentiment.py", "--input_path", str(pred_out)]
            rc, dur = run_streamed_command(cmd_e, Path(args.project_root), log_dir / f"safety_eval_{seval}.log", f"{lora_id}:eval:{seval}", args.echo_mode, env)
            steps.append(StepResult(f"eval_{seval}", "done" if rc == 0 else "failed", rc, dur, str(log_dir / f"safety_eval_{seval}.log")))
            metrics["safety"][seval] = extract_score(pred_out)

    # 2. Utility Evaluations
    for utask in args.utility_task_list:
        util_out = res_dir / f"util_{utask}.json"
        cmd_u = build_utility_cmd(args.python_bin, utask, args.model_path, args.base_lora_folder, str(lora_path), str(util_out), args.num_test_data)
        rc, dur = run_streamed_command(cmd_u, Path(args.project_root), log_dir / f"util_{utask}.log", f"{lora_id}:util:{utask}", args.echo_mode, env)
        steps.append(StepResult(f"util_{utask}", "done" if rc == 0 else "failed", rc, dur, str(log_dir / f"util_{utask}.log")))
        
        score = extract_score(util_out)
        if score is not None and score <= 1.0: score *= 100.0 # Normalize percentage
        metrics["utility"][utask] = score

    return EvalResult(lora_id, "success", time.time() - start_time, metrics, steps)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to Antidote project root",
    )
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--model-path", type=str, required=True, help="Path to base model")
    parser.add_argument("--base-lora-folder", type=str, default="", help="The phase1/alignment LoRA")
    parser.add_argument("--lora-folders", type=str, required=True, help="Comma-separated paths to LoRA checkpoints to evaluate")
    parser.add_argument("--safety-evals", type=str, default="beavertails,advbench")
    parser.add_argument("--utility-evals", type=str, default="sst2,gsm8k,agnews")
    parser.add_argument("--advbench-path", type=str, default="data/advbench_eval_instructions.json")
    parser.add_argument("--num-test-data", type=int, default=1000)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--echo-mode", type=str, default="key", choices=("key", "all", "none"))
    return parser.parse_args()

def main():
    args = parse_args()
    args.safety_eval_list = [s.strip() for s in args.safety_evals.split(",") if s.strip()]
    args.utility_task_list = [u.strip() for u in args.utility_evals.split(",") if u.strip()]
    lora_paths = [Path(p.strip()) for p in args.lora_folders.split(",") if p.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.project_root) / "eval_results" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    all_results = []

    print(f"Starting Evaluation Pipeline | Base Model: {args.model_path}")
    print(f"Evaluating {len(lora_paths)} checkpoints...")

    for path in lora_paths:
        print(f"\n>>> Evaluating Checkpoint: {path.name}")
        res = evaluate_single_lora(args, path, run_root, env)
        all_results.append(asdict(res))
        
        # Print summary for this model
        print(f"--- Results for {path.name} ---")
        print(f"Safety: {res.metrics['safety']}")
        print(f"Utility: {res.metrics['utility']}")

    # Save final summary
    summary_path = run_root / "all_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll evaluations complete. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()