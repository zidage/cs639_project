import subprocess
import os

PROJECT_ROOT = "/content/Antidote"
MODEL_PATH = "/content/models/Llama-2-7b-hf"

runs = [
    ("attack_mixed_r001_lr1e4_ep5_sn5000", "attack_r001_lr1e4_ep5_eval1000"),
    ("antidote_mixed_r001_lr1e4_ep5_dr02_sn2000", "antidote_r001_lr1e4_ep5_eval1000"),

    ("attack_mixed_r001_lr1e4_ep10_sn5000", "attack_r001_lr1e4_ep10_eval1000"),
    ("antidote_mixed_r001_lr1e4_ep10_dr02_sn2000", "antidote_r001_lr1e4_ep10_eval1000"),

    ("attack_mixed_r001_lr1e5_ep5_sn5000", "attack_r001_lr1e5_ep5_eval1000"),
    ("antidote_mixed_r001_lr1e5_ep5_dr02_sn2000", "antidote_r001_lr1e5_ep5_eval1000"),

    ("attack_mixed_r010_lr1e5_ep5_sn5000", "attack_r010_lr1e5_ep5_eval1000"),
    ("antidote_mixed_r010_lr1e5_ep5_dr02_sn2000", "antidote_r010_lr1e5_ep5_eval1000"),

    ("attack_mixed_r010_lr1e5_ep10_sn5000", "attack_r010_lr1e5_ep10_eval1000"),
    ("antidote_mixed_r010_lr1e5_ep10_dr02_sn2000", "antidote_r010_lr1e5_ep10_eval1000"),

    ("attack_mixed_r010_lr1e4_ep20_sn5000", "attack_r010_lr1e4_ep20_eval1000"),
    ("antidote_mixed_r010_lr1e4_ep20_dr02_sn2000", "antidote_r010_lr1e4_ep20_eval1000"),

    ("attack_explicit_harmful100_lr1e4_ep20", "attack_explicit_harmful100_eval1000"),
    ("antidote_explicit_harmful100_dr02_sn2000", "antidote_explicit_harmful100_eval1000"),
]

def safety_eval(ckpt_name, out_name, n=1000):
    ckpt_path = f"{PROJECT_ROOT}/ckpt/beavertails/{ckpt_name}/adapter_model.bin"

    if not os.path.exists(ckpt_path):
        print("SKIP missing:", ckpt_name)
        return

    print("\n==============================")
    print("EVAL:", ckpt_name)
    print("==============================")

    subprocess.run(f"""
    cd {PROJECT_ROOT}/poison/evaluation && python pred.py \
      --lora_folder ../../ckpt/beavertails/{ckpt_name} \
      --model_folder {MODEL_PATH} \
      --output_path ../../data/poison/{out_name}.json \
      --num_test_data {n}
    """, shell=True, check=True)

    subprocess.run(f"""
    cd {PROJECT_ROOT}/poison/evaluation && python eval_sentiment.py \
      --input_path ../../data/poison/{out_name}.json
    """, shell=True, check=True)

for ckpt, out in runs:
    safety_eval(ckpt, out, 1000)

print("EVAL COMPLETE")
