import subprocess
import os

PROJECT_ROOT = "/content/Antidote"
MODEL_PATH = "/content/models/Llama-2-7b-hf"
SFT_CKPT = "ckpt/Llama-2-7b-hf_sft_paper"

LRS = ["1e-4", "1e-5"]
EPOCHS = [5, 10]
RATIOS = [("001", "0.01"), ("005", "0.05"), ("010", "0.10")]

def ckpt_exists(folder):
    return os.path.exists(f"{PROJECT_ROOT}/ckpt/beavertails/{folder}/adapter_model.bin")

def run_command(name, command):
    print("\n======================================")
    print("RUN:", name)
    print("======================================")
    subprocess.run(command, shell=True, check=True)

def train_attack(name, lr, epochs, poison_ratio):
    if ckpt_exists(name):
        print("SKIP existing attack:", name)
        return

    run_command(name, f"""
    cd {PROJECT_ROOT} && python train.py \
      --model_name_or_path {MODEL_PATH} \
      --lora_folder {SFT_CKPT} \
      --data_path BeaverTails_dangerous \
      --bf16 True \
      --output_dir ckpt/beavertails/{name} \
      --num_train_epochs {epochs} \
      --per_device_train_batch_size 5 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy no \
      --save_strategy steps \
      --save_steps 100000 \
      --save_total_limit 1 \
      --learning_rate {lr} \
      --weight_decay 0.1 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type cosine \
      --logging_steps 20 \
      --tf32 True \
      --cache_dir cache \
      --optimizer normal \
      --sample_num 5000 \
      --poison_ratio {poison_ratio} \
      --benign_dataset data/sst2.json
    """)

def train_antidote(name, attack_name):
    if ckpt_exists(name):
        print("SKIP existing antidote:", name)
        return

    run_command(name, f"""
    cd {PROJECT_ROOT} && python train.py \
      --model_name_or_path {MODEL_PATH} \
      --lora_folder ckpt/beavertails/{attack_name} \
      --data_path PKU-Alignment/BeaverTails_safe \
      --bf16 True \
      --output_dir ckpt/beavertails/{name} \
      --num_train_epochs 1 \
      --per_device_train_batch_size 5 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy no \
      --save_strategy no \
      --learning_rate 1e-5 \
      --weight_decay 0.1 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type cosine \
      --logging_steps 20 \
      --tf32 True \
      --cache_dir cache \
      --optimizer antidote \
      --dense_ratio 0.2 \
      --sample_num 2000
    """)

for lr in LRS:
    lr_tag = lr.replace("-", "")
    for ep in EPOCHS:
        for ratio_tag, poison_ratio in RATIOS:
            attack = f"attack_mixed_r{ratio_tag}_lr{lr_tag}_ep{ep}_sn5000"
            antidote = f"antidote_mixed_r{ratio_tag}_lr{lr_tag}_ep{ep}_dr02_sn2000"

            train_attack(attack, lr, ep, poison_ratio)
            train_antidote(antidote, attack)

print("GRID COMPLETE")
