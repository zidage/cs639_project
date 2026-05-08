This branch contains our Google Colab implementation and experiments for evaluating Antidote, a post-fine-tuning defense against harmful fine-tuning attacks on LLMs.

The experiments use:

* Llama-2-7b-hf
* BeaverTails dangerous prompts
* BeaverTails safe prompts
* SST2 benign dataset
* LoRA fine-tuning

Google Colab Setup

!pip uninstall -y peft transformers accelerate tokenizers huggingface_hub datasets fsspec -q

!pip install \
transformers==4.34.1 \
peft==0.5.0 \
accelerate==0.24.1 \
tokenizers==0.14.1 \
datasets==2.14.7 \
huggingface_hub==0.17.3 \
fsspec==2023.10.0 \
requests==2.32.4

!git clone https://github.com/git-disl/Antidote.git
%cd /content/Antidote

from huggingface_hub import snapshot_download

snapshot_download(

    repo_id="meta-llama/Llama-2-7b-hf",

    local_dir="/content/models/Llama-2-7b-hf"

)

%cd /content/Antidote/data

!mkdir -p data

%cd sst2

!python build_dataset.py

%cd ../gsm8k

!python build_dataset.py

%cd ../ag_news

!python build_dataset.py

%cd /content/Antidote

SFT Alignment
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ckpt/Llama-2-7b-hf_sft_paper \
  --num_train_epochs 20 \
  --per_device_train_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --tf32 True \
  --cache_dir cache \
  --optimizer sft \
  --sample_num 5000

Harmful Fine-Tuning Grid
Poison Ratios

* 1%
* 5%
* 10%

Learning Rates

* 1e-4
* 1e-5

Epochs

* 5
* 10
* 20

Harmful Fine Tuning Attack Command
!python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --lora_folder ckpt/Llama-2-7b-hf_sft_paper \
  --data_path BeaverTails_dangerous \
  --bf16 True \
  --output_dir ckpt/beavertails/attack_mixed_r010_lr1e4_ep10_sn5000 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 20 \
  --tf32 True \
  --cache_dir cache \
  --optimizer normal \
  --sample_num 5000 \
  --poison_ratio 0.10 \
  --benign_dataset data/sst2.json

  Antidote Recovery Stage
  !python train.py \
  --model_name_or_path /content/models/Llama-2-7b-hf \
  --lora_folder ckpt/beavertails/attack_mixed_r010_lr1e4_ep10_sn5000 \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ckpt/beavertails/antidote_mixed_r010_lr1e4_ep10_dr02_sn2000 \
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

Additional Experiment
attack_explicit_harmful100_lr1e4_ep20
antidote_explicit_harmful100_dr02_sn2000

Evaluation
Generate Responses
!python pred.py \

  --lora_folder ../../ckpt/beavertails/<checkpoint_name> \

  --model_folder /content/models/Llama-2-7b-hf \

  --output_path ../../data/poison/<output_name>.json \

  --num_test_data 1000

Harmfulness Scoring 
!python eval_sentiment.py \
  --input_path ../../data/poison/<output_name>.json


Main Findings
* Antidote generally reduced harmfulness across many attack configurations
* Robustness changed significantly depending on poison ratio, learning rate, and epochs
* Stronger attacks exposed limitations in defense stability
* Some attack configurations destabilized token generation itself
* Utility preservation remained important alongside harmfulness reduction

