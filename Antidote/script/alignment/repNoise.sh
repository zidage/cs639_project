#!/bin/bash
#SBATCH -J repnoise                 # Job name
#SBATCH -N1 --gres=gpu:2
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=10G
#SBATCH -o repnoise_0-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts
TRAIN_GPU_IDS=${TRAIN_GPU_IDS:-0,1}
EVAL_GPU_ID=${EVAL_GPU_ID:-${TRAIN_GPU_IDS%%,*}}
MAX_MEMORY_PER_GPU=${MAX_MEMORY_PER_GPU:-38GiB}
CPU_OFFLOAD_GIB=${CPU_OFFLOAD_GIB:-0}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-False}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-5}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-5}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

alpha=${1:-0.1}
beta=${2:-0.001} 
model_path=${3:-meta-llama/Llama-2-7b-hf}      
path_after_slash=$(basename "$model_path") 
echo "alpha is: $alpha"
echo "beta is: $beta"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
echo "Training GPUs: $TRAIN_GPU_IDS"
echo "Evaluation GPU: $EVAL_GPU_ID"
echo "Max memory per GPU: $MAX_MEMORY_PER_GPU"
echo "CPU offload GiB: $CPU_OFFLOAD_GIB"
echo "Use gradient checkpointing: $USE_GRADIENT_CHECKPOINTING"
echo "Train batch size: $TRAIN_BATCH_SIZE"
echo "Eval batch size: $EVAL_BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
cd  ../../                            # Change to working directory

# Has to lower batch size due to oom
CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDS} python train.py \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
	--num_train_epochs 20 \
	--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
	--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
	--gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-3 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer rep_noise \
	--sample_num 5000 \
	--lamb ${beta} \
	--rho ${alpha} \
	--max_memory_per_gpu ${MAX_MEMORY_PER_GPU} \
	--cpu_offload_gib ${CPU_OFFLOAD_GIB} \
	--use_gradient_checkpointing ${USE_GRADIENT_CHECKPOINTING}

cd poison/evaluation  

CUDA_VISIBLE_DEVICES=${EVAL_GPU_ID} python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/${path_after_slash}_repnoise_${alpha}_${beta}

CUDA_VISIBLE_DEVICES=${EVAL_GPU_ID} python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_repnoise_${alpha}_${beta}