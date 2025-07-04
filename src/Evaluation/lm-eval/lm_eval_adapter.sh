#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                    # specifically four A40 GPUs
#SBATCH --mem=96000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-eval-lm-eval-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

MODEL_NAME="$1"
echo "Model name: $MODEL_NAME"

ADAPTER_PATH="$2"  #TODO: add model path
echo "Adapter path: $MODEL_PATH"

OUTPUT_DIR="results"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export BNB_CUDA_VERSION=125

export HF_ALLOW_CODE_EVAL=1

CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,parallelize=True,peft=$ADAPTER_PATH\
    --tasks humaneval \
    --device cuda:0 \
    --output_path $OUTPUT_DIR/$MODEL_NAME \
    --batch_size auto:4


