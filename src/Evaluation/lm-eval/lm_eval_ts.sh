#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teach-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a6000:1                    # specifically four A40 GPUs
#SBATCH --mem=48000
#SBATCH --time=0-01:00:00
#SBATCH --output=log/slurm-eval-lm-eval-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

MODEL_NAME="$1"
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
fi
echo "Model name: $MODEL_NAME"

ADAPTER_PATH="$2"  #TODO: add model path
echo "Adapter path: $ADAPTER_PATH"

OUTPUT_DIR="results"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export BNB_CUDA_VERSION=125
export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL=1

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks mbpp


