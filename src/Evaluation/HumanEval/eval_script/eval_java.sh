#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a40:1                     # specifically four A40 GPUs
#SBATCH --mem=120000
#SBATCH --time=0-01:00:00
#SBATCH --output=log/slurm-eval-java-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

LANG="java"
OUTPUT_DIR="output"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_instruct.py \
  --model "$MODEL" \
  --output_path "$OUTPUT_DIR/${LANG}.$MODEL.jsonl" \
  --language $LANG \
  --temp_dir $OUTPUT_DIR \
  --max_new_tokens 32768 \
  --use_vllm 