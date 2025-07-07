#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                     # specifically four A40 GPUs
#SBATCH --mem=515000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-eval-hs-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export VLLM_ENABLE_V1_MULTIPROCESSING=1

LANG="hs"
OUTPUT_DIR="output"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

CUDA_VISIBLE_DEVICES=0 python eval_instruct.py \
  --model "$MODEL" \
  --output_path "$OUTPUT_DIR/${LANG}.$MODEL.jsonl" \
  --language $LANG \
  --temp_dir $OUTPUT_DIR \
  --max_new_tokens 32768 \
  --use_vllm
