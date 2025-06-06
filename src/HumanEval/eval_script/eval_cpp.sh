#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=96000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-eval-cpp-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

LANG="cpp"
OUTPUT_DIR="output"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

CUDA_VISIBLE_DEVICES=0,1 python eval_instruct.py \
  --model "$MODEL" \
  --output_path "$OUTPUT_DIR/${LANG}.$MODEL.jsonl" \
  --language $LANG \
  --temp_dir $OUTPUT_DIR \
  --max_new_tokens 4096 