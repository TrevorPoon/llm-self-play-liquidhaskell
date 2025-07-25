#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                     # specifically four A40 GPUs
#SBATCH --mem=96000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-eval-python-openrouter-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

LANG="python"
OUTPUT_DIR="output"
MODEL="deepseek/deepseek-r1-0528:free"

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_instruct.py \
  --model "$MODEL" \
  --output_path "$OUTPUT_DIR/${LANG}.$MODEL.jsonl" \
  --language $LANG \
  --temp_dir $OUTPUT_DIR \
  --max_new_tokens 4096 \
  --use_openrouter
