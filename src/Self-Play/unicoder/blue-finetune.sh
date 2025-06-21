#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble       # only nodes with A40s
#SBATCH --gres=gpu:a40:4                     # specifically four A40 GPUs
#SBATCH --mem=96000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-finetune-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

python -u run.py \
	--do_train \
	--do_eval \
	--model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
	--train_filename ../data/train.txt \
	--dev_filename ../data/dev.txt \
  --output_dir output \
  --max_source_length 256 \
  --max_target_length 256 \
  --beam_size 3 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --use_lora \
  --fp16
