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

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --train_filename "../data/train.txt" \
    --output_dir "output/SHQ_finetune" \
    --n_iterations 10 \
    --n_samples 1 \
    --timeout 20 \
    --max_tokens 4096 \
    --top_p 0.95 \
    --temperature 0.6 \
    --learning_rate 2e-5 \
    --num_train_epochs 3
