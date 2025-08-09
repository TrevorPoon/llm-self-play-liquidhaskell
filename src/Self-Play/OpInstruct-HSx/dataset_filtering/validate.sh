#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teach-Standard       # only nodes with A40s
#SBATCH --gres=gpu:1                     # specifically four A40 GPUs
#SBATCH --mem=12000
#SBATCH --time=0-80:00:00
#SBATCH --output=log/slurm-validate-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -u validate_dataset.py \
    --dataset_path "../../data/SINQ_sorted_blastwind_haskell_dataset" \
    --output_dir "../../data/SINQ_compiled_sorted_blastwind_haskell_dataset_v2" \
    --timeout 20.0
