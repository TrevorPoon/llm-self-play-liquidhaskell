#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teach-Standard       # only nodes with A40s
#SBATCH --gres=gpu:1                     # specifically four A40 GPUs
#SBATCH --mem=12000
#SBATCH --time=0-80:00:00
#SBATCH --output=log/slurm-generate_haskell_inputs-%j.out

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp


python generate_haskell_inputs.py \
    --dataset_name "../../data/synthetic_haskell_dataset_nvidia_100000/synthetic_haskell_dataset_nvidia.jsonl" \
    --output_dir "../../data/SINQ_synthetic_haskell_dataset_nvidia" \
    --output_hf_dataset_dir "../../data/SINQ_synthetic_haskell_dataset_nvidia_hf"
