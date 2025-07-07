#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble    # only nodes with A40s
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive                      # Request exclusive node access for consistent performance
#SBATCH --mem=515000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-eval-adapter-%j.out

#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=12
#SBATCH --exclusive                      # Request exclusive node access for consistent performance
#SBATCH --mem=96000
#SBATCH --time=3-08:00:00
#SBATCH --output=log/slurm-eval-adapter-%j.out