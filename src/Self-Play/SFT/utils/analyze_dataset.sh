#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                  # specifically four A40 GPUs
#SBATCH --mem=192000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-sft-analyze-data-%j.out

# --- Environment Setup ---
# Find CUDA (not strictly necessary for this script, but good practice for consistency)
if [[ -z "$CUDA_HOME" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  else
    CUDA_HOME="$(ls -d /usr/local/cuda-* /opt/cuda-* 2>/dev/null | sort -V | tail -n1)"
  fi
fi

if [[ ! -d "$CUDA_HOME" ]]; then
  echo "ERROR: cannot locate CUDA_HOME ($CUDA_HOME)" >&2
  exit 1
fi
echo "CUDA_HOME: $CUDA_HOME"

# Set library paths
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}

# Set temporary directory for scratch space
export STUDENT_ID=$(whoami)
mkdir -p /disk/scratch/${STUDENT_ID}
export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# Activate Conda environment
source /home/$(whoami)/miniconda3/bin/activate llm_sp

# --- Job Configuration ---
# The model whose tokenizer will be used for analysis.
# This should match the model you intend to train.
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# This is the PROCESSED, but UNTOKENIZED dataset.
export DATASET_PATH="../data/sft_processed_haskell_dataset"

# Ensure log directory exists
mkdir -p ../log

# --- Run Data Analysis ---
echo "Running data analysis script..."
python utils/analyze_dataset.py \
    --dataset_path "$DATASET_PATH" \
    --model_name_or_path "$MODEL_NAME"

echo "Data analysis complete. Statistics printed above and histogram saved." 