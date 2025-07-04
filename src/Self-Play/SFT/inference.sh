#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard       # only nodes with A40s
#SBATCH --gres=gpu:a40:1                  # one A40 GPU is sufficient for inference
#SBATCH --mem=80000
#SBATCH --time=0-01:00:00                 # 1 hour should be plenty
#SBATCH --output=log/slurm-sft-inference-%j.out

# --- Environment Setup ---
# Find CUDA
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
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# This should point to the final adapter from your training run
export ADAPTER_PATH="output/SFT-Haskell-CodeLlama-7b/final_adapter"
export PROMPT_TEXT="Write a Haskell function named 'fib' that computes the nth Fibonacci number."

# Ensure log directory exists
mkdir -p log

# --- Run Inference ---
python src/Self-Play/SFT/inference.py \
    --model_name_or_path "$MODEL_NAME" \
    --adapter_path "$ADAPTER_PATH" \
    --prompt "$PROMPT_TEXT" 