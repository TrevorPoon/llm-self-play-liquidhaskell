#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                  # specifically four A40 GPUs
#SBATCH --mem=192000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-sft-train-%j.out

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

# Set BNB_CUDA_VERSION to match CUDA version
export BNB_CUDA_VERSION=125

# Activate Conda environment
source /home/$(whoami)/miniconda3/bin/activate llm_sp

# --- Job Configuration ---
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export DATASET_PATH="../data/sft_tokenized_haskell_dataset" # Path from prepare_data.py
export OUTPUT_DIR="output/$MODEL_NAME"  
export LEARNING_RATE=1e-5


# Ensure log directory exists
mkdir -p log
mkdir -p $OUTPUT_DIR

# --- Run Training ---
# Use accelerate to launch the training script on all 4 GPUs.
# The script will automatically use the FSDP strategy if you have configured accelerate.
echo "--- Running Training ---"
accelerate launch train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate $LEARNING_RATE \
    --save_steps 100 \
    --logging_steps 10 \
    --dataset_is_tokenized