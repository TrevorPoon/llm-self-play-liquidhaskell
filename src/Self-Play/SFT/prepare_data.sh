#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble
#SBATCH --mem=96000
#SBATCH --time=0-02:00:00                 # 2 hours should be sufficient
#SBATCH --output=log/slurm-sft-prepare-data-%j.out

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
# Match the model name from train.sh for the correct tokenizer
# export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # No longer needed for data prep
# The raw, unprocessed source dataset
export SOURCE_DATASET_NAME="../data/sorted_blastwind_haskell_dataset"
# The directory where the processed, tokenized dataset will be saved. Match from train.sh
export PROCESSED_DATA_DIR="../data/sft_processed_haskell_dataset"

# Ensure log directory exists
mkdir -p log

# --- Run Data Preparation ---
echo "Running data preparation script..."
python prepare_data.py \
    --dataset_name "$SOURCE_DATASET_NAME" \
    --output_dir "$PROCESSED_DATA_DIR"

echo "Data preparation complete. Processed data saved to $PROCESSED_DATA_DIR"

# --- Clean Cache ---
echo "Cleaning cache files from source dataset directory: $SOURCE_DATASET_NAME"
find "$SOURCE_DATASET_NAME" -type f -name "cache-*.arrow" -print -delete
echo "Cache cleaning complete." 