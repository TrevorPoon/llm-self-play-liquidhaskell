#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu
#SBATCH --mem=96000
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm-sft-process-reasoning-%j.out

# This script runs the processing and tokenization for the synthetic reasoning dataset.

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
  echo "WARN: cannot locate CUDA_HOME ($CUDA_HOME). This may be fine as the script is CPU-bound." >&2
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
# The model whose tokenizer will be used. Should match the model for fine-tuning.
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Data fraction for generation
DATA_FRACTION=0.1
# The raw, generated dataset from the previous step
RAW_DATASET_DIR="../../data/synthetic_reasoning_dataset_raw_DATA_FRACTION_${DATA_FRACTION}/synthetic_reasoning_dataset_DATA_FRACTION_${DATA_FRACTION}.jsonl"
# The directory for the processed, text-only dataset
PROCESSED_DATA_DIR="../../data/sft_processed_reasoning_dataset"
# The directory for the final, tokenized dataset
TOKENIZED_DATA_DIR="../../data/sft_tokenized_reasoning_dataset"
# Sequence length for tokenization
MAX_LENGTH=32768


# Ensure log directory exists
mkdir -p log

nvidia-smi

# --- Run Data Processing ---
echo "Running data processing and tokenization script..."
python process_reasoning_data.py \
    --dataset_path "$RAW_DATASET_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --processed_output_dir "$PROCESSED_DATA_DIR" \
    --tokenized_output_dir "$TOKENIZED_DATA_DIR" \
    --max_length $MAX_LENGTH \
    --interleaved_split

echo "Data processing complete. Text data saved to $PROCESSED_DATA_DIR"
echo "Tokenized data saved to $TOKENIZED_DATA_DIR"

# --- Clean Cache ---
# echo "Cleaning cache files from raw dataset directory: $RAW_DATASET_DIR"
# find "$RAW_DATASET_DIR" -type f -name "cache-*.arrow" -print -delete
# echo "Cache cleaning complete." 