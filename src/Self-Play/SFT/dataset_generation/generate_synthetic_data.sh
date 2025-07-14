#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a40:2                  # specifically four A40 GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=250000
#SBATCH --exclude=crannog04
#SBATCH --time=7-00:00:00          
#SBATCH --output=log/slurm-sft-generate-data-%j.out

# This script runs the data generation process directly, using vLLM to load the model
# on the allocated GPUs. It no longer requires a separate VLLM server.

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

# Set BNB_CUDA_VERSION to match CUDA version
export BNB_CUDA_VERSION=125

# Activate Conda environment
source /home/$(whoami)/miniconda3/bin/activate llm_sp

# --- Job Configuration ---
NUM_SAMPLES=500
MODEL_NAME="unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit"
GPU_MEM_UTIL=0.9
MAX_NEW_TOKENS=4096
MAX_MODEL_LEN=8192
DTYPE="bfloat16"
QUANTIZATION="bitsandbytes"
PIPELINE_PARALLEL_SIZE=2
OUTPUT_DIR="../data/synthetic_liquid_haskell_dataset"
OUTPUT_FILENAME_ARROW="synthetic_liquid_haskell_dataset.arrow"
OUTPUT_FILENAME_JSONL="synthetic_liquid_haskell_dataset.jsonl"

# --- Optional: Hugging Face Upload ---
# To enable, uncomment the following lines and set your details.
# UPLOAD_TO_HF=true
# HF_USERNAME="your-hf-username"
# HF_REPO_NAME="synthetic-liquid-haskell-dataset"

# Ensure log directory exists
mkdir -p log
mkdir -p $OUTPUT_DIR

# --- Run Data Generation ---
echo "Running data generation script..."

# Base command
CMD="python generate_synthetic_data.py \
    --num_samples $NUM_SAMPLES \
    --model \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --output_filename_arrow \"$OUTPUT_FILENAME_ARROW\" \
    --output_filename_jsonl \"$OUTPUT_FILENAME_JSONL\" \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_model_len $MAX_MODEL_LEN \
    --dtype $DTYPE \
    --quantization $QUANTIZATION \
    --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE"

# Add Hugging Face flags if the upload is enabled
if [ "$UPLOAD_TO_HF" = true ]; then
    CMD="$CMD --upload_to_hf --hf_username \"$HF_USERNAME\" --hf_repo_name \"$HF_REPO_NAME\""
fi

echo "Executing command: $CMD"
eval $CMD

echo "Data generation complete. Datasets potentially saved to $OUTPUT_DIR" 