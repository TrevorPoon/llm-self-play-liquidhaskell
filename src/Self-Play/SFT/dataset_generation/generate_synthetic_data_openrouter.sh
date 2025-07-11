#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard
#SBATCH --mem=16000
#SBATCH --time=7-00:00:00                 
#SBATCH --output=log/slurm-sft-generate-data-openrouter-%j.out

# This script runs the data generation process by making API calls to OpenRouter.
# It does not require local GPUs. Ensure you have set your OPENROUTER_API_KEY
# in a .env file at the root of the project.

# --- Environment Setup ---
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
MODEL_NAME="deepseek/deepseek-r1-0528:free" # Specify any valid model on OpenRouter
MAX_NEW_TOKENS=4096
OUTPUT_DIR="../data/synthetic_liquid_haskell_dataset_openrouter"
OUTPUT_FILENAME_ARROW="synthetic_liquid_haskell_dataset_openrouter.arrow"
OUTPUT_FILENAME_JSONL="synthetic_liquid_haskell_dataset_openrouter.jsonl"

# --- Optional: Hugging Face Upload ---
# To enable, uncomment the following lines and set your details.
# UPLOAD_TO_HF=true
# HF_USERNAME="your-hf-username"
# HF_REPO_NAME="synthetic-liquid-haskell-dataset-openrouter"

# Ensure log and output directories exist
mkdir -p log
mkdir -p $OUTPUT_DIR

# --- Run Data Generation ---
echo "Running data generation script via OpenRouter..."

# Base command
CMD="python generate_synthetic_data_openrouter.py \
    --num_samples $NUM_SAMPLES \
    --model \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --output_filename_arrow \"$OUTPUT_FILENAME_ARROW\" \
    --output_filename_jsonl \"$OUTPUT_FILENAME_JSONL\" \
    --max_new_tokens $MAX_NEW_TOKENS"

# The script will automatically load the OPENROUTER_API_KEY from the .env file.

# Add Hugging Face flags if the upload is enabled
if [ "$UPLOAD_TO_HF" = true ]; then
    CMD="$CMD --upload_to_hf --hf_username \"$HF_USERNAME\" --hf_repo_name \"$HF_REPO_NAME\""
fi

echo "Executing command: $CMD"
eval $CMD

echo "Data generation complete. Datasets potentially saved to $OUTPUT_DIR" 