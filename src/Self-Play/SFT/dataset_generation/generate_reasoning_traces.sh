#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a40:2                  # specifically four A40 GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=250000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-sft-generate-reasoning-%j.out

# This script runs the reasoning trace generation process using vLLM.

# --- Environment Setup ---
if [[ -z "$CUDA_HOME" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  else
    CUDA_HOME="$(ls -d /usr/local/cuda-* /opt/cuda-* 2>/dev/null | sort -V | tail -n1)"
  fi
fi

if [[ ! -d "$CUDA_HOME" ]]; then
  echo "WARN: cannot locate CUDA_HOME ($CUDA_HOME). This may be fine if not needed." >&2
fi
echo "CUDA_HOME: $CUDA_HOME"

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}

export STUDENT_ID=$(whoami)
mkdir -p /disk/scratch/${STUDENT_ID}
export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

export BNB_CUDA_VERSION=125
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

# Activate Conda environment
source /home/$(whoami)/miniconda3/bin/activate llm_sp

# --- Job Configuration ---
# The source dataset containing Haskell code
SOURCE_DATASET_PATH="../../data/SINQ_sorted_blastwind_haskell_dataset"
# The fraction of the source dataset to process
DATA_FRACTION=0.3
# Model for generation
MODEL_NAME="unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit" # unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit
# VLLM configuration
GPU_MEM_UTIL=0.9
MAX_NEW_TOKENS=4096
MAX_MODEL_LEN=8192
DTYPE="bfloat16"
QUANTIZATION="bitsandbytes" # bitsandbytes
PIPELINE_PARALLEL_SIZE=2
# Output directory for the raw generated data
OUTPUT_DIR="../../data/synthetic_reasoning_dataset_raw_DATA_FRACTION_${DATA_FRACTION}"
OUTPUT_FILENAME_ARROW="synthetic_reasoning_dataset_DATA_FRACTION_${DATA_FRACTION}.arrow"
OUTPUT_FILENAME_JSONL="synthetic_reasoning_dataset_DATA_FRACTION_${DATA_FRACTION}.jsonl"

# Ensure log directory exists
mkdir -p log
mkdir -p $OUTPUT_DIR

# --- Run Data Generation ---
echo "Running reasoning trace generation script..."

CUDA_VISIBLE_DEVICES=0,1 python generate_reasoning_traces.py \
    --source_dataset_path $SOURCE_DATASET_PATH \
    --data_fraction $DATA_FRACTION \
    --model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --output_filename_arrow $OUTPUT_FILENAME_ARROW \
    --output_filename_jsonl $OUTPUT_FILENAME_JSONL \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_model_len $MAX_MODEL_LEN \
    --dtype $DTYPE \
    --quantization $QUANTIZATION \
    --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE


echo "Reasoning trace generation complete. Raw data saved to $OUTPUT_DIR" 