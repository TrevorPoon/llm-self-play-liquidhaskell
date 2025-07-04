#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard       # only nodes with A40s
#SBATCH --gres=gpu:a40:4                  # specifically four A40 GPUs
#SBATCH --mem=192000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-finetune-%j.out

# 2) Fallback: detect CUDA_HOME
if [[ -z "$CUDA_HOME" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  else
    CUDA_HOME="$(ls -d /usr/local/cuda-* /opt/cuda-* 2>/dev/null | sort -V | tail -n1)"
  fi
fi

# 3) Detect CUDNN_HOME
# if [[ -z "$CUDNN_HOME" ]]; then
#   CUDNN_PATH="$(locate cudnn.h 2>/dev/null | head -n1)"
#   CUDNN_HOME="$(dirname "$(dirname "$CUDNN_PATH")")"
# fi

# 4) Sanity check
if [[ ! -d "$CUDA_HOME" ]]; then
  echo "ERROR: cannot locate CUDA_HOME ($CUDA_HOME)" >&2
  exit 1
fi

echo "CUDA_HOME: $CUDA_HOME"
# echo "CUDNN_HOME: $CUDNN_HOME"

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDA_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

source /home/$(whoami)/miniconda3/bin/activate llm_sp

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export BNB_CUDA_VERSION=125

#INPUTS
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME="../data/successfully_compiled_sorted_blastwind_haskell_dataset"
NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION=2
NUM_INITIAL_PROGRAMS=2 # Set 0 to use all programs
PER_DEVICE_TRAIN_BATCH_SIZE=1
OUTPUT_DIR="output/SHQ_finetune_${MODEL_NAME}_PROGRAMS${NUM_INITIAL_PROGRAMS}_EVALS${NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION}_without_difficulty_prediction"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run_blastwind_without_diff.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --n_iterations 3 \
    --n_samples 10 \
    --timeout 20 \
    --max_tokens 32768 \
    --top_p 0.95 \
    --temperature 0.6 \
    --top_k 20 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --gpu_memory_utilization 0.8 \
    --num_initial_programs $NUM_INITIAL_PROGRAMS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --n_humaneval_evaluations_per_iteration $NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION \
    --run_evaluation False
