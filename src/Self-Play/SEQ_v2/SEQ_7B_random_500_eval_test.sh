#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:l40s:1                # specifically four A40 GPUs
#SBATCH --mem=120000
#SBATCH --time=7-00:00:00
#SBATCH --exclude=scotia08
#SBATCH --output=log/slurm-seq-7B-eval-test-%j.out

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

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export BNB_CUDA_VERSION=125
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

# --- Configuration ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_NAME="../data/SINQ_synthetic_haskell_dataset_nvidia_hf"
NUM_INITIAL_PROGRAMS=500 # Set 0 to use all programs
INITIAL_ADAPTER_PATH=""
TIME="20250804"
NAME="no_initial_adapter_random_dataset_eval_test"
N_ITERATIONS=1

# --- Evaluation Paths---
LATEST_ALICE_ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_LR2e-4_EPOCHS3/iteration_7/alice_adapters/checkpoint-537
LATEST_BOB_ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_LR2e-4_EPOCHS3/iteration_7/bob_adapters/checkpoint-4548

# Generate a unique experiment name for this run
EXPERIMENT_NAME="SEQ_${MODEL_NAME}_TIME${TIME}_SEQ_PROGRAMS${NUM_INITIAL_PROGRAMS}_ITERATIONS${N_ITERATIONS}_${NAME}_LR${LEARNING_RATE}_EPOCHS${NUM_EPOCHS}"
OUTPUT_DIR="output/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# --- Self-Play Loop ---
ALICE_TRAINING_DATA_PATH="" # Start with empty, will be created in the first iteration
BOB_TRAINING_DATA_PATH="" # Start with empty, will be created in the first iteration

  echo "--- Starting Evaluation (Base) ---"
  
  ITERATION_DIR="${OUTPUT_DIR}/evaluation/base"
  mkdir -p "$ITERATION_DIR"

  CUDA_VISIBLE_DEVICES=0

  python SEQ_miceli_random_test.py \
      --model_name_or_path "$MODEL_NAME" \
      --dataset_name "$DATASET_NAME" \
      --output_dir "$OUTPUT_DIR" \
      --iteration_dir "$ITERATION_DIR" \
      --iteration "1" \
      --cumulative_alice_training_data_path "$ALICE_TRAINING_DATA_PATH" \
      --cumulative_bob_training_data_path "$BOB_TRAINING_DATA_PATH" \
      --alice_adapter_path "$LATEST_ALICE_ADAPTER_PATH" \
      --timeout 60 \
      --max_tokens 32768 \
      --top_p 0.95 \
      --temperature 0.6 \
      --top_k 20 \
      --gpu_memory_utilization 0.95 \
      --num_initial_programs $NUM_INITIAL_PROGRAMS \
      --tensor_parallel_size 1

  echo "--- Starting Evaluation (Bob) ---"

  ITERATION_DIR="${OUTPUT_DIR}/evaluation/bob"
  mkdir -p "$ITERATION_DIR"

  python SEQ_miceli_random_test.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --iteration_dir "$ITERATION_DIR" \
    --iteration "1" \
    --cumulative_alice_training_data_path "$ALICE_TRAINING_DATA_PATH" \
    --cumulative_bob_training_data_path "$BOB_TRAINING_DATA_PATH" \
    --alice_adapter_path "$LATEST_ALICE_ADAPTER_PATH" \
    --bob_adapter_path "$LATEST_BOB_ADAPTER_PATH" \
    --timeout 60 \
    --max_tokens 32768 \
    --top_p 0.95 \
    --temperature 0.6 \
    --top_k 20 \
    --gpu_memory_utilization 0.95 \
    --num_initial_programs $NUM_INITIAL_PROGRAMS \
    --tensor_parallel_size 1


echo "--- Self-Play complete ---"
