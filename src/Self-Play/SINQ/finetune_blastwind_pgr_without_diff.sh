#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                  # specifically four A40 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=515000
#SBATCH --time=7-00:00:00
#SBATCH --exclusive              # entire node exclusively yours
#SBATCH --output=log/slurm-finetune-%j.out

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

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export BNB_CUDA_VERSION=125

# INPUTS
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME="../data/compiled_sorted_blastwind_haskell_dataset_with_input"
NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION=3
NUM_INITIAL_PROGRAMS=2 # Set 0 to use all programs
INITIAL_ADAPTER_PATH="/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SFT/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_dataset_fraction_0.3_epochs_10_learning_rate_5e-4_batch_4_grad_steps_8/checkpoint-5000"

OUTPUT_DIR="output/SHQ_finetune_${MODEL_NAME}_PROGRAMS${NUM_INITIAL_PROGRAMS}_EVALS${NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION}_without_difficulty_prediction"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run_blastwind_without_diff.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --initial_adapter_path "$INITIAL_ADAPTER_PATH" \
    --n_iterations 3 \
    --n_samples 10 \
    --timeout 20 \
    --max_tokens 32768 \
    --top_p 0.95 \
    --temperature 0.6 \
    --top_k 20 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --gpu_memory_utilization 0.9 \
    --num_initial_programs $NUM_INITIAL_PROGRAMS \
    --per_device_train_batch_size 1 \
    --n_humaneval_evaluations_per_iteration $NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION \
    --run_evaluation False
