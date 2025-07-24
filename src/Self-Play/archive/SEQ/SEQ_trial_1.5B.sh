#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a40:1                 # specifically four A40 GPUs
#SBATCH --mem=128000
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm-seq-trial-1.5B-%j.out

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

export VLLM_HOST_IP=127.0.0.1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo


# INPUTS
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME="../data/SINQ_synthetic_haskell_dataset_nvidia_hf"
NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION=4
NUM_INITIAL_PROGRAMS=100 # Set 0 to use all programs
INITIAL_ADAPTER_PATH=""
NAME="no_initial_adapter"

OUTPUT_DIR="output/${MODEL_NAME}_PROGRAMS${NUM_INITIAL_PROGRAMS}_EVALS${NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION}_${NAME}_without_difficulty_prediction"

accelerate launch \
    --debug \
    SEQ.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --initial_adapter_path "$INITIAL_ADAPTER_PATH" \
    --n_iterations 3  \
    --n_samples 10 \
    --timeout 20 \
    --max_tokens 32768 \
    --top_p 0.95 \
    --temperature 0.6 \
    --top_k 20 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --gpu_memory_utilization 0.95 \
    --num_initial_programs $NUM_INITIAL_PROGRAMS \
    --per_device_train_batch_size 1 \
    --n_humaneval_evaluations_per_iteration $NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION \
    --tensor_parallel_size 1
