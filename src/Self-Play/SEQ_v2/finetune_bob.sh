#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:l40s:4                 # specifically four A40 GPUs
#SBATCH --mem=510000
#SBATCH --time=7-00:00:00
#SBATCH --exclude=scotia08
#SBATCH --array=0-8
#SBATCH --output=log/slurm-seq-7B-%j-array-%A_%a.out

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
TIME="20250806"
NAME="bob_finetune"
N_ITERATIONS=7
i=1

# Generate a unique experiment name for this run
EXPERIMENT_NAME="SEQ_${MODEL_NAME}_TIME${TIME}_SEQ_PROGRAMS${NUM_INITIAL_PROGRAMS}_ITERATIONS${N_ITERATIONS}_${NAME}_LR${LEARNING_RATE}_EPOCHS${NUM_EPOCHS}"
OUTPUT_DIR="output/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# --- Fine-tuning for Bob ---
BOB_TRAINING_DATA_PATH=""
lr_list=(2e-4 1e-4 5e-5)
epoch_list=(3 5 7)

task_id=$SLURM_ARRAY_TASK_ID
# Map linear task_id [0..8] to lr and epoch combination
lr=${lr_list[$(( task_id % ${#lr_list[@]} ))]}
epoch=${epoch_list[$(( task_id / ${#lr_list[@]} ))]}

echo "Learning rate: $lr"
echo "Number of epochs: $epoch"

echo "--- [Iteration ${i}] Running Fine-tuning for Bob ---"

CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config_file accelerate_config_bob.yaml \
    finetune_v2.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$BOB_TRAINING_DATA_PATH" \
    --model_type "bob" \
    --output_dir "${OUTPUT_DIR}/bob_adapters_lr${lr}_epoch${epoch}" \
    --previous_adapter_path "" \
    --iteration "$i" \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 1 \
    --learning_rate $lr

echo "--- Self-Play complete ---"
