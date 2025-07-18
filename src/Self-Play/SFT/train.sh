#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard    # only nodes with A40s
#SBATCH --gres=gpu:a40:4                  # specifically four A40 GPUs
#SBATCH --mem=515000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-sft-train-%j.out

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
export HF_HUB_OFFLINE=1

# Activate Conda environment
source /home/$(whoami)/miniconda3/bin/activate llm_sp

# --- Job Configuration ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH="../data/sft_tokenized_haskell_dataset" # Path from prepare_data.py
DATASET_FRACTION=0.3
NAME="lm_head"

# Hyperparameters
NUM_TRAIN_EPOCHS=10
LEARNING_RATE=5e-4 
PER_DEVICE_TRAIN_BATCH_SIZE=4 # 4 is the max for A40s for 4096 tokens 
PER_DEVICE_EVAL_BATCH_SIZE=PER_DEVICE_TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS=8
EVAL_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS

OUTPUT_DIR="output/${MODEL_NAME}_${NAME}_dataset_fraction_${DATASET_FRACTION}_epochs_${NUM_TRAIN_EPOCHS}_learning_rate_${LEARNING_RATE}_batch_${PER_DEVICE_TRAIN_BATCH_SIZE}_grad_steps_${GRADIENT_ACCUMULATION_STEPS}"  


# Ensure log directory exists
mkdir -p log
mkdir -p $OUTPUT_DIR

# --- Run Training ---
# Use accelerate to launch the training script on all 4 GPUs.
# The script will automatically use the FSDP strategy if you have configured accelerate.
echo "--- Running Training ---"
accelerate launch --config_file accelerate_config.yaml train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --save_steps 1000 \
    --logging_steps 50 \
    --dataset_fraction $DATASET_FRACTION \
    --dataset_is_tokenized \
    --run_humaneval_evaluation \
    --n_humaneval_evaluations 4 \
    --log_memory_usage