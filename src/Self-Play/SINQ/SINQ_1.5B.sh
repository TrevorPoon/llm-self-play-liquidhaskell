#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a40:1                  # specifically four A40 GPUs
#SBATCH --mem=128000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-sinq-trial-1.5B-%j.out

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

# --- Configuration ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME="../data/SINQ_synthetic_haskell_dataset_nvidia_hf"
NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION=0
NUM_INITIAL_PROGRAMS=1000 # Set 0 to use all programs
INITIAL_ADAPTER_PATH=""
NAME="no_initial_adapter_without_difficulty_prediction"
N_ITERATIONS=3
LEARNING_RATE=5e-4
NUM_EPOCHS=3

# Generate a unique experiment name for this run
EXPERIMENT_NAME="${MODEL_NAME}_PROGRAMS${NUM_INITIAL_PROGRAMS}_EVALS${NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION}_${NAME}_LR${LEARNING_RATE}_EPOCHS${NUM_EPOCHS}"
OUTPUT_DIR="output/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# --- Self-Play Loop ---
LATEST_ALICE_ADAPTER_PATH="$INITIAL_ADAPTER_PATH"
LATEST_BOB_ADAPTER_PATH="$INITIAL_ADAPTER_PATH" # Assuming Bob might also be trained in the future
ALICE_TRAINING_DATA_PATH="" # Start with empty, will be created in the first iteration
BOB_TRAINING_DATA_PATH="" # Start with empty, will be created in the first iteration

for i in $(seq 0 $((N_ITERATIONS - 1)))
do
    echo "--- Starting Self-Play Iteration ${i} ---"
    
    ITERATION_DIR="${OUTPUT_DIR}/iteration_${i}"
    mkdir -p "$ITERATION_DIR"

    # --- Step 1: Data Generation (vLLM on GPU 0) ---
    echo "--- [Iteration ${i}] Running Data Generation ---"
  
    
    python SINQ_wo_d_v2.py \
        --model_name_or_path "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --iteration_dir "$ITERATION_DIR" \
        --iteration "$i" \
        --cumulative_alice_training_data_path "$ALICE_TRAINING_DATA_PATH" \
        --cumulative_bob_training_data_path "$BOB_TRAINING_DATA_PATH" \
        --alice_adapter_path "$LATEST_ALICE_ADAPTER_PATH" \
        --bob_adapter_path "$LATEST_BOB_ADAPTER_PATH" \
        --n_samples 10 \
        --timeout 20 \
        --max_tokens 32768 \
        --top_p 0.95 \
        --temperature 0.6 \
        --top_k 20 \
        --gpu_memory_utilization 0.95 \
        --num_initial_programs $NUM_INITIAL_PROGRAMS \
        --tensor_parallel_size 1

    # Update programs file path for the next iteration
    ALICE_TRAINING_DATA_PATH="${ITERATION_DIR}/alice_training_data.jsonl"
    BOB_TRAINING_DATA_PATH="${ITERATION_DIR}/bob_training_data.jsonl"
    
    # --- Step 2: Fine-tuning (Accelerate on GPUs 1, 2, 3) ---
    if [ -f "$ALICE_TRAINING_DATA_PATH" ] && [ -s "$ALICE_TRAINING_DATA_PATH" ]; then
        echo "--- [Iteration ${i}] Running Fine-tuning for Alice ---"
        
        accelerate launch \
            finetune.py \
            --model_name_or_path "$MODEL_NAME" \
            --dataset_path "$ALICE_TRAINING_DATA_PATH" \
            --model_type "alice" \
            --output_dir "${ITERATION_DIR}/alice_adapters" \
            --previous_adapter_path "" \
            --iteration "$i" \
            --num_train_epochs $NUM_EPOCHS \
            --per_device_train_batch_size 1 \
            --learning_rate $LEARNING_RATE

        # Find the path to the latest adapter created by the fine-tuning script
        # Assuming the last epoch's adapter is the one to use next.
        # This part might need adjustment based on the exact output of finetune.py's SavePeftModelCallback
        LATEST_ALICE_ADAPTER_PATH=$(find "${ITERATION_DIR}/alice_adapters" -type d -name "checkpoint-*" | sort -V | tail -n 1)
        echo "Updated LATEST_ALICE_ADAPTER_PATH=${LATEST_ALICE_ADAPTER_PATH}"
    else
        echo "--- [Iteration ${i}] No new Alice training data generated. Skipping fine-tuning. ---"
    fi
    
    # Reset CUDA_VISIBLE_DEVICES to avoid affecting other scripts or subsequent iterations
    unset CUDA_VISIBLE_DEVICES
done

echo "--- Self-Play complete ---"
