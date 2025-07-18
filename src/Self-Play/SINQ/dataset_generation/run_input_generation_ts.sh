#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teach-Standard     # only nodes with A40s
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96000
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm-input-gen-%j.out

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

nvidia-smi

# OUTPUTS
OUTPUT_DIR="generated_inputs_output"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python -u generate_haskell_inputs.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset_name "../../data/SINQ_compiled_sorted_blastwind_haskell_dataset_with_input" \
    --output_dir "$OUTPUT_DIR" \
    --max_tokens 4096 \
    --temperature 0.6 \
    --top_p 0.9 \
    --top_k 20 \
    --presence_penalty 1.5 \
    --timeout 30.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95