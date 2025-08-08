#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble    # only nodes with A40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=120000
#SBATCH --time=7-00:00:00
#SBATCH --exclude=scotia08
#SBATCH --output=log/slurm-eval-adapter-%j.out

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

export HF_HUB_OFFLINE=1
export TORCHDYNAMO_VERBOSE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp


ADAPTER_PATH="$1"  #TODO: add model path
MODEL_NAME=$(basename "$ADAPTER_PATH")

echo "Adapter path: $ADAPTER_PATH"

MODEL="$2"
echo "Model: $MODEL"

NUM_ITERATION="$3"
echo "Number of iterations: $NUM_ITERATION"


for ((i=1; i<=NUM_ITERATION; i++)); do
  echo "Running evaluation $i of $NUM_ITERATION..."
  python main.py \
    --model "$MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --max_new_tokens 4096 \
    --num_iterations "$NUM_ITERATION"
done


# sbatch eval_s.sh "" deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 1
# sbatch eval_s.sh /home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_LR2e-4_EPOCHS3/iteration_7/bob_adapters/checkpoint-4548 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 1


