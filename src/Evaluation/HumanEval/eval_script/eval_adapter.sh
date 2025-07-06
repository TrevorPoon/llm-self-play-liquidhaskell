#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble    # only nodes with A40s
#SBATCH --gres=gpu:a40:4                   
#SBATCH --mem=515000
#SBATCH --time=0-168:00:00
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

mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

source /home/$(whoami)/miniconda3/bin/activate llm_sp


ADAPTER_PATH="$1"  #TODO: add model path
if [ -z "$ADAPTER_PATH" ]; then
    echo "Error: Model path argument is required."
    exit 1
fi
MODEL_NAME=$(basename "$ADAPTER_PATH")

echo "Adapter path: $ADAPTER_PATH"

LANG="$2"
echo "Language: $LANG"

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

MODEL="$3"
echo "Model: $MODEL"

NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION="$4"
echo "Number of HumanEval evaluations per iteration: $NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION"

for ((i=1; i<=NUM_HUMANEVAL_EVALUATIONS_PER_ITERATION; i++)); do
  echo "--- Starting Evaluation Run $i ---"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_instruct.py \
    --model "$MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --output_path "$OUTPUT_DIR/${LANG}.$MODEL.${MODEL_NAME}.jsonl" \
    --language "$LANG" \
    --temp_dir $OUTPUT_DIR \
    --max_new_tokens 32768 \
    --use_vllm
done

# sbatch eval_script/eval_adapter.sh /home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SFT/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_dataset_fraction_0.3_epochs_10_learning_rate_5e-4_batch_4_grad_steps_8/checkpoint-1000 hs deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 4


