#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble     # only nodes with A40s
#SBATCH --gres=gpu:a40:4                   
#SBATCH --mem=96000
#SBATCH --time=0-168:00:00
#SBATCH --output=log/slurm-eval-adapter-%j.out

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

# sbatch eval_script/eval_adapter.sh /home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SFT/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_dataset_fraction_0.05_epochs_5_learning_rate_2e-4_batch_4_grad_steps_8/final_adapter hs deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 16


