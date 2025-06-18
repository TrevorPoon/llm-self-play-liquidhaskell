#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard-Noble       # Partition with A40 GPUs
#SBATCH --gres=gpu:a40:4                     # Request four A40 GPUs
#SBATCH --mem=96000                          # Memory request
#SBATCH --time=0-168:00:00                     # Time limit
#SBATCH --output=log/slurm-eval-lcb-%j.out

# Set up a temporary directory on scratch space
mkdir -p /disk/scratch/$(whoami)
export TMPDIR=/disk/scratch/$(whoami)/

# Activate the correct conda environment
# Make sure the 'llm_sp' environment is configured with necessary libraries (torch, transformers, lighteval)
source /home/$(whoami)/miniconda3/bin/activate llm_sp

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

# --- Configuration ---
# Set the model to be evaluated
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B # Example model, change as needed
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Set the output directory relative to the script's location
OUTPUT_DIR=output/$MODEL
mkdir -p $OUTPUT_DIR
# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 


