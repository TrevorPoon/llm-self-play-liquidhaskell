#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:8
#SBATCH --mem=48000  # memory in Mb
#SBATCH --time=0-08:00:00
#SBATCH --output=log/slurm-eval-%j.out   # %j = Job ID

# export CUDA_HOME=/opt/cuda-9.0.176.1/

# export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

# export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

# export CPATH=${CUDNN_HOME}/include:$CPATH

# export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

source /home/${STUDENT_ID}/miniconda3/bin/activate llm_sp

LANG="python"
OUTPUT_DIR="output"
MODEL="DeepSeek-R1-Distill-Qwen-1.5B"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_instruct.py \
    --model "deepseek-ai/$MODEL" \
    --output_path "$OUTPUT_DIR/${LANG}.$MODEL.jsonl" \
    --language $LANG \
    --temp_dir $OUTPUT_DIR