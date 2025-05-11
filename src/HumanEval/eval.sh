#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:8
#SBATCH --mem=96000  # memory in Mb
#SBATCH --time=0-80:00:00
#SBATCH --output=log/slurm/slurm-eval-%j.out   # %j = Job ID

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

source /home/${STUDENT_ID}/miniconda3/bin/activate llm_sp

MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_ROOT="data/"
LANGUAGE="python"
python src/HumanEval/eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}

# python src/HumanEval/eval_pal.py --logdir deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --language python --dataroot data/