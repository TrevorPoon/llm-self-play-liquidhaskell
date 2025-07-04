#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PGR-Standard      # only nodes with A40s
#SBATCH --gres=gpu:a40:4               # specifically four A40 GPUs
#SBATCH --mem=192000
#SBATCH --time=7-00:00:00
#SBATCH --output=log/slurm-sft-install-%j.out

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

# Activate Conda environment
source /home/$(whoami)/miniconda3/bin/activate llm_sp

which pip 

python -c "import torch; print(f'torch version: {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}'); print(f'cuda version: {torch.version.cuda}')"

nvcc --version

python -c "from torch.utils import cpp_extension; print(f'cuda home: {cpp_extension.CUDA_HOME}')"

/home/s2652867/.conda/envs/llm_sp/bin/x86_64-conda-linux-gnu-c++ --version

ninja --version && echo $? 

echo nproc: $(nproc)

# cd /home/$(whoami)/flash-attention
# MAX_JOBS=4 pip install -e . --no-build-isolation --use-pep517

# MAX_JOBS=4 pip install flash-attn --no-build-isolation --use-pep517

# MAX_JOBS=4 pip install flash-attn==2.7.3 --no-build-isolation

MAX_JOBS=4 pip install flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl




