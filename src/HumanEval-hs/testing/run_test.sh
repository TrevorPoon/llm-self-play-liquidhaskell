#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teach-LongJobs       # only nodes with A40s
#SBATCH --gres=gpu:4                     # specifically four A40 GPUs
#SBATCH --mem=96000
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm-testing-haskell-%j.out

echo "--- SLURM JOB START ---"
echo "Debug: Slurm Job ID: $SLURM_JOB_ID"
echo "Debug: Running on host: $(hostname)"
echo "Debug: Current working directory: $(pwd)"
echo "Debug: Start time: $(date)"


# Clean PATH to avoid Conda linker interference
# export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.ghcup/bin"
# hash -r
# echo "After activate, ghc → $(which ghc)"  

source /home/$(whoami)/miniconda3/bin/activate llm_sp

echo "Environment activated: $(which python3)"


# Check if GHC is available
if ! command -v ghc &> /dev/null
then
    echo "Error: ghc could not be found. Please load the GHC module or ensure it's in your PATH."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please load a Python module or ensure it's in your PATH."
    exit 1
fi

if [ -z "$SLURM_SUBMIT_DIR" ]; then
    echo "Error: SLURM_SUBMIT_DIR is not set. This script should be run with sbatch."
    exit 1
fi

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/run_haskell.py"
HASKELL_FILE_PATH="$SCRIPT_DIR/tmp/main.hs"

echo "Debug: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Debug: Script directory set to: $SCRIPT_DIR"
echo "Debug: Python script path: $PYTHON_SCRIPT_PATH"
echo "Debug: Haskell file path: $HASKELL_FILE_PATH"

echo "Debug: Executing Python script to compile and run Haskell code..."
python3 "$PYTHON_SCRIPT_PATH" "$HASKELL_FILE_PATH"

echo "Debug: Python script finished."
echo "Debug: End time: $(date)"
echo "--- SLURM JOB END ---"
