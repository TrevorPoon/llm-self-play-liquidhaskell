#!/bin/bash

# Define the languages you want to run evaluations for
LANGUAGES=("python" "java" "sh" "cpp" "cs" "php" "ts" "js" "hs") # Add other languages here e.g., "cpp" "java"

# Directory where the eval_LANG.sh scripts are located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Script directory: $SCRIPT_DIR"

# Loop through the languages and submit sbatch jobs
for lang in "${LANGUAGES[@]}"; do
    eval_script="${SCRIPT_DIR}/eval_script/eval_${lang}.sh"
    if [ -f "$eval_script" ]; then
        echo "Submitting sbatch job for ${lang} using ${eval_script}..."
        sbatch "$eval_script"
    else
        echo "Warning: Evaluation script ${eval_script} not found. Skipping ${lang}."
    fi
done

echo "All specified sbatch jobs submitted." 