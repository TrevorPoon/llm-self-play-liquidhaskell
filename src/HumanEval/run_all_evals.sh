#!/bin/bash

# Define languages
LANGUAGES=("python" "java" "cpp" "cs" "php" "ts" "js" "hs")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
echo "📂 Script directory: $SCRIPT_DIR"

# Parameters
MAX_RETRIES=10
SLEEP_SECONDS=1

# Arrays to store job info
declare -A JOBS  # lang -> jobid

# Step 1: Submit all jobs
for lang in "${LANGUAGES[@]}"; do
    eval_script="${SCRIPT_DIR}/eval_script/eval_${lang}.sh"
    if [ -f "$eval_script" ]; then
        echo "🚀 Submitting job for $lang..."
        jobid=$(sbatch "$eval_script" | awk '{print $4}')
        if [ -n "$jobid" ]; then
            JOBS["$lang"]=$jobid
            echo "✅ Submitted $lang with Job ID $jobid"
        else
            echo "❌ Failed to submit $lang (no Job ID returned)"
        fi
    else
        echo "⚠️ Script not found: $eval_script — skipping $lang"
    fi
done

echo "⏳ Waiting $SLEEP_SECONDS seconds before verification..."
sleep $SLEEP_SECONDS

# Step 2: Check and retry failed jobs
for lang in "${!JOBS[@]}"; do
    jobid=${JOBS[$lang]}
    attempt=1
    while ! squeue -j "$jobid" | grep -q "$jobid"; do
        echo "⚠️ Job $jobid for $lang not found in queue. Retrying ($attempt)..."
        if [ $attempt -ge $MAX_RETRIES ]; then
            echo "❌ Max retries reached for $lang. Giving up."
            break
        fi
        eval_script="${SCRIPT_DIR}/eval_script/eval_${lang}.sh"
        jobid=$(sbatch "$eval_script" | awk '{print $4}')
        if [ -n "$jobid" ]; then
            JOBS["$lang"]=$jobid
            echo "🔄 Re-submitted $lang with new Job ID $jobid"
        else
            echo "❌ Failed to resubmit $lang"
        fi
        ((attempt++))
        sleep $SLEEP_SECONDS
    done

    if squeue -j "${JOBS[$lang]}" | grep -q "${JOBS[$lang]}"; then
        echo "✅ Final check passed: ${JOBS[$lang]} for $lang is now in the queue."
    fi
done

echo "✅ All job submissions and checks complete."
