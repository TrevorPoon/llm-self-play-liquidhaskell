#!/bin/bash

for i in {1..16}; do
    echo "Running evaluation iteration $i"
    # bash run_all_evals.sh
    sbatch eval_script/eval_hs.sh
    echo "Completed iteration $i"
    echo "----------------------------------"
done
