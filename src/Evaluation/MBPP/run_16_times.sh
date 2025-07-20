#!/bin/bash

for i in {1..16}; do
    echo "Running evaluation iteration $i"

    sbatch eval_script/eval_hs.sh
    sbatch eval_script/eval_python.sh
    
    echo "Completed iteration $i"
    echo "----------------------------------"
done
