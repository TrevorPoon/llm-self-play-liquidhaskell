#!/bin/bash

for i in {1..16}; do
    echo "Running evaluation iteration $i"
    # bash run_all_evals.sh
    sbatch eval_script/eval_hs.sh
    sbatch eval_script/eval_python.sh
    sbatch eval_script/eval_java.sh
    sbatch eval_script/eval_cpp.sh
    sbatch eval_script/eval_cs.sh
    sbatch eval_script/eval_php.sh
    sbatch eval_script/eval_ts.sh
    sbatch eval_script/eval_js.sh
    
    echo "Completed iteration $i"
    echo "----------------------------------"
done
