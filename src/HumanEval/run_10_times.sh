#!/bin/bash

for i in {1..10}; do
    echo "Running evaluation iteration $i"
    bash run_all_evals.sh
    echo "Completed iteration $i"
    echo "----------------------------------"
done
