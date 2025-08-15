#!/bin/bash
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
TRIALS=8

for i in $(seq 1 $TRIALS)
do
    ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250811_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_eq_096_LR2e-4_EPOCHS3_DIFF3
    
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-6 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-12 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-15 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-18 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-24 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-33 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-48 hs  $MODEL_NAME 1

    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-657 hs  $MODEL_NAME 1
done
