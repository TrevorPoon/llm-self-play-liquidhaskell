#!/bin/bash
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
TRIALS=8

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS100_ITERATIONS7_no_initial_adapter_random_dataset_LR5e-4_EPOCHS3

sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-96 hs $MODEL_NAME $TRIALS


