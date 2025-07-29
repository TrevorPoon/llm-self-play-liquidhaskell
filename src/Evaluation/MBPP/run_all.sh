#!/bin/bash

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS100_no_initial_adapter_LR5e-4_EPOCHS3
ITERATION=iteration_3
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
CHECKPOINT=checkpoint-90
TRIALS=8
# Haskell HumanEval
sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/$ITERATION/alice_adapters/$CHECKPOINT hs $MODEL_NAME $TRIALS





# Haskell HumanEval
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-4000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-5000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-6000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-7000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-8000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-9000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-10000 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS


# # Python HumanEval
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1000 python deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2000 python deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3000 python deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-4000 python deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
# sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-5000 python deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS


