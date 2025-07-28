#!/bin/bash

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SINQ/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_SINQ_PROGRAMS1000_EVALS0_no_initial_adapter_LR5e-4_EPOCHS3/iteration_1/alice_adapters/
TRIALS=8

# Haskell HumanEval
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-0 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS




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


