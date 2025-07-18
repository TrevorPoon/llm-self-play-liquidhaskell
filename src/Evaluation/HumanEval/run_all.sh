#!/bin/bash

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SFT/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_dataset_fraction_0.5_epochs_10_learning_rate_5e-4_batch_4_grad_steps_8
TRIALS=8

# Haskell HumanEval
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-0 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1063 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2126 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3189 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-4252 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-5315 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-6378 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-7441 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-8504 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-9567 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-10630 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS




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


