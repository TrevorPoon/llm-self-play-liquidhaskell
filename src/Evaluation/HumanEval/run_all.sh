#!/bin/bash

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SFT/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_dataset_fraction_0.3_epochs_10_learning_rate_1e-4_batch_4_grad_steps_8
TRIALS=4

# Haskell HumanEval
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-638 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1276 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1914 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2552 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3190 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3828 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-4466 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-5104 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-5742 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-6380 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS 

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


