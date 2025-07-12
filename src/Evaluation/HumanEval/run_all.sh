#!/bin/bash

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SFT/output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_reasoning_dataset_adapter_output_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_dataset_fraction_0.3_epochs_10_learning_rate_1e-4_batch_4_grad_steps_8_checkpoint-5742_fraction_1_epochs_10_learning_rate_1e-4_batch_4_grad_steps_8
TRIALS=4

# Haskell HumanEval
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-425 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-850 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1275 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-1700 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2125 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2550 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-2975 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3400 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-3825 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS
sbatch eval_script/eval_adapter.sh $ADAPTER_PATH/checkpoint-4250 hs  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $TRIALS


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


