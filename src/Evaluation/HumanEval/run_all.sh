#!/bin/bash
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
TRIALS=8

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_LR1e-4_EPOCHS3

# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-75 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-147 hs  $MODEL_NAME $TRIALS
sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-219 hs  $MODEL_NAME $TRIALS
sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-294 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-72 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-87 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-102 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-879 hs  $MODEL_NAME $TRIALS

ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_LR2e-4_EPOCHS3

# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-72 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-141 hs  $MODEL_NAME $TRIALS
sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-216 hs  $MODEL_NAME $TRIALS
sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-291 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-72 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-87 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-102 hs  $MODEL_NAME $TRIALS
# sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-879 hs  $MODEL_NAME $TRIALS


