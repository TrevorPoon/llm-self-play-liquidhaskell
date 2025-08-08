#!/bin/bash
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
TRIALS=8

for i in $(seq 1 $TRIALS)
do
    ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250806_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_diff5_LR2e-4_EPOCHS3
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-6 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-21 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-39 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-294 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-369 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-441 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-516 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-4236 hs  $MODEL_NAME 1

    ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250806_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_LR2e-4_EPOCHS3
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-9 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-21 hs  $MODEL_NAME 1
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-36 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-30 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-366 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-453 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-537 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-4548 hs  $MODEL_NAME 1

    ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250807_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_diff1_LR2e-4_EPOCHS3
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-33 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-15 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-27 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-30 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-366 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-453 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-537 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-4548 hs  $MODEL_NAME 1

    ADAPTER_PATH=/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250807_SEQ_PROGRAMS500_ITERATIONS7_pa_random_dataset_withdiffbiasing_diff5_LR2e-4_EPOCHS3
    sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_1/alice_adapters/checkpoint-12 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_2/alice_adapters/checkpoint-15 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_3/alice_adapters/checkpoint-27 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_4/alice_adapters/checkpoint-30 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_5/alice_adapters/checkpoint-366 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_6/alice_adapters/checkpoint-453 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/alice_adapters/checkpoint-537 hs  $MODEL_NAME 1
    # sbatch eval_script/eval_adapter_s.sh $ADAPTER_PATH/iteration_7/bob_adapters/checkpoint-4548 hs  $MODEL_NAME 1

done
