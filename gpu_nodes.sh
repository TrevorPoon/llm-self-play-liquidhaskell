#!/bin/bash

# Get all nodes and filter for A40 or A6000 GPUs
# sinfo -N -o "%N" | tail -n +2 | xargs -I{} bash -c \
#   'scontrol show node {} | grep -q "Gres=gpu:a40\|Gres=gpu:a6000\|Gres=gpu:l40s" && \
#    echo -e "{}\t$(scontrol show node {} | grep -oP "Gres=gpu:\K[a-zA-Z0-9]+")\t\
#    $(scontrol show node {} | grep -oP "Partitions=\K\S+")\t\
#    $(scontrol show node {} | grep -oP "State=\K\S+")"'

# echo "===================================================================================================="

# # sinfo -Ne --format="%N %G %P %t %c %m %E"



# echo "===================================================================================================="

# # To inspect a single node:
# scontrol show node node01 | grep -i Gres

# # To do all nodes in one go:
# for n in $(sinfo -h -N -o "%N"); do
#   echo “=== $n ===”
#   scontrol show node $n | grep -i Gres
# done


#!/bin/bash

echo -e "NODE\t\tGPU_TYPE\tTOTAL_GPUs\tALLOC_GPUs\tFREE_GPUs\tPARTITION\tSTATE"

for node in $(sinfo -N -h -o "%N"); do
  info=$(scontrol show node "$node")

  # Get GPU type (a40, a6000, etc.)
  gpu_type=$(echo "$info" | grep -oP "Gres=gpu:\K[^:\s]+")

  # Only show nodes with specific GPU types
  if [[ "$gpu_type" == "a40" || "$gpu_type" == "a6000" || "$gpu_type" == "l40s" ]]; then
    # Total GPUs
    total_gpus=$(echo "$info" | grep -oP "Gres=gpu:$gpu_type:\K\d+")

    # Allocated GPUs (from AllocTRES)
    alloc_gpus=$(echo "$info" | grep -oP "AllocTRES=.*?gres/gpu=\K\d+" || echo "0")

    # Free GPUs
    free_gpus=$((total_gpus - alloc_gpus))

    # Partition
    partition=$(echo "$info" | grep -oP "Partitions=\K\S+")

    # State
    state=$(echo "$info" | grep -oP "State=\K\S+")

    # Output result
    echo -e "$node\t$gpu_type\t$total_gpus\t\t$alloc_gpus\t\t$free_gpus\t\t$partition\t$state"
  fi
done



