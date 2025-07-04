#!/bin/bash

# Get all nodes and filter for A40 or A6000 GPUs
sinfo -N -o "%N" | tail -n +2 | xargs -I{} bash -c \
  'scontrol show node {} | grep -q "Gres=gpu:a40\|Gres=gpu:a6000" && \
   echo -e "{}\t$(scontrol show node {} | grep -oP "Gres=gpu:\K[a-zA-Z0-9]+")\t\
   $(scontrol show node {} | grep -oP "Partitions=\K\S+")\t\
   $(scontrol show node {} | grep -oP "State=\K\S+")"'

