# llm-self-play-liquidhaskell
Improving LLM code generation and reasoning with semantic equivalence self-play

The repo provides a pipeline for fine-tuning and running inference on a large language model. It parses command-line arguments to choose between generating text from a prompt or fine-tuning the model on a dataset formatted with system instructions and answers. The fine-tuning process includes data loading, batching, mixed precision, checkpointing, and early stopping, while logging training progress and GPU usage. Utility scripts also handle logging setup and snapshotting the project structure.

In future, the repo will be developed into self-play. 

Setup
```bash
mkdir -p log/output log/slurm model src/HumanEval/log

To run finetuning:
```bash
sbatch scripts/run_finetune_mlp.sh
```

To run inferencing:
```bash
sbatch scripts/run_inference_mlp.sh
```

To run self-play fine-tuning (to be developed):
```bash

```

To check GPU info: 
```bash
sbatch scripts/gpu_info.sh
```

Slurm Commands:
```bash
squeue -u s2652867
scancel -u <user_id>
scancel <job_id>
```
