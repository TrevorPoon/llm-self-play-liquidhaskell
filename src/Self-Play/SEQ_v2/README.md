## Semantic Equivalence Self-Play (SEQ v2)

This directory contains a self-play framework for improving code reasoning in LLMs via Haskell. Two agents interact adversarially:
- **Alice**: generates a Haskell variant `Q` from a reference program `P` and either proves equivalence (Liquid Haskell) or produces a diverging input for inequivalence.
- **Bob**: evaluates whether `(P, Q)` are semantically equivalent and induces a continuous difficulty score to guide curriculum.

At each iteration we (optionally) fine-tune LoRA adapters for both agents using the high-quality, auto-validated data produced in that round.


### How it works

- **Step 1 — Program selection and branching**: The pipeline loads a reference Haskell program `P` from a dataset `\mathcal{D}` (see Dataset section). It then randomly branches with 50/50 probability into:
  - SEQ (equivalence): Alice tries to generate `Q` such that for all inputs `x`, `P(x) == Q(x)` and must provide a Liquid Haskell proof.
  - SINQ (inequivalence): Alice tries to generate `Q` and a concrete `x_a` such that `P(x_a) != Q(x_a)`.

- **Step 2a — SEQ (Alice)**: Alice proposes `Q` and a Liquid Haskell proof of equivalence. We verify the proof by compiling a module that reflects `P` and `Q` and checks the lemma with the Liquid Haskell plugin. Only accepted proofs are kept.

- **Step 2b — SINQ (Alice)**: Alice proposes `Q` and a diverging input `x_a`. We compile `P` and `Q`, execute both on `x_a`, and keep the instance only if their outputs diverge.

- **Step 3 — Validation**:
  - SEQ: formal proof via Liquid Haskell must succeed.
  - SINQ: execution-based divergence must be observed.
  In both cases, non-compiling programs are discarded.

- **Step 4 — Bob (difficulty estimation)**: Bob is sampled `N` times on `(P, Q)`. The difficulty is `difficulty = 10 * (1 - N_success / N)`. This acts as an automatic curriculum signal.

- **Step 5 — Buffer update and fine-tuning**: Hard examples (difficulty > threshold `\tau`) are always retained; a stratified subset of easy ones is also kept. These data are used to fine-tune LoRA adapters for the next round.

- **Step 6 — Iterate**: Repeat for multiple rounds, updating adapters each time.


### Key components

- `SEQ.py`: End-to-end self-play iteration. Loads `P`, calls Alice to produce `Q` and a proof (SEQ) or diverging input (SINQ), verifies with Liquid Haskell or execution, queries Bob for difficulty, saves data, and updates cumulative buffers.
- `finetune.py`: Fine-tunes a base model with LoRA on JSONL conversation-style data (`system_prompt`, `user_prompt`, `output`). Saves adapters per epoch/checkpoint.
- `SEQ_7B_random_500.sh`: Example Slurm job orchestrating multiple self-play iterations and adapter fine-tuning across GPUs.


### Requirements
- GHC (Glasgow Haskell Compiler)
- Liquid Haskell plugin and `liquid-prelude`



### Dataset

You can use:
- A local Hugging Face dataset directory (created by `datasets` `save_to_disk`),
- A local JSONL file (`.jsonl`),
- Or a Hub dataset name.

The dataset OpInstruct-HSx could be found in `https://huggingface.co/datasets/Trevor0501/OpInstruct-HSx`


### Running a single iteration locally

```bash
python SEQ.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_name ../data/successfully_compiled_sorted_haskell_dataset \
  --iteration 1 \
  --iteration_dir output/iter_1 \
  --num_initial_programs 100 \
  --max_tokens 32768 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --gpu_memory_utilization 0.95 \
  --tensor_parallel_size 1 \
  --difficulty_threshold 5
```

Optional:
- `--alice_adapter_path` and `--bob_adapter_path` to start from prior LoRA adapters.
- `--cumulative_alice_training_data_path` and `--cumulative_bob_training_data_path` to continue from previous buffers.

The script will:
- Load and validate programs.
- For each `P`, with 50% chance run SEQ (equivalence + Liquid Haskell proof) or SINQ (inequivalence + diverging input).
- Query Bob for difficulty (multiple samples).
- Save per-iteration artifacts under `--iteration_dir`.


### Running multi-iteration self-play with Slurm

The provided Slurm script runs N iterations, then fine-tunes adapters between rounds if new data were saved:

```bash
sbatch SEQ_7B_random_500.sh
```

Before submitting, consider updating in `SEQ_7B_random_500.sh`:
- `MODEL_NAME`, `DATASET_NAME`, `NUM_INITIAL_PROGRAMS`, `N_ITERATIONS`
- Learning rate and epochs for fine-tuning
- GPU partition and resource settings
- `accelerate_config.yaml` path (used by `accelerate launch`)

Environment notes (already in the script):
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `HF_HUB_OFFLINE=1` for offline runs (remove if you need to fetch from Hub)


### Fine-tuning adapters

`finetune.py` expects JSONL with records of the form:
```json
{"system_prompt": "...", "user_prompt": "...", "output": "..."}
```
Basic usage:
```bash
accelerate launch --config_file accelerate_config.yaml \
  finetune.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path output/iter_1/alice_training_data.jsonl \
  --model_type alice \
  --output_dir output/iter_1/alice_adapters \
  --iteration 1 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1
```
Important flags (see `finetune.py`):
- `--lora_r` (default 8), `--lora_alpha` (16), `--lora_dropout` (0.05)
- `--previous_adapter_path` to continue from a prior checkpoint

The Slurm script demonstrates how to run fine-tuning after each iteration and how to pick the latest checkpoint directory to reuse in the next round.


### Outputs

For each iteration directory (e.g., `output/EXPERIMENT/iteration_1`):
- `iteration_data.jsonl`: Accepted examples with metadata (e.g., status, difficulty).
- `alice_training_data.jsonl`: Aggregated training data for Alice.
- `bob_training_data.jsonl`: Aggregated training data for Bob (if enabled).
- `seq_summary.jsonl`: Summary statistics for the iteration.
- `alice_adapters/` or `bob_adapters/`: LoRA checkpoints created by fine-tuning.

### Reference

This framework follows the semantic self-play formulation inspired by Miceli et al. (2025). The difficulty-driven curriculum has no theoretical upper bound on improvement, enabling continual learning about complex program semantics via adversarial interaction between agents.
