# LLM Self-Play with Liquid Haskell (SEQ/SINQ)

[![Haskell](https://img.shields.io/badge/Haskell-%235e5086?logo=haskell&logoColor=white)](https://www.haskell.org/)
[![Liquid Haskell](https://img.shields.io/badge/Liquid%20Haskell-%23007ACC)](https://github.com/ucsd-progsys/liquidhaskell)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## What this project is

A self-play framework that teaches language models to reason about program semantics using Haskell and Liquid Haskell. Two agents interact adversarially per iteration:
- **Alice** proposes a Haskell variant `Q` of a reference program `P`, and either proves equivalence with a Liquid Haskell proof (SEQ) or provides a counterexample input showing inequivalence (SINQ).
- **Bob** evaluates the semantic difficulty of `(P, Q)` by repeated sampling and scores how often models succeed, inducing a curriculum signal.

Optionally, we fine-tune lightweight LoRA adapters (via PyTorch) using the highest-quality, auto-validated data from each iteration, enabling continual improvement.

- Detailed SEQ/SINQ pipeline: `src/Self-Play/SEQ_v2/README.md`
- Dataset pipeline (OpInstruct-HSx): `src/Self-Play/OpInstruct-HSx/README.md`

### Why Haskell + Liquid Haskell + PyTorch?
- **Haskell**: Strong static types and purity make programs concise and analyzable.
- **Liquid Haskell**: SMT-backed refinement typing enables machine-checkable equivalence proofs.
- **PyTorch**: Efficiently fine-tunes LoRA adapters on the generated conversation-style data for both agents.

### How it works (high level)
1. Load a reference program `P` from a dataset.
2. With 50/50 probability, branch to:
   - **SEQ** (equivalence): Alice proposes `Q` and a proof; we verify with Liquid Haskell.
   - **SINQ** (inequivalence): Alice proposes `Q` and a diverging input; we compile and run both `P` and `Q` on that input to verify divergence.
3. Discard any non-compiling or non-validated cases.
4. **Bob** samples the difficulty of `(P, Q)` and outputs a continuous score: `difficulty = 10 * (1 - success_rate)`.
5. Keep hard examples and a stratified sample of easy ones; fine-tune adapters for the next round.

## Key components
- `src/Self-Play/SEQ_v2/SEQ_miceli_random_v2.py`: Orchestrates one self-play iteration (load data, run SEQ/SINQ, validate via Liquid Haskell or execution, query Bob for difficulty, save outputs, and optionally fine-tune).
- `src/Self-Play/SEQ_v2/finetune.py`: LoRA adapter fine-tuning script (PyTorch + Accelerate).
- `src/Self-Play/SEQ_v2/SEQ_7B_random_500.sh`: Example Slurm script for multi-iteration runs and interleaved fine-tuning.
- `src/Self-Play/OpInstruct-HSx/`: End-to-end dataset generation, validation, and filtering pipeline for runnable Haskell snippets.

### Requirements
- GHC (Glasgow Haskell Compiler)
- Liquid Haskell
- Python 3.10+ with PyTorch, `transformers`, `datasets`, `accelerate`, and optionally vLLM for fast inference

### Quickstart: run a single SEQ/SINQ iteration
```bash
python src/Self-Play/SEQ_v2/SEQ_miceli_random_v2.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_name ./data/successfully_compiled_sorted_haskell_dataset \
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

### Outputs per iteration
- `iteration_data.jsonl`: Accepted examples and metadata
- `alice_training_data.jsonl`, `bob_training_data.jsonl`: Conversation-style training data
- `seq_summary.jsonl`: Summary metrics
- `alice_adapters/`, `bob_adapters/`: LoRA checkpoints (if fine-tuned)

## Dataset: OpInstruct-HSx
- Public dataset: `https://huggingface.co/datasets/Trevor0501/OpInstruct-HSx`
- This repo includes scripts to synthesize Haskell code from high-quality Python instruction data and filter to a runnable subset via compilation and execution.
  - Generate: `src/Self-Play/OpInstruct-HSx/synthetic_dataset_generation/generate_synthetic_data.py`
  - Validate & run-filter: `src/Self-Play/OpInstruct-HSx/dataset_filtering/{validate_dataset.py, generate_haskell_inputs.py}`

## Learn more
- SEQ/SINQ framework details: `src/Self-Play/SEQ_v2/README.md`
- Dataset pipeline and tiers: `src/Self-Play/OpInstruct-HSx/README.md`
