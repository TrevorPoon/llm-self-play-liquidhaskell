## OpInstruct-HSx: Synthetic and Validated Haskell Dataset for SEQ/SINQ

OpInstruct-HSx is a reproducible pipeline and dataset for generating, filtering, and validating Haskell functions for semantic reasoning tasks such as SEQ and SINQ self-play. It generates simple, self-contained Haskell code from high-quality Python instruction data and then compile- and run-validates the results.

- **Dataset (Hugging Face)**: `https://huggingface.co/datasets/Trevor0501/OpInstruct-HSx`
- **Paper context**: High-quality Haskell data is scarce; we synthesize Haskell from `nvidia/OpenCodeInstruct` prompts using `DeepSeek-R1-Distill-Llama-70B` and keep only compile- and run-validated programs.
- **Scale**: Approximately 28,000 validated, executable Haskell functions after filtering.

### Why this dataset?
- **Synthetic from strong supervision**: We adapt high-quality Python instruction–completion pairs (OpenCodeInstruct) into simple Haskell tasks, then compile- and run-validate them. This yields an executable subset suitable for training and evaluation in reasoning-heavy settings.

## Repo structure (this folder)
- `synthetic_dataset_generation/generate_synthetic_data.py`: Generate Haskell code from `nvidia/OpenCodeInstruct` via vLLM and `DeepSeek-R1-Distill-Llama-70B`.
- `dataset_filtering/validate_dataset.py`: Compile-check Haskell programs (GHC -c) and save the validated subset.
- `dataset_filtering/generate_haskell_inputs.py`: Synthesize simple, type-correct inputs and keep only programs that both compile and execute successfully.

Ensure GHC is installed and available on your system path. The executable validation step invokes GHC to compile and run small programs.

## End-to-end pipeline

### A. Synthetic Haskell from OpenCodeInstruct
We adapt Python instruction–completion pairs from `nvidia/OpenCodeInstruct` into simple Haskell tasks using vLLM and `DeepSeek-R1-Distill-Llama-70B`.

Generate synthetic Haskell (adjust `--num_samples` to your budget):
```bash
python src/Self-Play/OpInstruct-HSx/synthetic_dataset_generation/generate_synthetic_data.py \
  --num_samples 5000 \
  --output_dir ./data/synthetic_haskell_dataset \
  --output_filename_arrow synthetic_haskell_dataset_OpenCode_Instruct.arrow \
  --output_filename_jsonl synthetic_haskell_dataset_OpenCode_Instruct.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --gpu_memory_utilization 0.8 \
  --max_new_tokens 4096 \
  --max_model_len 8192 \
  --dtype bfloat16
```
Outputs:
- Arrow: `./data/synthetic_haskell_dataset/synthetic_haskell_dataset_OpenCode_Instruct.arrow`
- JSONL: `./data/synthetic_haskell_dataset/synthetic_haskell_dataset_OpenCode_Instruct.jsonl`


### B. Executable subset via input synthesis and run-validation
We heuristically extract function name and argument type, synthesize a small literal input (supporting base types like `Int`, `Bool`, `String`, lists, tuples, `Maybe`, `Either`), compile with GHC, run with the synthesized input, and keep only programs that both compile and execute.

Run on JSONL (easiest):
```bash
python src/Self-Play/OpInstruct-HSx/dataset_filtering/generate_haskell_inputs.py \
  --dataset_name ./data/synthetic_haskell_dataset/synthetic_haskell_dataset_OpenCode_Instruct.jsonl \
  --output_dir ./data/SINQ_synthetic_haskell_dataset \
  --output_hf_dataset_dir ./data/SINQ_synthetic_haskell_dataset_hf \
  --timeout 20
```

Alternatively, you can point `--dataset_name` to a dataset saved with `datasets.save_to_disk(...)` and the script will load it from disk.

Outputs:
- JSONL with code, synthesized input, and run output: `./data/SINQ_synthetic_haskell_dataset/haskell_dataset_with_generated_inputs.jsonl`
- HF dataset on disk with the successful subset: `./data/SINQ_synthetic_haskell_dataset_hf`


## Three-tier data strategy (recommended)
Do not destructively filter; structure into tiers for different uses:

- **A: Raw**: All raw code after light de-dupe/parse. Use for pretraining, code infilling, and syntax repair; prioritize recall.
- **B: Typechecked / Needs-Input**: Passes `validate_dataset.py` (has type sig, binding, GHC -fno-code equivalent). Use for theory-phase self-play (reason about types, generate candidate inputs symbolically).
- **C: Runnable Sample**: Subset of B where `generate_haskell_inputs.py` produced a value and the harness compiled & ran. Use for high-trust evaluation, automated semantic inequivalence scoring, and RL signals.

## Reproducibility and notes
- The sorting step uses output length and average test quality within OpenCodeInstruct to prioritize simpler and higher-quality prompts before generation.
- The input synthesizer uses a recursive literal generator to produce parsable values for common types; non-parsable or effectful types are skipped.
- Only programs that compile and execute on the synthesized input are retained.
- The full production pipeline is implemented in this repository and was used to produce the public dataset.

## References
- OpenCodeInstruct: `https://huggingface.co/datasets/nvidia/OpenCodeInstruct`
- DeepSeek-R1-Distill-Llama-70B: `https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B`

