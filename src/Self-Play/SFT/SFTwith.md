# Supervised Fine-Tuning Pipeline for Haskell Code Generation

This document outlines the steps to build a supervised fine-tuning (SFT) pipeline to train a Large Language Model (LLM) on a Haskell dataset. The final output will be a LoRA adapter that can be used for efficient inference. The pipeline will be optimized for a multi-GPU setup with 4x A40 GPUs.

## 1. Project Goal

The primary goal is to fine-tune a pre-trained LLM to improve its Haskell code generation capabilities using a given Haskell dataset. We will leverage LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

## 2. Code Design and Modularization

To ensure the codebase is maintainable, reusable, and easy to debug, the Python scripts will be modularized. Core logic will be encapsulated into functions and classes.

- **`utils.py`**: A new file for shared utilities, such as loading datasets, prompt formatting, and model loading logic that might be used across training and inference.
- **`prepare_data.py`**: Focuses solely on processing the raw dataset and saving the splits.
- **`train.py`**: Contains the main training loop, leveraging utility functions.
- **`inference.py`**: A clean script for running predictions, reusing code from `utils.py`.

## 3. Environment Setup

- **Hardware**: 4x NVIDIA A40 GPUs.
- **Software**:
    - PyTorch with CUDA support.
    - Hugging Face libraries: `transformers`, `peft`, `accelerate`, `datasets`.
    - `flash-attn` for Flash Attention 2 optimization.
    - `optuna` and `scipy` for hyperparameter optimization.

### Configure `accelerate` for Multi-GPU
Before running the training, configure `accelerate` to leverage all 4 GPUs efficiently. FSDP (Fully Sharded Data Parallelism) is recommended for large models.

Run this in your terminal and follow the prompts:
```bash
accelerate config
```
**Example Configuration for 4 GPUs with FSDP:**
- `compute_environment`: `This machine`
- `distributed_type`: `FSDP`
- `num_machines`: `1`
- `num_processes`: `4` (one per GPU)
- `use_cpu`: `False`
- FSDP specific settings can be fine-tuned for performance.

## 4. Data Preparation

### Dataset
The pipeline will use a dataset of Haskell code. This script will split the data into **training, development (validation), and test sets** (e.g., in an 80/10/10 ratio) to ensure robust evaluation. We assume each sample has a `code` field.

### Prompting Strategy
To teach the model effectively, we need to format the raw code into a detailed instruction-following format.

**Example Prompt Template:**
```
You are an expert Haskell programmer. Your task is to write a correct and efficient Haskell function that solves the given problem. Please provide a type signature for your function.

### Instruction:
Write a Haskell function named `sumEven` that takes a list of integers and returns the sum of all the even numbers in the list.

### Response:
```haskell
sumEven :: [Int] -> Int
sumEven xs = sum (filter even xs)
```
```

### `prepare_data.py`
This script will:
1. Load the initial dataset using `datasets.load_dataset` or `load_from_disk`.
2. Perform a train-validation-test split (e.g., 80%/10%/10%) using `dataset.train_test_split()`.
3. Apply the detailed prompting template to format each example in all splits.
4. Tokenize the formatted text using the model's tokenizer.
5. Save the processed splits to disk for fast loading during training and evaluation.

## 5. Model Training (`train.py`)

This will be the main script to run the fine-tuning job.

### 5.1. Key Components
- **Argument Parsing**: Use `argparse` to configure model name, dataset path, training parameters, etc.
- **Model Loading**:
    - Load tokenizer: `AutoTokenizer.from_pretrained`. Add a padding token if it's missing.
    - Load base model: `AutoModelForCausalLM.from_pretrained`.
        - Use `torch_dtype=torch.bfloat16` which is optimal for A40 GPUs.
        - Add `attn_implementation="flash_attention_2"` to use Flash Attention.
- **LoRA Configuration**:
    - Use `peft.LoraConfig` to define which modules to apply LoRA to (e.g., `q_proj`, `v_proj`, `k_proj`, `o_proj`).
    - Wrap the base model with `peft.get_peft_model`.
- **Training Arguments**:
    - Use `transformers.TrainingArguments`.
    - `bf16=True`: Enable bfloat16 mixed-precision training.
    - `output_dir`: Where to save checkpoints and the final adapter.
    - `per_device_train_batch_size`: Adjust based on VRAM (e.g., 1, 2, or 4).
    - `gradient_accumulation_steps`: Use to increase effective batch size without increasing memory.
    - `gradient_checkpointing=True`: To save VRAM at the cost of some re-computation.
    - Set learning rate, number of epochs, logging steps, etc.
- **Trainer**:
    - Instantiate `transformers.Trainer` with the model, training args, and datasets.
    - Call `trainer.train()`.
- **Saving**:
    - After training, save the LoRA adapter using `model.save_pretrained(output_dir)`. The tokenizer should also be saved.

### 5.2. Optimization for 4x A40 GPUs
1.  **Distributed Training with `accelerate`**: The `train.py` script must be launched using `accelerate launch train.py [args...]`. `Trainer` will automatically use the `accelerate` environment.
2.  **FSDP**: The `accelerate` configuration will enable FSDP, which shards model parameters, gradients, and optimizer states across GPUs. This is the most effective way to train large models on multi-GPU setups.
3.  **Flash Attention 2**: When loading the model, specify `attn_implementation="flash_attention_2"`. This requires `flash-attn` to be installed and can speed up training and reduce memory usage significantly.
4.  **`bfloat16` Mixed Precision**: A40s have Tensor Cores that excel with `bfloat16`. It offers a good balance of speed and precision. Enable with `bf16=True` in `TrainingArguments`.
5.  **Gradient Accumulation & Checkpointing**: These are crucial memory-saving techniques that allow for larger models and batch sizes.

### 5.3. Hyperparameter Tuning with Bayesian Optimization
To find the optimal hyperparameters, we will use Bayesian Optimization, which is more efficient than grid search or random search.

- **Library**: `optuna` will be integrated with the `transformers.Trainer`.
- **Process**:
    1.  **Define a Search Space**: Specify ranges for key hyperparameters (e.g., `learning_rate`, `lora_r`, `lora_alpha`, `lora_dropout`).
    2.  **Objective Function**: The `Trainer`'s objective will be to minimize the validation loss.
    3.  **Run Search**: Use the `trainer.hyperparameter_search()` method with `backend="optuna"` to automate the search process. The best trial's parameters will be used for the final training run.

## 6. Inference (`inference.py`)

A script to test the trained LoRA adapter.
1.  Load the base model in `bfloat16`.
2.  Load the LoRA adapter from the training output directory using `peft.PeftModel.from_pretrained`.
3.  Merge LoRA weights for faster inference: `model = model.merge_and_unload()`.
4.  Use a `transformers.pipeline` or `model.generate()` to generate Haskell code from a test prompt.

## 7. Proposed Directory Structure

```
SFT-Haskell/
├── SFTwith.md            # This outline
├── requirements.txt      # Project dependencies
├── utils.py              # Shared utility functions
├── prepare_data.py       # Script to preprocess and tokenize data
├── train.py              # Main training script
└── inference.py          # Script for testing the adapter
``` 