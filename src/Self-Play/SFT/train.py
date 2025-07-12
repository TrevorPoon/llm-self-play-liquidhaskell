# Main training script
import os
import argparse
import torch
import subprocess
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    TrainerCallback
)
from utils import load_model_for_training

class MemoryUsageCallback(TrainerCallback):
    """
    A TrainerCallback that logs GPU memory usage during training.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if state.is_world_process_zero and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            if logs is not None:
                logs['memory_allocated_gb'] = round(allocated, 4)
                logs['memory_reserved_gb'] = round(reserved, 4)
                logs['max_memory_allocated_gb'] = round(max_allocated, 4)

class HumanEvalOnSaveCallback(TrainerCallback):
    """
    A TrainerCallback that runs the HumanEval evaluation script via sbatch on each save.
    """
    def __init__(self, model_name, n_humaneval_evaluations):
        self.model_name = model_name
        self.n_humaneval_evaluations = n_humaneval_evaluations
        self.eval_working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Evaluation/HumanEval'))
        self.eval_script_path = os.path.join(self.eval_working_dir, 'eval_script/eval_adapter.sh')

        print("--- HumanEval Callback Initialized ---")
        print(f"  Evaluation script path: {self.eval_script_path}")
        print(f"  Evaluation working directory: {self.eval_working_dir}")
        print("------------------------------------")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            adapter_path_abs = os.path.abspath(checkpoint_dir)

            print(f"\n--- Submitting HumanEval script for adapter: {adapter_path_abs} ---")

            try:
                # Submit Haskell evaluation
                # hs_command = ['sbatch', self.eval_script_path, adapter_path_abs, "hs", self.model_name, str(self.n_humaneval_evaluations)]
                # print(f"  Running command: {' '.join(hs_command)}")
                # subprocess.run(hs_command, check=True, cwd=self.eval_working_dir)
                # print(f"  Successfully submitted Haskell evaluation script.")

                # # Submit Python evaluation
                # py_command = ['sbatch', self.eval_script_path, adapter_path_abs, "python", self.model_name, str(self.n_humaneval_evaluations)]
                # print(f"  Running command: {' '.join(py_command)}")
                # subprocess.run(py_command, check=True, cwd=self.eval_working_dir)
                # print(f"  Successfully submitted Python evaluation script.")

                pass

            except subprocess.CalledProcessError as e:
                print(f"  ERROR: Failed to submit evaluation script. Error: {e}")
            except FileNotFoundError:
                print(f"  ERROR: 'sbatch' command not found. Make sure you are in a Slurm environment.")
                
                print("-------------------------------------------------------------------\n")

def main():
    # Recommended by torchdynamo logs for dealing with tensor.item() graph breaks
    torch._dynamo.config.capture_scalar_outputs = True

    parser = argparse.ArgumentParser(description="Fine-tune a model on the Haskell dataset.")
    
    # Model and Data Arguments
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the processed dataset directory.")
    parser.add_argument('--dataset_is_tokenized', action='store_true', help="Flag to indicate if the dataset is already tokenized.")
    parser.add_argument('--output_dir', type=str, default='./sft-output', help="Directory to save the final adapter and any checkpoints.")
    parser.add_argument('--dataset_fraction', type=float, default=1.0, help="Fraction of the dataset to use for training and validation (e.g., 0.1 for 10%).")
    
    # LoRA Arguments
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r.")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")

    # Training Arguments
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size per device during training.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, help="Batch size per device during evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument('--eval_accumulation_steps', type=int, default=8, help="Number of evaluation steps to accumulate predictions for before moving to CPU.")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate for fine-tuning.")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument('--saving_strategy', type=str, default="epoch", help="Saving strategy.")
    
    # HumanEval Arguments
    parser.add_argument('--run_humaneval_evaluation', action='store_true', help="Flag to run HumanEval evaluation on each save.")
    parser.add_argument('--n_humaneval_evaluations', type=int, default=4, help="Number of HumanEval problems to evaluate for each language.")
    parser.add_argument('--log_memory_usage', action='store_true', help="Flag to log GPU memory usage during training.")
    parser.add_argument('--adapter_path', type=str, default=None, help="Path to an existing adapter to continue fine-tuning.")
    
    args = parser.parse_args()

    print("Training arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # --- Load Datasets ---
    print(f"Loading dataset from {args.dataset_path}")
    processed_dataset = load_from_disk(args.dataset_path)

    # --- Optional: Subsample the dataset for quick tests ---
    if args.dataset_fraction < 1.0:
        print(f"Using {args.dataset_fraction * 100:.0f}% of the training and validation datasets.")
        
        train_subset_size = int(len(processed_dataset['train']) * args.dataset_fraction)
        processed_dataset['train'] = processed_dataset['train'].select(range(train_subset_size))
        
        eval_subset_size = int(len(processed_dataset['validation']) * args.dataset_fraction)
        processed_dataset['validation'] = processed_dataset['validation'].select(range(eval_subset_size))

    if args.dataset_is_tokenized:
        print("Dataset is pre-tokenized. Loading tokenizer from dataset directory.")
        # When tokenized, the tokenizer is saved with the dataset
        tokenizer = AutoTokenizer.from_pretrained(args.dataset_path)
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["validation"]
    else:
        # --- Load Tokenizer ---
        print("Loading tokenizer for on-the-fly tokenization...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Tokenize Dataset ---
        def tokenize_function(examples):
            # We tokenize the 'text' field which contains our formatted prompt
            # Padding is now handled dynamically by the DataCollatorForLanguageModeling.
            return tokenizer(examples["text"], truncation=True, max_length=4096)

        print("Tokenizing dataset...")
        tokenized_dataset = processed_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )

        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["validation"]
    
    # --- Training ---
    print("Initializing model for training...")
    peft_model, _ = load_model_for_training(
        args.model_name_or_path,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        tokenizer,
        args.adapter_path # Pass the adapter_path here
    )
    peft_model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        eval_strategy=args.saving_strategy,
        eval_steps=args.save_steps,
        save_strategy=args.saving_strategy,
        save_steps=args.save_steps,
        bf16=True,
        tf32=True,
        torch_compile=True,
        save_total_limit=None,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        weight_decay=0.01 # Added for regularization to combat overfitting
    )

    callbacks = []
    if args.run_humaneval_evaluation:
        callbacks.append(HumanEvalOnSaveCallback(
            model_name=args.model_name_or_path,
            n_humaneval_evaluations=args.n_humaneval_evaluations
        ))
    
    if args.log_memory_usage:
        callbacks.append(MemoryUsageCallback())

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks
    )

    print("Starting training...")
    trainer.train()

    # Manually load the best model adapter
    if trainer.state.best_model_checkpoint:
        print(f"Loading best model from {trainer.state.best_model_checkpoint}")
        adapter_name = list(trainer.model.peft_config.keys())[0]
        trainer.model.load_adapter(trainer.state.best_model_checkpoint, adapter_name=adapter_name)

    print("Saving final LoRA adapter...")
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    print(f"Training complete. Final adapter saved to {final_adapter_path}")


if __name__ == "__main__":
    main() 