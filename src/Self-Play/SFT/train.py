# Main training script
import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
from utils import load_model_for_training

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on the Haskell dataset.")
    
    # Model and Data Arguments
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the processed dataset directory.")
    parser.add_argument('--dataset_is_tokenized', action='store_true', help="Flag to indicate if the dataset is already tokenized.")
    parser.add_argument('--output_dir', type=str, default='./sft-output', help="Directory to save the final adapter and any checkpoints.")
    
    # LoRA Arguments
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r.")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")

    # Training Arguments
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate for fine-tuning.")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every X updates steps.")
    # parser.add_argument('--early_stopping_patience', type=int, default=2, help="Number of evaluation steps with no improvement to wait before stopping.")
    
    args = parser.parse_args()

    print("Training arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # --- Load Datasets ---
    print(f"Loading dataset from {args.dataset_path}")
    processed_dataset = load_from_disk(args.dataset_path)

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
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=4096)

        print("Tokenizing dataset...")
        tokenized_dataset = processed_dataset.map(
            tokenize_function, 
            batched=True, 
            num_proc=os.cpu_count(), 
            remove_columns=["text"]
        )

        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["validation"]
    
    # --- Training ---
    print("Initializing model for training...")
    model, _ = load_model_for_training(
        args.model_name_or_path,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        tokenizer
    )
    model.enable_input_require_grads()
    # Recommended for gradient checkpointing
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=args.save_steps,
        bf16=True,
        tf32=True,
        torch_compile=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to="tensorboard",
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting training...")
    trainer.train()

    print("Saving final LoRA adapter...")
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    print(f"Training complete. Final adapter saved to {final_adapter_path}")


if __name__ == "__main__":
    main() 