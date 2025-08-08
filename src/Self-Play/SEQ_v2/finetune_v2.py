import os
import torch
import argparse
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import Dataset
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SavePeftModelCallback(TrainerCallback):
    def __init__(self, model_type=None, iteration=None):
        self.model_type = model_type
        self.iteration = iteration

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        output_dir = args.output_dir
        model = kwargs['model']
        
        if self.model_type is not None and self.iteration is not None:
            adapter_dir_name = f"{self.model_type}-adapter-iter-{self.iteration}-epoch-{int(epoch)}"
        else:
            adapter_dir_name = f"adapter-epoch-{int(epoch)}"

        adapter_path = os.path.join(output_dir, adapter_dir_name)
        model.save_pretrained(adapter_path)
        logger.info(f"Saved adapter for epoch {int(epoch)} to {adapter_path}")

class SInQ_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_tokens):
        self.data = data
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        system = ex.get("system_prompt", "").strip()
        user = ex.get("user_prompt", "").strip()
        output = ex.get("output", "").strip()

        # Build prompt and full sequence. Model sees prompt+output, but loss only on output.
        prompt = system + "\n" + user + "\n"
        full_sequence = prompt + output + self.tokenizer.eos_token

        # Tokenize prompt alone to get its length (no padding here).
        prompt_tok = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_tokens,
        )
        prompt_len = len(prompt_tok["input_ids"])

        # Tokenize full sequence with padding/truncation to max_tokens.
        full_tok = self.tokenizer(
            full_sequence,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = full_tok["input_ids"]

        # Prepare labels: mask out prompt tokens.
        labels = input_ids.copy()
        if prompt_len >= len(labels):
            # prompt consumed entire context window; no output left. 
            # To avoid spurious loss, make all labels -100.
            labels = [-100] * len(labels)
        else:
            for i in range(prompt_len):
                labels[i] = -100  # no loss on prompt

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def load_training_data(file_path):
    return load_dataset('json', data_files=file_path, split='train')

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script for SInQ")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the training data file (.txt).")
    parser.add_argument('--model_type', type=str, required=True, choices=['alice', 'bob'], help="Type of model to fine-tune.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained adapter.")
    parser.add_argument('--previous_adapter_path', type=str, default=None, help="Path to a previous LoRA adapter to continue fine-tuning from.")
    parser.add_argument('--iteration', type=int, required=True, help="Current self-play iteration number.")
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for fine-tuning.")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r.")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Maximum number of tokens for the model's context window.")

    args = parser.parse_args()

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    # --- Tokenizer and Dataset Loading ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    training_data = load_training_data(args.dataset_path)
    if not training_data:
        logger.error(f"No training data found in {args.dataset_path}. Exiting.")
        return

    train_dataset = SInQ_Dataset(training_data, tokenizer, args.max_tokens)

    # --- Model Initialization ---
    logger.info("Initializing base model for fine-tuning...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    if args.previous_adapter_path and os.path.exists(args.previous_adapter_path):
        logger.info(f"Loading adapter from {args.previous_adapter_path} to continue fine-tuning.")
        peft_model = PeftModel.from_pretrained(base_model, args.previous_adapter_path, is_trainable=True)
    else:
        if args.previous_adapter_path:
            logger.warning(f"Adapter path {args.previous_adapter_path} not found. Creating a new adapter.")
        logger.info("Creating a new PeftModel for training from base model.")
        peft_model = get_peft_model(base_model, lora_config)

    peft_model.enable_input_require_grads()
    peft_model.config.use_cache = False

    # --- Trainer Setup and Execution ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=10,
        save_strategy="epoch",
        save_steps=500,
        bf16=True,
        tf32=True,
        torch_compile=False,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    logger.info(f"Starting fine-tuning for {args.model_type}...")
    trainer.train()
    logger.info("Fine-tuning complete.")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
