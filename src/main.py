# Main.py

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)
from utils.logger import setup_logger
from inference import run_inference
from finetune import run_finetuning
from evaluate import run_evaluation

def main():

    parser = argparse.ArgumentParser(description="Run inference or fine-tuning")
    parser.add_argument('--mode', choices=['inference', 'finetune', 'evaluate'], required=True,
                        help="Mode to run: inference, finetune or evaluate")
    parser.add_argument('--prompt', type=str, default="What is the capital of France?",
                        help="Prompt for inference")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for fine-tuning")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--max_length', type=int, default=2048, help="Maximum sequence length")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name")
    parser.add_argument('--warmup_steps', type=int, default=100, help="Warmup steps")
    parser.add_argument('--early_stopping_patience', type=int, default=2, help="Early stopping patience")
    parser.add_argument('--use_subset', action='store_true', default=True, help="Use a subset of the data for fine-tuning")
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    model_name = args.model_name
    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)

    # 8-bit quantization config
    # bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    # Load the model in 8-bit and shard across available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_cfg,
        device_map="auto"
    )

    # Compile with Torch 2.0 for kernel fusion
    model = torch.compile(model)

    # Log GPU count
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {num_gpus}")
    else:
        logger.info("No GPU available, using CPU")

    # Dispatch based on mode
    if args.mode == 'inference':
        run_inference(model, tokenizer, args.prompt, device, logger)

    elif args.mode == 'finetune':
        run_finetuning(
            model=model,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
            model_name=model_name, 
            epochs=args.epochs,
            lr=args.lr, 
            warmup_steps=args.warmup_steps,
            max_length=args.max_length,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            use_subset=args.use_subset 
        )
    
    elif args.mode == 'evaluate':
        run_evaluation(model, tokenizer, device, logger)

if __name__ == "__main__":
    main()
