import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import setup_logger
from inference import run_inference
from finetune import run_finetuning

def main():
    
    parser = argparse.ArgumentParser(description="Run inference or fine-tuning")
    parser.add_argument('--mode', choices=['inference', 'finetune'], required=True,
                        help="Mode to run: inference or finetune")
    parser.add_argument('--prompt', type=str, default="What is the capital of France?",
                        help="Prompt for inference")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs for fine-tuning")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--max_length', type=int, default=2048, help="Maximum sequence length")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    
    # Model and tokenizer loading in main.py
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else {"": device}
    )

    # Log the number of GPUs being used
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {num_gpus}")
    else:
        logger.info("No GPU available, using CPU")

    if torch.cuda.is_available():
        logger.info(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=False))

    
    if args.mode == 'inference':
        run_inference(model, tokenizer, args.prompt, device, logger)

    elif args.mode == 'finetune':
        run_finetuning(
            model=model,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            lr=args.lr
        )

if __name__ == "__main__":
    main()
