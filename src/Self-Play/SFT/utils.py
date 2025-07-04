# Shared utility functions 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import Optional

def load_model_for_training(
    model_name_or_path: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    tokenizer: Optional[AutoTokenizer] = None
):
    """
    Loads a model and tokenizer for training, applying LoRA configuration.

    Args:
        model_name_or_path (str): The name or path of the base model.
        lora_r (int): The rank for LoRA.
        lora_alpha (int): The alpha parameter for LoRA.
        lora_dropout (float): The dropout rate for LoRA.
        tokenizer (Optional[AutoTokenizer]): An optional pre-loaded tokenizer.

    Returns:
        tuple: A tuple containing the configured model and tokenizer.
    """
    
    # --- Load Tokenizer ---
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

    # --- Load Model ---
    # Optimized for A40 GPUs with bfloat16 and Flash Attention 2
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": torch.cuda.current_device()} # Ensures model is on the correct device for FSDP
    )

    # --- LoRA Configuration ---
    # Target modules can vary by model architecture.
    # For Llama-like models, these are common targets.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    print("LoRA configured model ready for training.")
    model.print_trainable_parameters()
    
    return model, tokenizer 