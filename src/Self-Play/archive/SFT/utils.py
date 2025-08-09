# Shared utility functions 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel # Added PeftModel
from typing import Optional

def load_model_for_training(
    model_name_or_path: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    tokenizer: Optional[AutoTokenizer] = None,
    adapter_path: Optional[str] = None # Added adapter_path
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
        attn_implementation="flash_attention_2"
    )
    model.to('cuda')

    # --- LoRA Configuration ---
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    if adapter_path: # Load existing adapter
        print(f"Loading existing adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else: # Create new PEFT model
        model = get_peft_model(model, lora_config)
    
    print("LoRA configured model ready for training.")
    model.print_trainable_parameters()
    
    return model, tokenizer 