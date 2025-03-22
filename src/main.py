from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Model setup
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
logger.info(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else {"": device}
)

# Prompt and inference
prompt = "What is the capital of France?"
logger.info(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_tokens = model.generate(**inputs, max_length=100)

output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
logger.info(f"Model Output: {output_text}")
