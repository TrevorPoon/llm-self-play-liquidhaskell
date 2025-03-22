from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in half-precision (to save memory)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")

# Define prompt
prompt = "What is the capital of France?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
with torch.no_grad():
    output_tokens = model.generate(**inputs, max_length=100)

# Decode the output
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(output_text)
