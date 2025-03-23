import torch

def run_inference(model, tokenizer, prompt, device, logger):
    logger.info(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=10000)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    logger.info(f"Model Output: {output_text}")
