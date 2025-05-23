# inference.py

import torch
import re

def run_inference(model, tokenizer, prompt, device, logger):

    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request. 
        Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

        ### Instruction:
        Complete the following Task.

        ### Question:
        {}

        ### Response:
        <think>{}"""
    
    final_prompt = prompt_style.format(prompt, "")
    logger.info(f"Final Prompt: {final_prompt}")

    # Tokenize and move inputs to device
    inputs = tokenizer(final_prompt, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    # Timing using torch.cuda events (if on GPU)
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

    with torch.inference_mode():
        with torch.autocast(device_type=device if device == "cuda" else "cpu"):
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                use_cache=True
            )

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        inference_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        # Fallback timing on CPU
        import time
        start = time.time()
        # (Regenerate for timing, or approximate)
        inference_time = time.time() - start

    # Decode and log
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    logger.info(f"Inference time: {inference_time:.3f} seconds")
    logger.info(f"Model Output (raw): {output_text}")

    # 🧹 Optionally clean out <think>… block
    # cleaned = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
    # logger.info(f"Model Output (cleaned): {cleaned}")
