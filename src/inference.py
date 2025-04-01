import torch

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
    
    final_prompt = prompt_style.format(prompt, "", "")
    logger.info(f"Final Prompt: {final_prompt}")

    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  # Ensure all GPU ops are finished before starting timing
    start_event.record()
    
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=10000)
    
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded
    inference_time = start_event.elapsed_time(end_event) / 1000.0 # elapsed_time returns milliseconds; convert to seconds

    logger.info(f"Output Tokens: {output_tokens}")
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    logger.info(f"Inference time: {inference_time:.3f} seconds")
    logger.info(f"Model Output: {output_text}")
