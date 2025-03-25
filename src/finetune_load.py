import os
import random
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from utils.logger import log_gpu_info
import datetime
from datasets import load_dataset

def batchify(data, batch_size):
    """
    Yields batches from a list of data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def evaluate(model, tokenizer, text_data, device, max_length, batch_size, logger):
    """
    Evaluates the given model on a list of text samples using mixed precision,
    and returns the average loss. Additionally, logs the inputs and predicted outputs
    for the last 5 batches for manual inspection.
    
    It logs:
    1) Eval set input: The part before "Answer:".
    2) Eval set output: The expected output (the part after "Answer:").
    3) Model prediction: The model's predicted sequence.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    last_batches = []  # Will store info for the last 5 batches

    with torch.no_grad():
        batches = list(batchify(text_data, batch_size))
        progress_bar = tqdm(batches, desc="Evaluation", mininterval=60, leave=False)
        for idx, batch in enumerate(progress_bar):
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                               padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**inputs, labels=inputs["input_ids"])
            loss_value = outputs.loss.item()
            total_loss += loss_value * len(batch)
            count += len(batch)
            progress_bar.set_postfix(loss=loss_value)

            # For the last 5 batches, capture additional info.
            if idx >= len(batches) - 5:
                # Get the predicted token ids (take argmax over logits)
                preds = outputs.logits.argmax(dim=-1)
                # Decode predictions for each sample
                decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
                # For each sample in the batch, parse out the eval input and expected output.
                batch_info = []
                for orig_text, pred_text in zip(batch, decoded_preds):
                    # We assume the sample is constructed with "Answer:" as the separator.
                    parts = orig_text.split("Answer:")
                    eval_input = parts[0].strip() if parts else orig_text
                    eval_output = parts[1].strip() if len(parts) > 1 else ""
                    batch_info.append({
                        "eval_input": eval_input,
                        "eval_output": eval_output,
                        "model_prediction": pred_text
                    })
                last_batches.append({
                    "batch_index": idx,
                    "samples": batch_info
                })

    torch.cuda.empty_cache()
    model.train()

    # Log the last 5 batches for manual inspection.
    logger.info("----- Last 5 Batches Inspection -----")
    for batch_info in last_batches:
        logger.info(f"Batch Index: {batch_info['batch_index']}")
        for sample in batch_info["samples"]:
            logger.info("Eval Set Input:")
            logger.info(sample["eval_input"])
            logger.info("Eval Set Output (Ground Truth):")
            logger.info(sample["eval_output"])
            logger.info("Model Prediction:")
            logger.info(sample["model_prediction"])
            logger.info("-----")
            
    return total_loss / count if count > 0 else 0.0



def run_finetuning(
    model,
    tokenizer,
    device,
    logger,
    model_name,
    epochs=3,
    lr=5e-5,
    warmup_steps=100,
    max_length=2048,
    batch_size=8,
    early_stopping_patience=2, 
    use_subset=True 
):
    """
    Fine-tunes the model on the jtatman/python-code-dataset-500k dataset loaded via Hugging Face.
    Each sample is constructed as:
        System: <system>
        Instruction: <instruction>
        Answer: <output>
    
    The data is split into train (80%), validation (10%), and test (10%) sets.
    The function uses mixed precision, gradient checkpointing, data shuffling, learning rate scheduling,
    checkpointing, early stopping, and logs GPU utilization information.
    """
    logger.info("Loading dataset jtatman/python-code-dataset-500k from Hugging Face")
    ds = load_dataset("jtatman/python-code-dataset-500k", split="train")
    logger.info(f"Total samples in dataset: {len(ds)}")

    # Build the training text using the three columns: system, instruction, output.
    def build_sample(row):
        return f"System: {row['system']}\nInstruction: {row['instruction']}\nAnswer: {row['output']}"
    
    all_texts = [build_sample(row) for row in ds]

    # Shuffle and split into train (80%), validation (10%), test (10%)
    random.shuffle(all_texts)
    n = len(all_texts)

    if use_subset:
        n = int(n * 0.01)
        logger.info(f"Using a subset of the data for testing: {n} samples.")
        all_texts = all_texts[:n]
    else: 
        logger.info(f"Using the entire dataset for training: {n} samples.")

    # Generate a unique run identifier that includes the model name, training data label, subset flag, and a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model_name = model_name.replace("/", "-")
    subset_tag = "subset" if use_subset else "full"
    run_id = f"{sanitized_model_name}_{subset_tag}_{timestamp}"
    save_dir = os.path.join("model", run_id)
    logger.info(f"Checkpoints and best model will be saved in {save_dir}")

    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    train_texts = all_texts[:train_end]
    val_texts = all_texts[train_end:val_end]
    test_texts = all_texts[val_end:]

    logger.info(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")
    logger.info("Starting fine-tuning process.")

    # Log GPU utilization at the start
    log_gpu_info(logger)

    # Freeze all parameters except the final lm_head
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False
    logger.info("Frozen all layers except lm_head.")

    # Enable gradient checkpointing if supported
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing.")
    
    # Log all hyperparameters
    logger.info(f"Hyperparameters: epochs={epochs}, lr={lr}, warmup_steps={warmup_steps}, "
                f"max_length={max_length}, batch_size={batch_size}, "
                f"early_stopping_patience={early_stopping_patience}, use_subset={use_subset}")

    # Evaluate model before fine-tuning (validation set)
    logger.info("Evaluating model before fine-tuning...")
    val_loss_before = evaluate(model, tokenizer, val_texts, device, max_length, batch_size, logger)
    logger.info(f"Pre-Fine-tuning - Validation Loss: {val_loss_before:.4f}")

    # Set up optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = (len(train_texts) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler()  

    best_val_loss = float('inf')
    epochs_without_improve = 0
    best_epoch = -1

    model.train()
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} - Training started")
        epoch_loss = 0.0
        batch_count = 0

        # Shuffle training data each epoch
        random.shuffle(train_texts)
        batches = list(batchify(train_texts, batch_size))
        progress_bar = tqdm(batches, desc=f"Epoch {epoch+1} Training", mininterval=60, leave=False)
        for batch in progress_bar:
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                               padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() * len(batch)
            batch_count += len(batch)
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / batch_count
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Log GPU usage at the end of the epoch
        log_gpu_info(logger)

        val_loss = evaluate(model, tokenizer, val_texts, device, max_length, batch_size, logger)
        logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"finetuned_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improve = 0
            best_model_path = os.path.join(save_dir, "finetuned_best.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
            }, best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            epochs_without_improve += 1
            logger.info(f"No improvement for {epochs_without_improve} epoch(s)")

        if epochs_without_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs. Best epoch: {best_epoch} with val loss {best_val_loss:.4f}.")
            break

        torch.cuda.empty_cache()

    logger.info("Evaluating on test set...")
    test_loss = evaluate(model, tokenizer, test_texts, device, max_length, batch_size, logger)
    logger.info(f"Test Loss after fine-tuning: {test_loss:.4f}")

    logger.info("Fine-tuning complete.")
