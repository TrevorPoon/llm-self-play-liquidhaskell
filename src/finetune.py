import os
import glob
import random
import torch
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from utils.logger import log_gpu_info


def batchify(data, batch_size):
    """
    Yields batches from a list of data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def evaluate(model, tokenizer, text_data, device, max_length=2048, batch_size=1):
    """
    Evaluates the given model on a list of text samples using mixed precision,
    and returns the average loss.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        batches = list(batchify(text_data, batch_size))
        progress_bar = tqdm(batches, desc="Evaluation", mininterval=60, leave=False)
        for batch in progress_bar:
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                               padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * len(batch)
            count += len(batch)
    torch.cuda.empty_cache()
    model.train()
    return total_loss / count if count > 0 else 0.0

def run_finetuning(
    model,
    tokenizer,
    device,
    logger,
    epochs=3,
    lr=5e-5,
    warmup_steps=100,
    max_length=2048,
    batch_size=1,
    early_stopping_patience=2
):
    """
    Fine-tunes the model on local Parquet files from data/jtatman-python-code-dataset-500k.
    Each sample is constructed as:
        System: <system>
        Instruction: <instruction>
        Answer: <output>
    
    The data is split into train (80%), validation (10%), and test (10%) sets.
    The function uses mixed precision, gradient checkpointing, data shuffling, learning rate scheduling,
    checkpointing, early stopping, and logs GPU utilization information.
    """
    logger.info("Loading all Parquet files from data/jtatman-python-code-dataset-500k/")
    
    # Read all parquet files in the folder
    parquet_files = glob.glob("data/jtatman-python-code-dataset-500k/*.parquet")
    if not parquet_files:
        logger.error("No parquet files found in the specified directory.")
        return

    df_list = []
    for file in parquet_files:
        table = pq.read_table(file)
        df_list.append(table.to_pandas())
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total samples in concatenated DataFrame: {len(df)}")

    # Build the training text using the three columns: system, instruction, output.
    def build_sample(row):
        return f"System: {row['system']}\nInstruction: {row['instruction']}\nAnswer: {row['output']}"
    
    all_texts = df.apply(build_sample, axis=1).tolist()

    # Shuffle and split into train (80%), validation (10%), test (10%)
    random.shuffle(all_texts)
    n = len(all_texts)

    # =============================================================================
    # Comment this block to use a smaller subset of the data for testing
    n = int(n * 0.01)
    logger.info(f"Using a subset of the data for testing: {n} samples.")
    all_texts = all_texts[:n]
    # =============================================================================

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

    # Evaluate model before fine-tuning (validation set)
    logger.info("Evaluating model before fine-tuning...")
    val_loss_before = evaluate(model, tokenizer, val_texts, device, max_length=max_length, batch_size=batch_size)
    logger.info(f"Pre-Fine-tuning - Validation Loss: {val_loss_before:.4f}")

    # Set up optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = (len(train_texts) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler()  # For mixed precision training

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

        val_loss = evaluate(model, tokenizer, val_texts, device, max_length=max_length, batch_size=batch_size)
        logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")

        # Save checkpoint every epoch
        save_dir = "model"
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
    test_loss = evaluate(model, tokenizer, test_texts, device, max_length=max_length, batch_size=batch_size)
    logger.info(f"Test Loss after fine-tuning: {test_loss:.4f}")

    logger.info("Fine-tuning complete.")
