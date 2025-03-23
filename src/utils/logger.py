import os
import logging
from datetime import datetime
import torch

def setup_logger(log_dir="log/output"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"output_{timestamp}.log")

    # Configure logger
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Add console output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger()

def log_gpu_info(logger):
    """Logs the number of GPUs and their memory usage (in percent)."""
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        allocated_pct = (allocated / total_mem) * 100
        reserved_pct = (reserved / total_mem) * 100
        logger.info(f"GPU {i}: Total Memory = {total_mem/1e9:.2f} GB, "
                    f"Allocated = {allocated_pct:.1f}% ({allocated/1e9:.2f} GB), "
                    f"Reserved = {reserved_pct:.1f}% ({reserved/1e9:.2f} GB)")
