from datasets import load_from_disk
from huggingface_hub import HfApi

# --- Step 1: Load dataset from disk ---
dataset_path = "/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/data/SINQ_synthetic_haskell_dataset_nvidia_hf"
dataset = load_from_disk(dataset_path)

# --- Step 2: Split into train/test ---
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# --- Step 3: Push to Hugging Face Hub ---
repo_id = "OpInstruct-HSx"  # Your dataset repo
split_dataset.push_to_hub(
    repo_id,
    commit_message="Add train/test splits",
)

print(f"✅ Train/Test splits uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
