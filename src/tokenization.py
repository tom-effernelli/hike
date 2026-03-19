# src/tokenization.py
import os
from datasets import load_from_disk
from transformers import AutoTokenizer
import config

def run_tokenization():
    """
    Loads the clean dataset and transforms C++ code into 
    numerical input IDs using the CodeBERT tokenizer.
    """
    print(f"[*] Loading clean dataset from: {config.CLEAN_DATASET_PATH}")
    if not os.path.exists(config.CLEAN_DATASET_PATH):
        raise FileNotFoundError("Clean dataset not found. Run dataset_acquisition.py first.")

    dataset = load_from_disk(config.CLEAN_DATASET_PATH)

    print(f"[*] Initializing tokenizer: {config.MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)

    def tokenize_function(examples):
        """
        Processes a batch of code snippets.
        - Truncation ensures we don't exceed the model's 512 token limit.
        - Padding ensures all vectors in a batch have the same length for the GPU.
        """
        return tokenizer(
            examples["code"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        )

    print("[*] Tokenizing dataset (this uses multiprocessing if available)...")
    # We use .map() to keep it scalable and save RAM
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=config.BATCH_SIZE
    )

    # Hugging Face Trainer expects the target column to be named 'labels'
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Set format to PyTorch tensors for the training phase
    tokenized_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels"]
    )

    print(f"[*] Saving tokenized dataset to: {config.TOKENIZED_DATASET_PATH}")
    tokenized_dataset.save_to_disk(config.TOKENIZED_DATASET_PATH)
    print("[*] Tokenization complete!")

if __name__ == "__main__":
    run_tokenization()