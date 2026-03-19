# src/train.py
import os
import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# Import your centralized configuration
import config

def compute_metrics(eval_pred):
    """
    Calculates Accuracy and F1-Score during evaluation.
    F1-Score is critical in cybersecurity to balance False Positives and False Negatives.
    """
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)
    
    return {**acc, **f1}

def run_training():
    """
    Main orchestration function for the fine-tuning process.
    """
    # 1. Hardware Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware detected: {device.upper()}")
    if device == "cpu":
        print("[!] WARNING: Training on CPU will be extremely slow. Consider using a GPU.")

    # 2. Load the GPU-ready dataset
    print(f"[*] Loading tokenized dataset from: {config.TOKENIZED_DATASET_PATH}")
    if not os.path.exists(config.TOKENIZED_DATASET_PATH):
        raise FileNotFoundError("Tokenized dataset not found. Please run tokenization.py first.")
    
    dataset = load_from_disk(config.TOKENIZED_DATASET_PATH)

    # 3. Create Train / Validation Splits (90% Train, 10% Eval)
    # We use a fixed seed (42) for reproducibility
    print("[*] Splitting dataset into Train and Validation sets...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]
    print(f"    -> Train size: {len(train_ds)} samples")
    print(f"    -> Eval size:  {len(eval_ds)} samples")

    # 4. Initialize the Model
    # num_labels=2 because we are doing Binary Classification (0: Safe, 1: Vulnerable)
    print(f"[*] Loading pre-trained model: {config.MODEL_CHECKPOINT}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_CHECKPOINT, 
        num_labels=2
    )

    # 5. Define Training Hyperparameters
    print("[*] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN_BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=0.01,                  # Regularization to prevent overfitting
        evaluation_strategy="epoch",        # Evaluate at the end of each epoch
        save_strategy="epoch",              # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True,        # Keep the best performing model based on eval
        metric_for_best_model="f1",         # Use F1-score to define "best"
        logging_steps=50,                   # Print logs every 50 steps
        report_to="none"                    # Disable 3rd party loggers (like WandB) for now
    )

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    # 7. Launch Training
    print("\n========================================================")
    print("[*] STARTING TRAINING LOOP")
    print("========================================================\n")
    trainer.train()

    # 8. Save the Final Output
    final_model_path = os.path.join(config.MODEL_OUTPUT_DIR, "final_vulnerability_model")
    print(f"\n[*] Training complete! Saving the best model to: {final_model_path}")
    trainer.save_model(final_model_path)
    print("[*] Pipeline finished successfully.")

if __name__ == "__main__":
    run_training()