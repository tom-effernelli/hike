# src/config.py
import os

# ==========================================
# 1. DIRECTORY STRUCTURE
# ==========================================

# Get the absolute path to the project root (cyber_ai_project/)
# This ensures paths work regardless of where you launch the script from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Main data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 2. DATASET PATHS
# ==========================================

# Path for the normalized/deduplicated dataset from Step 1
CLEAN_DATASET_PATH = os.path.join(DATA_DIR, "security_breach_dataset_arrow")

# Path for the tokenized dataset ready for the GPU (Step 2)
TOKENIZED_DATASET_PATH = os.path.join(DATA_DIR, "tokenized_dataset_arrow")

# Directory to save model checkpoints and the final trained model
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "model_checkpoints")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 3. MODEL & TRAINING HYPERPARAMETERS
# ==========================================

# The pre-trained model checkpoint from Hugging Face
# CodeBERT is optimized for programming languages like C/C++
MODEL_CHECKPOINT = "microsoft/codebert-base"

# Maximum number of tokens per code snippet (512 is standard for CodeBERT)
MAX_SEQ_LENGTH = 512

# Training settings
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
AUGMENT_THRESHOLD = 0.66

# ==========================================
# 4. ACQUISITION SETTINGS (Step 1)
# ==========================================

BATCH_SIZE = 1000
MIN_HASH_THRESHOLD = 0.90
NUM_PERM = 128
SCRIPT_TESTING = True  # Set to False to process the full dataset