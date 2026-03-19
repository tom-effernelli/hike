# src/dataset_aquisition.py
import re
import random
from typing import Dict, List, Any
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH
import config

# ==========================================
# 1. HELPER FUNCTIONS (Transformation & Hashing)
# ==========================================

def get_minhash(code_string: str, num_perm: int = config.NUM_PERM) -> MinHash:
    """
    Create a MinHash signature for a code snippet based on trigrams.
    This allows for fuzzy matching of code snippets.
    """
    m = MinHash(num_perm=num_perm)
    
    # Remove repeated whitespace and perform basic tokenization
    tokens = re.sub(r'\s+', ' ', code_string).split(' ')
    
    # Create 3-grams (shingles) to capture structural semantics
    for i in range(len(tokens) - 2):
        shingle = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}".encode('utf8')
        m.update(shingle)
        
    return m

def augment_code(code_string: str) -> str:
    """
    Apply semantically neutral transformations to prevent the AI 
    from overfitting to specific coding styles or variable names.
    """
    augmented = code_string
    
    # Strategy A: Inject dead comments (noise)
    if random.random() > 0.5:
        noise = ["// TODO: Check bounds", "/* Auto-generated review */", "// Refactored"]
        augmented = random.choice(noise) + "\n" + augmented
        
    # Strategy B: Basic obfuscation of common variable names
    # (e.g., change 'buf' or 'buffer' into 'data_chunk')
    if random.random() > 0.5:
        augmented = re.sub(r'\bbuffer\b', 'data_chunk', augmented)
        augmented = re.sub(r'\bbuf\b', 'd_buf', augmented)
        
    return augmented

# ==========================================
# 2. BATCH PROCESSING FUNCTIONS (Hugging Face mapped functions)
# ==========================================

def batch_normalize(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Process a batch of raw BigVul rows and map them to a unified format.
    Splits the 'before' and 'after' code states into distinct entries.
    """
    new_codes, new_labels, new_cwes, new_sources = [], [], [], []
    
    # Iterate through the lists provided in the batch
    for before, after, cwe in zip(batch['func_before'], batch['func_after'], batch['CWE ID']):
        # Vulnerable entry (Label = 1)
        if before:
            new_codes.append(before)
            new_labels.append(1)
            new_cwes.append(cwe)
            new_sources.append("bigvul")
            
        # Patched/Safe entry (Label = 0)
        if after:
            new_codes.append(after)
            new_labels.append(0)
            new_cwes.append(cwe)
            new_sources.append("bigvul")
            
    return {
        "code": new_codes, 
        "label": new_labels, 
        "cwe": new_cwes, 
        "source": new_sources
    }

def batch_augment(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Process a batch of normalized code and randomly generate 
    augmented versions to enrich the dataset.
    """
    aug_codes, aug_labels, aug_cwes, aug_sources = [], [], [], []
    
    for code, label, cwe, source in zip(batch['code'], batch['label'], batch['cwe'], batch['source']):
        # Always keep the original entry
        aug_codes.append(code)
        aug_labels.append(label)
        aug_cwes.append(cwe)
        aug_sources.append(source)
        
        # Create an augmented version ~(1-AUGMENT_THRESHOLD)*100% of the time to avoid extreme inflation
        if random.random() > config.AUGMENT_THRESHOLD:
            aug_codes.append(augment_code(code))
            aug_labels.append(label)
            aug_cwes.append(cwe)
            aug_sources.append(source + "_augmented")
            
    return {
        "code": aug_codes, 
        "label": aug_labels, 
        "cwe": aug_cwes, 
        "source": aug_sources
    }

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

def build_dataset() -> Dataset:
    """
    Main orchestration function.
    Downloads, normalizes, deduplicates, augments, and saves the dataset.
    """
    print("[*] Loading raw dataset (Lazy Loading)...")
    # Load the complete dataset. It is not loaded into RAM, just memory-mapped from disk.
    if config.SCRIPT_TESTING:
        raw_ds = load_dataset("bstee615/bigvul", split="train[:10000]")
    else:
        raw_ds = load_dataset("bstee615/bigvul", split="train")
    
    # --- STEP 1: SCALABLE NORMALIZATION ---
    print("[*] 1. Normalization in progress (on disk)...")
    # batched=True allows changing the number of rows (e.g., 1 row -> 2 rows)
    # remove_columns destroys the old schema to enforce the new one
    norm_ds = raw_ds.map(
        batch_normalize, 
        batched=True, 
        batch_size=config.BATCH_SIZE, 
        remove_columns=raw_ds.column_names
    )
    print(f"    -> {len(norm_ds)} entries after normalization.")

    # --- STEP 2: SCALABLE DEDUPLICATION ---
    print("[*] 2. Deduplication using MinHash LSH...")
    lsh = MinHashLSH(threshold=config.MIN_HASH_THRESHOLD, num_perm=config.NUM_PERM)
    
    def is_unique(example: Dict[str, Any], idx: int) -> bool:
        """
        Stateful filter: keeps the LSH index in RAM while reading code from disk.
        Returns True if the snippet is unique, False if it's a duplicate.
        """
        m = get_minhash(example["code"])
        if lsh.query(m):
            return False  # Near-duplicate found, drop it
            
        lsh.insert(str(idx), m)
        return True  # Unseen snippet, keep it

    # The dataset is filtered on the fly.
    dedup_ds = norm_ds.filter(is_unique, with_indices=True)
    print(f"    -> {len(dedup_ds)} entries after deduplication.")

    # --- STEP 3: SCALABLE AUGMENTATION ---
    print("[*] 3. Data Augmentation in progress...")
    final_ds = dedup_ds.map(batch_augment, batched=True, batch_size=config.BATCH_SIZE)
    print(f"    -> {len(final_ds)} final entries ready for training.")
    
    # --- STEP 4: SAVE TO DISK ---
    output_path = config.CLEAN_DATASET_PATH
    final_ds.save_to_disk(output_path)
    print(f"[*] Dataset successfully saved to '{output_path}'")
    
    return final_ds

if __name__ == "__main__":
    random.seed(42)
    # Execute the pipeline
    dataset = build_dataset()