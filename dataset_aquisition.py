import re
import random
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

# ==========================================
# 1. NORMALIZATION (Unified Format)
# ==========================================
def normalize_bigvul(example):
    """Convert one BigVul row into two entries (Vulnerable = 1, Patched = 0)"""
    records = []
    # Vulnerable entry
    if example['func_before']:
        records.append({
            "code": example['func_before'],
            "label": 1,
            "cwe": example['cwe_id'],
            "source": "bigvul"
        })
    # Patched (safe) entry
    if example['func_after']:
        records.append({
            "code": example['func_after'],
            "label": 0,
            "cwe": example['cwe_id'], # The patch relates to this CWE
            "source": "bigvul"
        })
    return records

# ==========================================
# 2. SMART DEDUPLICATION (MinHash LSH)
# ==========================================
def get_minhash(code_string, num_perm=128):
    """Create a MinHash signature for a code snippet (based on trigrams)."""
    m = MinHash(num_perm=num_perm)
    # Remove repeated whitespace and do a basic tokenization
    tokens = re.sub(r'\s+', ' ', code_string).split(' ')
    # Create 3-grams to capture structure
    for i in range(len(tokens) - 2):
        shingle = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}".encode('utf8')
        m.update(shingle)
    return m

# ==========================================
# 3. DATA AUGMENTATION
# ==========================================
def augment_code(code_string):
    """
    Apply semantically neutral transformations to make
    the AI robust to coding style.
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
# MAIN SCRIPT
# ==========================================
def build_dataset():
    print("[*] Loading raw datasets...")
    # Load BigVul (take only the first 2000 for testing)
    bigvul_raw = load_dataset("bstee615/bigvul", split="train[:2000]")
    
    print("[*] 1. Normalization...")
    unified_data = []
    for row in bigvul_raw:
        unified_data.extend(normalize_bigvul(row))
        
    print(f"    -> {len(unified_data)} entries after normalization.")

    print("[*] 2. Deduplication (MinHash LSH)...")
    # threshold=0.9 means if two code snippets are 90% similar, we drop the second
    lsh = MinHashLSH(threshold=0.90, num_perm=128)
    deduplicated_data = []
    seen_signatures = set()

    for idx, item in enumerate(unified_data):
        code = item["code"]
        m = get_minhash(code)
        
        # Check whether a 90%+ similar snippet already exists
        result = lsh.query(m)
        if not result:
            # New snippet, keep it
            lsh.insert(str(idx), m)
            deduplicated_data.append(item)
        else:
            # Near-duplicate, drop it
            pass 

    print(f"    -> {len(deduplicated_data)} entries after deduplication.")

    print("[*] 3. Data Augmentation...")
    final_dataset = []
    for item in deduplicated_data:
        # Always keep the original
        final_dataset.append(item)
        
        # Create an augmented version ~1 time out of 3 (to avoid inflating the dataset too much)
        if random.random() > 0.66:
            aug_item = item.copy()
            aug_item["code"] = augment_code(item["code"])
            aug_item["source"] = item["source"] + "_augmented"
            final_dataset.append(aug_item)

    print(f"    -> {len(final_dataset)} final entries ready for training.")
    
    # Quick preview of the result
    print("\n[*] Example final entry:")
    print(final_dataset[-1])
    
    return final_dataset

if __name__ == "__main__":
    my_cyber_dataset = build_dataset()
    # You can then save it with pandas:
    # import pandas as pd
    # df = pd.DataFrame(my_cyber_dataset)
    # df.to_parquet("my_cyber_dataset_v1.parquet")