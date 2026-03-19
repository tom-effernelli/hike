# Vulnerability Detection Pipeline (BigVul + CodeBERT)

This repository contains a scalable data engineering and machine learning pipeline designed to detect security vulnerabilities in C/C++ source code using Transformer-based models.

## Overview

The project processes the BigVul dataset to train a binary classifier. It addresses common challenges in AI for Cyber, such as data duplication in open-source repositories and memory management for large-scale source code datasets.

## Key Components

- **Memory-Mapped Storage**: Uses Apache Arrow (via Hugging Face Datasets) to handle data larger than RAM.
- **MinHash LSH Deduplication**: Removes near-duplicate code snippets (Jaccard similarity > 0.9) to prevent data leakage between training and testing sets.
- **Code Augmentation**: Injects synthetic noise (comments and variable renaming) to improve model generalization.
- **Fine-tuning**: Adapts the `microsoft/codebert-base` model for sequence classification.

## Project Structure

```text
.
├── data/                       # Local data storage (Git ignored)
│   ├── security_breach_dataset_arrow/
│   └── tokenized_dataset_arrow/
├── model_checkpoints/          # Model checkpoints (Git ignored)
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── ...
│   └── final_model/
├── src/
│   ├── __init__.py
│   ├── config.py               # Global constants and hyperparameters
│   ├── dataset_acquisition.py  # Step 1: Normalization and Deduplication
│   ├── tokenization.py         # Step 2: BPE Tokenization
│   └── train.py                # Step 3: PyTorch/Trainer loop
├── requirements.txt
└── README.md

## Setup

1. **Clone the repository and install dependencies**  
   Requires Python 3.9+.

   ```bash
   git clone https://github.com/yourusername/vuln-detect-pipeline.git
   cd vuln-detect-pipeline
   pip install -r requirements.txt
   ```

   *(For GPU users: Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to install the appropriate torch/CUDA version for your setup.)*

2. **Configure hyperparameters**  
   Adjust settings in `src/config.py` as needed for your hardware and experimental needs. Key configs include paths, batch size, sequence length, and deduplication threshold.

---

## Usage

### 1. **Dataset Acquisition & Preprocessing**

Processes the BigVul dataset, applies normalization, deduplication, and augmentation.  
This will save a clean Arrow-formatted dataset to disk.

```bash
python src/dataset_aquisition.py
```

### 2. **Tokenization**

Tokenizes code snippets using the CodeBERT tokenizer and saves the processed dataset.

```bash
python src/tokenization.py
```

### 3. **Model Training**

Fine-tunes CodeBERT on the tokenized dataset for binary classification.

```bash
python src/train.py
```

For additional options (e.g., epochs, learning rate), modify or extend `src/config.py`.

---

## Notes

- **Reproducibility**: The pipeline seeds random number generators where feasible.
- **Scalability**: All heavy-duty mapping/filtering is disk-based and batched, so processing >1M samples is feasible on a consumer machine.
- **Extensibility**: You can plug in new datasets, apply alternative tokenizers, or swap in different Transformers models with minimal changes.

---

## Citation

If you use this pipeline, please cite the original [BigVul dataset](https://zenodo.org/record/4424123) and [CodeBERT](https://arxiv.org/abs/2002.08155).

---

## License

This project is open source under the MIT License.