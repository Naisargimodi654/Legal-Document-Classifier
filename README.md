# Legal Document Classifier

A rephrased, ready-to-use repository README for a multi-label legal document classification pipeline built on transformer models. This document is designed to be dropped into a GitHub project as `README.md` or used as the basis for project documentation.

---

## Table of contents

1. Project Overview
2. Objective & Motivation
3. Abstract
4. Terminology & Concepts
5. Repository Structure
6. Quickstart (Run locally)
7. Environment & Requirements
8. Data — Source, Format & Preparation
9. Data Preprocessing (Detailed)
10. Labeling & MultiLabelBinarizer
11. Model Architecture & Rationale
12. Training Pipeline (Step-by-step)
13. Hyperparameters & Tips
14. Evaluation, Metrics & Reporting
15. Inference & Chunking Strategy
16. Streamlit App (Usage & Features)
17. Exporting & Using the Saved Model
18. Optimization & Production-readiness
19. Reproducibility & Experiments
20. Troubleshooting & Common Errors
21. Security, Privacy & Ethics
22. Extensions & Next Steps
23. References & Resources
24. License
25. Acknowledgements
26. Appendix

    * Example queries / sample inputs
    * Example outputs (format)
    * Useful commands

---

# 1. Project Overview

This repository provides a full pipeline for **multi-label classification of long-form legal documents**. It covers data ingestion (PDF/DOCX/TXT), preprocessing and chunking for transformer models, fine-tuning, evaluation, inference and a lightweight Streamlit UI to upload documents and inspect predictions. The pipeline is modular so you can swap models, change chunking strategies, or plug in a different front-end.

# 2. Objective & Motivation

Large legal documents frequently exceed transformer token limits. The aim of this project is to provide a practical, reproducible system that:

* Enables labeling of legal documents with multiple tags (e.g., contract types, risk categories, legal topics).
* Handles arbitrarily long texts by splitting into overlapping chunks, classifying chunks, then aggregating into document-level labels.
* Ships with a simple UI for non-technical reviewers and an inference pipeline ready for production optimization.

# 3. Abstract

We present a flexible, production-aware pipeline: read documents, preprocess and chunk them, fine-tune a transformer backbone with a multi-label head, and aggregate chunk predictions to generate document-level labels. The approach balances performance and compute cost and includes best-practices for evaluation, threshold selection and model export.

# 4. Terminology & Concepts

* **Chunking / Sliding Window**: Splitting long text into token-limited segments with configurable overlap (stride).
* **Backbone**: Transformer encoder (e.g., `legal-bert`, `bert-base-uncased`, `roberta`) used to produce contextual embeddings.
* **Multi-label classification**: Each document may have zero, one or several labels simultaneously; the model uses sigmoid activations per label.
* **Aggregation**: Combining chunk-level predictions into a single document-level output (mean, max, or weighted pooling).
* **Thresholding**: Converting continuous probabilities into discrete label assignments using per-label thresholds.

# 5. Repository Structure

```
├── data/                      # sample data, data schema, and ingestion helpers
├── notebooks/                 # EDA and experiment analysis notebooks
├── src/
│   ├── preprocessing.py       # text cleaning, OCR, tokenization helpers
│   ├── chunking.py            # chunk & stride logic
│   ├── dataset.py             # Dataset class for PyTorch
│   ├── model.py               # model definition and utilities
│   ├── train.py               # training loop, scheduler, metrics
│   ├── inference.py           # document inference & aggregation
│   ├── eval.py                # evaluation routines and reporting
│   └── utils.py               # misc helpers (logging, seeding)
├── streamlit_app/             # Streamlit UI for demoing inference
├── experiments/               # saved checkpoints and experiment metadata
├── requirements.txt
├── environment.yml            # optional conda env
├── README.md                  # this file
└── LICENSE
```

# 6. Quickstart (Run locally)

```bash
# 1. clone
git clone https://github.com/<username>/legal-document-classifier.git
cd legal-document-classifier

# 2. create env
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 3. run a quick training on sample data (toy mode)
python src/train.py --config configs/toy_config.yaml

# 4. run the Streamlit demo
streamlit run streamlit_app/app.py
```

# 7. Environment & Requirements

* Python 3.9+ recommended.
* Key packages: `transformers`, `datasets`, `torch` (1.12+), `scikit-learn`, `pandas`, `numpy`, `pdfminer.six` or `pytesseract` (OCR), `streamlit`.
* Optional for speed: `onnxruntime`, `torchscript`, `accelerate`.

Install via pip: `pip install -r requirements.txt`.

# 8. Data — Source, Format & Preparation

## Sources

* Public datasets (court opinions, contract corpora), internal corpora, or scraped regulatory filings.
* For scanned documents, run OCR first (Tesseract recommended).

## Expected format

A CSV/JSONL where each row/document contains:

* `doc_id` — unique identifier
* `text` — raw textual content (or path to original file)
* `labels` — list of label strings (can be empty)

Example (JSONL): `{"doc_id": "D001", "text": "...", "labels": ["Employment","IP"]}`

# 9. Data Preprocessing (Detailed)

1. **Ingestion**: Convert PDF/DOCX to plain text. Use PDFMiner for text PDFs and Tesseract for image PDFs.
2. **Cleaning**: Remove boilerplate (page headers/footers), collapse whitespace, normalize quotes/Unicode, remove non-informative sections if required.
3. **Segmentation**: Optionally detect and keep document sections (headings, clauses) as metadata for specialized models.
4. **Tokenization**: Use the tokenizer matching the chosen transformer.
5. **Chunking**: Create overlapping chunks of `max_length` tokens with a `stride` overlap. Store chunk metadata (start_token, end_token, chunk_index).

# 10. Labeling & MultiLabelBinarizer

* Labels should be consistent and curated. Keep a label catalog (`labels.txt`) mapping label name → index.
* Convert label lists to binary vectors using `sklearn.preprocessing.MultiLabelBinarizer` or a custom map.
* For extreme label imbalance, consider label grouping, up/down-sampling, or using focal loss.

# 11. Model Architecture & Rationale

* **Backbone**: A transformer encoder (LegalBERT recommended for legal texts).
* **Pooling**: Use `[CLS]` token or mean-pooling over token embeddings. Experiment to see which suits your data.
* **Head**: Dense layer(s) with dropout and a sigmoid output per label.
* **Loss**: Binary cross-entropy (BCEWithLogitsLoss) for multi-label training.

Rationale: transformers capture contextual semantics; chunking lets us apply those models to long documents without specialized long-range transformers.

# 12. Training Pipeline (Step-by-step)

1. Create train/val/test splits at the **document** level (avoid leaking chunks across splits).
2. Instantiate tokenizer and dataset (chunks → examples).
3. Build model and optimizer (AdamW) and a scheduler (linear warmup + decay).
4. Train for `N` epochs, validate at intervals, and save best checkpoint by validation metric (e.g., macro-F1).
5. After training, run evaluation on the held-out test set and produce a per-label report.

# 13. Hyperparameters & Tips

* `max_length`: 512 tokens for standard BERT; smaller backbones might need 256.
* `stride`: 128–256 tokens (balance between context overlap and compute).
* `batch_size`: as large as fits in GPU memory — use gradient accumulation if needed.
* Learning rate: `2e-5`–`5e-5` for fine-tuning transformers.
* Regularization: dropout 0.1–0.3, weight decay 0.01.
* Use mixed precision (AMP) to reduce memory and speed up training.

# 14. Evaluation, Metrics & Reporting

* Primary metrics: **micro/macro F1**, **precision**, **recall** (per-label and averaged).
* Additional diagnostics: ROC-AUC per label, PR curves for rare labels, confusion matrices for single-label subsets.
* Report calibration and choose per-label thresholds (global `0.5` is a start but per-label thresholds often improve results).
* Save prediction CSVs with columns: `doc_id`, `true_labels`, `pred_probs`, `pred_labels`.

# 15. Inference & Chunking Strategy

* Tokenize and chunk input document the same way as during training.
* For each chunk, obtain label probabilities.
* Aggregate chunk-level probabilities into document-level scores using `max`, `mean`, or a learned attention-weighted pooling.
* Apply thresholds to get final labels.
* Optionally, return chunk-level heatmaps or highlighted spans for explainability.

# 16. Streamlit App (Usage & Features)

* Upload a PDF/DOCX/TXT and run inference locally.
* View predicted labels with probabilities and per-chunk scores.
* Options to change threshold, aggregation method, and to display chunk text for inspection.

# 17. Exporting & Using the Saved Model

* Save PyTorch checkpoints and a `model_card.json` describing labels, tokenizer, and training config.
* Optionally export to TorchScript or ONNX for faster CPU inference.
* Bundle the tokenizer and label map with the model artifact for reproducible inference.

# 18. Optimization & Production-readiness

* Techniques: model quantization (post-training static/dynamic), weight pruning, distillation to a smaller student model.
* Serve with a lightweight API (FastAPI/Flask) and batch requests for throughput.
* Use ONNXRuntime for CPU inference or optimized GPU containers for low-latency.

# 19. Reproducibility & Experiments

* Fix random seeds for `numpy`, `torch`, and Python `random`.
* Log experiments (weights, configs, metrics) with tools like MLflow, Weights & Biases, or simple CSVs.
* Keep `configs/` for each experiment and store checkpoints under `experiments/<exp-id>/`.

# 20. Troubleshooting & Common Errors

* **OOM (out of memory)**: reduce `batch_size`, use gradient accumulation, or shorten `max_length`.
* **Tokenization mismatch**: ensure inference tokenizer == training tokenizer.
* **Label leakage**: ensure split is by document, not chunk.
* **Poor performance on rare labels**: consider oversampling, class-weighted loss, or focal loss.

# 21. Security, Privacy & Ethics

* Keep sensitive documents local; avoid uploading private legal documents to public services.
* Document limitations: this tool aids classification and should not replace legal judgment.
* Maintain an audit trail for automated decisions and allow human review for high-risk labels.

# 22. Extensions & Next Steps

* Replace chunking + aggregation with a long-range transformer (Longformer, BigBird) for native long-context modeling.
* Add a supervised attention head that learns to weight chunks.
* Build active learning loops to prioritize human annotation where the model is uncertain.
* Multi-modal extension: include metadata (dates, parties) or tables extracted from documents.

# 23. References & Resources

* Papers and resources on transformers, multi-label loss functions, and long-document modeling.
* Tooling: Hugging Face Transformers, ONNXRuntime, Tesseract OCR.

# 24. License

Include an open-source license (e.g., MIT, Apache-2.0) and update `LICENSE` accordingly.

# 25. Acknowledgements

Credit authors of datasets, models (LegalBERT), and any libraries used.

# 26. Appendix

## Example queries / sample inputs

* Upload `contract_123.pdf` → expect labels: `["Non-Disclosure", "Service Agreement"]`
* Raw JSONL row: `{ "doc_id": "C-001", "text": "This Agreement...", "labels": ["Employment"] }`

## Example outputs (format)

```csv
doc_id,true_labels,pred_probs,pred_labels
C-001,"[Employment]","[0.04, 0.87, 0.12]","[Employment]"
```

## Useful commands

```bash
# Run training (example config)
python src/train.py --config configs/finetune_legalbert.yaml

# Evaluate a checkpoint
python src/eval.py --checkpoint experiments/exp123/best.pt --test-file data/test.jsonl

# Run inference on single file
python src/inference.py --model experiments/exp123/best.pt --file examples/contract.pdf

# Start Streamlit demo
streamlit run streamlit_app/app.py
```

---

If you want, I can also generate ready-to-commit files: `README.md`, a sample `requirements.txt`, `configs/toy_config.yaml`, and a minimal `src/train.py` scaffold. Tell me which files you'd like and I will create them.
