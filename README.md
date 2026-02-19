# Hate Speech Classification with GPT-2, DistilBERT, and LoRA

A Natural Language Processing project that investigates whether lightweight transformer models can effectively detect hate speech while remaining computationally efficient.

This study compares GPT-2 and DistilBERT models, with and without Low-Rank Adaptation (LoRA), using the HateXplain dataset.

## Overview

Large language models achieve strong performance but require significant computational resources.  
This project explores parameter-efficient fine-tuning techniques to determine whether smaller models can provide practical performance for real-world deployment.

Key objectives:

- Fine-tune lightweight transformer models for text classification
- Evaluate parameter-efficient training with LoRA
- Compare performance across models
- Analyze model limitations and classification errors

## Models Evaluated

- GPT-2 (fine-tuned)
- GPT-2 + LoRA
- DistilBERT (fine-tuned)
- DistilBERT baseline

## Dataset

HateXplain dataset — annotated tweets labeled as:

- Hate Speech
- Offensive Language
- Neutral Speech

Dataset paper: https://arxiv.org/abs/2012.10289

## Methodology

- Data preprocessing and cleaning
- Tokenization using model-specific tokenizers
- Padding/truncation to fixed sequence length
- Fine-tuning using AdamW optimizer
- Parameter-efficient tuning with LoRA
- Evaluation using classification metrics

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Precision-Recall Curve

## Results (Summary)

- Smaller models can achieve useful performance with careful fine-tuning
- LoRA reduces training cost while maintaining effectiveness
- Class imbalance significantly impacts classification accuracy
- Neutral class proved most difficult to classify

## Project Structure

src/ Training and evaluation scripts
data/ Processed datasets and predictions
results/ Model outputs and visualizations
report/ Final project report


## Setup

### Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- pandas
- scikit-learn

Install dependencies:

```bashd
pip install -r requirements.txt
```

## Running the Project
```bashd
pyton src/train_gpt2.py
```

## Report
See the full project report for detailed methodology and results:
```bashd
report/Fine Tuning Small Models for Hate Speech Classification A.pdf
```

