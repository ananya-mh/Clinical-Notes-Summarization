# Clinical Text Summarization with Causal Inference

A pipeline for summarizing doctor-patient dialogues into clinical notes using fine-tuned Transformer models (T5, BART), with causal feature analysis to study the effect of clinical events on patient outcomes.

## Dataset

**MTS-Dialog** 1,701 doctor-patient encounter dialogues with corresponding clinical note sections across six medical specialties.

Source: [Ben Abacha et al., EACL 2023](https://aclanthology.org/2023.eacl-main.168)


## Pipeline Overview

### 1. Preprocessing & Tokenization
- Clean dialogue text (normalize speaker labels, whitespace)
- Clean target clinical notes
- Tokenize for T5 (with `"summarize: "` prefix) and BART
- Compute dataset statistics and compression ratios

### 2. Summarization (T5 & BART fine-tuning)
- Fine-tune T5 and BART to generate clinical notes from dialogues
- Target: ~40% compression while retaining >90% of key medical entities
- Evaluate with ROUGE, BERTScore, and medical entity retention

### 3. Causal Inference & Explainability
- Extract structured clinical features from notes (diagnoses, treatments, symptoms)
- Estimate treatment effects using propensity score methods
- Apply SHAP to separate causal signals from correlations

## Tech Stack

Python, PyTorch, HuggingFace Transformers, SHAP, scikit-learn, pandas
