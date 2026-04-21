# Clinical Text Summarization & Entity Analysis

An end-to-end NLP pipeline that fine-tunes Transformer models on doctor-patient dialogues to generate clinical note summaries, extracts medical entities with biomedical NER, classifies clinical note sections using engineered features, and explains predictions with SHAP.

## Dataset

**MTS-Dialog** — 1,201 train / 100 validation / 200 test doctor-patient dialogues paired with clinical note summaries across six section types.

| Section | Distribution |
|---------|-------------|
| GENHX | ~60% |
| NEURO | ~17% |
| ORTHO | ~12% |
| SOAP | ~5% |
| DERM | ~3% |
| ALLERGY | ~3% |

Source: [Ben Abacha et al., "An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters", EACL 2023](https://aclanthology.org/2023.eacl-main.168)

## Pipeline

![Pipeline](https://img.shields.io/badge/Phase_1-Summarization-blue) ![Pipeline](https://img.shields.io/badge/Phase_2-Entity_Extraction-green) ![Pipeline](https://img.shields.io/badge/Phase_3-Classification-orange) ![Pipeline](https://img.shields.io/badge/Phase_4-Explainability-red)

### Phase 1: Fine-Tune T5 & BART for Clinical Summarization
- Fine-tuned **T5-small** and **BART-base** on the training split to generate clinical notes from dialogues
- Trained with AdamW optimizer, linear warmup schedule, and fp16 mixed precision on GPU
- Evaluated with ROUGE, BERTScore, compression ratio, and medical entity retention

### Phase 2: Clinical Entity Extraction
- Extracted medical entities from dialogues and generated summaries using **ScispaCy** (`en_core_sci_sm`)
- Categorized entities into symptoms, medications, diagnoses, and procedures
- Computed entity overlap between source dialogues and summaries to measure information retention

### Phase 3: Section-Type Classification
- Engineered features from entity counts, dialogue length, speaker turns, and top-entity presence flags
- Trained **Logistic Regression**, **Random Forest**, and **XGBoost** with stratified 5-fold cross-validation
- Used macro F1 as primary metric to handle class imbalance

### Phase 4: SHAP Explainability
- Applied SHAP to the best classifier to identify which features drive section-type predictions
- Generated summary plots, per-class breakdowns, dependence plots, and individual force plots

## Results

### Summarization — T5-small vs BART-base

| Metric | T5-small | BART-base |
|--------|----------|-----------|
| ROUGE-1 | 0.252 | 0.338 |
| ROUGE-2 | 0.099 | 0.158 |
| ROUGE-L | 0.195 | 0.279 |
| BERTScore F1 | 0.778 | 0.817 |
| Compression Ratio | 51.9% | 75.5% |
| Entity Retention | 38.8% | 21.7% |

**Key finding:** BART-base produces more fluent summaries (higher ROUGE and BERTScore), but compresses too aggressively and loses medical entities. T5-small retains nearly **2x more medical entities** and compresses closer to the 40% target. Standard text metrics alone would favor BART — the entity retention metric reveals that T5 better preserves clinically important information.

### Entity Extraction

| Entity Type | Avg per Dialogue |
|-------------|-----------------|
| Symptoms | 1.4 |
| Medications | 0.4 |
| Diagnoses | 0.5 |
| Procedures | 0.4 |
| Other | 16.2 |
| **Total** | **18.9** |

### Section-Type Classification

| Model | CV Accuracy | CV Macro F1 | Test Accuracy | Test Macro F1 |
|-------|-------------|-------------|---------------|---------------|
| Logistic Regression | 46.6% | 22.3% | 42.5% | 22.5% |
| Random Forest | 57.8% | 22.4% | 52.5% | 19.7% |
| XGBoost | 57.7% | 25.6% | 53.5% | 21.2% |

Macro F1 is limited by class imbalance (GENHX dominates at ~60%) and the discriminative power of entity count features. Results suggest that richer representations (entity embeddings, TF-IDF, or contextual features) would be needed for stronger classification.

### SHAP Feature Importance

Top features by mean |SHAP value|:

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | symptom_count | 0.777 |
| 2 | summary_word_count | 0.573 |
| 3 | diagnosis_count | 0.564 |
| 4 | medication_count | 0.556 |
| 5 | dialogue_word_count | 0.541 |
| 6 | speaker_turn_count | 0.514 |
| 7 | procedure_count | 0.415 |

Medical entity counts — especially symptom and diagnosis counts — are the strongest signals for distinguishing section types. Conversational tokens picked up by ScispaCy (`yeah`, `okay`) contribute noise, highlighting an opportunity for improved NER post-processing.

## Project Structure

```
Clinical-Notes-Summarization/
├── data/
│   ├── raw/                    # MTS-Dialog training set
│   ├── val/                    # Validation split
│   ├── test/                   # Test split
│   └── processed/              # Generated summaries, entities, feature matrices
├── src/
│   ├── preprocess.py           # Data loading, cleaning, tokenization
│   ├── train_summarizer.py     # Fine-tune T5 and BART
│   ├── evaluate_summaries.py   # ROUGE, BERTScore, entity retention
│   ├── entity_extractor.py     # ScispaCy NER pipeline
│   ├── feature_engineer.py     # Build feature matrix from entities
│   ├── classifier.py           # Train section-type classifiers
│   └── explainability.py       # SHAP analysis and plots
├── results/
│   ├── summarization/          # Evaluation metrics, training history
│   └── classification/         # Confusion matrices, SHAP plots
└── models/                     # Saved fine-tuned checkpoints
```

## Setup

```bash
pip install torch transformers datasets evaluate rouge-score bert-score
pip install scispacy pandas numpy matplotlib seaborn scikit-learn xgboost shap
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

## Usage

```bash
# Phase 1: Fine-tune and generate summaries
python src/train_summarizer.py

# Phase 2: Extract medical entities
python src/entity_extractor.py --splits train validation test

# Phase 1 (cont): Evaluate summaries
python src/evaluate_summaries.py

# Phase 3: Build features and train classifiers
python src/feature_engineer.py
python src/classifier.py

# Phase 4: SHAP explainability
python src/explainability.py
```

## Tech Stack

Python, PyTorch, HuggingFace Transformers, ScispaCy, scikit-learn, XGBoost, SHAP, pandas, matplotlib, seaborn
