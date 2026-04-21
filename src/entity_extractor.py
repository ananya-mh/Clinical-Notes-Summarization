"""
Clinical Entity Extraction
===========================
Extracts medical entities from dialogues and generated summaries using ScispaCy.
"""

import os
import json
import argparse
import pandas as pd
import spacy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODELS = ["t5-small", "t5-base", "bart-base", "bart-large-cnn"]

ENTITY_CATEGORIES = {
    "SYMPTOM": {
        "pain", "ache", "fever", "cough", "nausea", "vomiting", "fatigue",
        "dizziness", "headache", "swelling", "rash", "itching", "bleeding",
        "numbness", "tingling", "weakness", "soreness", "stiffness",
        "congestion", "shortness of breath", "chest pain", "palpitations",
        "diarrhea", "constipation", "insomnia", "anxiety", "depression",
        "chills", "cramps", "burning", "discharge", "bruising",
    },
    "MEDICATION": {
        "mg", "tablet", "capsule", "pill", "dose", "medication", "medicine",
        "prescription", "drug", "antibiotic", "steroid", "aspirin",
        "ibuprofen", "acetaminophen", "metformin", "lisinopril", "atorvastatin",
        "omeprazole", "amlodipine", "metoprolol", "losartan", "gabapentin",
        "prednisone", "insulin", "albuterol", "levothyroxine", "hydrocodone",
    },
    "DIAGNOSIS": {
        "diabetes", "hypertension", "asthma", "arthritis", "cancer",
        "infection", "disease", "disorder", "syndrome", "fracture",
        "pneumonia", "bronchitis", "anemia", "obesity", "osteoporosis",
        "hypothyroidism", "hyperthyroidism", "copd", "gerd", "migraine",
        "allergic rhinitis", "dermatitis", "eczema", "psoriasis",
    },
    "PROCEDURE": {
        "mri", "ct scan", "x-ray", "xray", "ultrasound", "biopsy",
        "surgery", "blood test", "lab", "ecg", "ekg", "eeg", "endoscopy",
        "colonoscopy", "mammogram", "vaccination", "injection", "physical therapy",
        "referral", "screening", "examination", "follow-up", "check-up",
    },
}


def load_scispacy_model():
    try:
        nlp = spacy.load("en_core_sci_sm")
    except OSError:
        print("ERROR: ScispaCy model not found. Install with:")
        print("  pip install scispacy")
        print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
        raise
    return nlp


def categorize_entity(entity_text: str) -> str:
    lower = entity_text.lower().strip()
    for category, keywords in ENTITY_CATEGORIES.items():
        for kw in keywords:
            if kw in lower or lower in kw:
                return category
    return "OTHER"


def extract_entities(nlp, text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        return []

    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        ent_text = ent.text.strip()
        if len(ent_text) < 2 or ent_text.lower() in seen:
            continue
        seen.add(ent_text.lower())

        category = categorize_entity(ent_text)
        entities.append({
            "text": ent_text,
            "label": ent.label_,
            "category": category,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return entities


def compute_entity_stats(entities: list[dict]) -> dict:
    stats = {
        "total_entities": len(entities),
        "symptom_count": 0,
        "medication_count": 0,
        "diagnosis_count": 0,
        "procedure_count": 0,
        "other_count": 0,
        "unique_entities": [],
    }

    for ent in entities:
        cat = ent["category"].lower() + "_count"
        if cat in stats:
            stats[cat] += 1

    stats["unique_entities"] = list({e["text"].lower() for e in entities})
    return stats


def process_split(nlp, split_name: str, models: list[str]) -> pd.DataFrame:
    from preprocess import load_mts_dialog, clean_dialogue

    dataset = load_mts_dialog()
    split = dataset[split_name]
    df = split.to_pandas()
    df["dialogue_clean"] = df["dialogue"].apply(clean_dialogue)

    rows = []

    print(f"\nExtracting entities from {split_name} dialogues ({len(df)} examples)...")
    for i, row in df.iterrows():
        dialogue_ents = extract_entities(nlp, row["dialogue_clean"])
        dialogue_stats = compute_entity_stats(dialogue_ents)

        entry = {
            "ID": row["ID"],
            "section_header": row["section_header"],
            "dialogue_entities": json.dumps(dialogue_ents),
            "dialogue_total": dialogue_stats["total_entities"],
            "dialogue_symptom_count": dialogue_stats["symptom_count"],
            "dialogue_medication_count": dialogue_stats["medication_count"],
            "dialogue_diagnosis_count": dialogue_stats["diagnosis_count"],
            "dialogue_procedure_count": dialogue_stats["procedure_count"],
            "dialogue_other_count": dialogue_stats["other_count"],
        }

        for model_key in models:
            summary_path = os.path.join(PROCESSED_DIR, f"summaries_{model_key}_{split_name}.csv")
            if not os.path.exists(summary_path):
                continue

            summary_df = pd.read_csv(summary_path)
            match = summary_df[summary_df["ID"] == row["ID"]]
            if match.empty:
                continue

            gen_summary = str(match.iloc[0]["generated_summary"])
            summary_ents = extract_entities(nlp, gen_summary)
            summary_stats = compute_entity_stats(summary_ents)

            dialogue_unique = {e["text"].lower() for e in dialogue_ents}
            summary_unique = {e["text"].lower() for e in summary_ents}
            overlap = dialogue_unique & summary_unique
            retention = len(overlap) / len(dialogue_unique) if dialogue_unique else 0.0

            entry[f"{model_key}_summary_entities"] = json.dumps(summary_ents)
            entry[f"{model_key}_summary_total"] = summary_stats["total_entities"]
            entry[f"{model_key}_entity_overlap"] = len(overlap)
            entry[f"{model_key}_entity_retention"] = round(retention, 4)

        rows.append(entry)

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"  [{i+1}/{len(df)}]")

    result_df = pd.DataFrame(rows)
    output_path = os.path.join(PROCESSED_DIR, f"entities_{split_name}.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")

    print_summary(result_df, models)
    return result_df


def print_summary(df: pd.DataFrame, models: list[str]):
    print(f"\n{'='*60}")
    print("ENTITY EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples: {len(df)}")
    print(f"Avg entities per dialogue: {df['dialogue_total'].mean():.1f}")
    print(f"  Symptoms:   {df['dialogue_symptom_count'].mean():.1f}")
    print(f"  Medications:{df['dialogue_medication_count'].mean():.1f}")
    print(f"  Diagnoses:  {df['dialogue_diagnosis_count'].mean():.1f}")
    print(f"  Procedures: {df['dialogue_procedure_count'].mean():.1f}")

    for model_key in models:
        col = f"{model_key}_entity_retention"
        if col in df.columns:
            print(f"\n  {model_key} entity retention: {df[col].mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Extract medical entities from dialogues and summaries")
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        help="Dataset splits to process (default: validation test)",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help="Summary models to compute retention for",
    )
    args = parser.parse_args()

    nlp = load_scispacy_model()
    print(f"Loaded ScispaCy model: {nlp.meta['name']}")

    for split_name in args.splits:
        available_models = []
        for m in args.models:
            path = os.path.join(PROCESSED_DIR, f"summaries_{m}_{split_name}.csv")
            if os.path.exists(path):
                available_models.append(m)
            else:
                print(f"  [skip] No summaries found for {m}/{split_name}")

        process_split(nlp, split_name, available_models)


if __name__ == "__main__":
    main()
