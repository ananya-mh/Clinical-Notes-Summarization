"""
Feature Engineering
====================
Builds a feature matrix from extracted entities for section-type classification.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np

from preprocess import load_mts_dialog, clean_dialogue, clean_section_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def count_speaker_turns(dialogue: str) -> int:
    return len(re.findall(r"(Doctor|Patient)\s*:", dialogue))


def get_top_entities(entity_col: pd.Series, top_n: int = 20) -> list[str]:
    all_entities = []
    for raw in entity_col.dropna():
        try:
            ents = json.loads(raw)
            all_entities.extend(e["text"].lower() for e in ents)
        except (json.JSONDecodeError, TypeError):
            continue

    freq = pd.Series(all_entities).value_counts()
    return freq.head(top_n).index.tolist()


def build_features(split_name: str, top_entities: list[str] = None) -> pd.DataFrame:
    entity_path = os.path.join(PROCESSED_DIR, f"entities_{split_name}.csv")
    if not os.path.exists(entity_path):
        print(f"  Entity file not found: {entity_path}")
        print(f"  Run entity_extractor.py first.")
        return None

    df = pd.read_csv(entity_path)

    dataset = load_mts_dialog()
    split = dataset[split_name]
    raw_df = split.to_pandas()
    raw_df["dialogue_clean"] = raw_df["dialogue"].apply(clean_dialogue)
    raw_df["summary_clean"] = raw_df["section_text"].apply(clean_section_text)

    features = pd.DataFrame()
    features["ID"] = df["ID"]
    features["section_header"] = df["section_header"]

    features["symptom_count"] = df["dialogue_symptom_count"]
    features["medication_count"] = df["dialogue_medication_count"]
    features["diagnosis_count"] = df["dialogue_diagnosis_count"]
    features["procedure_count"] = df["dialogue_procedure_count"]
    features["other_count"] = df["dialogue_other_count"]
    features["total_entities"] = df["dialogue_total"]

    features["dialogue_word_count"] = raw_df["dialogue_clean"].apply(lambda x: len(str(x).split()))
    features["summary_word_count"] = raw_df["summary_clean"].apply(lambda x: len(str(x).split()))
    features["speaker_turn_count"] = raw_df["dialogue_clean"].apply(count_speaker_turns)

    features["unique_entity_ratio"] = 0.0
    for i, raw in enumerate(df["dialogue_entities"]):
        try:
            ents = json.loads(raw)
            total = len(ents)
            unique = len({e["text"].lower() for e in ents})
            features.loc[i, "unique_entity_ratio"] = unique / total if total > 0 else 0.0
        except (json.JSONDecodeError, TypeError):
            pass

    if top_entities is None:
        top_entities = get_top_entities(df["dialogue_entities"])

    for ent_name in top_entities:
        col_name = f"has_{ent_name.replace(' ', '_')}"
        flags = []
        for raw in df["dialogue_entities"]:
            try:
                ents = json.loads(raw)
                entity_texts = {e["text"].lower() for e in ents}
                flags.append(1 if ent_name in entity_texts else 0)
            except (json.JSONDecodeError, TypeError):
                flags.append(0)
        features[col_name] = flags

    return features, top_entities


def main():
    parser = argparse.ArgumentParser(description="Build feature matrix from extracted entities")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"],
        help="Splits to build features for",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of top entities for presence flags")
    args = parser.parse_args()

    top_entities = None

    for split_name in args.splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*60}")

        result = build_features(split_name, top_entities)
        if result is None:
            continue

        features, top_entities = result

        output_path = os.path.join(PROCESSED_DIR, f"features_{split_name}.csv")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        features.to_csv(output_path, index=False)
        print(f"Saved → {output_path}")
        print(f"Shape: {features.shape}")
        print(f"\nFeature columns: {list(features.columns)}")
        print(f"\nSection header distribution:")
        print(features["section_header"].value_counts().to_string())


if __name__ == "__main__":
    main()
