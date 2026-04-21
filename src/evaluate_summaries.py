"""
Summary Evaluation
==================
Computes ROUGE, BERTScore, compression ratio, and entity retention
for generated summaries vs reference clinical notes.
"""

import os
import argparse
import pandas as pd
import numpy as np
import evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "summarization")

MODELS = ["t5-small", "bart-base"]


def load_summaries(model_key: str, split_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"summaries_{model_key}_{split_name}.csv")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    return pd.read_csv(path)


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return {
        "rouge1": round(results["rouge1"], 4),
        "rouge2": round(results["rouge2"], 4),
        "rougeL": round(results["rougeL"], 4),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="distilbert-base-uncased",
    )
    return {
        "bertscore_precision": round(np.mean(results["precision"]), 4),
        "bertscore_recall": round(np.mean(results["recall"]), 4),
        "bertscore_f1": round(np.mean(results["f1"]), 4),
    }


def compute_compression(df: pd.DataFrame) -> dict:
    src_words = df["dialogue"].apply(lambda x: len(str(x).split()))
    gen_words = df["generated_summary"].apply(lambda x: len(str(x).split()))
    compression = 1 - (gen_words / src_words)
    return {
        "avg_compression": round(compression.mean(), 4),
        "avg_src_words": round(src_words.mean(), 1),
        "avg_gen_words": round(gen_words.mean(), 1),
        "avg_ref_words": round(df["reference_summary"].apply(lambda x: len(str(x).split())).mean(), 1),
    }


def compute_entity_retention(model_key: str, split_name: str) -> dict:
    entity_path = os.path.join(PROCESSED_DIR, f"entities_{split_name}.csv")
    if not os.path.exists(entity_path):
        return {}

    df = pd.read_csv(entity_path)
    col = f"{model_key}_entity_retention"
    if col not in df.columns:
        return {}

    retention = df[col].dropna()
    if retention.empty:
        return {}

    return {
        "entity_retention": round(retention.mean(), 4),
        "entity_retention_median": round(retention.median(), 4),
    }


def evaluate_model(model_key: str, split_name: str) -> dict:
    df = load_summaries(model_key, split_name)
    if df is None:
        return None

    predictions = df["generated_summary"].fillna("").astype(str).tolist()
    references = df["reference_summary"].fillna("").astype(str).tolist()

    print(f"  Computing ROUGE...")
    metrics = compute_rouge(predictions, references)

    print(f"  Computing BERTScore...")
    metrics.update(compute_bertscore(predictions, references))

    print(f"  Computing compression...")
    metrics.update(compute_compression(df))

    print(f"  Computing entity retention...")
    metrics.update(compute_entity_retention(model_key, split_name))

    metrics["model"] = model_key
    metrics["split"] = split_name
    metrics["n_examples"] = len(df)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated summaries")
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help="Models to evaluate",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        help="Splits to evaluate",
    )
    args = parser.parse_args()

    all_results = []

    for split_name in args.splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*60}")

        for model_key in args.models:
            print(f"\n--- {model_key} ---")
            metrics = evaluate_model(model_key, split_name)
            if metrics:
                all_results.append(metrics)

    if not all_results:
        print("\nNo results to report.")
        return

    results_df = pd.DataFrame(all_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved → {output_path}")

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")

    display_cols = ["model", "split", "rouge1", "rouge2", "rougeL",
                    "bertscore_f1", "avg_compression", "entity_retention"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
