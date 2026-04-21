"""
Zero-Shot Summarization Benchmark
==================================
Runs 4 pre-trained models on MTS-Dialog val/test splits and saves generated summaries.
"""

import os
import time
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from preprocess import load_mts_dialog, clean_dialogue, clean_section_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "summarization")

MODELS = {
    "t5-base": {
        "name": "t5-base",
        "max_input": 512,
        "max_output": 256,
        "prefix": "summarize: ",
    },
    "bart-large-cnn": {
        "name": "facebook/bart-large-cnn",
        "max_input": 1024,
        "max_output": 256,
        "prefix": "",
    },
    "medical-t5": {
        "name": "Falconsai/medical_summarization",
        "max_input": 512,
        "max_output": 256,
        "prefix": "summarize: ",
    },
    "pegasus-xsum": {
        "name": "google/pegasus-xsum",
        "max_input": 512,
        "max_output": 256,
        "prefix": "",
    },
}


def load_split(split_name: str) -> pd.DataFrame:
    dataset = load_mts_dialog()
    split = dataset[split_name]
    df = split.to_pandas()
    df["dialogue_clean"] = df["dialogue"].apply(clean_dialogue)
    df["summary_clean"] = df["section_text"].apply(clean_section_text)
    return df


def run_model(model_key: str, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cfg = MODELS[model_key]
    output_path = os.path.join(PROCESSED_DIR, f"summaries_{model_key}_{split_name}.csv")

    if os.path.exists(output_path):
        print(f"  [skip] {output_path} already exists. Delete to regenerate.")
        return pd.read_csv(output_path)

    print(f"  Loading {cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"])
    model.eval()

    generated = []
    total = len(df)
    start = time.time()

    for i, row in df.iterrows():
        source = cfg["prefix"] + row["dialogue_clean"]

        inputs = tokenizer(
            source,
            return_tensors="pt",
            max_length=cfg["max_input"],
            truncation=True,
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg["max_output"],
                min_length=min(30, cfg["max_output"]),
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated.append(summary)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{total}] {rate:.1f} ex/s  elapsed={elapsed:.0f}s")

    result_df = pd.DataFrame({
        "ID": df["ID"],
        "section_header": df["section_header"],
        "dialogue": df["dialogue_clean"],
        "reference_summary": df["summary_clean"],
        "generated_summary": generated,
        "model": model_key,
    })

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")

    return result_df


def build_comparison_table(split_name: str) -> pd.DataFrame:
    rows = []
    for model_key in MODELS:
        path = os.path.join(PROCESSED_DIR, f"summaries_{model_key}_{split_name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        gen_words = df["generated_summary"].apply(lambda x: len(str(x).split()))
        ref_words = df["reference_summary"].apply(lambda x: len(str(x).split()))
        src_words = df["dialogue"].apply(lambda x: len(str(x).split()))
        compression = 1 - (gen_words / src_words)

        rows.append({
            "model": model_key,
            "n_examples": len(df),
            "avg_gen_words": gen_words.mean(),
            "avg_ref_words": ref_words.mean(),
            "avg_src_words": src_words.mean(),
            "avg_compression": compression.mean(),
        })

    table = pd.DataFrame(rows)
    table_path = os.path.join(RESULTS_DIR, f"summary_stats_{split_name}.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"\nComparison table saved → {table_path}")
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description="Zero-shot summarization benchmark")
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        help="Dataset splits to run (default: validation test)",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to benchmark (default: all)",
    )
    args = parser.parse_args()

    for split_name in args.splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*60}")
        df = load_split(split_name)
        print(f"Loaded {len(df)} examples\n")

        for model_key in args.models:
            print(f"\n--- {model_key} ---")
            run_model(model_key, df, split_name)

        build_comparison_table(split_name)


if __name__ == "__main__":
    main()
