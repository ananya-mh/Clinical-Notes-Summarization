"""
Preprocessing & Tokenization Pipeline for Clinical Text Summarization
=====================================================================
Cleans MTS-Dialog data and tokenizes for T5 and BART fine-tuning.
"""

import os
import re
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# 1. RAW DATA LOADING
# ---------------------------------------------------------------------------

def load_mts_dialog(data_dir: str = None) -> DatasetDict:
    """Load MTS-Dialog dataset from local CSV files."""
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

    train_df = pd.read_csv(os.path.join(data_dir, "raw", "MTS-Dialog-TrainingSet.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val", "MTS-Dialog-ValidationSet.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test", "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"))

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })
    return dataset


# ---------------------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_dialogue(text: str) -> str:
    """Clean doctor-patient dialogue text."""
    if not isinstance(text, str):
        return ""

    # Normalize line breaks and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize speaker labels to consistent format
    text = re.sub(r"(Doctor|Patient)\s*:\s*", r"\1: ", text)

    # Collapse multiple spaces/newlines
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_section_text(text: str) -> str:
    """Clean the target clinical note / summary."""
    if not isinstance(text, str):
        return ""

    # Normalize whitespace
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def preprocess_example(example: dict) -> dict:
    """Clean a single example (applied via dataset.map)."""
    return {
        "dialogue_clean": clean_dialogue(example["dialogue"]),
        "summary_clean": clean_section_text(example["section_text"]),
        "section_header": example["section_header"].strip().upper(),
    }


# ---------------------------------------------------------------------------
# 3. DATA STATISTICS  (helps verify ~40% compression target)
# ---------------------------------------------------------------------------

def compute_stats(dataset: DatasetDict) -> pd.DataFrame:
    """Compute length stats for input dialogues and target summaries."""
    rows = []
    for split_name, split_data in dataset.items():
        for ex in split_data:
            src_len = len(ex["dialogue_clean"].split())
            tgt_len = len(ex["summary_clean"].split())
            compression = 1 - (tgt_len / src_len) if src_len > 0 else 0
            rows.append({
                "split": split_name,
                "section_header": ex["section_header"],
                "src_words": src_len,
                "tgt_words": tgt_len,
                "compression_ratio": round(compression, 3),
            })

    stats_df = pd.DataFrame(rows)

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Per-split summary
    for split in ["train", "validation", "test"]:
        s = stats_df[stats_df["split"] == split]
        print(f"\n--- {split.upper()} ({len(s)} examples) ---")
        print(f"  Source words  : mean={s['src_words'].mean():.0f}, "
              f"median={s['src_words'].median():.0f}, "
              f"max={s['src_words'].max()}")
        print(f"  Target words  : mean={s['tgt_words'].mean():.0f}, "
              f"median={s['tgt_words'].median():.0f}, "
              f"max={s['tgt_words'].max()}")
        print(f"  Compression   : mean={s['compression_ratio'].mean():.1%}")

    # Section header distribution (train only)
    print("\n--- SECTION HEADERS (train) ---")
    header_counts = stats_df[stats_df["split"] == "train"]["section_header"].value_counts()
    for header, count in header_counts.items():
        print(f"  {header:<20s} {count:>5d}  ({count/len(stats_df[stats_df['split']=='train']):.1%})")

    return stats_df


# ---------------------------------------------------------------------------
# 4. TOKENIZATION FOR T5 / BART
# ---------------------------------------------------------------------------

# Max token lengths (tunable based on stats output)
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 256

# T5 requires a task prefix
T5_PREFIX = "summarize: "


def get_tokenizer(model_name: str):
    """Load tokenizer for the chosen model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded tokenizer: {model_name} (vocab size: {tokenizer.vocab_size})")
    return tokenizer


def tokenize_for_t5(example: dict, tokenizer) -> dict:
    """Tokenize a single example for T5 fine-tuning."""
    source = T5_PREFIX + example["dialogue_clean"]

    model_inputs = tokenizer(
        source,
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        text_target=example["summary_clean"],
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )

    # Replace pad token ids in labels with -100 so they are ignored in loss
    labels["input_ids"] = [
        (lid if lid != tokenizer.pad_token_id else -100)
        for lid in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_for_bart(example: dict, tokenizer) -> dict:
    """Tokenize a single example for BART fine-tuning."""
    source = example["dialogue_clean"]

    model_inputs = tokenizer(
        source,
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        text_target=example["summary_clean"],
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )

    labels["input_ids"] = [
        (lid if lid != tokenizer.pad_token_id else -100)
        for lid in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_dataset(dataset: DatasetDict, model_name: str) -> DatasetDict:
    """
    Full pipeline: clean → tokenize → return model-ready DatasetDict.

    Args:
        dataset:    raw DatasetDict from load_mts_dialog()
        model_name: e.g. "t5-small", "t5-base", "facebook/bart-base"
    """
    # Step 1: Clean text
    print("[1/3] Cleaning text...")
    dataset = dataset.map(preprocess_example, desc="Cleaning")

    # Step 2: Show stats
    print("[2/3] Computing stats...")
    compute_stats(dataset)

    # Step 3: Tokenize
    print(f"\n[3/3] Tokenizing for {model_name}...")
    tokenizer = get_tokenizer(model_name)

    is_t5 = "t5" in model_name.lower()
    tok_fn = (
        lambda ex: tokenize_for_t5(ex, tokenizer)
        if is_t5
        else lambda ex: tokenize_for_bart(ex, tokenizer)
    )

    # Pick the right function cleanly
    if is_t5:
        tokenized = dataset.map(
            lambda ex: tokenize_for_t5(ex, tokenizer),
            desc="Tokenizing (T5)",
        )
    else:
        tokenized = dataset.map(
            lambda ex: tokenize_for_bart(ex, tokenizer),
            desc="Tokenizing (BART)",
        )

    # Keep only columns needed for training
    keep_cols = ["input_ids", "attention_mask", "labels"]
    remove_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)

    tokenized.set_format("torch")

    print(f"\nDone! Final dataset:")
    print(tokenized)
    print(f"\nSample input_ids shape : {tokenized['train'][0]['input_ids'].shape}")
    print(f"Sample labels shape    : {tokenized['train'][0]['labels'].shape}")

    return tokenized, tokenizer


# ---------------------------------------------------------------------------
# 5. MAIN — run standalone to verify the pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MTS-Dialog for summarization")
    parser.add_argument(
        "--model", type=str, default="t5-small",
        help="Model name for tokenizer (e.g. t5-small, t5-base, facebook/bart-base)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data/ folder (auto-detected if not set)"
    )
    args = parser.parse_args()

    # Load raw data
    print("Loading MTS-Dialog dataset...\n")
    raw_dataset = load_mts_dialog(args.data_dir)
    print(raw_dataset)
    print()

    # Run full pipeline
    tokenized_dataset, tokenizer = prepare_dataset(raw_dataset, args.model)