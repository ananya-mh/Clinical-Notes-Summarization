"""
Fine-Tune T5 & BART for Clinical Summarization
================================================
Trains T5 and BART on MTS-Dialog train split, evaluates on val,
generates summaries on val/test, and saves checkpoints.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset

from preprocess import load_mts_dialog, clean_dialogue, clean_section_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "summarization")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CONFIGS = {
    "t5-small": {
        "name": "t5-small",
        "max_input": 512,
        "max_output": 256,
        "prefix": "summarize: ",
        "batch_size": 16,
        "lr": 3e-5,
        "epochs": 3,
    },
    "t5-base": {
        "name": "t5-base",
        "max_input": 512,
        "max_output": 256,
        "prefix": "summarize: ",
        "batch_size": 8,
        "lr": 2e-5,
        "epochs": 3,
    },
    "bart-base": {
        "name": "facebook/bart-base",
        "max_input": 512,
        "max_output": 256,
        "prefix": "",
        "batch_size": 8,
        "lr": 3e-5,
        "epochs": 3,
    },
    "bart-large-cnn": {
        "name": "facebook/bart-large-cnn",
        "max_input": 1024,
        "max_output": 256,
        "prefix": "",
        "batch_size": 4,
        "lr": 2e-5,
        "epochs": 3,
    },
}


def prepare_split(split_name: str) -> pd.DataFrame:
    dataset = load_mts_dialog()
    split = dataset[split_name]
    df = split.to_pandas()
    df["dialogue_clean"] = df["dialogue"].apply(clean_dialogue)
    df["summary_clean"] = df["section_text"].apply(clean_section_text)
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer, cfg: dict) -> Dataset:
    sources = [cfg["prefix"] + d for d in df["dialogue_clean"].tolist()]
    targets = df["summary_clean"].tolist()

    model_inputs = tokenizer(
        sources,
        max_length=cfg["max_input"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    labels = tokenizer(
        text_target=targets,
        max_length=cfg["max_output"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    label_ids = labels["input_ids"]
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    dataset = Dataset.from_dict({
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": label_ids,
    })
    dataset.set_format("torch")
    return dataset


def train(model_key: str, device: str = None):
    cfg = CONFIGS[model_key]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading tokenizer and model: {cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["name"])
    model.to(device)

    print("Preparing datasets...")
    train_df = prepare_split("train")
    val_df = prepare_split("validation")

    train_dataset = tokenize_dataset(train_df, tokenizer, cfg)
    val_dataset = tokenize_dataset(val_df, tokenizer, cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16 = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    print(f"\nTraining {model_key} for {cfg['epochs']} epochs")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples:   {len(val_dataset)}")
    print(f"  Batch size:     {cfg['batch_size']}")
    print(f"  Total steps:    {total_steps}")
    print(f"  Warmup steps:   {warmup_steps}")
    print(f"  Mixed precision: {use_fp16}")

    history = []
    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg = train_loss / (step + 1)
                elapsed = time.time() - start
                print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)} "
                      f"loss={avg:.4f} elapsed={elapsed:.0f}s")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start

        print(f"  Epoch {epoch+1}/{cfg['epochs']} — "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"time={elapsed:.0f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = os.path.join(MODELS_DIR, model_key, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  Saved best checkpoint → {save_dir}")

    save_dir = os.path.join(MODELS_DIR, model_key, "final")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved final checkpoint → {save_dir}")

    history_df = pd.DataFrame(history)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_df.to_csv(os.path.join(RESULTS_DIR, f"training_history_{model_key}.csv"), index=False)

    return model, tokenizer


def generate_summaries(model_key: str, split_name: str, device: str = None):
    cfg = CONFIGS[model_key]
    checkpoint_dir = os.path.join(MODELS_DIR, model_key, "best")

    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint found at {checkpoint_dir}. Train first.")
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = os.path.join(PROCESSED_DIR, f"summaries_{model_key}_{split_name}.csv")
    if os.path.exists(output_path):
        print(f"  [skip] {output_path} already exists. Delete to regenerate.")
        return pd.read_csv(output_path)

    print(f"\nLoading best checkpoint for {model_key}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    df = prepare_split(split_name)
    generated = []
    start = time.time()

    print(f"Generating summaries for {split_name} ({len(df)} examples)...")
    for i, row in df.iterrows():
        source = cfg["prefix"] + row["dialogue_clean"]

        inputs = tokenizer(
            source,
            return_tensors="pt",
            max_length=cfg["max_input"],
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg["max_output"],
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated.append(summary)

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(df)}] {elapsed:.0f}s")

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
    print(f"Saved → {output_path}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5/BART for clinical summarization")
    parser.add_argument(
        "--models", nargs="+", default=["t5-small", "bart-base"],
        choices=list(CONFIGS.keys()),
        help="Models to train (default: t5-small bart-base)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        help="Splits to generate summaries for after training",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only generate summaries")
    args = parser.parse_args()

    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key}")
        print(f"{'='*60}")

        if not args.skip_train:
            train(model_key, args.device)

        for split_name in args.splits:
            generate_summaries(model_key, split_name, args.device)


if __name__ == "__main__":
    main()
