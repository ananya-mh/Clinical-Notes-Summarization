import os
import pandas as pd
from datasets import Dataset, DatasetDict

def load_mts_dialog():
    """Load MTS-Dialog dataset from local CSV files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # Load each split
    train_df = pd.read_csv(os.path.join(data_dir, "raw", "MTS-Dialog-TrainingSet.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val", "MTS-Dialog-ValidationSet.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test", "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"))

    # Preview the data
    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"\nSample:\n{train_df.head(2)}")

    # Convert to HuggingFace DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

    return dataset


if __name__ == "__main__":
    ds = load_mts_dialog()
    print(ds)