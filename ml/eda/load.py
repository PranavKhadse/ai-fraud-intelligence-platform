"""
Data loader for processed benchmark partitions.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd

PROCESSED_DIR = Path("data/processed/benchmark")


def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Train, Validation, and Test Parquet partitions without modifying source files.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed Parquet partitions not found in {PROCESSED_DIR}. "
            "Please run Phase 1 data pipeline first."
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    return train_df, val_df, test_df
