"""
Automated unit tests for Phase 1 Data Pipeline & Ingestion.

Validates:
- Schema validation and column mapping
- Data type enforcement and non-nullable field integrity
- Non-decreasing multi-key chronological sorting
- Strict Out-of-Time temporal splitting with zero leakage
- Boundary constraints (coordinates, amounts, labels)
- Parquet storage serialization
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ml.data.schema import (
    CANONICAL_COLUMNS,
    NON_NULLABLE_COLUMNS,
    SPARKOV_COLUMN_MAPPING,
    validate_raw_schema,
    transform_to_canonical,
    audit_and_clean_data,
)
from ml.data.pipeline import time_aware_split


@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    """Create a realistic mock raw DataFrame matching Sparkov structure."""
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3, 4],
        "trans_date_trans_time": [
            "2020-01-01 00:00:15",
            "2020-01-01 00:00:15", # Same second transaction (tests secondary sort)
            "2020-01-01 01:15:30",
            "2020-01-01 03:45:00",
            "2020-01-01 05:00:00",
        ],
        "cc_num": [1001, 1002, 1001, 1003, 1004],
        "merchant": ["fraud_StoreA", "fraud_StoreB", "fraud_StoreA", "fraud_StoreC", "fraud_StoreD"],
        "category": ["grocery_pos", "shopping_net", "gas_transport", "misc_net", "entertainment"],
        "amt": [45.50, 120.00, 35.10, 850.00, 15.00],
        "first": ["Alice", "Bob", "Alice", "Charlie", "David"],
        "last": ["Smith", "Jones", "Smith", "Brown", "Taylor"],
        "gender": ["F", "M", "F", "M", "M"],
        "street": ["123 Main St", "456 Elm St", "123 Main St", "789 Oak St", "321 Pine St"],
        "city": ["New York", "Chicago", "New York", "Houston", "Phoenix"],
        "state": ["NY", "IL", "NY", "TX", "AZ"],
        "zip": [10001, 60601, 10001, 77001, 85001],
        "lat": [40.7128, 41.8781, 40.7128, 29.7604, 33.4484],
        "long": [-74.0060, -87.6298, -74.0060, -95.3698, -112.0740],
        "city_pop": [8000000, 2700000, 8000000, 2300000, 1600000],
        "job": ["Engineer", "Teacher", "Engineer", "Accountant", "Doctor"],
        "dob": ["1985-05-12", "1990-08-22", "1985-05-12", "1978-01-15", "1995-11-30"],
        "trans_num": ["TXN_002", "TXN_001", "TXN_003", "TXN_004", "TXN_005"],
        "unix_time": [1577836815, 1577836815, 1577841330, 1577850300, 1577854800],
        "merch_lat": [40.7589, 41.8819, 40.7484, 29.7499, 33.4500],
        "merch_long": [-73.9851, -87.6278, -73.9857, -95.3584, -112.0700],
        "is_fraud": [0, 0, 0, 1, 0],
    })


def test_raw_schema_validation_success(mock_raw_df: pd.DataFrame):
    """Ensure valid raw DataFrame passes schema check."""
    is_valid, missing = validate_raw_schema(mock_raw_df)
    assert is_valid is True
    assert len(missing) == 0


def test_raw_schema_validation_missing_columns():
    """Ensure missing columns are properly detected."""
    invalid_df = pd.DataFrame({"amt": [10.0], "trans_num": ["TXN_1"]})
    is_valid, missing = validate_raw_schema(invalid_df)
    assert is_valid is False
    assert "cc_num" in missing
    assert "unix_time" in missing


def test_canonical_transformation(mock_raw_df: pd.DataFrame):
    """Verify transformation maps exactly to the 15 canonical columns with correct types."""
    canonical_df = transform_to_canonical(mock_raw_df)
    
    assert list(canonical_df.columns) == CANONICAL_COLUMNS
    assert len(canonical_df) == len(mock_raw_df)
    assert canonical_df["currency"].iloc[0] == "USD"
    assert canonical_df["account_id"].dtype == object  # string cast
    assert canonical_df["transaction_id"].dtype == object
    assert np.issubdtype(canonical_df["timestamp"].dtype, np.datetime64)
    assert canonical_df["amount"].dtype == np.float64
    assert canonical_df["is_fraud"].dtype == np.int64


def test_multi_key_chronological_sorting(mock_raw_df: pd.DataFrame):
    """Verify transactions sort non-decreasingly by unix_time and secondarily by transaction_id."""
    canonical_df = transform_to_canonical(mock_raw_df)
    clean_df, audit = audit_and_clean_data(canonical_df)

    # Check unix_time monotonicity
    unix_times = clean_df["unix_time"].values
    assert np.all(unix_times[:-1] <= unix_times[1:]), "unix_time must be non-decreasing"

    # Check secondary sort when unix_time is identical (first two rows)
    assert clean_df.iloc[0]["transaction_id"] == "TXN_001"
    assert clean_df.iloc[1]["transaction_id"] == "TXN_002"


def test_data_cleaning_and_range_validation():
    """Verify invalid records (negative amount, out-of-bounds coords) are filtered and logged."""
    dirty_df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "account_id": ["A1", "A2", "A3", "A4"],
        "timestamp": pd.to_datetime(["2020-01-01 01:00:00"] * 4),
        "unix_time": [1000, 1001, 1002, 1003],
        "amount": [50.0, -10.0, 75.0, 100.0],  # T2 is negative
        "currency": ["USD"] * 4,
        "merchant_id": ["M1"] * 4,
        "merchant_category": ["cat1"] * 4,
        "cardholder_lat": [40.0, 40.0, 95.0, 40.0],  # T3 has invalid latitude > 90
        "cardholder_long": [-74.0] * 4,
        "merchant_lat": [40.0] * 4,
        "merchant_long": [-74.0] * 4,
        "city_pop": [100000] * 4,
        "job_category": ["Job"] * 4,
        "is_fraud": [0, 0, 0, 0],
    })

    clean_df, audit = audit_and_clean_data(dirty_df)
    assert len(clean_df) == 2
    assert audit["negative_amounts"] == 1
    assert audit["invalid_cardholder_lat"] == 1
    assert "T2" not in clean_df["transaction_id"].values
    assert "T3" not in clean_df["transaction_id"].values


def test_time_aware_split_no_leakage():
    """Verify Out-of-Time temporal splitting guarantees zero lookahead leakage."""
    n = 100
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i:04d}" for i in range(n)],
        "account_id": [f"ACC_{i%10}" for i in range(n)],
        "timestamp": times,
        "unix_time": [int(t.timestamp()) for t in times],
        "amount": [10.0 + i for i in range(n)],
        "currency": ["USD"] * n,
        "merchant_id": ["M1"] * n,
        "merchant_category": ["grocery"] * n,
        "cardholder_lat": [40.0] * n,
        "cardholder_long": [-74.0] * n,
        "merchant_lat": [40.0] * n,
        "merchant_long": [-74.0] * n,
        "city_pop": [50000] * n,
        "job_category": ["Engineer"] * n,
        "is_fraud": [1 if i % 20 == 0 else 0 for i in range(n)],
    })

    train_df, val_df, test_df, meta = time_aware_split(df, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strict temporal boundary checks
    assert train_df["unix_time"].max() <= val_df["unix_time"].min(), "Train/Val temporal leakage detected"
    assert val_df["unix_time"].max() <= test_df["unix_time"].min(), "Val/Test temporal leakage detected"
    assert meta["leakage_checks"]["train_val_disjoint"] is True
    assert meta["leakage_checks"]["val_test_disjoint"] is True


def test_processed_parquet_partitions():
    """Verify generated Parquet files match schema and temporal disjointness."""
    processed_dir = Path("data/processed/benchmark")
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed parquet partitions not generated yet.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Schema integrity
    for partition_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert list(df.columns) == CANONICAL_COLUMNS, f"{partition_name} columns mismatch"
        assert len(df) > 0, f"{partition_name} partition is empty"
        assert df["is_fraud"].sum() > 0, f"{partition_name} contains zero fraud samples"
        assert df.isnull().sum().sum() == 0, f"{partition_name} contains null values"

    # Temporal non-overlap
    assert train_df["unix_time"].max() <= val_df["unix_time"].min()
    assert val_df["unix_time"].max() <= test_df["unix_time"].min()

