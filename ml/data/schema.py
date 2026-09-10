"""
Canonical Schema and Validation Engine for Fraud Transaction Data.

Dataset Provenance Note:
------------------------
The primary benchmark used is the simulated Sparkov Credit Card dataset
(Brandon Harris / kartik2112). It is a synthetic benchmark simulating credit card
transactions, cardholder demographics, and merchant interactions. It serves as a
baseline benchmark for imbalanced learning and behavioral modeling and is NOT
real-world production banking data.

The source `cc_num` field is a simulated cardholder/account identifier and is
mapped to `account_id` without assuming cryptographic hashing.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


# Canonical schema field names aligned with PROJECT_SPEC.md
CANONICAL_COLUMNS: List[str] = [
    "transaction_id",
    "account_id",
    "timestamp",
    "unix_time",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "cardholder_lat",
    "cardholder_long",
    "merchant_lat",
    "merchant_long",
    "city_pop",
    "job_category",
    "is_fraud",
]

# Source-to-canonical column mapping for Sparkov dataset
SPARKOV_COLUMN_MAPPING: Dict[str, str] = {
    "trans_num": "transaction_id",
    "cc_num": "account_id",
    "trans_date_trans_time": "timestamp",
    "unix_time": "unix_time",
    "amt": "amount",
    "merchant": "merchant_id",
    "category": "merchant_category",
    "lat": "cardholder_lat",
    "long": "cardholder_long",
    "merch_lat": "merchant_lat",
    "merch_long": "merchant_long",
    "city_pop": "city_pop",
    "job": "job_category",
    "is_fraud": "is_fraud",
}

# Required critical columns that must never contain nulls
NON_NULLABLE_COLUMNS: List[str] = [
    "transaction_id",
    "account_id",
    "timestamp",
    "unix_time",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "is_fraud",
]


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the raw source dataframe contains expected Sparkov fields.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of missing fields)
    """
    required_source_cols = set(SPARKOV_COLUMN_MAPPING.keys())
    missing = [col for col in required_source_cols if col not in df.columns]
    return len(missing) == 0, missing


def transform_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Sparkov source DataFrame to the canonical schema.
    Preserves source timestamp semantics without forcing unverified UTC assumptions.
    """
    out = df.copy()

    # Drop index column if present (e.g., unnamed 0 in raw csv)
    for col in ["Unnamed: 0", "index"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    # Rename source columns to canonical names
    rename_dict = {k: v for k, v in SPARKOV_COLUMN_MAPPING.items() if k in out.columns}
    out = out.rename(columns=rename_dict)

    # Deterministic currency assignment (USD as documented dataset context)
    if "currency" not in out.columns:
        out["currency"] = "USD"

    # Type casting and parsing
    out["transaction_id"] = out["transaction_id"].astype(str)
    out["account_id"] = out["account_id"].astype(str)
    
    # Parse timestamps preserving local datetime format as recorded
    if not np.issubdtype(out["timestamp"].dtype, np.datetime64):
        out["timestamp"] = pd.to_datetime(out["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    out["unix_time"] = out["unix_time"].astype(np.int64)
    out["amount"] = out["amount"].astype(np.float64)
    out["currency"] = out["currency"].astype(str)
    out["merchant_id"] = out["merchant_id"].astype(str)
    out["merchant_category"] = out["merchant_category"].astype(str)
    out["cardholder_lat"] = out["cardholder_lat"].astype(np.float64)
    out["cardholder_long"] = out["cardholder_long"].astype(np.float64)
    out["merchant_lat"] = out["merchant_lat"].astype(np.float64)
    out["merchant_long"] = out["merchant_long"].astype(np.float64)
    out["city_pop"] = out["city_pop"].astype(np.int64)
    out["job_category"] = out["job_category"].astype(str)
    out["is_fraud"] = out["is_fraud"].astype(np.int64)

    # Select only canonical columns in defined order
    out = out[CANONICAL_COLUMNS]
    return out


def audit_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform rigorous validation checks on canonical DataFrame:
    1. Null value counts and handling.
    2. Range checks: amount >= 0, coordinates within [-90, 90] and [-180, 180].
    3. Label check: is_fraud in {0, 1}.
    4. Duplicate transaction_id check.
    5. Multi-key chronological sort by unix_time (non-decreasing) and transaction_id.
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned sorted DataFrame and audit metrics dict.
    """
    initial_rows = len(df)
    audit: Dict[str, Any] = {
        "initial_row_count": initial_rows,
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_txn_ids": int(df.duplicated(subset=["transaction_id"]).sum()),
        "negative_amounts": int((df["amount"] < 0).sum()),
        "invalid_labels": int((~df["is_fraud"].isin([0, 1])).sum()),
        "invalid_cardholder_lat": int((~df["cardholder_lat"].between(-90, 90)).sum()),
        "invalid_cardholder_long": int((~df["cardholder_long"].between(-180, 180)).sum()),
        "invalid_merchant_lat": int((~df["merchant_lat"].between(-90, 90)).sum()),
        "invalid_merchant_long": int((~df["merchant_long"].between(-180, 180)).sum()),
        "removed_rows": 0,
        "cleaning_actions": [],
    }

    clean_df = df.copy()

    # Drop exact duplicate transaction IDs if any exist (keeping first occurrence)
    if audit["duplicate_txn_ids"] > 0:
        clean_df = clean_df.drop_duplicates(subset=["transaction_id"], keep="first")
        removed = initial_rows - len(clean_df)
        audit["removed_rows"] += removed
        audit["cleaning_actions"].append(f"Deduplicated {removed} duplicate transaction_id records (kept first).")

    # Filter invalid records if any exist
    valid_mask = (
        (clean_df["amount"] >= 0)
        & (clean_df["is_fraud"].isin([0, 1]))
        & (clean_df["cardholder_lat"].between(-90, 90))
        & (clean_df["cardholder_long"].between(-180, 180))
        & (clean_df["merchant_lat"].between(-90, 90))
        & (clean_df["merchant_long"].between(-180, 180))
        & (clean_df["timestamp"].notnull())
    )

    invalid_count = int((~valid_mask).sum())
    if invalid_count > 0:
        clean_df = clean_df[valid_mask]
        audit["removed_rows"] += invalid_count
        audit["cleaning_actions"].append(f"Filtered {invalid_count} records failing range/null constraints.")

    # Deterministic multi-key sort: unix_time (non-decreasing) then transaction_id
    clean_df = clean_df.sort_values(
        by=["unix_time", "transaction_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    audit["final_row_count"] = len(clean_df)
    audit["fraud_count"] = int(clean_df["is_fraud"].sum())
    audit["fraud_percentage"] = float((clean_df["is_fraud"].mean() * 100))
    audit["earliest_timestamp"] = str(clean_df["timestamp"].min())
    audit["latest_timestamp"] = str(clean_df["timestamp"].max())
    audit["earliest_unix"] = int(clean_df["unix_time"].min())
    audit["latest_unix"] = int(clean_df["unix_time"].max())

    return clean_df, audit
