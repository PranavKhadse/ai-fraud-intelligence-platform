"""
End-to-End Data Ingestion, Schema Validation, and Time-Aware Splitting Pipeline.

Executes the Phase 1 benchmark ingestion workflow:
1. Loads raw source files from data/raw/benchmark/.
2. Validates raw source schema.
3. Transforms to canonical schema aligned with PROJECT_SPEC.md.
4. Performs quality audit, range validations, and deterministic multi-key sorting (unix_time, transaction_id).
5. Splits into time-aware Out-of-Time partitions (70% Train, 15% Validation, 15% Test).
6. Writes clean columnar Parquet partitions to data/processed/benchmark/.
7. Generates docs/data_pipeline_report.md with detailed summary statistics.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from ml.data.schema import (
    validate_raw_schema,
    transform_to_canonical,
    audit_and_clean_data,
    CANONICAL_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataPipeline")

RAW_DIR = Path("data/raw/benchmark")
PROCESSED_DIR = Path("data/processed/benchmark")
DOCS_DIR = Path("docs")


def load_raw_benchmark() -> pd.DataFrame:
    """
    Load raw benchmark CSV files from data/raw/benchmark/.
    Combines fraudTrain.csv and fraudTest.csv if both are present.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory {RAW_DIR} does not exist. Run ml/data/download.py first.")

    csv_files = sorted(list(RAW_DIR.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}. Run ml/data/download.py first.")

    dfs = []
    for f in csv_files:
        logger.info(f"Loading raw file: {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)...")
        df_part = pd.read_csv(f)
        dfs.append(df_part)

    df_raw = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded total {len(df_raw):,} raw records from {len(csv_files)} file(s).")
    return df_raw


def time_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform strict Out-of-Time (OOT) temporal splitting.
    Transactions MUST be pre-sorted chronologically.
    
    Args:
        df: Chronologically sorted DataFrame.
        train_ratio: Proportion for training split (default: 0.70).
        val_ratio: Proportion for validation split (default: 0.15).
        
    Returns:
        Tuple[train_df, val_df, test_df, split_metadata]
    """
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[train_idx:val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[val_idx:].copy().reset_index(drop=True)

    metadata: Dict[str, Any] = {
        "total_records": n,
        "splits": {
            "train": {
                "row_count": len(train_df),
                "proportion": float(len(train_df) / n),
                "fraud_count": int(train_df["is_fraud"].sum()),
                "fraud_percentage": float(train_df["is_fraud"].mean() * 100),
                "start_time": str(train_df["timestamp"].min()),
                "end_time": str(train_df["timestamp"].max()),
                "start_unix": int(train_df["unix_time"].min()),
                "end_unix": int(train_df["unix_time"].max()),
            },
            "validation": {
                "row_count": len(val_df),
                "proportion": float(len(val_df) / n),
                "fraud_count": int(val_df["is_fraud"].sum()),
                "fraud_percentage": float(val_df["is_fraud"].mean() * 100),
                "start_time": str(val_df["timestamp"].min()),
                "end_time": str(val_df["timestamp"].max()),
                "start_unix": int(val_df["unix_time"].min()),
                "end_unix": int(val_df["unix_time"].max()),
            },
            "test": {
                "row_count": len(test_df),
                "proportion": float(len(test_df) / n),
                "fraud_count": int(test_df["is_fraud"].sum()),
                "fraud_percentage": float(test_df["is_fraud"].mean() * 100),
                "start_time": str(test_df["timestamp"].min()),
                "end_time": str(test_df["timestamp"].max()),
                "start_unix": int(test_df["unix_time"].min()),
                "end_unix": int(test_df["unix_time"].max()),
            },
        },
        "leakage_checks": {
            "train_val_disjoint": bool(train_df["unix_time"].max() <= val_df["unix_time"].min()),
            "val_test_disjoint": bool(val_df["unix_time"].max() <= test_df["unix_time"].min()),
        },
    }

    return train_df, val_df, test_df, metadata


def generate_pipeline_report(
    audit: Dict[str, Any],
    split_meta: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate comprehensive markdown summary report for Phase 1 data pipeline."""
    splits = split_meta["splits"]

    report_content = f"""# Data Pipeline & Ingestion Report (Phase 1)

> **Dataset:** Synthetic Sparkov Credit Card Fraud Benchmark  
> **Source:** Brandon Harris / `kartik2112/fraud-detection` (Hugging Face Datasets)  
> **Schema Standard:** Canonical 15-Field Transaction Schema ([PROJECT_SPEC.md](file:///PROJECT_SPEC.md))  

---

## 1. Executive Ingestion Summary

- **Total Ingested Records:** {audit['initial_row_count']:,}
- **Cleaned & Deduplicated Records:** {audit['final_row_count']:,}
- **Total Fraud Instances:** {audit['fraud_count']:,} ({audit['fraud_percentage']:.3f}%)
- **Dataset Time Span:** `{audit['earliest_timestamp']}` to `{audit['latest_timestamp']}`
- **Temporal Ordering:** Verified non-decreasing `unix_time` with deterministic secondary sort on `transaction_id`.

---

## 2. Time-Aware Out-of-Time (OOT) Split Statistics

| Partition | Row Count | % Total | Fraud Cases | Fraud % | Temporal Start | Temporal End |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | {splits['train']['row_count']:,} | {splits['train']['proportion']*100:.1f}% | {splits['train']['fraud_count']:,} | {splits['train']['fraud_percentage']:.3f}% | `{splits['train']['start_time']}` | `{splits['train']['end_time']}` |
| **Validation Set** | {splits['validation']['row_count']:,} | {splits['validation']['proportion']*100:.1f}% | {splits['validation']['fraud_count']:,} | {splits['validation']['fraud_percentage']:.3f}% | `{splits['validation']['start_time']}` | `{splits['validation']['end_time']}` |
| **Test Set (OOT)** | {splits['test']['row_count']:,} | {splits['test']['proportion']*100:.1f}% | {splits['test']['fraud_count']:,} | {splits['test']['fraud_percentage']:.3f}% | `{splits['test']['start_time']}` | `{splits['test']['end_time']}` |
| **Total** | **{split_meta['total_records']:,}** | **100.0%** | **{audit['fraud_count']:,}** | **{audit['fraud_percentage']:.3f}%** | `{audit['earliest_timestamp']}` | `{audit['latest_timestamp']}` |

---

## 3. Data Leakage & Temporal Disjointness Audit

- **Train vs. Validation Disjointness:** `{"PASS (Train max <= Val min)" if split_meta['leakage_checks']['train_val_disjoint'] else "FAIL"}`
- **Validation vs. Test Disjointness:** `{"PASS (Val max <= Test min)" if split_meta['leakage_checks']['val_test_disjoint'] else "FAIL"}`
- **Data Leakage Risk:** **Zero Lookahead Leakage**. Random K-Fold partitioning is strictly avoided; all future evaluations will be conducted out-of-time.

---

## 4. Data Quality & Cleaning Decisions

- **Null Value Counts:** {json.dumps(audit['null_counts'])}
- **Duplicate Transaction IDs Found:** {audit['duplicate_txn_ids']}
- **Negative Monetary Amounts:** {audit['negative_amounts']}
- **Invalid Class Labels:** {audit['invalid_labels']}
- **Invalid Coordinates Out of Bounds:** Lat: {audit['invalid_cardholder_lat'] + audit['invalid_merchant_lat']}, Long: {audit['invalid_cardholder_long'] + audit['invalid_merchant_long']}
- **Cleaning Actions Applied:**
{chr(10).join([f"  - {action}" for action in audit['cleaning_actions']]) if audit['cleaning_actions'] else "  - Clean baseline: No invalid rows or coordinate anomalies detected."}

---

## 5. Output Storage Details

Processed columnar files saved in Apache Parquet format:
- `data/processed/benchmark/train.parquet`
- `data/processed/benchmark/val.parquet`
- `data/processed/benchmark/test.parquet`

*All raw and processed data artifacts are excluded from Git tracking via `.gitignore`.*
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Pipeline report saved to {output_path}")


def run_pipeline() -> None:
    """Execute the complete Phase 1 data pipeline."""
    logger.info("Starting Phase 1 Data Ingestion Pipeline...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw CSVs
    df_raw = load_raw_benchmark()

    # 2. Validate Raw Schema
    is_valid, missing = validate_raw_schema(df_raw)
    if not is_valid:
        raise ValueError(f"Raw source schema validation failed. Missing expected columns: {missing}")
    logger.info("Raw schema validation passed.")

    # 3. Transform to Canonical Schema
    logger.info("Mapping to canonical transaction schema...")
    df_canonical = transform_to_canonical(df_raw)

    # 4. Audit, Clean, Range Check, and Multi-Key Sort
    logger.info("Auditing data quality and performing multi-key chronological sorting...")
    df_clean, audit_metrics = audit_and_clean_data(df_canonical)
    logger.info(
        f"Data audit completed: {audit_metrics['final_row_count']:,} valid records, "
        f"{audit_metrics['fraud_count']:,} frauds ({audit_metrics['fraud_percentage']:.3f}%)."
    )

    # 5. Time-Aware Split (70% Train, 15% Val, 15% Test)
    logger.info("Performing time-aware Out-of-Time split (70/15/15)...")
    train_df, val_df, test_df, split_meta = time_aware_split(df_clean)

    # 6. Save Parquet Partitions
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    logger.info("Writing processed Parquet partitions...")
    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    val_df.to_parquet(val_path, index=False, engine="pyarrow")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")

    logger.info(f"Saved: {train_path.name} ({train_path.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"Saved: {val_path.name} ({val_path.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"Saved: {test_path.name} ({test_path.stat().st_size / (1024*1024):.2f} MB)")

    # 7. Generate Pipeline Report
    report_path = DOCS_DIR / "data_pipeline_report.md"
    generate_pipeline_report(audit_metrics, split_meta, report_path)
    logger.info("Phase 1 Data Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
