"""
Dataset overview and descriptive profile module.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def compute_partition_overview(df: pd.DataFrame, partition_name: str) -> Dict[str, Any]:
    """
    Compute comprehensive summary metrics for a single partition.
    """
    n = len(df)
    fraud_count = int(df["is_fraud"].sum())
    legit_count = n - fraud_count
    fraud_pct = float(fraud_count / n * 100) if n > 0 else 0.0

    amt = df["amount"]
    amt_stats = {
        "count": int(amt.count()),
        "mean": float(amt.mean()),
        "std": float(amt.std()),
        "median": float(amt.median()),
        "min": float(amt.min()),
        "max": float(amt.max()),
        "p25": float(amt.quantile(0.25)),
        "p50": float(amt.quantile(0.50)),
        "p75": float(amt.quantile(0.75)),
        "p90": float(amt.quantile(0.90)),
        "p95": float(amt.quantile(0.95)),
        "p99": float(amt.quantile(0.99)),
    }

    ts = pd.to_datetime(df["timestamp"])
    time_span_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400.0)

    return {
        "partition": partition_name,
        "row_count": n,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_percentage": fraud_pct,
        "imbalance_ratio": f"1:{int(legit_count / fraud_count)}" if fraud_count > 0 else "N/A",
        "unique_accounts": int(df["account_id"].nunique()),
        "unique_merchants": int(df["merchant_id"].nunique()),
        "unique_categories": int(df["merchant_category"].nunique()),
        "unique_jobs": int(df["job_category"].nunique()),
        "amount_stats": amt_stats,
        "start_time": str(ts.min()),
        "end_time": str(ts.max()),
        "time_span_days": round(time_span_days, 1),
        "txns_per_day": round(n / time_span_days, 1),
        "txns_per_month": round(n / (time_span_days / 30.4375), 1),
    }


def compute_all_overviews(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute summary profiles across all three temporal partitions.
    """
    return {
        "train": compute_partition_overview(train_df, "Train Set (70%)"),
        "val": compute_partition_overview(val_df, "Validation Set (15%)"),
        "test": compute_partition_overview(test_df, "Test Set (15% OOT)"),
    }
