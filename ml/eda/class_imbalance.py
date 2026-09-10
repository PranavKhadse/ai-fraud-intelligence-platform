"""
Class imbalance analysis module.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_class_imbalance(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Perform deep statistical analysis of the extreme class imbalance across partitions and time.
    """
    results: Dict[str, Any] = {"partitions": {}}

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = len(df)
        frauds = int(df["is_fraud"].sum())
        legit = n - frauds
        pct = float(frauds / n * 100) if n > 0 else 0.0
        ratio = float(legit / frauds) if frauds > 0 else 0.0

        results["partitions"][name] = {
            "total": n,
            "fraud_count": frauds,
            "legit_count": legit,
            "fraud_percentage": round(pct, 4),
            "imbalance_ratio": f"1:{ratio:.1f}",
            "numeric_ratio": ratio,
        }

    # Temporal breakdown on training dataset
    train_copy = train_df.copy()
    train_copy["year_month"] = pd.to_datetime(train_copy["timestamp"]).dt.to_period("M").astype(str)
    monthly_stats = (
        train_copy.groupby("year_month")
        .agg(
            total_txns=("is_fraud", "count"),
            fraud_txns=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100)),
        )
        .reset_index()
    )

    results["train_monthly_breakdown"] = monthly_stats.to_dict(orient="records")
    return results
