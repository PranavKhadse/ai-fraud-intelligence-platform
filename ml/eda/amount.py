"""
Transaction amount analysis module.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_amount_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare transaction amount distribution between legitimate and fraudulent transactions.
    """
    legit_amt = df[df["is_fraud"] == 0]["amount"]
    fraud_amt = df[df["is_fraud"] == 1]["amount"]

    def stats_dict(s: pd.Series) -> Dict[str, float]:
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
        }

    # Amount Buckets Analysis
    bins = [0, 25, 50, 100, 200, 500, 1000, 5000, float("inf")]
    labels = ["$0-$25", "$25-$50", "$50-$100", "$100-$200", "$200-$500", "$500-$1000", "$1000-$5000", "$5000+"]

    df_copy = df.copy()
    df_copy["amount_bucket"] = pd.cut(df_copy["amount"], bins=bins, labels=labels, right=False)

    bucket_stats = (
        df_copy.groupby("amount_bucket", observed=False)
        .agg(
            total_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100) if len(x) > 0 else 0.0),
        )
        .reset_index()
    )

    return {
        "legitimate_stats": stats_dict(legit_amt),
        "fraudulent_stats": stats_dict(fraud_amt),
        "overall_stats": stats_dict(df["amount"]),
        "bucket_analysis": bucket_stats.to_dict(orient="records"),
    }
