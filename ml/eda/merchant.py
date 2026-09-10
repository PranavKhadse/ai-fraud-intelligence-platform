"""
Merchant and category fraud analysis module.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_merchants_and_categories(
    df: pd.DataFrame,
    min_support_threshold: int = 500,
) -> Dict[str, Any]:
    """
    Analyze fraud concentration by merchant category and merchant ID.
    Enforces a minimum-support threshold to avoid misleading rates from tiny sample sizes.
    """
    # 1. Category-level analysis
    cat_summary = (
        df.groupby("merchant_category")
        .agg(
            total_txns=("is_fraud", "count"),
            fraud_txns=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100)),
            mean_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            total_fraud_spend=("amount", lambda x: float(x[df.loc[x.index, "is_fraud"] == 1].sum())),
        )
        .reset_index()
    )

    # Sort categories
    top_cats_by_count = cat_summary.sort_values(by="fraud_txns", ascending=False).to_dict(orient="records")
    top_cats_by_rate = cat_summary.sort_values(by="fraud_rate", ascending=False).to_dict(orient="records")

    # 2. Merchant-level concentration analysis
    merch_summary = (
        df.groupby("merchant_id")
        .agg(
            total_txns=("is_fraud", "count"),
            fraud_txns=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100)),
        )
        .reset_index()
    )

    unique_merchants = int(df["merchant_id"].nunique())
    merchants_with_fraud = int((merch_summary["fraud_txns"] > 0).sum())

    # Supported merchants (exceeding min_support_threshold)
    supported_merchants = merch_summary[merch_summary["total_txns"] >= min_support_threshold]
    top_merchants_by_rate = supported_merchants.sort_values(by="fraud_rate", ascending=False).head(15).to_dict(orient="records")
    top_merchants_by_count = merch_summary.sort_values(by="fraud_txns", ascending=False).head(15).to_dict(orient="records")

    return {
        "min_support_threshold": min_support_threshold,
        "unique_categories": int(df["merchant_category"].nunique()),
        "category_summary": cat_summary.to_dict(orient="records"),
        "top_categories_by_fraud_count": top_cats_by_count,
        "top_categories_by_fraud_rate": top_cats_by_rate,
        "unique_merchants": unique_merchants,
        "merchants_with_fraud": merchants_with_fraud,
        "merchant_fraud_prevalence": round(float(merchants_with_fraud / unique_merchants * 100), 2),
        "top_merchants_by_fraud_count": top_merchants_by_count,
        "top_merchants_by_fraud_rate": top_merchants_by_rate,
    }
