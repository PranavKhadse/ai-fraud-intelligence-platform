"""
Account-level descriptive analysis and behavioral signals discovery module.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_account_behavior(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Descriptive account-level aggregations and behavioral patterns.
    """
    account_grp = (
        df.groupby("account_id")
        .agg(
            txn_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            total_spend=("amount", "sum"),
            mean_spend=("amount", "mean"),
            std_spend=("amount", "std"),
        )
        .reset_index()
    )

    total_accounts = len(account_grp)
    accounts_with_fraud = int((account_grp["fraud_count"] > 0).sum())
    accounts_without_fraud = total_accounts - accounts_with_fraud
    account_fraud_prevalence = float(accounts_with_fraud / total_accounts * 100) if total_accounts > 0 else 0.0

    counts = account_grp["txn_count"]
    total_spends = account_grp["total_spend"]
    mean_spends = account_grp["mean_spend"]

    def summary_stats(s: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()) if not s.isnull().all() else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
        }

    # Compare accounts with fraud vs without fraud
    fraud_accs = account_grp[account_grp["fraud_count"] > 0]
    clean_accs = account_grp[account_grp["fraud_count"] == 0]

    return {
        "total_unique_accounts": total_accounts,
        "accounts_with_fraud": accounts_with_fraud,
        "accounts_without_fraud": accounts_without_fraud,
        "account_fraud_prevalence_pct": round(account_fraud_prevalence, 2),
        "txns_per_account_stats": summary_stats(counts),
        "total_spend_per_account_stats": summary_stats(total_spends),
        "mean_spend_per_account_stats": summary_stats(mean_spends),
        "fraud_accounts_profile": {
            "mean_txns": float(fraud_accs["txn_count"].mean()) if len(fraud_accs) > 0 else 0.0,
            "mean_total_spend": float(fraud_accs["total_spend"].mean()) if len(fraud_accs) > 0 else 0.0,
            "mean_avg_spend": float(fraud_accs["mean_spend"].mean()) if len(fraud_accs) > 0 else 0.0,
        },
        "clean_accounts_profile": {
            "mean_txns": float(clean_accs["txn_count"].mean()) if len(clean_accs) > 0 else 0.0,
            "mean_total_spend": float(clean_accs["total_spend"].mean()) if len(clean_accs) > 0 else 0.0,
            "mean_avg_spend": float(clean_accs["mean_spend"].mean()) if len(clean_accs) > 0 else 0.0,
        },
    }
