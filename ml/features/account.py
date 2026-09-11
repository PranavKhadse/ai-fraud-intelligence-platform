"""
Group 5: Account History Feature Engineering Module.

Computes expanding lifetime transaction history, aggregate spending, max amount,
and distinct merchant / category counts for each account strictly prior (timestamp_j < timestamp_i).
"""

from typing import Dict
import pandas as pd
import numpy as np

from ml.features.config import ACCOUNT_HISTORY_FEATURES


def compute_account_history_features(
    unix_times: np.ndarray,
    amounts: np.ndarray,
    merchant_ids: np.ndarray,
    category_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized calculation of account history features for a single account.
    
    Args:
        unix_times: 1D int64 array of unix timestamps (sorted non-decreasingly).
        amounts: 1D float64 array of transaction amounts.
        merchant_ids: 1D array of merchant string identifiers.
        category_ids: 1D array of merchant category string identifiers.
        
    Returns:
        Dict mapping feature names to 1D numpy arrays.
    """
    m = len(unix_times)
    if m == 0:
        return {
            "account_txn_count_before": np.empty(0, dtype=np.int32),
            "account_total_spend_before": np.empty(0, dtype=np.float32),
            "account_avg_amount_before": np.empty(0, dtype=np.float32),
            "account_max_amount_before": np.empty(0, dtype=np.float32),
            "account_unique_merchant_count_before": np.empty(0, dtype=np.int32),
            "account_unique_category_count_before": np.empty(0, dtype=np.int32),
        }

    # Strict point-in-time index
    idx_strict = np.searchsorted(unix_times, unix_times, side="left")
    
    # Cumulative spend and max
    cum_amt = np.empty(m + 1, dtype=np.float64)
    cum_amt[0] = 0.0
    np.cumsum(amounts, out=cum_amt[1:])
    
    cum_max = np.empty(m + 1, dtype=np.float64)
    cum_max[0] = 0.0
    np.maximum.accumulate(amounts, out=cum_max[1:])
    
    txn_count_before = idx_strict.astype(np.int32)
    total_spend_before = cum_amt[idx_strict].astype(np.float32)
    max_amount_before = cum_max[idx_strict].astype(np.float32)
    
    avg_amount_before = np.zeros(m, dtype=np.float32)
    np.divide(total_spend_before, txn_count_before, out=avg_amount_before, where=txn_count_before > 0)
    
    # Efficient O(M) monotonic pointer for tracking distinct merchants & categories
    unique_merch_count = np.zeros(m, dtype=np.int32)
    unique_cat_count = np.zeros(m, dtype=np.int32)
    
    seen_merchants = set()
    seen_categories = set()
    cursor = 0
    
    for i in range(m):
        target_k = idx_strict[i]
        while cursor < target_k:
            seen_merchants.add(merchant_ids[cursor])
            seen_categories.add(category_ids[cursor])
            cursor += 1
        unique_merch_count[i] = len(seen_merchants)
        unique_cat_count[i] = len(seen_categories)
        
    return {
        "account_txn_count_before": txn_count_before,
        "account_total_spend_before": total_spend_before,
        "account_avg_amount_before": avg_amount_before,
        "account_max_amount_before": max_amount_before,
        "account_unique_merchant_count_before": unique_merch_count,
        "account_unique_category_count_before": unique_cat_count,
    }


def extract_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract account history features across all accounts in a DataFrame.
    """
    n = len(df)
    results = {
        "account_txn_count_before": np.zeros(n, dtype=np.int32),
        "account_total_spend_before": np.zeros(n, dtype=np.float32),
        "account_avg_amount_before": np.zeros(n, dtype=np.float32),
        "account_max_amount_before": np.zeros(n, dtype=np.float32),
        "account_unique_merchant_count_before": np.zeros(n, dtype=np.int32),
        "account_unique_category_count_before": np.zeros(n, dtype=np.int32),
    }
    
    grouped = df.groupby("account_id", sort=False)
    for _, group in grouped:
        indices = group.index.values
        u_times = group["unix_time"].values
        amts = group["amount"].values
        merchs = group["merchant_id"].values
        cats = group["merchant_category"].values
        
        acc_feats = compute_account_history_features(u_times, amts, merchs, cats)
        for col in ACCOUNT_HISTORY_FEATURES:
            results[col][indices] = acc_feats[col]
            
    out = pd.DataFrame(results, index=df.index)
    return out[ACCOUNT_HISTORY_FEATURES]
