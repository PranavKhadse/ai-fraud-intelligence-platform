"""
Group 2: Velocity Feature Engineering Module.

Computes rolling transaction frequency bursts and time elapsed since previous transaction
for each account, enforcing strict point-in-time inequality (timestamp_j < timestamp_i).
"""

from typing import Dict, List
import pandas as pd
import numpy as np

from ml.features.config import VELOCITY_FEATURES, WINDOW_SECONDS


def compute_velocity_features_for_account(
    unix_times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized calculation of velocity features for a single account's chronologically sorted transactions.
    
    Args:
        unix_times: 1D int64 array of unix timestamps for the account (sorted non-decreasingly).
        
    Returns:
        Dict mapping feature names to 1D numpy arrays.
    """
    m = len(unix_times)
    if m == 0:
        return {
            "txn_count_1h": np.empty(0, dtype=np.int32),
            "txn_count_6h": np.empty(0, dtype=np.int32),
            "txn_count_24h": np.empty(0, dtype=np.int32),
            "txn_count_7d": np.empty(0, dtype=np.int32),
            "txn_count_30d": np.empty(0, dtype=np.int32),
            "time_since_prev_txn_seconds": np.empty(0, dtype=np.float32),
            "is_first_account_txn": np.empty(0, dtype=np.int8),
        }

    # Strict point-in-time index: first index with timestamp >= current timestamp
    # All elements in 0..idx_strict-1 satisfy timestamp_j < timestamp_i
    # Identical timestamps start at idx_strict and are strictly excluded
    idx_strict = np.searchsorted(unix_times, unix_times, side="left")
    
    is_first = (idx_strict == 0).astype(np.int8)
    
    prev_idx = idx_strict - 1
    has_prev = prev_idx >= 0
    time_since_prev = np.zeros(m, dtype=np.float32)
    if np.any(has_prev):
        time_since_prev[has_prev] = (unix_times[has_prev] - unix_times[prev_idx[has_prev]]).astype(np.float32)
        
    # Window counts: (t - W, t)
    w_1h = WINDOW_SECONDS["1h"]
    w_6h = WINDOW_SECONDS["6h"]
    w_24h = WINDOW_SECONDS["24h"]
    w_7d = WINDOW_SECONDS["7d"]
    w_30d = WINDOW_SECONDS["30d"]
    
    start_1h = np.searchsorted(unix_times, unix_times - w_1h, side="right")
    start_6h = np.searchsorted(unix_times, unix_times - w_6h, side="right")
    start_24h = np.searchsorted(unix_times, unix_times - w_24h, side="right")
    start_7d = np.searchsorted(unix_times, unix_times - w_7d, side="right")
    start_30d = np.searchsorted(unix_times, unix_times - w_30d, side="right")
    
    count_1h = np.maximum(0, idx_strict - start_1h).astype(np.int32)
    count_6h = np.maximum(0, idx_strict - start_6h).astype(np.int32)
    count_24h = np.maximum(0, idx_strict - start_24h).astype(np.int32)
    count_7d = np.maximum(0, idx_strict - start_7d).astype(np.int32)
    count_30d = np.maximum(0, idx_strict - start_30d).astype(np.int32)
    
    return {
        "txn_count_1h": count_1h,
        "txn_count_6h": count_6h,
        "txn_count_24h": count_24h,
        "txn_count_7d": count_7d,
        "txn_count_30d": count_30d,
        "time_since_prev_txn_seconds": time_since_prev,
        "is_first_account_txn": is_first,
    }


def extract_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract velocity features across all accounts in a DataFrame.
    Assumes DataFrame is sorted by unix_time and transaction_id.
    """
    n = len(df)
    results = {col: np.zeros(n, dtype=np.int32 if "count" in col else (np.int8 if col == "is_first_account_txn" else np.float32)) for col in VELOCITY_FEATURES}
    
    # Group by account_id
    grouped = df.groupby("account_id", sort=False)
    for _, group in grouped:
        indices = group.index.values
        u_times = group["unix_time"].values
        
        acc_feats = compute_velocity_features_for_account(u_times)
        for col in VELOCITY_FEATURES:
            results[col][indices] = acc_feats[col]
            
    out = pd.DataFrame(results, index=df.index)
    return out[VELOCITY_FEATURES]
