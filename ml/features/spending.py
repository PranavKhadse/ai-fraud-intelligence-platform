"""
Group 3: Spending Feature Engineering Module.

Computes rolling amount sums, means, maximums, and medians for each account,
enforcing strict point-in-time inequality (timestamp_j < timestamp_i).
"""

from typing import Dict
import pandas as pd
import numpy as np

from ml.features.config import SPENDING_FEATURES, WINDOW_SECONDS


def compute_spending_features_for_account(
    unix_times: np.ndarray,
    amounts: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized calculation of spending features for a single account's chronologically sorted transactions.
    
    Args:
        unix_times: 1D int64 array of unix timestamps (sorted non-decreasingly).
        amounts: 1D float64 array of transaction amounts.
        
    Returns:
        Dict mapping feature names to 1D numpy arrays.
    """
    m = len(unix_times)
    if m == 0:
        return {col: np.empty(0, dtype=np.float32) for col in SPENDING_FEATURES}

    # Strict point-in-time boundary (excludes current and equal timestamp rows)
    idx_strict = np.searchsorted(unix_times, unix_times, side="left")
    
    # Cumulative sum with leading 0 for O(1) interval sums
    cum_amt = np.empty(m + 1, dtype=np.float64)
    cum_amt[0] = 0.0
    np.cumsum(amounts, out=cum_amt[1:])
    
    w_1h = WINDOW_SECONDS["1h"]
    w_24h = WINDOW_SECONDS["24h"]
    w_7d = WINDOW_SECONDS["7d"]
    w_30d = WINDOW_SECONDS["30d"]
    
    start_1h = np.searchsorted(unix_times, unix_times - w_1h, side="right")
    start_24h = np.searchsorted(unix_times, unix_times - w_24h, side="right")
    start_7d = np.searchsorted(unix_times, unix_times - w_7d, side="right")
    start_30d = np.searchsorted(unix_times, unix_times - w_30d, side="right")
    
    sum_1h = np.maximum(0.0, cum_amt[idx_strict] - cum_amt[start_1h]).astype(np.float32)
    sum_24h = np.maximum(0.0, cum_amt[idx_strict] - cum_amt[start_24h]).astype(np.float32)
    sum_7d = np.maximum(0.0, cum_amt[idx_strict] - cum_amt[start_7d]).astype(np.float32)
    sum_30d = np.maximum(0.0, cum_amt[idx_strict] - cum_amt[start_30d]).astype(np.float32)
    
    count_24h = np.maximum(0, idx_strict - start_24h)
    count_7d = np.maximum(0, idx_strict - start_7d)
    
    mean_24h = np.zeros(m, dtype=np.float32)
    np.divide(sum_24h, count_24h, out=mean_24h, where=count_24h > 0)
    
    mean_7d = np.zeros(m, dtype=np.float32)
    np.divide(sum_7d, count_7d, out=mean_7d, where=count_7d > 0)
    
    # Max in 24h and median in 30d
    max_24h = np.zeros(m, dtype=np.float32)
    median_30d = np.zeros(m, dtype=np.float32)
    
    for i in range(m):
        s24 = start_24h[i]
        e = idx_strict[i]
        if s24 < e:
            max_24h[i] = np.max(amounts[s24:e])
            
        s30 = start_30d[i]
        if s30 < e:
            median_30d[i] = np.median(amounts[s30:e])
            
    return {
        "amt_sum_1h": sum_1h,
        "amt_sum_24h": sum_24h,
        "amt_sum_7d": sum_7d,
        "amt_sum_30d": sum_30d,
        "amt_mean_24h": mean_24h,
        "amt_mean_7d": mean_7d,
        "amt_max_24h": max_24h,
        "amt_median_30d": median_30d,
    }


def extract_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract spending features across all accounts in a DataFrame.
    """
    n = len(df)
    results = {col: np.zeros(n, dtype=np.float32) for col in SPENDING_FEATURES}
    
    grouped = df.groupby("account_id", sort=False)
    for _, group in grouped:
        indices = group.index.values
        u_times = group["unix_time"].values
        amts = group["amount"].values
        
        acc_feats = compute_spending_features_for_account(u_times, amts)
        for col in SPENDING_FEATURES:
            results[col][indices] = acc_feats[col]
            
    out = pd.DataFrame(results, index=df.index)
    return out[SPENDING_FEATURES]
