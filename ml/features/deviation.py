"""
Group 4: Spending Deviation Feature Engineering Module.

Calculates account baseline spending deviations, expanding mean, standard deviation,
median, Z-scores, and amount ratios strictly against prior transactions (timestamp_j < timestamp_i).
"""

from typing import Dict
import pandas as pd
import numpy as np

from ml.features.config import DEVIATION_FEATURES


def compute_deviation_features_for_account(
    unix_times: np.ndarray,
    amounts: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized calculation of spending deviation features for a single account.
    
    Args:
        unix_times: 1D int64 array of unix timestamps (sorted non-decreasingly).
        amounts: 1D float64 array of transaction amounts.
        
    Returns:
        Dict mapping feature names to 1D numpy arrays.
    """
    m = len(unix_times)
    if m == 0:
        return {col: np.empty(0, dtype=np.float32) for col in DEVIATION_FEATURES}

    # Strict point-in-time boundary: number of transactions strictly before current timestamp
    idx_strict = np.searchsorted(unix_times, unix_times, side="left")
    
    # Cumulative sums of A and A^2 for O(1) expanding mean and sample variance
    cum_amt = np.empty(m + 1, dtype=np.float64)
    cum_amt[0] = 0.0
    np.cumsum(amounts, out=cum_amt[1:])
    
    cum_sq = np.empty(m + 1, dtype=np.float64)
    cum_sq[0] = 0.0
    np.cumsum(amounts ** 2, out=cum_sq[1:])
    
    hist_mean = np.zeros(m, dtype=np.float32)
    hist_std = np.zeros(m, dtype=np.float32)
    hist_median = np.zeros(m, dtype=np.float32)
    amount_zscore = np.zeros(m, dtype=np.float32)
    amount_ratio = np.ones(m, dtype=np.float32)  # Default neutral ratio is 1.0
    
    last_n = -1
    cached_median = 0.0
    
    for i in range(m):
        n_prior = idx_strict[i]
        curr_amt = amounts[i]
        
        if n_prior > 0:
            sum_val = cum_amt[n_prior]
            mean_val = sum_val / n_prior
            hist_mean[i] = mean_val
            
            # Ratio to historical mean (default 1.0 if mean <= 0)
            if mean_val > 1e-6:
                amount_ratio[i] = curr_amt / mean_val
            else:
                amount_ratio[i] = 1.0
                
            # Sample variance and standard deviation (requires N >= 2)
            if n_prior >= 2:
                sq_sum = cum_sq[n_prior]
                # Unbiased sample variance formula
                var = (sq_sum - (sum_val ** 2 / n_prior)) / (n_prior - 1)
                std_val = np.sqrt(max(0.0, var))
                hist_std[i] = std_val
                
                if std_val > 1e-6:
                    amount_zscore[i] = (curr_amt - mean_val) / std_val
                else:
                    amount_zscore[i] = 0.0
            else:
                hist_std[i] = 0.0
                amount_zscore[i] = 0.0
                
            # Expanding median
            if n_prior == last_n:
                hist_median[i] = cached_median
            else:
                cached_median = float(np.median(amounts[:n_prior]))
                hist_median[i] = cached_median
                last_n = n_prior
        else:
            # Cold start (0 prior transactions)
            hist_mean[i] = 0.0
            hist_std[i] = 0.0
            hist_median[i] = 0.0
            amount_zscore[i] = 0.0
            amount_ratio[i] = 1.0
            
    return {
        "historical_amount_mean": hist_mean,
        "historical_amount_std": hist_std,
        "historical_amount_median": hist_median,
        "amount_zscore": amount_zscore,
        "amount_ratio_to_historical_mean": amount_ratio,
    }


def extract_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract spending deviation features across all accounts in a DataFrame.
    """
    n = len(df)
    results = {col: np.zeros(n, dtype=np.float32) for col in DEVIATION_FEATURES}
    
    grouped = df.groupby("account_id", sort=False)
    for _, group in grouped:
        indices = group.index.values
        u_times = group["unix_time"].values
        amts = group["amount"].values
        
        acc_feats = compute_deviation_features_for_account(u_times, amts)
        for col in DEVIATION_FEATURES:
            results[col][indices] = acc_feats[col]
            
    out = pd.DataFrame(results, index=df.index)
    return out[DEVIATION_FEATURES]
