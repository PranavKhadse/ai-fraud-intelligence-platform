"""
Group 6: Merchant & Category Interaction Feature Engineering Module.

Computes behavioral interaction history between accounts and specific merchants / categories,
as well as global merchant / category volume counters, strictly prior (timestamp_j < timestamp_i).
Zero fraud label usage.
"""

from typing import List, Union, Tuple
import pandas as pd
import numpy as np

from ml.features.config import MERCHANT_INTERACTION_FEATURES


def _compute_group_history(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    compute_spend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic vectorized helper to calculate count and spend strictly prior to current transaction.
    
    Returns:
        Tuple[count_array, spend_array] aligned with df index.
    """
    n = len(df)
    counts = np.zeros(n, dtype=np.int32)
    spends = np.zeros(n, dtype=np.float32) if compute_spend else np.empty(0, dtype=np.float32)
    
    grouped = df.groupby(group_cols, sort=False)
    for _, group in grouped:
        indices = group.index.values
        u_times = group["unix_time"].values
        m = len(u_times)
        
        idx_strict = np.searchsorted(u_times, u_times, side="left")
        counts[indices] = idx_strict.astype(np.int32)
        
        if compute_spend:
            amts = group["amount"].values
            cum_amt = np.empty(m + 1, dtype=np.float64)
            cum_amt[0] = 0.0
            np.cumsum(amts, out=cum_amt[1:])
            spends[indices] = cum_amt[idx_strict].astype(np.float32)
            
    return counts, spends


def extract_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 6 merchant & category interaction features:
    - account_merchant_txn_count_before (int32)
    - account_category_txn_count_before (int32)
    - account_merchant_spend_before (float32)
    - account_category_spend_before (float32)
    - merchant_txn_count_before (int32)
    - category_txn_count_before (int32)
    """
    # 1. Account x Merchant interaction
    acc_merch_cnt, acc_merch_spd = _compute_group_history(
        df, group_cols=["account_id", "merchant_id"], compute_spend=True
    )
    
    # 2. Account x Category interaction
    acc_cat_cnt, acc_cat_spd = _compute_group_history(
        df, group_cols=["account_id", "merchant_category"], compute_spend=True
    )
    
    # 3. Global Merchant transaction volume before
    merch_cnt, _ = _compute_group_history(
        df, group_cols="merchant_id", compute_spend=False
    )
    
    # 4. Global Category transaction volume before
    cat_cnt, _ = _compute_group_history(
        df, group_cols="merchant_category", compute_spend=False
    )
    
    out = pd.DataFrame(
        {
            "account_merchant_txn_count_before": acc_merch_cnt,
            "account_category_txn_count_before": acc_cat_cnt,
            "account_merchant_spend_before": acc_merch_spd,
            "account_category_spend_before": acc_cat_spd,
            "merchant_txn_count_before": merch_cnt,
            "category_txn_count_before": cat_cnt,
        },
        index=df.index,
    )
    
    return out[MERCHANT_INTERACTION_FEATURES]
