"""
Point-in-Time Leakage Audit and Feature Schema Validation Suite.

Performs exhaustive verification on engineered feature dataframes:
- Schema conformance: exactly 62 columns in deterministic order
- Non-null and finite numerical bounds (no NaN, no Inf)
- Behavioral threshold logic consistency (e.g. speed > 800 km/h)
- 1,000-sample point-in-time ground truth reconstruction audit
- Identical-timestamp isolation verification
"""

from typing import Dict, Any, List
import logging
import pandas as pd
import numpy as np

from ml.features.config import (
    FULL_FEATURE_DATASET_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    CANONICAL_COLUMNS,
    IMPOSSIBLE_SPEED_THRESHOLD_KMH,
)

logger = logging.getLogger("FeatureValidation")


def validate_feature_schema_and_finiteness(df: pd.DataFrame, partition_name: str = "dataset") -> Dict[str, Any]:
    """
    Validate that a feature DataFrame satisfies all schema and numerical validity rules.
    """
    col_list = list(df.columns)
    is_cols_valid = (col_list == FULL_FEATURE_DATASET_COLUMNS)
    
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    
    # Check for Inf values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {col: int(np.isinf(df[col]).sum()) for col in numeric_cols if np.isinf(df[col]).sum() > 0}
    total_infs = sum(inf_counts.values())
    
    # Check geographic threshold consistency
    speed_cond = (df["implied_travel_speed_kmh"] > IMPOSSIBLE_SPEED_THRESHOLD_KMH).astype(np.int8)
    speed_flag_mismatches = int((df["is_impossible_travel_speed"] != speed_cond).sum())
    
    # Check first transaction indicator consistency
    first_txn_cond = (df["account_txn_count_before"] == 0).astype(np.int8)
    first_txn_mismatches = int((df["is_first_account_txn"] != first_txn_cond).sum())
    
    is_valid = (
        is_cols_valid
        and total_nulls == 0
        and total_infs == 0
        and speed_flag_mismatches == 0
        and first_txn_mismatches == 0
    )
    
    metrics: Dict[str, Any] = {
        "partition": partition_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns_match_expected_62": is_cols_valid,
        "total_nulls": total_nulls,
        "total_infs": total_infs,
        "inf_columns": inf_counts,
        "speed_flag_mismatches": speed_flag_mismatches,
        "first_txn_mismatches": first_txn_mismatches,
        "validation_passed": is_valid,
    }
    
    if not is_valid:
        logger.error(f"Schema/finiteness validation failed for {partition_name}: {metrics}")
    else:
        logger.info(f"Schema and numerical validity PASSED for {partition_name} ({len(df):,} rows).")
        
    return metrics


def audit_point_in_time_leakage(
    full_df: pd.DataFrame,
    sample_size: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform a rigorous, ground-truth point-in-time audit on a random sample of transactions.
    
    For each sampled transaction i at time t_i:
    1. Filters raw dataset to strictly historical rows: unix_time < t_i for that account_id.
    2. Recomputes independent ground-truth counts, sums, distinct entities, and rolling windows.
    3. Confirms that no rows with timestamp >= t_i (including identical timestamps) were used.
    4. Compares pipeline features against ground-truth values.
    
    Args:
        full_df: The feature-engineered DataFrame (e.g. combined or partition).
        sample_size: Number of transactions to independently reconstruct and audit.
        random_seed: Seed for reproducible sampling.
        
    Returns:
        Dict detailing audit metrics, passed tests, and errors if any.
    """
    logger.info(f"Starting Point-in-Time Leakage Audit on {sample_size} sample transactions...")
    rng = np.random.default_rng(random_seed)
    
    n = len(full_df)
    actual_sample_size = min(sample_size, n)
    sample_indices = rng.choice(n, size=actual_sample_size, replace=False)
    
    audited_features = [
        "account_txn_count_before",
        "account_total_spend_before",
        "txn_count_1h",
        "txn_count_24h",
        "amt_sum_24h",
        "account_unique_merchant_count_before",
        "account_unique_category_count_before",
    ]
    
    passed_samples = 0
    failed_samples = 0
    failures: List[Dict[str, Any]] = []
    
    # Pre-index for fast ground truth lookup
    account_groups = {acc: grp for acc, grp in full_df.groupby("account_id")}
    
    for idx in sample_indices:
        row = full_df.iloc[idx]
        acc_id = row["account_id"]
        t_curr = row["unix_time"]
        
        acc_history = account_groups[acc_id]
        # Strict inequality filter
        strictly_prior = acc_history[acc_history["unix_time"] < t_curr]
        
        # Ground truth values
        gt_count_before = len(strictly_prior)
        gt_spend_before = float(strictly_prior["amount"].sum()) if gt_count_before > 0 else 0.0
        
        # 1h window: (t_curr - 3600, t_curr)
        gt_1h = strictly_prior[strictly_prior["unix_time"] > (t_curr - 3600)]
        gt_count_1h = len(gt_1h)
        
        # 24h window: (t_curr - 86400, t_curr)
        gt_24h = strictly_prior[strictly_prior["unix_time"] > (t_curr - 86400)]
        gt_count_24h = len(gt_24h)
        gt_amt_sum_24h = float(gt_24h["amount"].sum()) if gt_count_24h > 0 else 0.0
        
        gt_unique_merch = int(strictly_prior["merchant_id"].nunique())
        gt_unique_cat = int(strictly_prior["merchant_category"].nunique())
        
        # Pipeline computed values
        p_count_before = int(row["account_txn_count_before"])
        p_spend_before = float(row["account_total_spend_before"])
        p_count_1h = int(row["txn_count_1h"])
        p_count_24h = int(row["txn_count_24h"])
        p_amt_sum_24h = float(row["amt_sum_24h"])
        p_unique_merch = int(row["account_unique_merchant_count_before"])
        p_unique_cat = int(row["account_unique_category_count_before"])
        
        # Comparison: exact integer equality on counts/entities; appropriate float32 tolerance on sums
        is_match = (
            p_count_before == gt_count_before
            and bool(np.isclose(p_spend_before, gt_spend_before, rtol=1e-4, atol=1e-2))
            and p_count_1h == gt_count_1h
            and p_count_24h == gt_count_24h
            and bool(np.isclose(p_amt_sum_24h, gt_amt_sum_24h, rtol=1e-4, atol=1e-2))
            and p_unique_merch == gt_unique_merch
            and p_unique_cat == gt_unique_cat
        )
        
        if is_match:
            passed_samples += 1
        else:
            failed_samples += 1
            if len(failures) < 5:
                failures.append({
                    "transaction_id": row["transaction_id"],
                    "account_id": acc_id,
                    "unix_time": int(t_curr),
                    "expected": {
                        "count_before": gt_count_before,
                        "spend_before": gt_spend_before,
                        "count_1h": gt_count_1h,
                        "count_24h": gt_count_24h,
                        "amt_sum_24h": gt_amt_sum_24h,
                        "unique_merch": gt_unique_merch,
                        "unique_cat": gt_unique_cat,
                    },
                    "actual": {
                        "count_before": p_count_before,
                        "spend_before": p_spend_before,
                        "count_1h": p_count_1h,
                        "count_24h": p_count_24h,
                        "amt_sum_24h": p_amt_sum_24h,
                        "unique_merch": p_unique_merch,
                        "unique_cat": p_unique_cat,
                    },
                })
                
    audit_results: Dict[str, Any] = {
        "audited_samples": actual_sample_size,
        "passed_samples": passed_samples,
        "failed_samples": failed_samples,
        "pass_rate_pct": float(passed_samples / actual_sample_size * 100),
        "audited_features": audited_features,
        "leakage_detected": failed_samples > 0,
        "failures_sample": failures,
    }
    
    if failed_samples == 0:
        logger.info(f"Point-in-Time Leakage Audit: 100% PASSED ({passed_samples}/{actual_sample_size} samples match ground truth exactly).")
    else:
        logger.error(f"Point-in-Time Leakage Audit FAILED on {failed_samples} samples: {failures}")
        
    return audit_results
