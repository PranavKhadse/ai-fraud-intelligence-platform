"""
Master Feature Engineering Pipeline and Cross-Partition Historical Context Orchestrator.

Enforces zero-leakage cross-partition context:
- Train: History = earlier Train only
- Validation: History = all Train + earlier Validation
- Test: History = all Train + all Validation + earlier Test
Zero usage of fraud labels during feature extraction.
"""

from typing import Tuple, List
import logging
import pandas as pd
import numpy as np

from ml.features.config import (
    CANONICAL_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    FULL_FEATURE_DATASET_COLUMNS,
)
from ml.features.temporal import extract_temporal_features
from ml.features.velocity import extract_velocity_features
from ml.features.spending import extract_spending_features
from ml.features.deviation import extract_deviation_features
from ml.features.account import extract_account_features
from ml.features.merchant import extract_merchant_features
from ml.features.geographic import extract_geographic_features

logger = logging.getLogger("FeaturePipeline")


def extract_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all 47 engineered features across all 7 feature groups for an input DataFrame.
    Guarantees that `is_fraud` is NEVER accessed by feature extraction logic.
    
    Args:
        df: Chronologically sorted DataFrame conforming to the canonical schema.
        
    Returns:
        pd.DataFrame containing exactly the 47 engineered features.
    """
    # Defensive measure: Remove `is_fraud` from working DataFrame to prevent any accidental target leakage
    feature_input_cols = [c for c in df.columns if c != "is_fraud"]
    df_clean = df[feature_input_cols].copy()
    
    # 1. Temporal Features (11)
    df_temporal = extract_temporal_features(df_clean)
    
    # 2. Velocity Features (7)
    df_velocity = extract_velocity_features(df_clean)
    
    # 3. Spending Features (8)
    df_spending = extract_spending_features(df_clean)
    
    # 4. Spending Deviation Features (5)
    df_deviation = extract_deviation_features(df_clean)
    
    # 5. Account History Features (6)
    df_account = extract_account_features(df_clean)
    
    # 6. Merchant / Category Interaction Features (6)
    df_merchant = extract_merchant_features(df_clean)
    
    # 7. Geographic / Travel Features (4)
    df_geographic = extract_geographic_features(df_clean)
    
    # Concatenate all 47 engineered feature blocks
    engineered_df = pd.concat(
        [
            df_temporal,
            df_velocity,
            df_spending,
            df_deviation,
            df_account,
            df_merchant,
            df_geographic,
        ],
        axis=1,
    )
    
    # Enforce exact column order
    return engineered_df[ENGINEERED_FEATURE_COLUMNS]


def engineer_features_for_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a single partition DataFrame with all 47 engineered features,
    producing the full 62-column output dataset.
    """
    engineered_feats = extract_all_engineered_features(df)
    
    # Combine canonical columns (including is_fraud) with engineered features
    out = pd.concat([df[CANONICAL_COLUMNS], engineered_feats], axis=1)
    return out[FULL_FEATURE_DATASET_COLUMNS]


def build_cross_partition_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate feature dataframes for Train, Validation, and Test with seamless
    cross-partition point-in-time history preservation:
    
    - Train: Calculated solely on Train stream (N_train)
    - Validation: Calculated on [Train + Val] stream (N_train + N_val), then sliced for Val
    - Test: Calculated on [Train + Val + Test] stream (N_total), then sliced for Test
    
    Returns:
        Tuple[train_features_df, val_features_df, test_features_df]
    """
    logger.info("Generating features for Train partition (Train history only)...")
    train_feat_df = engineer_features_for_partition(train_df)
    
    logger.info("Generating features for Validation partition (Train + Val history)...")
    val_stream = pd.concat([train_df, val_df], ignore_index=True)
    full_val_feats = engineer_features_for_partition(val_stream)
    val_feat_df = full_val_feats.iloc[len(train_df):].reset_index(drop=True)
    
    logger.info("Generating features for Test partition (Train + Val + Test history)...")
    test_stream = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_test_feats = engineer_features_for_partition(test_stream)
    test_feat_df = full_test_feats.iloc[(len(train_df) + len(val_df)):].reset_index(drop=True)
    
    # Assert row counts match original inputs exactly
    assert len(train_feat_df) == len(train_df), "Train row count mismatch!"
    assert len(val_feat_df) == len(val_df), "Validation row count mismatch!"
    assert len(test_feat_df) == len(test_df), "Test row count mismatch!"
    
    # Assert transaction_ids align exactly
    assert (train_feat_df["transaction_id"].values == train_df["transaction_id"].values).all(), "Train txn IDs misaligned!"
    assert (val_feat_df["transaction_id"].values == val_df["transaction_id"].values).all(), "Val txn IDs misaligned!"
    assert (test_feat_df["transaction_id"].values == test_df["transaction_id"].values).all(), "Test txn IDs misaligned!"
    
    return train_feat_df, val_feat_df, test_feat_df
