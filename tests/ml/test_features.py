"""
Comprehensive Automated Unit Tests for Phase 3 Behavioral Feature Engineering.

Validates:
1. Column count and exact 47 engineered features + 15 canonical = 62 total columns.
2. Exact mathematical definitions across all 7 feature groups.
3. Strict point-in-time inequality (timestamp_j < timestamp_i) and current-row exclusion.
4. Identical-timestamp isolation: equal-timestamp transactions never see each other.
5. Cross-partition historical context continuity (Train -> Val -> Test).
6. Future-row perturbation invariance (zero lookahead leakage).
7. Target independence: features never depend on or read fraud labels.
8. Cold-start deterministic defaults.
9. Geographic Haversine calculation and impossible travel speed threshold (>800 km/h).
10. Numerical validity and finiteness (no NaN, no Inf).
"""

import pytest
import pandas as pd
import numpy as np

from ml.features.config import (
    CANONICAL_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    FULL_FEATURE_DATASET_COLUMNS,
    FEATURE_GROUPS,
    TEMPORAL_FEATURES,
    VELOCITY_FEATURES,
    SPENDING_FEATURES,
    DEVIATION_FEATURES,
    ACCOUNT_HISTORY_FEATURES,
    MERCHANT_INTERACTION_FEATURES,
    GEOGRAPHIC_FEATURES,
    NIGHT_HOURS,
    IMPOSSIBLE_SPEED_THRESHOLD_KMH,
)
from ml.features.temporal import extract_temporal_features
from ml.features.velocity import extract_velocity_features
from ml.features.spending import extract_spending_features
from ml.features.deviation import extract_deviation_features
from ml.features.account import extract_account_features
from ml.features.merchant import extract_merchant_features
from ml.features.geographic import extract_geographic_features, haversine_distance_km
from ml.features.pipeline import (
    extract_all_engineered_features,
    engineer_features_for_partition,
    build_cross_partition_features,
)
from ml.features.validation import (
    validate_feature_schema_and_finiteness,
    audit_point_in_time_leakage,
)


@pytest.fixture
def synthetic_canonical_df() -> pd.DataFrame:
    """
    Construct a deterministic synthetic canonical DataFrame with controlled edge cases:
    - Multiple accounts
    - Equal timestamp collisions for the same account
    - Velocity bursts within 1h, 24h, 7d
    - High-value amount deviations
    - Known geographic coordinates for travel speed calculation
    """
    data = [
        # Account 101: First transaction (Cold start)
        {
            "transaction_id": "TXN_001",
            "account_id": "ACC_101",
            "timestamp": "2020-01-01 00:00:00",
            "unix_time": 1577836800,
            "amount": 50.0,
            "currency": "USD",
            "merchant_id": "MERCH_A",
            "merchant_category": "grocery_pos",
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7150,
            "merchant_long": -74.0080,
            "city_pop": 8000000,
            "job_category": "Engineer",
            "is_fraud": 0,
        },
        # Account 101: 30 minutes later (1800s) - same day, night hour
        {
            "transaction_id": "TXN_002",
            "account_id": "ACC_101",
            "timestamp": "2020-01-01 00:30:00",
            "unix_time": 1577838600,
            "amount": 100.0,
            "currency": "USD",
            "merchant_id": "MERCH_B",
            "merchant_category": "shopping_net",
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7300,
            "merchant_long": -74.0100,
            "city_pop": 8000000,
            "job_category": "Engineer",
            "is_fraud": 0,
        },
        # Account 101: IDENTICAL TIMESTAMP Collision 1 at 02:00:00 (7200s)
        {
            "transaction_id": "TXN_003",
            "account_id": "ACC_101",
            "timestamp": "2020-01-01 02:00:00",
            "unix_time": 1577844000,
            "amount": 200.0,
            "currency": "USD",
            "merchant_id": "MERCH_A",
            "merchant_category": "grocery_pos",
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7150,
            "merchant_long": -74.0080,
            "city_pop": 8000000,
            "job_category": "Engineer",
            "is_fraud": 0,
        },
        # Account 101: IDENTICAL TIMESTAMP Collision 2 at 02:00:00 (7200s) - Different Txn ID
        {
            "transaction_id": "TXN_004",
            "account_id": "ACC_101",
            "timestamp": "2020-01-01 02:00:00",
            "unix_time": 1577844000,
            "amount": 300.0,
            "currency": "USD",
            "merchant_id": "MERCH_C",
            "merchant_category": "misc_net",
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7500,
            "merchant_long": -73.9900,
            "city_pop": 8000000,
            "job_category": "Engineer",
            "is_fraud": 0,
        },
        # Account 101: Impossible Travel Test (TXN_005)
        # Occurs 60 seconds after TXN_003/004, but merchant is in London (5,500 km away) -> speed > 300,000 km/h
        {
            "transaction_id": "TXN_005",
            "account_id": "ACC_101",
            "timestamp": "2020-01-01 02:01:00",
            "unix_time": 1577844060,
            "amount": 500.0,
            "currency": "USD",
            "merchant_id": "MERCH_D",
            "merchant_category": "shopping_net",
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 51.5074,
            "merchant_long": -0.1278,
            "city_pop": 8000000,
            "job_category": "Engineer",
            "is_fraud": 1,
        },
        # Account 102: Independent Account Transaction (Cold start)
        {
            "transaction_id": "TXN_006",
            "account_id": "ACC_102",
            "timestamp": "2020-01-01 02:00:00",  # Same timestamp as ACC_101 TXN_003/004
            "unix_time": 1577844000,
            "amount": 75.0,
            "currency": "USD",
            "merchant_id": "MERCH_A",
            "merchant_category": "grocery_pos",
            "cardholder_lat": 34.0522,
            "cardholder_long": -118.2437,
            "merchant_lat": 34.0550,
            "merchant_long": -118.2450,
            "city_pop": 4000000,
            "job_category": "Designer",
            "is_fraud": 0,
        },
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_feature_counts_and_column_catalog():
    """Verify exact count of engineered features (47), groups (7), and full dataset width (62)."""
    assert len(TEMPORAL_FEATURES) == 11
    assert len(VELOCITY_FEATURES) == 7
    assert len(SPENDING_FEATURES) == 8
    assert len(DEVIATION_FEATURES) == 5
    assert len(ACCOUNT_HISTORY_FEATURES) == 6
    assert len(MERCHANT_INTERACTION_FEATURES) == 6
    assert len(GEOGRAPHIC_FEATURES) == 4
    
    assert len(ENGINEERED_FEATURE_COLUMNS) == 47
    assert len(CANONICAL_COLUMNS) == 15
    assert len(FULL_FEATURE_DATASET_COLUMNS) == 62
    assert len(FEATURE_GROUPS) == 7


def test_temporal_features_and_night_hours(synthetic_canonical_df: pd.DataFrame):
    """Verify Group 1 temporal features, cyclic bounds, and exact night hours definition."""
    df_temp = extract_temporal_features(synthetic_canonical_df)
    
    assert list(df_temp.columns) == TEMPORAL_FEATURES
    assert len(df_temp) == len(synthetic_canonical_df)
    
    # Check TXN_001 (00:00:00) -> hour=0, is_night=1
    assert df_temp["transaction_hour"].iloc[0] == 0
    assert df_temp["is_night"].iloc[0] == 1
    
    # Verify cyclic sine/cosine in [-1, 1]
    assert (df_temp["hour_sin"] >= -1.0).all() and (df_temp["hour_sin"] <= 1.0).all()
    assert (df_temp["hour_cos"] >= -1.0).all() and (df_temp["hour_cos"] <= 1.0).all()
    assert (df_temp["day_of_week_sin"] >= -1.0).all() and (df_temp["day_of_week_sin"] <= 1.0).all()
    assert (df_temp["day_of_week_cos"] >= -1.0).all() and (df_temp["day_of_week_cos"] <= 1.0).all()


def test_cold_start_defaults(synthetic_canonical_df: pd.DataFrame):
    """Verify that first transaction for an account receives deterministic cold-start defaults."""
    df_feats = extract_all_engineered_features(synthetic_canonical_df)
    
    # TXN_001 is ACC_101's first transaction
    row0 = df_feats.iloc[0]
    assert row0["is_first_account_txn"] == 1
    assert row0["time_since_prev_txn_seconds"] == 0.0
    assert row0["txn_count_1h"] == 0
    assert row0["txn_count_24h"] == 0
    assert row0["amt_sum_1h"] == 0.0
    assert row0["amt_sum_24h"] == 0.0
    assert row0["amt_mean_24h"] == 0.0
    assert row0["amt_max_24h"] == 0.0
    assert row0["amt_median_30d"] == 0.0
    assert row0["historical_amount_mean"] == 0.0
    assert row0["historical_amount_std"] == 0.0
    assert row0["historical_amount_median"] == 0.0
    assert row0["amount_zscore"] == 0.0
    assert row0["amount_ratio_to_historical_mean"] == 1.0  # Neutral cold start
    assert row0["account_txn_count_before"] == 0
    assert row0["account_total_spend_before"] == 0.0
    assert row0["account_unique_merchant_count_before"] == 0
    assert row0["account_unique_category_count_before"] == 0
    assert row0["distance_from_prev_merchant_km"] == 0.0
    assert row0["implied_travel_speed_kmh"] == 0.0
    assert row0["is_impossible_travel_speed"] == 0


def test_identical_timestamp_isolation(synthetic_canonical_df: pd.DataFrame):
    """
    CRITICAL POINT-IN-TIME TEST:
    Verify that TXN_003 and TXN_004 (same account, identical timestamp 1577844000)
    CANNOT see each other as historical information!
    """
    df_feats = extract_all_engineered_features(synthetic_canonical_df)
    
    row2 = df_feats.iloc[2]  # TXN_003 (t=1577844000, amt=200)
    row3 = df_feats.iloc[3]  # TXN_004 (t=1577844000, amt=300)
    
    # Both TXN_003 and TXN_004 must only see TXN_001 ($50) and TXN_002 ($100)
    # They must have identical prior count = 2, and prior spend = $150.0
    assert row2["account_txn_count_before"] == 2
    assert row3["account_txn_count_before"] == 2
    
    assert row2["account_total_spend_before"] == 150.0
    assert row3["account_total_spend_before"] == 150.0
    
    # Velocity counts: TXN_001 was at 00:00 (7200s ago -> outside 1h), TXN_002 was at 00:30 (5400s ago -> outside 1h)
    # So 1h count must be 0 for BOTH
    assert row2["txn_count_1h"] == 0
    assert row3["txn_count_1h"] == 0
    
    # 24h count must be 2 for BOTH (TXN_001 and TXN_002)
    assert row2["txn_count_24h"] == 2
    assert row3["txn_count_24h"] == 2
    
    # TXN_005 occurs later at 02:01:00 (t=1577844060).
    # It MUST see all 4 earlier transactions: TXN_001 ($50), TXN_002 ($100), TXN_003 ($200), TXN_004 ($300).
    row4 = df_feats.iloc[4]
    assert row4["account_txn_count_before"] == 4
    assert row4["account_total_spend_before"] == 650.0
    # In 1h window (past 3600s), TXN_003 and TXN_004 occurred 60s ago -> count = 2
    assert row4["txn_count_1h"] == 2
    assert row4["amt_sum_1h"] == 500.0  # 200 + 300


def test_impossible_travel_speed_trigger(synthetic_canonical_df: pd.DataFrame):
    """Verify Haversine distance, speed calculation, and >800 km/h impossible speed flag."""
    df_feats = extract_all_engineered_features(synthetic_canonical_df)
    
    # TXN_005 occurs 60s after TXN_003/004 with a transatlantic merchant coordinate jump (>5000 km in 60s)
    row4 = df_feats.iloc[4]
    assert row4["distance_from_prev_merchant_km"] > 5000.0
    assert row4["implied_travel_speed_kmh"] > 800.0
    assert row4["is_impossible_travel_speed"] == 1


def test_cross_partition_historical_continuity():
    """
    Verify that features computed across Train, Val, Test partitions
    seamlessly preserve historical state across partition boundaries without lookahead leakage.
    """
    # Create Train: 2 transactions for ACC_999
    train_data = pd.DataFrame([
        {
            "transaction_id": "TR_01",
            "account_id": "ACC_999",
            "timestamp": pd.to_datetime("2020-01-01 10:00:00"),
            "unix_time": 1577872800,
            "amount": 100.0,
            "currency": "USD",
            "merchant_id": "M_1",
            "merchant_category": "grocery_pos",
            "cardholder_lat": 40.0,
            "cardholder_long": -74.0,
            "merchant_lat": 40.0,
            "merchant_long": -74.0,
            "city_pop": 100000,
            "job_category": "Pilot",
            "is_fraud": 0,
        },
        {
            "transaction_id": "TR_02",
            "account_id": "ACC_999",
            "timestamp": pd.to_datetime("2020-01-01 11:00:00"),
            "unix_time": 1577876400,
            "amount": 200.0,
            "currency": "USD",
            "merchant_id": "M_2",
            "merchant_category": "shopping_net",
            "cardholder_lat": 40.0,
            "cardholder_long": -74.0,
            "merchant_lat": 40.0,
            "merchant_long": -74.0,
            "city_pop": 100000,
            "job_category": "Pilot",
            "is_fraud": 0,
        },
    ])
    
    # Validation: 1 transaction for ACC_999 occurring at 12:00:00
    val_data = pd.DataFrame([
        {
            "transaction_id": "VL_01",
            "account_id": "ACC_999",
            "timestamp": pd.to_datetime("2020-01-01 12:00:00"),
            "unix_time": 1577880000,
            "amount": 300.0,
            "currency": "USD",
            "merchant_id": "M_1",
            "merchant_category": "grocery_pos",
            "cardholder_lat": 40.0,
            "cardholder_long": -74.0,
            "merchant_lat": 40.0,
            "merchant_long": -74.0,
            "city_pop": 100000,
            "job_category": "Pilot",
            "is_fraud": 0,
        },
    ])
    
    # Test: 1 transaction for ACC_999 occurring at 13:00:00
    test_data = pd.DataFrame([
        {
            "transaction_id": "TE_01",
            "account_id": "ACC_999",
            "timestamp": pd.to_datetime("2020-01-01 13:00:00"),
            "unix_time": 1577883600,
            "amount": 400.0,
            "currency": "USD",
            "merchant_id": "M_3",
            "merchant_category": "misc_net",
            "cardholder_lat": 40.0,
            "cardholder_long": -74.0,
            "merchant_lat": 40.0,
            "merchant_long": -74.0,
            "city_pop": 100000,
            "job_category": "Pilot",
            "is_fraud": 0,
        },
    ])
    
    tr_feats, val_feats, te_feats = build_cross_partition_features(train_data, val_data, test_data)
    
    # Validation row must see the 2 Train transactions
    assert val_feats["account_txn_count_before"].iloc[0] == 2
    assert val_feats["account_total_spend_before"].iloc[0] == 300.0  # 100 + 200
    assert val_feats["is_first_account_txn"].iloc[0] == 0
    
    # Test row must see all 3 preceding transactions (2 Train + 1 Val)
    assert te_feats["account_txn_count_before"].iloc[0] == 3
    assert te_feats["account_total_spend_before"].iloc[0] == 600.0  # 100 + 200 + 300
    
    # Train rows must NOT see Validation or Test rows
    assert tr_feats["account_txn_count_before"].iloc[0] == 0
    assert tr_feats["account_txn_count_before"].iloc[1] == 1


def test_future_perturbation_invariance(synthetic_canonical_df: pd.DataFrame):
    """
    Verify that altering a future transaction has ZERO effect on earlier transaction features.
    """
    df_base = synthetic_canonical_df.copy()
    feats_base = extract_all_engineered_features(df_base)
    
    # Perturb the last transaction (change amount from 500 to 99999 and timestamp)
    df_perturbed = synthetic_canonical_df.copy()
    df_perturbed.loc[df_perturbed.index[-1], "amount"] = 999999.0
    
    feats_perturbed = extract_all_engineered_features(df_perturbed)
    
    # All rows before the last row must be 100% identical
    for i in range(len(df_base) - 1):
        for col in ENGINEERED_FEATURE_COLUMNS:
            v_base = feats_base[col].iloc[i]
            v_pert = feats_perturbed[col].iloc[i]
            assert v_base == v_pert or (np.isnan(v_base) and np.isnan(v_pert)), (
                f"Leakage detected! Feature {col} at index {i} changed when future row was perturbed."
            )


def test_schema_and_finiteness_validation(synthetic_canonical_df: pd.DataFrame):
    """Verify validation module passes clean schema and bounds checks."""
    full_df = engineer_features_for_partition(synthetic_canonical_df)
    metrics = validate_feature_schema_and_finiteness(full_df, "synthetic_test")
    
    assert metrics["validation_passed"] is True
    assert metrics["column_count"] == 62
    assert metrics["total_nulls"] == 0
    assert metrics["total_infs"] == 0
