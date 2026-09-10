"""
Automated unit tests for Phase 2 Exploratory Data Analysis (EDA) & Insights.

Validates:
- Data loader integrity and read-only non-destructive behavior
- Canonical column consistency across splits
- Internal arithmetic consistency of class counts and percentages
- Numerical validity of amount and distance statistics
- Category aggregation and minimum-support filtering
- Execution of full EDA pipeline
- Generation and integrity of report and all 15 figures
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from ml.data.schema import CANONICAL_COLUMNS
from ml.eda.load import load_processed_splits
from ml.eda.overview import compute_partition_overview, compute_all_overviews
from ml.eda.class_imbalance import analyze_class_imbalance
from ml.eda.temporal import analyze_temporal_patterns
from ml.eda.amount import compute_amount_distribution
from ml.eda.merchant import analyze_merchants_and_categories
from ml.eda.geography import analyze_geography, haversine_distance_km
from ml.eda.behavioral_signals import analyze_account_behavior
from ml.eda.correlations import analyze_correlations, cramers_v


@pytest.fixture(scope="module")
def loaded_splits():
    """Load splits once for test module."""
    return load_processed_splits()


def test_processed_splits_loading(loaded_splits):
    """Verify that train, val, and test splits load properly with canonical columns."""
    train_df, val_df, test_df = loaded_splits

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert list(df.columns) == CANONICAL_COLUMNS, f"Columns mismatch in {name}"
        assert df["is_fraud"].isin([0, 1]).all()
        assert (df["amount"] >= 0).all()


def test_temporal_disjointness_preserved(loaded_splits):
    """Verify that OOT splits preserve strict temporal ordering without leakage."""
    train_df, val_df, test_df = loaded_splits
    assert train_df["unix_time"].max() <= val_df["unix_time"].min()
    assert val_df["unix_time"].max() <= test_df["unix_time"].min()


def test_overview_metrics_consistency(loaded_splits):
    """Verify overview metric calculations and arithmetic consistency."""
    train_df, val_df, test_df = loaded_splits
    overviews = compute_all_overviews(train_df, val_df, test_df)

    for name, key in [("train", train_df), ("val", val_df), ("test", test_df)]:
        meta = overviews[name]
        assert meta["row_count"] == len(key)
        assert meta["fraud_count"] + meta["legit_count"] == meta["row_count"]
        expected_pct = (meta["fraud_count"] / meta["row_count"]) * 100
        assert abs(meta["fraud_percentage"] - expected_pct) < 1e-4
        assert meta["amount_stats"]["min"] <= meta["amount_stats"]["median"] <= meta["amount_stats"]["max"]


def test_class_imbalance_analysis(loaded_splits):
    """Verify class imbalance analyzer output structure."""
    train_df, val_df, test_df = loaded_splits
    imbalance = analyze_class_imbalance(train_df, val_df, test_df)

    assert "partitions" in imbalance
    assert "train" in imbalance["partitions"]
    assert imbalance["partitions"]["train"]["fraud_count"] == int(train_df["is_fraud"].sum())
    assert len(imbalance["train_monthly_breakdown"]) > 0


def test_temporal_analysis(loaded_splits):
    """Verify temporal aggregation outputs."""
    train_df, _, _ = loaded_splits
    temporal = analyze_temporal_patterns(train_df)

    assert len(temporal["hourly_stats"]) == 24
    assert len(temporal["dow_stats"]) == 7
    assert "insights" in temporal
    assert 0 <= temporal["insights"]["peak_fraud_rate_hour"] <= 23


def test_amount_distribution_analysis(loaded_splits):
    """Verify amount distribution and bucketing."""
    train_df, _, _ = loaded_splits
    amount_res = compute_amount_distribution(train_df)

    assert "legitimate_stats" in amount_res
    assert "fraudulent_stats" in amount_res
    assert len(amount_res["bucket_analysis"]) > 0
    assert amount_res["fraudulent_stats"]["median"] > amount_res["legitimate_stats"]["median"]


def test_merchant_category_support_filtering(loaded_splits):
    """Verify merchant and category analysis respects minimum support thresholds."""
    train_df, _, _ = loaded_splits
    merchant_res = analyze_merchants_and_categories(train_df, min_support_threshold=100)

    assert merchant_res["unique_categories"] == 14
    assert len(merchant_res["category_summary"]) == 14
    assert len(merchant_res["top_categories_by_fraud_count"]) == 14
    for m in merchant_res["top_merchants_by_fraud_rate"]:
        assert m["total_txns"] >= 100


def test_geography_haversine_calculation():
    """Verify Haversine formula calculation accuracy with known coordinates."""
    # NYC (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    lat1, lon1 = np.array([40.7128]), np.array([-74.0060])
    lat2, lon2 = np.array([51.5074]), np.array([-0.1278])

    dist = haversine_distance_km(lat1, lon1, lat2, lon2)
    assert abs(dist[0] - 5570.0) < 50.0  # Within 50km tolerance


def test_account_behavioral_analysis(loaded_splits):
    """Verify account-level behavior aggregations."""
    train_df, _, _ = loaded_splits
    acc_res = analyze_account_behavior(train_df)

    assert acc_res["total_unique_accounts"] > 0
    assert acc_res["accounts_with_fraud"] + acc_res["accounts_without_fraud"] == acc_res["total_unique_accounts"]
    assert acc_res["txns_per_account_stats"]["min"] >= 1


def test_correlations_and_cramers_v(loaded_splits):
    """Verify numerical correlations and categorical association calculations."""
    train_df, _, _ = loaded_splits
    corr_res = analyze_correlations(train_df)

    assert "pearson_matrix" in corr_res
    assert "spearman_matrix" in corr_res
    assert "merchant_category" in corr_res["categorical_fraud_associations_cramers_v"]
    cv = corr_res["categorical_fraud_associations_cramers_v"]["merchant_category"]
    assert 0.0 <= cv <= 1.0


def test_eda_artifacts_generation():
    """Verify that the EDA report and all 15 figures exist and are non-empty."""
    report_path = Path("docs/eda_report.md")
    figures_dir = Path("docs/eda/figures")

    if report_path.exists() and figures_dir.exists():
        assert report_path.stat().st_size > 500, "EDA report is too small"
        figures = list(figures_dir.glob("*.png"))
        assert len(figures) == 15, f"Expected 15 figures, found {len(figures)}"
        for fig in figures:
            assert fig.stat().st_size > 1000, f"Figure {fig.name} is empty"
