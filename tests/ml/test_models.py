"""
Automated Unit Tests for Phase 4 Machine Learning Models & Evaluation Pipeline.

Validates:
1. Feature matrix extraction: exactly 55 predictive features, zero target or ID leakage.
2. Preprocessor fitting: fitted strictly on Train, graceful handling of unseen categories.
3. Metrics calculation: PR-AUC, ROC-AUC, F1, Precision, Recall, Confusion Matrix arithmetic.
4. Probability validity: all outputs are strictly bounded in [0, 1] with zero NaN/Inf.
5. All 4 model architectures: Baseline LR, Random Forest, XGBoost, LightGBM fit and predict.
6. Process isolation on Windows: prevents cross-DLL OpenMP collisions between XGBoost and LightGBM.
7. Threshold analysis: sweep monotonically navigates precision/recall trade-off.
"""

import sys
import subprocess
import pytest
import numpy as np
import pandas as pd

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    MODEL_CONFIGS,
)
from ml.models.preprocessing import (
    prepare_features_and_target,
    TreePreprocessor,
    LinearPreprocessor,
)
from ml.models.baseline import BaselineLogisticRegression
from ml.models.random_forest import RandomForestBaseline
from ml.evaluation.metrics import compute_classification_metrics
from ml.evaluation.threshold_analysis import evaluate_threshold_sweep
from ml.features.config import FULL_FEATURE_DATASET_COLUMNS


@pytest.fixture
def mock_feature_df() -> pd.DataFrame:
    """Construct a synthetic 62-column feature DataFrame with severe class imbalance."""
    rng = np.random.default_rng(42)
    n = 200
    
    data = {}
    # Canonical columns
    data["transaction_id"] = [f"TXN_{i:04d}" for i in range(n)]
    data["account_id"] = [f"ACC_{rng.integers(1, 20)}" for _ in range(n)]
    data["timestamp"] = pd.date_range("2020-01-01", periods=n, freq="h")
    data["unix_time"] = (data["timestamp"].astype(np.int64) // 10**9).values
    data["amount"] = rng.exponential(scale=50.0, size=n)
    data["currency"] = "USD"
    data["merchant_id"] = [f"MERCH_{rng.integers(1, 10)}" for _ in range(n)]
    data["merchant_category"] = rng.choice(["grocery_pos", "shopping_net", "misc_net", "gas_transport"], size=n)
    data["cardholder_lat"] = 40.7128 + rng.normal(0, 0.05, size=n)
    data["cardholder_long"] = -74.0060 + rng.normal(0, 0.05, size=n)
    data["merchant_lat"] = 40.7128 + rng.normal(0, 0.05, size=n)
    data["merchant_long"] = -74.0060 + rng.normal(0, 0.05, size=n)
    data["city_pop"] = rng.integers(50000, 1000000, size=n)
    data["job_category"] = rng.choice(["Engineer", "Teacher", "Doctor", "Artist"], size=n)
    # Severe imbalance: ~5% fraud in mock fixture
    data["is_fraud"] = rng.choice([0, 1], p=[0.95, 0.05], size=n)
    
    # 47 Engineered features
    from ml.features.config import ENGINEERED_FEATURE_COLUMNS
    for col in ENGINEERED_FEATURE_COLUMNS:
        if "count" in col or "first" in col or "night" in col or "weekend" in col or "impossible" in col:
            data[col] = rng.integers(0, 5, size=n)
        else:
            data[col] = rng.uniform(0.0, 100.0, size=n)
            
    df = pd.DataFrame(data)
    assert list(df.columns) == FULL_FEATURE_DATASET_COLUMNS
    return df


def test_feature_separation_and_leakage_exclusion(mock_feature_df: pd.DataFrame):
    """Verify feature separation yields 55 columns with zero target or ID leakage."""
    X, y = prepare_features_and_target(mock_feature_df)
    
    assert len(X.columns) == 55
    assert list(X.columns) == PREDICTIVE_FEATURE_COLUMNS
    assert TARGET_COLUMN not in X.columns
    assert "transaction_id" not in X.columns
    for col in EXCLUDED_COLUMNS:
        assert col not in X.columns
        
    assert len(y) == len(mock_feature_df)
    assert set(y.unique()).issubset({0, 1})


def test_tree_preprocessor_fitting(mock_feature_df: pd.DataFrame):
    """Verify TreePreprocessor fits on train, encodes categoricals, and handles unseen categories."""
    X, _ = prepare_features_and_target(mock_feature_df)
    
    train_X = X.iloc[:150]
    test_X = X.iloc[150:].copy()
    test_X.loc[test_X.index[0], "merchant_category"] = "unseen_category_xyz"
    
    preproc = TreePreprocessor()
    X_train_trans = preproc.fit_transform(train_X)
    X_test_trans = preproc.transform(test_X)
    
    assert X_train_trans.shape == (150, 55)
    assert X_test_trans.shape == (50, 55)
    assert np.isfinite(X_train_trans).all()
    assert np.isfinite(X_test_trans).all()


def test_linear_preprocessor_fitting(mock_feature_df: pd.DataFrame):
    """Verify LinearPreprocessor scales numeric features and one-hot encodes categoricals."""
    X, _ = prepare_features_and_target(mock_feature_df)
    
    preproc = LinearPreprocessor()
    X_trans = preproc.fit_transform(X)
    
    assert X_trans.shape[0] == len(X)
    assert X_trans.shape[1] >= 53  # numeric + one-hot categories
    assert np.isfinite(X_trans).all()


def test_metrics_calculation_correctness():
    """Verify classification metric calculations against analytical ground truth."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    
    assert metrics["pr_auc"] == 1.0  # Perfect ranking
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 4
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0
    assert metrics["confusion_matrix"]["tn"] == 4


def test_baseline_logistic_regression_fit_and_predict(mock_feature_df: pd.DataFrame):
    """Verify Logistic Regression trains and produces valid probabilities."""
    X, y = prepare_features_and_target(mock_feature_df)
    lin_preproc = LinearPreprocessor().fit(X)
    X_lin = lin_preproc.transform(X)
    
    lr = BaselineLogisticRegression(max_iter=100).fit(X_lin, y)
    p_lr = lr.predict_proba(X_lin)
    
    assert p_lr.shape == (len(X),)
    assert np.all((p_lr >= 0.0) & (p_lr <= 1.0))
    assert len(lr.get_feature_importance(PREDICTIVE_FEATURE_COLUMNS)) > 0


def test_random_forest_baseline_fit_and_predict(mock_feature_df: pd.DataFrame):
    """Verify Random Forest trains and produces valid probabilities."""
    X, y = prepare_features_and_target(mock_feature_df)
    tree_preproc = TreePreprocessor().fit(X)
    X_tree = tree_preproc.transform(X)
    
    rf = RandomForestBaseline(n_estimators=10, max_depth=4).fit(X_tree, y)
    p_rf = rf.predict_proba(X_tree)
    
    assert p_rf.shape == (len(X),)
    assert np.all((p_rf >= 0.0) & (p_rf <= 1.0))
    assert len(rf.get_feature_importance(PREDICTIVE_FEATURE_COLUMNS)) == 55


def test_xgboost_isolated_worker_execution(tmp_path, mock_feature_df: pd.DataFrame):
    """Verify XGBoost executes successfully in process isolation with valid outputs."""
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    mock_feature_df.to_parquet(train_path)
    mock_feature_df.to_parquet(val_path)
    
    from ml.models.runner import run_isolated_model_job
    res = run_isolated_model_job(
        model_name="xgboost",
        train_features_path=train_path,
        val_features_path=val_path,
        hyperparams={"n_estimators": 10, "max_depth": 3},
    )
    
    assert res["model_name"] == "xgboost"
    assert res["val_prob"].shape == (len(mock_feature_df),)
    assert np.all(np.isfinite(res["val_prob"]))
    assert np.all((res["val_prob"] >= 0.0) & (res["val_prob"] <= 1.0))
    assert len(res["feature_importances"]) == 55


def test_lightgbm_isolated_worker_execution(tmp_path, mock_feature_df: pd.DataFrame):
    """Verify LightGBM executes successfully in process isolation with zero native collisions."""
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    mock_feature_df.to_parquet(train_path)
    mock_feature_df.to_parquet(val_path)
    
    from ml.models.runner import run_isolated_model_job
    res = run_isolated_model_job(
        model_name="lightgbm",
        train_features_path=train_path,
        val_features_path=val_path,
        hyperparams={"n_estimators": 15, "num_leaves": 15, "num_threads": 1},
    )
    
    assert res["model_name"] == "lightgbm"
    assert res["val_prob"].shape == (len(mock_feature_df),)
    assert np.all(np.isfinite(res["val_prob"]))
    assert np.all((res["val_prob"] >= 0.0) & (res["val_prob"] <= 1.0))
    assert len(res["feature_importances"]) == 55


def test_all_four_models_process_isolation_pipeline(tmp_path, mock_feature_df: pd.DataFrame):
    """
    Verify all 4 models train sequentially through process-isolated execution runner
    with zero native OpenMP access violations on Windows.
    """
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    mock_feature_df.to_parquet(train_path)
    mock_feature_df.to_parquet(val_path)
    
    from ml.models.runner import run_isolated_model_job
    
    models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    for m in models:
        res = run_isolated_model_job(
            model_name=m,
            train_features_path=train_path,
            val_features_path=val_path,
            hyperparams={"n_estimators": 5} if m != "logistic_regression" else {"max_iter": 50},
        )
        assert res["model_name"] == m
        assert res["val_prob"].shape == (len(mock_feature_df),)
        assert np.all(np.isfinite(res["val_prob"]))
        assert np.all((res["val_prob"] >= 0.0) & (res["val_prob"] <= 1.0))
        assert len(res["feature_importances"]) > 0


def test_threshold_analysis_sweep():
    """Verify threshold sweep identifies optimal F1 threshold on validation probabilities."""
    y_val = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.6, 0.7, 0.8, 0.9])
    
    sweep_results = evaluate_threshold_sweep(y_val, y_prob, step=0.05)
    
    assert "best_f1_threshold" in sweep_results
    assert "best_f1_value" in sweep_results
    assert sweep_results["best_f1_value"] == 1.0
    assert 0.3 < sweep_results["best_f1_threshold"] <= 0.6
    assert len(sweep_results["sweep_table"]) > 0
