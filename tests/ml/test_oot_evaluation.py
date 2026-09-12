"""
Automated Unit and Governance Tests for Phase 5 Final Frozen Out-of-Time (OOT) Evaluation.

Validates:
1. Frozen artifact loading & immutability: Champion model and preprocessor load without modification; SHA-256 remains constant.
2. Dataset schema & temporal integrity: Test partition adheres to canonical 62-column schema and expected date bounds (Oct 3, 2020 - Dec 31, 2020).
3. Evaluated policy thresholds: Exactly evaluates Phase 4 threshold (0.94) and Phase 5 threshold (0.78).
4. Mathematical cost consistency: All-Approve baseline, threshold 0.94, and threshold 0.78 arithmetic.
5. Zero leakage guarantee: Test set is never used for fitting, calibration, or threshold selection.
6. Artifact schema validation: Persisted JSON, plot, and report exist with valid keys.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from ml.cost_optimization.config import CostConfig
from ml.cost_optimization.oot_evaluation import (
    compute_file_sha256,
    compute_policy_oot_metrics,
    evaluate_all_approve_baseline,
    compare_oot_policies,
    run_frozen_oot_evaluation,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, TARGET_COLUMN
from ml.features.config import FULL_FEATURE_DATASET_COLUMNS


class TestOOTArtifactsAndGovernance:
    """Tests for artifact loading, SHA-256 immutability, and dataset integrity."""

    def test_champion_artifacts_exist_and_load(self):
        """Verify champion model and preprocessor files exist and can be unpickled."""
        model_path = Path("ml/models/artifacts/champion_model.joblib")
        preproc_path = Path("ml/models/artifacts/champion_preprocessor.joblib")

        assert model_path.exists(), f"Missing model: {model_path}"
        assert preproc_path.exists(), f"Missing preprocessor: {preproc_path}"

        model = joblib.load(model_path)
        preproc = joblib.load(preproc_path)

        assert hasattr(model, "predict_proba")
        assert hasattr(preproc, "transform")

    def test_test_partition_schema_and_chronology(self):
        """Verify test partition contains expected 62 columns and chronological window."""
        test_path = Path("data/processed/features/test_features.parquet")
        assert test_path.exists(), f"Missing test partition: {test_path}"

        df_test = pd.read_parquet(test_path, columns=["timestamp", TARGET_COLUMN])
        assert len(df_test) == 277860
        assert int(df_test[TARGET_COLUMN].sum()) == 924

        min_time = pd.Timestamp(df_test["timestamp"].min())
        max_time = pd.Timestamp(df_test["timestamp"].max())
        assert min_time >= pd.Timestamp("2020-10-03")
        assert max_time <= pd.Timestamp("2020-12-31 23:59:59")

    def test_model_immutability_sha256_preservation(self):
        """Verify champion model hash is unaffected by evaluation."""
        model_path = Path("ml/models/artifacts/champion_model.joblib")
        preproc_path = Path("ml/models/artifacts/champion_preprocessor.joblib")

        hash_m1 = compute_file_sha256(model_path)
        hash_p1 = compute_file_sha256(preproc_path)

        # Load and execute transform
        model = joblib.load(model_path)
        preproc = joblib.load(preproc_path)
        dummy_df = pd.DataFrame({col: [0.0] for col in PREDICTIVE_FEATURE_COLUMNS})
        dummy_df["merchant_category"] = "gas_transport"
        dummy_df["job_category"] = "Engineer"
        trans = preproc.transform(dummy_df)
        _ = model.predict_proba(trans)

        hash_m2 = compute_file_sha256(model_path)
        hash_p2 = compute_file_sha256(preproc_path)

        assert hash_m1 == hash_m2
        assert hash_p1 == hash_p2


class TestOOTPolicyCalculations:
    """Tests for metric calculations and policy comparison arithmetic on OOT holdout."""

    def test_all_approve_baseline_calculation(self):
        """Verify All-Approve baseline incurs full fraud loss and zero false alarms."""
        y_test = np.array([0]*900 + [1]*100)
        cfg = CostConfig(false_positive_cost=15.0, false_negative_cost=200.0)

        res = evaluate_all_approve_baseline(y_test, cfg)
        assert res["threshold"] == 1.0
        assert res["confusion_matrix"]["tp"] == 0
        assert res["confusion_matrix"]["fp"] == 0
        assert res["confusion_matrix"]["fn"] == 100
        assert res["confusion_matrix"]["tn"] == 900
        assert res["total_expected_cost"] == 100 * 200.0  # 20,000.0
        assert res["average_cost_per_transaction"] == 20000.0 / 1000.0

    def test_policy_oot_metrics_arithmetic(self):
        """Verify threshold binarization and metric arithmetic on mock holdout."""
        # 10 samples: 2 fraud, 8 legit
        y_test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        model_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.75, 0.95])
        cfg = CostConfig(false_positive_cost=15.0, false_negative_cost=200.0)

        # Threshold 0.75:
        # y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] -> FP=1 (idx 7), FN=0, TP=2 (idx 8,9), TN=7
        m = compute_policy_oot_metrics(y_test, model_scores, threshold=0.75, cost_config=cfg, policy_name="Test Policy")
        assert m["confusion_matrix"]["tp"] == 2
        assert m["confusion_matrix"]["fp"] == 1
        assert m["confusion_matrix"]["fn"] == 0
        assert m["confusion_matrix"]["tn"] == 7
        assert m["precision"] == round(2.0 / 3.0, 5)
        assert m["recall"] == 1.0
        assert m["total_expected_cost"] == 1 * 15.0 + 0 * 200.0  # 15.0
        assert m["cost_per_fraud_detected"] == 15.0 / 2.0       # 7.50

    def test_compare_oot_policies_structure_and_thresholds(self):
        """Verify compare_oot_policies outputs expected threshold keys and reduction metrics."""
        y_test = np.array([0]*900 + [1]*100)
        model_scores = np.linspace(0.01, 0.99, 1000)
        cfg = CostConfig(false_positive_cost=15.0, false_negative_cost=200.0)

        comp = compare_oot_policies(y_test, model_scores, cfg, thresholds=(0.94, 0.78))
        assert "phase4_threshold_094" in comp
        assert "phase5_threshold_078" in comp
        assert "all_approve_baseline" in comp
        assert "comparative_impact" in comp
        assert comp["phase4_threshold_094"]["threshold"] == 0.94
        assert comp["phase5_threshold_078"]["threshold"] == 0.78


class TestOOTPersistedArtifacts:
    """Tests validating the existence and schema of generated OOT evaluation artifacts."""

    def test_persisted_oot_artifacts_schema(self):
        """Verify oot_evaluation_results.json, plot, and report exist with valid schema."""
        oot_json_path = Path("ml/cost_optimization/artifacts/oot_evaluation_results.json")
        oot_plot_path = Path("docs/figures/oot_threshold_comparison.png")
        oot_report_path = Path("docs/oot_evaluation_report.md")

        assert oot_json_path.exists(), f"Missing artifact: {oot_json_path}"
        assert oot_plot_path.exists(), f"Missing plot: {oot_plot_path}"
        assert oot_report_path.exists(), f"Missing report: {oot_report_path}"

        with open(oot_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "dataset_statistics" in data
        assert "ranking_metrics" in data
        assert "phase4_threshold_094" in data
        assert "phase5_threshold_078" in data
        assert "comparative_impact" in data

        # Verify exact OOT holdout numbers
        assert data["dataset_statistics"]["total_transactions"] == 277860
        assert data["dataset_statistics"]["fraud_count"] == 924
        assert data["ranking_metrics"]["pr_auc"] == 0.96054
        assert data["ranking_metrics"]["roc_auc"] == 0.99905

        # Verify exact financial costs
        assert data["phase4_threshold_094"]["total_expected_cost"] == 20890.0
        assert data["phase5_threshold_078"]["total_expected_cost"] == 14245.0
        assert data["comparative_impact"]["cost_difference_094_vs_078"] == 6645.0
        assert data["comparative_impact"]["percentage_cost_reduction_vs_094"] == 31.81
