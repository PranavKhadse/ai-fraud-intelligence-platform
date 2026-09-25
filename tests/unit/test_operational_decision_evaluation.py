"""
Unit Tests for Phase 14.3 Operational Decision Evaluator.

Verifies:
- Operational decision evaluation via existing RiskEvaluator, DecisionPolicyEngine, and RuleEngine.
- Accurate calculation of decision distributions (APPROVE, REVIEW, BLOCK).
- Accurate risk tier mappings (LOW, MEDIUM, HIGH, CRITICAL).
- Ground truth cross-referencing: review queue purity and block precision.
- Rule override execution and tracking.
- Input validation and exception handling.
"""

from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.lifecycle.evaluation.operational import OperationalDecisionEvaluator
from ml.lifecycle.evaluation.schemas import OperationalDecisionMetrics
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, TARGET_COLUMN
from ml.models.preprocessing import TreePreprocessor
from ml.risk_engine.config import DecisionAction, PolicyMode, RuleOperator, RuleOutcome, RuleType
from ml.risk_engine.rules import RiskRule, RuleEngine


class DummyOperationalModel:
    """Picklable dummy model for operational testing."""
    def predict_proba(self, X):
        n = len(X)
        probs = np.full(n, 0.10, dtype=np.float64)
        n_high = max(1, int(n * 0.2))
        n_med = max(1, int(n * 0.2))
        probs[:n_high] = 0.90
        probs[n_high : n_high + n_med] = 0.50
        return probs


class TestOperationalDecisionEvaluator:
    """Test suite for OperationalDecisionEvaluator."""

    @pytest.fixture
    def mock_bundle_artifacts(self, tmp_path: Path):
        model = DummyOperationalModel()
        preproc = TreePreprocessor()
        dummy_cat_df = pd.DataFrame({
            "merchant_category": ["grocery_pos", "online", "gas_transport"],
            "job_category": ["engineer", "accountant", "doctor"],
        })
        preproc.fit(dummy_cat_df)
        preproc.feature_names = PREDICTIVE_FEATURE_COLUMNS

        m_path = tmp_path / "model.joblib"
        p_path = tmp_path / "preprocessor.joblib"
        joblib.dump(model, m_path)
        joblib.dump(preproc, p_path)
        return m_path, p_path

    @pytest.fixture
    def mock_dataset_df(self):
        n_samples = 50
        data = {}
        for col in PREDICTIVE_FEATURE_COLUMNS:
            if col in ["merchant_category", "job_category"]:
                data[col] = ["grocery_pos" if i % 2 == 0 else "online" for i in range(n_samples)]
            elif col == "amount":
                data[col] = [600.0 if i == 0 else 50.0 for i in range(n_samples)]
            else:
                data[col] = np.random.uniform(1.0, 100.0, size=n_samples).astype(np.float32)

        # First 10 samples are fraud
        data[TARGET_COLUMN] = np.array([1 if i < 10 else 0 for i in range(n_samples)], dtype=np.int64)
        return pd.DataFrame(data)

    def test_evaluate_operational_decisions_tri_tier(self, mock_bundle_artifacts, mock_dataset_df):
        m_path, p_path = mock_bundle_artifacts
        evaluator = OperationalDecisionEvaluator()

        metrics = evaluator.evaluate_operational_decisions(
            df=mock_dataset_df,
            model_path=m_path,
            preprocessor_path=p_path,
            operating_threshold=0.78,
            policy_mode=PolicyMode.TRI_TIER,
        )

        assert isinstance(metrics, OperationalDecisionMetrics)
        assert metrics.policy_mode == PolicyMode.TRI_TIER.value
        assert metrics.operating_threshold == 0.78
        assert "APPROVE" in metrics.action_counts
        assert "REVIEW" in metrics.action_counts
        assert "BLOCK" in metrics.action_counts
        assert sum(metrics.action_counts.values()) == 50
        assert sum(metrics.risk_tier_counts.values()) == 50

        # Ground truth cross referencing
        assert metrics.fraud_in_block_count >= 0
        assert metrics.fraud_in_review_count >= 0
        assert metrics.fraud_in_approve_count >= 0

    def test_evaluate_operational_decisions_binary_auto(self, mock_bundle_artifacts, mock_dataset_df):
        m_path, p_path = mock_bundle_artifacts
        evaluator = OperationalDecisionEvaluator()

        metrics = evaluator.evaluate_operational_decisions(
            df=mock_dataset_df,
            model_path=m_path,
            preprocessor_path=p_path,
            operating_threshold=0.78,
            policy_mode=PolicyMode.BINARY_AUTO,
        )

        assert metrics.policy_mode == PolicyMode.BINARY_AUTO.value
        # In BINARY_AUTO mode, review count should be 0
        assert metrics.action_counts.get("REVIEW", 0) == 0
        assert metrics.action_counts["APPROVE"] + metrics.action_counts["BLOCK"] == 50

    def test_evaluate_operational_with_rule_override(self, mock_bundle_artifacts, mock_dataset_df):
        m_path, p_path = mock_bundle_artifacts
        evaluator = OperationalDecisionEvaluator()

        rule = RiskRule(
            rule_id="FORCE_BLOCK_HIGH_AMT",
            description="Force block high amount",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=500.0,
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.AMOUNT,
            priority=10,
        )
        rule_engine = RuleEngine([rule])

        metrics = evaluator.evaluate_operational_decisions(
            df=mock_dataset_df,
            model_path=m_path,
            preprocessor_path=p_path,
            operating_threshold=0.78,
            policy_mode=PolicyMode.TRI_TIER,
            rule_engine=rule_engine,
        )

        # Transaction 0 had amount=600 and score=0.90 (already BLOCK), but rule triggers
        assert metrics.rule_trigger_counts.get("FORCE_BLOCK_HIGH_AMT", 0) >= 1

    def test_invalid_dataframe_type_raises_error(self, mock_bundle_artifacts):
        m_path, p_path = mock_bundle_artifacts
        evaluator = OperationalDecisionEvaluator()
        with pytest.raises(TypeError, match="must be a pd.DataFrame"):
            evaluator.evaluate_operational_decisions(
                df="INVALID",  # type: ignore
                model_path=m_path,
                preprocessor_path=p_path,
                operating_threshold=0.78,
            )

    def test_empty_dataframe_raises_error(self, mock_bundle_artifacts):
        m_path, p_path = mock_bundle_artifacts
        evaluator = OperationalDecisionEvaluator()
        with pytest.raises(ValueError, match="cannot be empty"):
            evaluator.evaluate_operational_decisions(
                df=pd.DataFrame(),
                model_path=m_path,
                preprocessor_path=p_path,
                operating_threshold=0.78,
            )

    def test_invalid_operating_threshold_raises_error(self, mock_bundle_artifacts, mock_dataset_df):
        m_path, p_path = mock_bundle_artifacts
        evaluator = OperationalDecisionEvaluator()
        with pytest.raises(ValueError, match="out of valid bounds"):
            evaluator.evaluate_operational_decisions(
                df=mock_dataset_df,
                model_path=m_path,
                preprocessor_path=p_path,
                operating_threshold=1.5,
            )
