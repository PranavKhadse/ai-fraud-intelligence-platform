"""
Unit Tests for Phase 13.4 Ground-Truth Performance Monitoring Engine.

Verifies:
- Ground-truth label mapping (CONFIRMED_FRAUD -> 1, FALSE_POSITIVE/LEGITIMATE -> 0).
- Exclusion of SUSPICIOUS_RESOLVED and unresolved cases from binary classification metrics.
- Strict relational join: Case -> evaluation_id -> RiskEvaluation -> transaction_id.
- Pure ML model performance at primary threshold tau* = 0.78 and comparison threshold tau = 0.50.
- Operational decision metrics: BLOCK precision, intervention recall, and review queue purity.
- 3-Tier sample-size semantics:
  * N < 20: INSUFFICIENT_DATA (suppressed alerts)
  * 20 <= N < 100: LOW_SAMPLE (informational / no strong CRITICAL alerts)
  * N >= 100: NORMAL_CONFIDENCE (full degradation alerts active)
- Mathematical degradation calculations for higher-is-better and lower-is-better (FPR) metrics.
- Handling of zero-division, single-class batches, and serialization roundtrips.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
import pytest

from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.performance import ModelPerformanceCalculator
from ml.monitoring.schemas import (
    ConfusionMatrixData,
    MetricDegradationResult,
    OperationalDecisionMetrics,
    PerformanceBaselineProfile,
    PerformanceReport,
    ThresholdPerformanceMetrics,
)


class MockRiskEvaluation:
    """Mock RiskEvaluation ORM object for unit testing."""
    def __init__(self, id_val: uuid.UUID, transaction_id: uuid.UUID, model_score: float, decision_action: str = "APPROVE"):
        self.id = id_val
        self.transaction_id = transaction_id
        self.model_score = Decimal(str(model_score))
        self.decision_action = decision_action


class MockCase:
    """Mock Case ORM object for unit testing."""
    def __init__(
        self,
        case_number: str,
        transaction_id: uuid.UUID,
        evaluation_id: uuid.UUID,
        disposition: Optional[str] = None,
        evaluation: Optional[MockRiskEvaluation] = None,
    ):
        self.id = uuid.uuid4()
        self.case_number = case_number
        self.transaction_id = transaction_id
        self.evaluation_id = evaluation_id
        self.disposition = disposition
        self.evaluation = evaluation


class TestModelPerformanceEngine:
    """Test suite for ModelPerformanceCalculator."""

    @pytest.fixture
    def baseline_profile(self) -> PerformanceBaselineProfile:
        return PerformanceBaselineProfile.load(default_monitoring_config.performance_profile_path)

    @pytest.fixture
    def calculator(self, baseline_profile: PerformanceBaselineProfile) -> ModelPerformanceCalculator:
        return ModelPerformanceCalculator(baseline_profile=baseline_profile)

    def test_calculator_initialization(self, calculator: ModelPerformanceCalculator):
        """Verify calculator initializes and loads verified OOT baseline benchmarks."""
        assert calculator.baseline_profile.model_version == "1.0.0"
        assert calculator.baseline_profile.default_operating_threshold == 0.78
        assert np.isclose(calculator.baseline_profile.primary_operating_metrics.precision, 0.78187, atol=1e-4)
        assert np.isclose(calculator.baseline_profile.primary_operating_metrics.recall, 0.94264, atol=1e-4)
        assert np.isclose(calculator.baseline_profile.primary_operating_metrics.fpr, 0.00088, atol=1e-4)
        assert np.isclose(calculator.baseline_profile.operational_decision_metrics.review_queue_purity, 0.01932, atol=1e-4)

    def test_ground_truth_label_extraction_and_disposition_mapping(self, calculator: ModelPerformanceCalculator):
        """Verify disposition mapping, SUSPICIOUS_RESOLVED exclusion, and unresolved case omission."""
        txn_id = uuid.uuid4()
        eval_id = uuid.uuid4()
        mock_eval = MockRiskEvaluation(eval_id, txn_id, model_score=0.85, decision_action="BLOCK")

        cases = [
            MockCase("CASE-001", txn_id, eval_id, disposition="CONFIRMED_FRAUD", evaluation=mock_eval),
            MockCase("CASE-002", txn_id, eval_id, disposition="FALSE_POSITIVE", evaluation=mock_eval),
            MockCase("CASE-003", txn_id, eval_id, disposition="LEGITIMATE", evaluation=mock_eval),
            MockCase("CASE-004", txn_id, eval_id, disposition="SUSPICIOUS_RESOLVED", evaluation=mock_eval),
            MockCase("CASE-005", txn_id, eval_id, disposition=None, evaluation=mock_eval),  # OPEN case
        ]

        labeled_records, suspicious_count = calculator.extract_ground_truth_labels(cases)

        assert len(labeled_records) == 3
        assert suspicious_count == 1

        assert labeled_records[0]["y_true"] == 1  # CONFIRMED_FRAUD
        assert labeled_records[1]["y_true"] == 0  # FALSE_POSITIVE
        assert labeled_records[2]["y_true"] == 0  # LEGITIMATE
        assert labeled_records[0]["evaluation_id"] == str(eval_id)
        assert labeled_records[0]["model_score"] == 0.85
        assert labeled_records[0]["decision_action"] == "BLOCK"

    def test_performance_metrics_at_default_operating_threshold_0_78(
        self, calculator: ModelPerformanceCalculator
    ):
        """Verify threshold metrics computation at tau* = 0.78."""
        # 100 labeled cases
        y_true = np.array([1] * 50 + [0] * 50)
        # 40 true positives (score >= 0.78, y=1), 10 false negatives (score < 0.78, y=1)
        # 5 false positives (score >= 0.78, y=0), 45 true negatives (score < 0.78, y=0)
        scores = np.array([0.90] * 40 + [0.20] * 10 + [0.85] * 5 + [0.10] * 45)

        metrics = calculator.compute_threshold_metrics(y_true, scores, threshold=0.78)

        assert metrics.threshold == 0.78
        assert metrics.confusion_matrix.tp == 40
        assert metrics.confusion_matrix.fp == 5
        assert metrics.confusion_matrix.fn == 10
        assert metrics.confusion_matrix.tn == 45
        assert metrics.confusion_matrix.total == 100

        assert np.isclose(metrics.precision, 40.0 / 45.0)  # ~0.8889
        assert np.isclose(metrics.recall, 40.0 / 50.0)     # 0.80
        assert np.isclose(metrics.fpr, 5.0 / 50.0)        # 0.10
        assert np.isclose(metrics.accuracy, 85.0 / 100.0) # 0.85

    def test_operational_decision_metrics_and_queue_purity(self, calculator: ModelPerformanceCalculator):
        """Verify hybrid decision metrics and review queue purity."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        actions = ["BLOCK", "BLOCK", "BLOCK", "REVIEW", "REVIEW", "REVIEW", "APPROVE", "APPROVE"]

        ops = calculator.compute_operational_metrics(y_true, actions)

        assert ops.total_blocks_count == 3
        assert ops.fraud_in_block_count == 2
        assert np.isclose(ops.decision_precision_block, 2.0 / 3.0)

        assert ops.total_reviews_count == 3
        assert ops.fraud_in_review_count == 1
        assert np.isclose(ops.review_queue_purity, 1.0 / 3.0)

        # Total fraud = 4. Fraud in BLOCK + REVIEW = 2 + 1 = 3. Recall = 3/4 = 0.75
        assert np.isclose(ops.decision_recall_intervention, 3.0 / 4.0)

    def test_sample_size_semantics_tier_a_insufficient_data(self, calculator: ModelPerformanceCalculator):
        """N < 20 labeled cases triggers INSUFFICIENT_DATA and suppresses degradation alerts."""
        y_true = [1] * 5 + [0] * 5  # 10 cases (< 20)
        scores = [0.1] * 10
        actions = ["APPROVE"] * 10

        report = calculator.compute_performance_from_labels(
            y_true=y_true,
            model_scores=scores,
            actions=actions,
            total_raw_count=15,
            suspicious_count=2,
        )

        assert report.labeled_sample_count == 10
        assert report.confidence == MetricConfidence.INSUFFICIENT_DATA
        assert report.overall_performance_status == DriftSeverity.INSUFFICIENT_DATA
        assert len(report.active_degradation_alerts) == 0

    def test_sample_size_semantics_tier_b_low_sample(self, calculator: ModelPerformanceCalculator):
        """20 <= N < 100 labeled cases triggers LOW_SAMPLE; severe degradation is capped to WARNING (no strong CRITICAL alert)."""
        # 50 cases (20 <= N < 100)
        # Severe model failure: all scores 0.05, 0 TPs -> Precision = 0.0, Recall = 0.0 (baseline precision 0.78)
        y_true = [1] * 25 + [0] * 25
        scores = [0.05] * 50
        actions = ["APPROVE"] * 50

        report = calculator.compute_performance_from_labels(
            y_true=y_true,
            model_scores=scores,
            actions=actions,
            total_raw_count=50,
        )

        assert report.labeled_sample_count == 50
        assert report.confidence == MetricConfidence.LOW_SAMPLE
        # Under LOW_SAMPLE, degradation triggers WARNING, not CRITICAL
        assert report.overall_performance_status == DriftSeverity.WARNING
        for alert in report.active_degradation_alerts:
            assert alert.severity == DriftSeverity.WARNING
            assert alert.confidence == MetricConfidence.LOW_SAMPLE

    def test_sample_size_semantics_tier_c_normal_confidence(self, calculator: ModelPerformanceCalculator):
        """N >= 100 labeled cases triggers NORMAL_CONFIDENCE; severe degradation triggers CRITICAL alert."""
        # 150 cases (>= 100)
        y_true = [1] * 75 + [0] * 75
        scores = [0.05] * 150  # Complete model miss
        actions = ["APPROVE"] * 150

        report = calculator.compute_performance_from_labels(
            y_true=y_true,
            model_scores=scores,
            actions=actions,
            total_raw_count=150,
        )

        assert report.labeled_sample_count == 150
        assert report.confidence == MetricConfidence.NORMAL_CONFIDENCE
        assert report.overall_performance_status == DriftSeverity.CRITICAL

        crit_alerts = [a for a in report.active_degradation_alerts if a.severity == DriftSeverity.CRITICAL]
        assert len(crit_alerts) > 0
        assert any(a.metric_name == "model_recall" for a in crit_alerts)

    def test_mathematical_degradation_higher_is_better(self, calculator: ModelPerformanceCalculator):
        """Higher-is-better metrics degrade when observed drops significantly below baseline."""
        base_prec = 0.80

        # Normal: drop <= 10% (observed = 0.75 -> delta_rel = 0.05 / 0.80 = 6.25%)
        res_norm = calculator.evaluate_degradation(0.75, base_prec, "precision", is_higher_better=True)
        assert res_norm.severity == DriftSeverity.NORMAL

        # Warning: 10% < drop <= 25% (observed = 0.65 -> delta_rel = 0.15 / 0.80 = 18.75%)
        res_warn = calculator.evaluate_degradation(0.65, base_prec, "precision", is_higher_better=True)
        assert res_warn.severity == DriftSeverity.WARNING
        assert "Performance degradation warning" in (res_warn.alert_message or "")

        # Critical: drop > 25% (observed = 0.50 -> delta_rel = 0.30 / 0.80 = 37.5%)
        res_crit = calculator.evaluate_degradation(0.50, base_prec, "precision", is_higher_better=True)
        assert res_crit.severity == DriftSeverity.CRITICAL
        assert "Critical performance degradation" in (res_crit.alert_message or "")

    def test_mathematical_degradation_lower_is_better_fpr(self, calculator: ModelPerformanceCalculator):
        """FPR degrades when observed increases significantly above baseline."""
        base_fpr = 0.0010

        # Normal: FPR = 0.0012 -> delta_rel = 20% (< 50%) and FPR <= 0.0050
        res_norm = calculator.evaluate_degradation(0.0012, base_fpr, "fpr", is_higher_better=False)
        assert res_norm.severity == DriftSeverity.NORMAL

        # Warning: FPR = 0.0020 (100% relative increase, between 50% and 150%)
        res_warn = calculator.evaluate_degradation(0.0020, base_fpr, "fpr", is_higher_better=False)
        assert res_warn.severity == DriftSeverity.WARNING

        # Critical: FPR = 0.0200 (> 150% and > 0.0150)
        res_crit = calculator.evaluate_degradation(0.0200, base_fpr, "fpr", is_higher_better=False)
        assert res_crit.severity == DriftSeverity.CRITICAL

    def test_pr_auc_and_roc_auc_computation(self, calculator: ModelPerformanceCalculator):
        """PR-AUC and ROC-AUC are properly computed on balanced batches."""
        y_true = [1] * 50 + [0] * 50
        # High scores for fraud, low for legit
        scores = [0.9] * 50 + [0.1] * 50
        actions = ["BLOCK"] * 50 + ["APPROVE"] * 50

        report = calculator.compute_performance_from_labels(
            y_true=y_true,
            model_scores=scores,
            actions=actions,
            total_raw_count=100,
        )

        assert report.roc_auc is not None
        assert report.pr_auc is not None
        assert np.isclose(report.roc_auc, 1.0)
        assert np.isclose(report.pr_auc, 1.0)

    def test_zero_division_guardrails(self, calculator: ModelPerformanceCalculator):
        """Zero predictions in a bucket/metric do not produce divide-by-zero errors or NaNs."""
        # 100 cases, all y=0, all scores 0.1, all actions APPROVE
        y_true = [0] * 100
        scores = [0.1] * 100
        actions = ["APPROVE"] * 100

        report = calculator.compute_performance_from_labels(
            y_true=y_true,
            model_scores=scores,
            actions=actions,
            total_raw_count=100,
        )

        assert report.primary_metrics.precision == 0.0
        assert report.primary_metrics.recall == 0.0
        assert report.primary_metrics.f1 == 0.0
        assert report.operational_metrics.decision_precision_block == 0.0
        assert report.operational_metrics.review_queue_purity == 0.0

    def test_evaluate_from_case_orm_mock_objects(self, calculator: ModelPerformanceCalculator):
        """Evaluating from mock Case ORM objects with linked RiskEvaluations."""
        cases = []
        for i in range(120):
            txn_id = uuid.uuid4()
            eval_id = uuid.uuid4()
            if i < 40:
                mock_eval = MockRiskEvaluation(eval_id, txn_id, model_score=0.92, decision_action="BLOCK")
                cases.append(MockCase(f"CASE-{i}", txn_id, eval_id, disposition="CONFIRMED_FRAUD", evaluation=mock_eval))
            else:
                mock_eval = MockRiskEvaluation(eval_id, txn_id, model_score=0.10, decision_action="APPROVE")
                cases.append(MockCase(f"CASE-{i}", txn_id, eval_id, disposition="LEGITIMATE", evaluation=mock_eval))

        report = calculator.compute_performance_report(cases, window_type="30d")
        assert report.dataset_row_count == 120
        assert report.labeled_sample_count == 120
        assert report.fraud_cases_count == 40
        assert report.legitimate_cases_count == 80
        assert report.confidence == MetricConfidence.NORMAL_CONFIDENCE
        assert report.primary_metrics.confusion_matrix.tp == 40
        assert report.primary_metrics.precision == 1.0

    def test_performance_report_serialization_roundtrip(self, calculator: ModelPerformanceCalculator):
        """PerformanceReport to_dict() and from_dict() maintain 100% data fidelity."""
        y_true = [1] * 60 + [0] * 60
        scores = [0.85] * 60 + [0.15] * 60
        actions = ["BLOCK"] * 60 + ["APPROVE"] * 60

        report = calculator.compute_performance_from_labels(
            y_true=y_true,
            model_scores=scores,
            actions=actions,
            total_raw_count=130,
            suspicious_count=10,
            window_type="24h",
            window_start="2026-09-01T00:00:00Z",
            window_end="2026-09-02T00:00:00Z",
        )

        d = report.to_dict()
        reconstructed = PerformanceReport.from_dict(d)

        assert reconstructed.model_version == report.model_version
        assert reconstructed.dataset_row_count == 130
        assert reconstructed.labeled_sample_count == 120
        assert reconstructed.suspicious_resolved_count == 10
        assert reconstructed.overall_performance_status == report.overall_performance_status
        assert reconstructed.confidence == report.confidence
        assert reconstructed.operating_threshold == 0.78
        assert reconstructed.window_type == "24h"
        assert reconstructed.window_start == "2026-09-01T00:00:00Z"
        assert reconstructed.window_end == "2026-09-02T00:00:00Z"

        # Check primary metrics equality
        assert reconstructed.primary_metrics.precision == report.primary_metrics.precision
        assert reconstructed.primary_metrics.confusion_matrix.tp == report.primary_metrics.confusion_matrix.tp

        # Check operational metrics equality
        assert reconstructed.operational_metrics.decision_precision_block == report.operational_metrics.decision_precision_block

        # Check degradation results equality
        assert len(reconstructed.degradation_results) == len(report.degradation_results)
        for k in report.degradation_results:
            assert reconstructed.degradation_results[k].relative_delta == report.degradation_results[k].relative_delta
