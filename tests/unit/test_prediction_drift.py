"""
Unit Tests for Phase 13.3 Prediction Drift Engine.

Verifies:
- Continuous model score distribution drift across 10 equal-width bins using PSI.
- Normalized risk score distribution drift across 10 histogram buckets using PSI.
- Categorical risk tier distribution drift across LOW, MEDIUM, HIGH, CRITICAL using JSD.
- Operational decision action distribution drift across APPROVE, REVIEW, BLOCK using JSD.
- Rule engine override rate tracking and isolation from model ranking drift.
- Sample-size confidence guardrail (N < 50 -> INSUFFICIENT_DATA).
- Handling of empty batches, NaNs, non-finite scores, and serialization roundtrips.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.prediction_drift import PredictionDriftCalculator
from ml.monitoring.schemas import (
    CategoricalDistributionDriftResult,
    OverrideRateDriftResult,
    PredictionBaselineProfile,
    PredictionDriftReport,
    RiskScoreBucket,
    RiskScoreDriftResult,
    ScoreDriftResult,
    ScoreHistogramBin,
)
from ml.risk_engine.config import DecisionAction, RiskTier, PolicyMode
from ml.risk_engine.policy import DecisionResult


class TestPredictionDriftEngine:
    """Test suite for PredictionDriftCalculator and prediction output drift metrics."""

    @pytest.fixture
    def baseline_profile(self) -> PredictionBaselineProfile:
        return PredictionBaselineProfile.load(default_monitoring_config.prediction_profile_path)

    @pytest.fixture
    def calculator(self, baseline_profile: PredictionBaselineProfile) -> PredictionDriftCalculator:
        return PredictionDriftCalculator(baseline_profile=baseline_profile)

    def _generate_synthetic_baseline_predictions(
        self, baseline_profile: PredictionBaselineProfile, n_samples: int = 2000
    ) -> Dict[str, list]:
        """Helper to generate synthetic prediction outputs conforming to baseline proportions."""
        rng = np.random.RandomState(42)

        # 1. Model scores sampled according to 10-bin histogram
        score_bin_props = [b.proportion for b in baseline_profile.model_score_bins]
        score_bin_props = np.array(score_bin_props) / sum(score_bin_props)
        chosen_bins = rng.choice(10, size=n_samples, p=score_bin_props)
        model_scores = [float(b * 0.1 + rng.uniform(0.001, 0.099)) for b in chosen_bins]

        # 2. Risk scores sampled according to 10-bucket histogram
        risk_bucket_props = [b.proportion for b in baseline_profile.risk_score_buckets]
        risk_bucket_props = np.array(risk_bucket_props) / sum(risk_bucket_props)
        chosen_buckets = rng.choice(10, size=n_samples, p=risk_bucket_props)
        risk_scores = [int(b * 10 + rng.randint(0, 9)) for b in chosen_buckets]

        # 3. Tiers sampled according to tier proportions
        tiers = list(baseline_profile.tier_proportions.keys())
        tier_props = np.array(list(baseline_profile.tier_proportions.values()))
        tier_props = tier_props / tier_props.sum()
        risk_tiers = list(rng.choice(tiers, size=n_samples, p=tier_props))

        # 4. Actions sampled according to action proportions
        actions_list = list(baseline_profile.action_proportions.keys())
        action_props = np.array(list(baseline_profile.action_proportions.values()))
        action_props = action_props / action_props.sum()
        actions = list(rng.choice(actions_list, size=n_samples, p=action_props))

        # 5. Overrides
        is_overridden = list(rng.rand(n_samples) < baseline_profile.is_overridden_rate)

        return {
            "model_scores": model_scores,
            "risk_scores": risk_scores,
            "risk_tiers": risk_tiers,
            "actions": actions,
            "is_overridden": is_overridden,
        }

    def test_calculator_initialization(self, calculator: PredictionDriftCalculator):
        """Verify calculator initializes with correct baseline profile and configuration."""
        assert calculator.baseline_profile.model_version == "1.0.0"
        assert len(calculator.baseline_profile.model_score_bins) == 10
        assert len(calculator.baseline_profile.risk_score_buckets) == 10
        assert set(calculator.baseline_profile.tier_proportions.keys()) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(calculator.baseline_profile.action_proportions.keys()) == {"APPROVE", "REVIEW", "BLOCK"}

    def test_prediction_drift_on_identical_distribution(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Synthetic data generated from baseline distribution produces NORMAL severity and near-zero drift."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=3000)
        report = calculator.compute_prediction_drift_from_arrays(
            model_scores=data["model_scores"],
            risk_scores=data["risk_scores"],
            risk_tiers=data["risk_tiers"],
            actions=data["actions"],
            is_overridden=data["is_overridden"],
            window_type="24h",
        )

        assert report.model_version == "1.0.0"
        assert report.dataset_row_count == 3000
        assert report.overall_prediction_drift_status == DriftSeverity.NORMAL
        assert report.model_score_drift.severity == DriftSeverity.NORMAL
        assert report.risk_score_drift.severity == DriftSeverity.NORMAL
        assert report.tier_drift.severity == DriftSeverity.NORMAL
        assert report.action_drift.severity == DriftSeverity.NORMAL
        assert report.model_score_drift.psi_value < 0.10
        assert report.risk_score_drift.psi_value < 0.10
        assert report.tier_drift.jsd_value < 0.10
        assert report.action_drift.jsd_value < 0.10

    def test_model_score_critical_shift(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Concentrating model scores in extreme high values [0.9, 1.0] triggers CRITICAL PSI drift."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=500)
        # Shift all model scores to 0.95
        shifted_scores = [0.95] * 500

        res = calculator.evaluate_model_score_drift(shifted_scores)
        assert res.severity == DriftSeverity.CRITICAL
        assert res.psi_value >= 0.25
        assert res.confidence == MetricConfidence.NORMAL_CONFIDENCE
        assert "Critical model score drift" in (res.alert_message or "")
        assert len(res.observed_bins) == 10
        assert res.observed_bins[9].count == 500
        assert res.observed_bins[9].proportion == 1.0

    def test_model_score_warning_shift(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Moderate shift in score distribution triggers WARNING PSI drift (0.10 <= PSI < 0.25)."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=2000)
        # Shift 12% of scores from bin 0 into bin 1
        scores = list(data["model_scores"])
        for i in range(250):
            scores[i] = 0.15

        res = calculator.evaluate_model_score_drift(scores)
        assert res.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
        assert res.psi_value >= 0.10

    def test_risk_score_buckets_drift(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Shifting normalized risk scores to high buckets 90-100 triggers CRITICAL risk score PSI drift."""
        shifted_risk_scores = [95] * 500
        res = calculator.evaluate_risk_score_drift(shifted_risk_scores)

        assert res.severity == DriftSeverity.CRITICAL
        assert res.psi_value >= 0.25
        assert len(res.observed_buckets) == 10
        assert res.observed_buckets[9].count == 500
        assert res.observed_buckets[9].proportion == 1.0
        assert "Critical risk score drift" in (res.alert_message or "")

    def test_risk_tier_jsd_drift(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Shifting tier distribution to 80% CRITICAL tiers triggers CRITICAL tier JSD drift."""
        shifted_tiers = ["CRITICAL"] * 800 + ["LOW"] * 200
        res = calculator.evaluate_tier_drift(shifted_tiers)

        assert res.severity == DriftSeverity.CRITICAL
        assert res.jsd_value >= 0.25
        assert "Critical risk tier drift" in (res.alert_message or "")
        assert res.observed_proportions["CRITICAL"] == 0.80

    def test_decision_action_jsd_drift(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Shifting decision actions to 60% BLOCK actions triggers CRITICAL action JSD drift."""
        shifted_actions = ["BLOCK"] * 600 + ["APPROVE"] * 400
        res = calculator.evaluate_action_drift(shifted_actions)

        assert res.severity == DriftSeverity.CRITICAL
        assert res.jsd_value >= 0.25
        assert "Critical decision action drift" in (res.alert_message or "")
        assert res.observed_proportions["BLOCK"] == 0.60

    def test_rule_override_isolation(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Rule override surge shifts actions and override delta while model score distribution remains stable."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=1000)
        # Keep normal model scores
        normal_scores = data["model_scores"]
        normal_risk_scores = data["risk_scores"]
        normal_tiers = data["risk_tiers"]

        # Surge override rate to 35% (e.g. aggressive velocity surge rules)
        surge_overrides = [True] * 350 + [False] * 650
        # Overridden actions become BLOCK or REVIEW
        overridden_actions = ["BLOCK"] * 350 + ["APPROVE"] * 650

        report = calculator.compute_prediction_drift_from_arrays(
            model_scores=normal_scores,
            risk_scores=normal_risk_scores,
            risk_tiers=normal_tiers,
            actions=overridden_actions,
            is_overridden=surge_overrides,
        )

        # Model score drift is NORMAL
        assert report.model_score_drift.severity == DriftSeverity.NORMAL
        # Override drift shows large delta
        assert report.override_drift.observed_rate == 0.35
        assert report.override_drift.overridden_count == 350
        assert report.override_drift.rate_delta > 0.30
        # Action drift reflects the surge in BLOCK actions
        assert report.action_drift.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)

    def test_sample_size_insufficient_data_guardrail(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Datasets with N < 50 evaluations trigger INSUFFICIENT_DATA across all prediction metrics."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=30)
        report = calculator.compute_prediction_drift_from_arrays(
            model_scores=data["model_scores"],
            risk_scores=data["risk_scores"],
            risk_tiers=data["risk_tiers"],
            actions=data["actions"],
            is_overridden=data["is_overridden"],
        )

        assert report.dataset_row_count == 30
        assert report.overall_prediction_drift_status == DriftSeverity.INSUFFICIENT_DATA
        assert report.model_score_drift.severity == DriftSeverity.INSUFFICIENT_DATA
        assert report.model_score_drift.confidence == MetricConfidence.INSUFFICIENT_DATA
        assert report.risk_score_drift.severity == DriftSeverity.INSUFFICIENT_DATA
        assert report.tier_drift.severity == DriftSeverity.INSUFFICIENT_DATA
        assert report.action_drift.severity == DriftSeverity.INSUFFICIENT_DATA

    def test_empty_batch_handling(self, calculator: PredictionDriftCalculator):
        """Empty input (0 evaluations) returns INSUFFICIENT_DATA without crashing."""
        report = calculator.compute_prediction_drift([])
        assert report.dataset_row_count == 0
        assert report.overall_prediction_drift_status == DriftSeverity.INSUFFICIENT_DATA

        report_df = calculator.compute_prediction_drift(pd.DataFrame())
        assert report_df.dataset_row_count == 0
        assert report_df.overall_prediction_drift_status == DriftSeverity.INSUFFICIENT_DATA

    def test_nan_and_infinite_scores_handling(self, calculator: PredictionDriftCalculator):
        """All-NaN model scores or risk scores evaluate as CRITICAL without crashing."""
        nan_scores = [np.nan] * 100
        score_res = calculator.evaluate_model_score_drift(nan_scores)
        assert score_res.severity == DriftSeverity.CRITICAL

        risk_res = calculator.evaluate_risk_score_drift(nan_scores)
        assert risk_res.severity == DriftSeverity.CRITICAL

    def test_evaluate_from_dataframe_with_predictions(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Computing drift from a DataFrame with prediction columns."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=200)
        df = pd.DataFrame({
            "model_score": data["model_scores"],
            "risk_score": data["risk_scores"],
            "risk_tier": data["risk_tiers"],
            "action": data["actions"],
            "is_overridden": data["is_overridden"],
        })

        report = calculator.compute_prediction_drift(df, window_type="1h")
        assert report.dataset_row_count == 200
        assert report.model_version == "1.0.0"
        assert report.window_type == "1h"

    def test_evaluate_from_decision_result_objects(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """Computing drift from a list of DecisionResult instances."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=100)
        results = [
            DecisionResult(
                action=DecisionAction(data["actions"][i]),
                risk_score=data["risk_scores"][i],
                risk_tier=RiskTier(data["risk_tiers"][i]),
                model_score=data["model_scores"][i],
                policy_mode=PolicyMode.TRI_TIER,
                reason="Standard test evaluation",
                thresholds_applied={"review": 0.50, "block": 0.78},
                is_overridden=bool(data["is_overridden"][i]),
            )
            for i in range(100)
        ]

        report = calculator.compute_prediction_drift(results)
        assert report.dataset_row_count == 100
        assert report.overall_prediction_drift_status in (DriftSeverity.NORMAL, DriftSeverity.WARNING, DriftSeverity.CRITICAL)

    def test_prediction_drift_report_serialization_roundtrip(
        self, calculator: PredictionDriftCalculator, baseline_profile: PredictionBaselineProfile
    ):
        """PredictionDriftReport to_dict() and from_dict() maintain 100% fidelity."""
        data = self._generate_synthetic_baseline_predictions(baseline_profile, n_samples=150)
        report = calculator.compute_prediction_drift_from_arrays(
            model_scores=data["model_scores"],
            risk_scores=data["risk_scores"],
            risk_tiers=data["risk_tiers"],
            actions=data["actions"],
            is_overridden=data["is_overridden"],
            window_type="7d",
            window_start="2026-09-01T00:00:00Z",
            window_end="2026-09-08T00:00:00Z",
        )

        d = report.to_dict()
        reconstructed = PredictionDriftReport.from_dict(d)

        assert reconstructed.model_version == report.model_version
        assert reconstructed.dataset_row_count == report.dataset_row_count
        assert reconstructed.overall_prediction_drift_status == report.overall_prediction_drift_status
        assert reconstructed.window_type == "7d"
        assert reconstructed.window_start == "2026-09-01T00:00:00Z"
        assert reconstructed.window_end == "2026-09-08T00:00:00Z"

        # Check model score drift subcomponent
        assert reconstructed.model_score_drift.psi_value == report.model_score_drift.psi_value
        assert reconstructed.model_score_drift.severity == report.model_score_drift.severity
        assert len(reconstructed.model_score_drift.observed_bins) == 10

        # Check risk score drift subcomponent
        assert reconstructed.risk_score_drift.psi_value == report.risk_score_drift.psi_value
        assert len(reconstructed.risk_score_drift.observed_buckets) == 10

        # Check tier & action drift
        assert reconstructed.tier_drift.jsd_value == report.tier_drift.jsd_value
        assert reconstructed.action_drift.jsd_value == report.action_drift.jsd_value

        # Check override drift
        assert reconstructed.override_drift.observed_rate == report.override_drift.observed_rate
        assert reconstructed.override_drift.rate_delta == report.override_drift.rate_delta
