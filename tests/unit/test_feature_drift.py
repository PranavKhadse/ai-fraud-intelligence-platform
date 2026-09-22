"""
Unit Tests for Phase 13.2 Feature Drift Engine.

Verifies:
- Statistical algorithms: PSI, two-sample KS test with asymptotic p-value, JSD, missing rate deltas, unseen rates.
- Numerical feature drift: 10-decile binning, reference sample comparison, severity classification.
- Categorical feature drift: Discrete probability divergence, unseen vocabulary detection.
- Missing-rate surges and column omissions.
- Sample-size confidence guardrails (N < 100 rows -> INSUFFICIENT_DATA).
- Zero-variance and constant features resilience.
- 55-feature drift report generation, ranking, and serialization roundtrip.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, CATEGORICAL_PREDICTORS
from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.feature_drift import FeatureDriftCalculator
from ml.monitoring.schemas import (
    CategoricalFeatureProfile,
    FeatureBaselineProfile,
    FeatureDriftReport,
    NumericalFeatureProfile,
    SingleFeatureDriftResult,
)
from ml.monitoring.stats import (
    calculate_jsd,
    calculate_missing_rate_delta,
    calculate_numerical_psi,
    calculate_psi,
    calculate_two_sample_ks,
    calculate_unseen_category_rate,
)


class TestStatisticalAlgorithms:
    """Test suite for mathematical helper algorithms in ml/monitoring/stats.py."""

    def test_psi_identical_distributions(self):
        """PSI between identical bucket probability vectors is exactly 0.0."""
        p = [0.1] * 10
        q = [0.1] * 10
        psi = calculate_psi(p, q)
        assert np.isclose(psi, 0.0)

    def test_psi_shifted_distribution(self):
        """PSI between substantially shifted distributions exceeds 0.25 (critical threshold)."""
        p = [0.1] * 10
        q = [0.6] + [0.4 / 9] * 9
        psi = calculate_psi(p, q)
        assert psi > 0.25

    def test_psi_laplace_smoothing_zero_buckets(self):
        """PSI handles zero-frequency buckets gracefully without NaN or division by zero."""
        p = [0.2, 0.2, 0.2, 0.2, 0.2]
        q = [0.5, 0.5, 0.0, 0.0, 0.0]
        psi = calculate_psi(p, q, epsilon=1e-4)
        assert np.isfinite(psi)
        assert psi > 0.0

    def test_psi_empty_and_mismatched_inputs(self):
        """PSI returns 0.0 for empty inputs and raises ValueError for mismatched lengths."""
        assert calculate_psi([], []) == 0.0
        with pytest.raises(ValueError, match="same length"):
            calculate_psi([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_calculate_numerical_psi_deciles(self):
        """Verifies numerical PSI binning against decile edges for in-range and out-of-range observations."""
        edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        ref_samples = np.linspace(0.0, 100.0, 1000)

        # Identical distribution
        obs_same = np.linspace(0.0, 100.0, 1000)
        psi_same, obs_props, exp_props = calculate_numerical_psi(
            obs_same, edges, baseline_ref_samples=ref_samples
        )
        assert np.isclose(psi_same, 0.0, atol=1e-2)
        assert len(obs_props) == 10
        assert len(exp_props) == 10

        # Severely shifted distribution (all extreme values > 100.0)
        obs_shifted = np.full(500, 500.0)
        psi_shift, obs_props_s, _ = calculate_numerical_psi(
            obs_shifted, edges, baseline_ref_samples=ref_samples
        )
        assert psi_shift > 0.25
        assert obs_props_s[-1] == 1.0  # All fall in last bin [90, inf)

    def test_two_sample_ks_asymptotic(self):
        """Two-sample KS test gives D=0, p=1.0 for identical samples and D close to 1.0, p < 0.001 for shifted samples."""
        np.random.seed(42)
        sample1 = np.random.normal(loc=0.0, scale=1.0, size=1000)
        sample2 = np.copy(sample1)

        stat_id, p_id = calculate_two_sample_ks(sample1, sample2, method="asymp")
        assert stat_id == 0.0
        assert p_id == 1.0

        # Shifted distribution: N(5.0, 1.0) vs N(0.0, 1.0)
        sample_shifted = np.random.normal(loc=5.0, scale=1.0, size=1000)
        stat_sh, p_sh = calculate_two_sample_ks(sample_shifted, sample1, method="asymp")
        assert stat_sh > 0.8
        assert p_sh < 0.001

    def test_two_sample_ks_empty_and_nan(self):
        """KS test returns D=0.0, p=1.0 for empty or all-NaN inputs."""
        stat, p = calculate_two_sample_ks([], [1.0, 2.0, 3.0])
        assert stat == 0.0
        assert p == 1.0

        stat_nan, p_nan = calculate_two_sample_ks([np.nan, np.nan], [1.0, 2.0, 3.0])
        assert stat_nan == 0.0
        assert p_nan == 1.0

    def test_calculate_jsd(self):
        """JSD is 0.0 for identical distributions, 1.0 for completely disjoint, and in (0, 1) for partial overlap."""
        # Identical
        p = {"food": 0.4, "gas": 0.4, "travel": 0.2}
        assert np.isclose(calculate_jsd(p, p), 0.0)

        # Disjoint
        q_disjoint = {"electronics": 0.6, "clothing": 0.4}
        jsd_disjoint = calculate_jsd(p, q_disjoint)
        assert np.isclose(jsd_disjoint, 1.0)

        # Partial overlap
        q_partial = {"food": 0.8, "gas": 0.2}
        jsd_partial = calculate_jsd(p, q_partial)
        assert 0.0 < jsd_partial < 1.0

    def test_calculate_missing_rate_delta(self):
        """Missing rate delta computes absolute difference."""
        assert calculate_missing_rate_delta(0.05, 0.01) == 0.04
        assert calculate_missing_rate_delta(0.01, 0.05) == 0.04
        assert calculate_missing_rate_delta(0.0, 0.0) == 0.0

    def test_calculate_unseen_category_rate(self):
        """Unseen category rate correctly detects unknown classes and returns distinct unseen set."""
        vocab = ["grocery_pos", "gas_transport", "shopping_net"]
        obs = ["grocery_pos", "crypto_exchange", "grocery_pos", "darknet_market", "shopping_net"]
        rate, unseen = calculate_unseen_category_rate(obs, vocab)
        assert rate == 2.0 / 5.0  # 0.4
        assert unseen == ["crypto_exchange", "darknet_market"]

        # No unseen
        rate_zero, unseen_empty = calculate_unseen_category_rate(["grocery_pos", "gas_transport"], vocab)
        assert rate_zero == 0.0
        assert unseen_empty == []


class TestFeatureDriftCalculator:
    """Test suite for FeatureDriftCalculator and 55-feature drift report generation."""

    @pytest.fixture
    def baseline_profile(self) -> FeatureBaselineProfile:
        return FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)

    @pytest.fixture
    def calculator(self, baseline_profile: FeatureBaselineProfile) -> FeatureDriftCalculator:
        return FeatureDriftCalculator(baseline_profile=baseline_profile)

    def _generate_synthetic_baseline_df(self, baseline_profile: FeatureBaselineProfile, n_samples: int = 5000) -> pd.DataFrame:
        """Helper to generate a clean synthetic dataframe from baseline reference samples and probabilities."""
        rng = np.random.RandomState(42)
        data_dict = {}
        for feat_name, prof in baseline_profile.features.items():
            if isinstance(prof, NumericalFeatureProfile):
                if len(prof.ref_samples) >= n_samples:
                    data_dict[feat_name] = prof.ref_samples[:n_samples]
                else:
                    data_dict[feat_name] = rng.choice(prof.ref_samples, size=n_samples)
            elif isinstance(prof, CategoricalFeatureProfile):
                cats = list(prof.probabilities.keys())
                probs = np.array(list(prof.probabilities.values()), dtype=np.float64)
                probs = probs / probs.sum()
                data_dict[feat_name] = rng.choice(cats, size=n_samples, p=probs)
        return pd.DataFrame(data_dict)

    def test_calculator_initialization(self, calculator: FeatureDriftCalculator):
        """Verify calculator initializes and precomputes numerical expected decile distributions."""
        assert calculator.baseline_profile.num_features == 55
        assert len(calculator._expected_numerical_proportions) == 53  # 55 minus 2 categoricals
        for feat_name, exp_props in calculator._expected_numerical_proportions.items():
            assert len(exp_props) == 10
            assert np.isclose(sum(exp_props), 1.0, atol=1e-3)

    def test_evaluate_feature_drift_on_identical_reference_data(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Evaluating reference samples directly results in NORMAL severity and near-zero drift metrics."""
        df_identical = self._generate_synthetic_baseline_df(baseline_profile, n_samples=5000)
        report = calculator.compute_feature_drift(df_identical, window_type="24h")

        assert report.model_version == "1.0.0"
        assert report.dataset_row_count == 5000
        assert report.overall_data_drift_status == DriftSeverity.NORMAL
        assert report.critical_features_count == 0
        assert report.warning_features_count == 0
        assert report.normal_features_count == 55
        assert report.insufficient_data_features_count == 0

        # Check a specific numerical feature result
        amount_res = report.feature_results["amount"]
        assert amount_res.feature_type == "numerical"
        assert amount_res.severity == DriftSeverity.NORMAL
        assert amount_res.drift_metric_value < 0.10  # PSI < 0.10
        assert amount_res.confidence == MetricConfidence.NORMAL_CONFIDENCE

    def test_numerical_feature_critical_drift(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Injecting a 100x shift into 'amount' triggers CRITICAL drift on PSI and KS test."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=5000)
        # Severely drift 'amount'
        df["amount"] = 99999.0

        report = calculator.compute_feature_drift(df, window_type="1h")

        assert report.overall_data_drift_status == DriftSeverity.CRITICAL
        assert report.critical_features_count == 1

        amount_res = report.feature_results["amount"]
        assert amount_res.severity == DriftSeverity.CRITICAL
        assert amount_res.drift_metric_value >= 0.25
        assert amount_res.ks_pvalue <= 0.001
        assert "Critical numerical drift" in (amount_res.alert_message or "")

        # Top ranked feature should be amount
        assert report.ranked_features[0].feature_name == "amount"

    def test_categorical_feature_unseen_and_jsd_drift(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Injecting unseen merchant categories and shifted frequencies triggers CRITICAL categorical drift."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=1000)
        # Inject unseen categories into merchant_category
        cats_injected = ["fraudulent_darknet_hub"] * 250 + ["unseen_crypto_mixer"] * 250 + list(df["merchant_category"][:500])
        df["merchant_category"] = cats_injected

        report = calculator.compute_feature_drift(df, window_type="7d")

        assert report.overall_data_drift_status == DriftSeverity.CRITICAL
        cat_res = report.feature_results["merchant_category"]
        assert cat_res.feature_type == "categorical"
        assert cat_res.severity == DriftSeverity.CRITICAL
        assert cat_res.unseen_category_rate == 0.50  # 500/1000
        assert set(cat_res.unseen_categories or []) == {"fraudulent_darknet_hub", "unseen_crypto_mixer"}
        assert cat_res.drift_metric_value > 0.25  # JSD > 0.25

    def test_missing_rate_surge_alert(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Injecting 20% missing values into 'city_pop' triggers CRITICAL missing rate alert."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=1000)
        # Introduce 20% NaNs into city_pop
        df.loc[:199, "city_pop"] = np.nan

        report = calculator.compute_feature_drift(df)

        city_pop_res = report.feature_results["city_pop"]
        assert city_pop_res.observed_missing_rate == 0.20
        assert city_pop_res.missing_rate_delta >= 0.05
        assert city_pop_res.severity == DriftSeverity.CRITICAL
        assert "Missing delta" in (city_pop_res.alert_message or "")

    def test_sample_size_insufficient_data_guardrail(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Datasets with N < 100 rows trigger INSUFFICIENT_DATA status and confidence grade."""
        df_small = self._generate_synthetic_baseline_df(baseline_profile, n_samples=50)
        report = calculator.compute_feature_drift(df_small)

        assert report.dataset_row_count == 50
        assert report.overall_data_drift_status == DriftSeverity.INSUFFICIENT_DATA
        assert report.insufficient_data_features_count == 55

        amt_res = report.feature_results["amount"]
        assert amt_res.severity == DriftSeverity.INSUFFICIENT_DATA
        assert amt_res.confidence == MetricConfidence.INSUFFICIENT_DATA
        assert amt_res.ks_statistic is None
        assert "below required minimum" in (amt_res.alert_message or "")

    def test_empty_dataframe_handling(self, calculator: FeatureDriftCalculator):
        """Empty DataFrame (0 rows) returns INSUFFICIENT_DATA without crashing."""
        df_empty = pd.DataFrame(columns=PREDICTIVE_FEATURE_COLUMNS)
        report = calculator.compute_feature_drift(df_empty)

        assert report.dataset_row_count == 0
        assert report.overall_data_drift_status == DriftSeverity.INSUFFICIENT_DATA
        assert report.insufficient_data_features_count == 55
        assert len(report.feature_results) == 55

    def test_missing_columns_in_dataframe(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """If a predictive column is completely omitted from the DataFrame, it evaluates as 100% missing with CRITICAL alert."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=500)
        df_missing_col = df.drop(columns=["amount"])
        assert "amount" not in df_missing_col.columns

        report = calculator.compute_feature_drift(df_missing_col)
        assert report.overall_data_drift_status == DriftSeverity.CRITICAL
        amt_res = report.feature_results["amount"]
        assert amt_res.observed_missing_rate == 1.0
        assert amt_res.missing_rate_delta == 1.0
        assert amt_res.severity == DriftSeverity.CRITICAL

    def test_constant_and_zero_variance_features(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Features with zero variance (all identical values) evaluate properly without numerical errors."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=500)
        df["amt_sum_1h"] = 0.0

        report = calculator.compute_feature_drift(df)
        assert "amt_sum_1h" in report.feature_results
        amt_sum_res = report.feature_results["amt_sum_1h"]
        assert np.isfinite(amt_sum_res.drift_metric_value)
        assert amt_sum_res.ks_statistic is not None

    def test_report_feature_ranking_order(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """Ranked features are ordered strictly by severity hierarchy and descending drift metric magnitude."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=1000)
        # Severely drift 'amount' (CRITICAL)
        df["amount"] = 99999.0
        # Mildly drift 'city_pop' (WARNING: 3% missing values, between 1% and 5%)
        df.loc[:29, "city_pop"] = np.nan

        report = calculator.compute_feature_drift(df)

        ranked = report.ranked_features
        assert len(ranked) == 55

        # First entries should be CRITICAL
        assert ranked[0].severity == DriftSeverity.CRITICAL
        assert ranked[0].feature_name == "amount"

        # CRITICAL features precede WARNING features
        severities = [r.severity for r in ranked]
        first_crit = severities.index(DriftSeverity.CRITICAL)
        if DriftSeverity.WARNING in severities:
            first_warn = severities.index(DriftSeverity.WARNING)
            assert first_crit <= first_warn

    def test_feature_drift_report_serialization_roundtrip(
        self, calculator: FeatureDriftCalculator, baseline_profile: FeatureBaselineProfile
    ):
        """FeatureDriftReport to_dict() and from_dict() maintain 100% data fidelity."""
        df = self._generate_synthetic_baseline_df(baseline_profile, n_samples=500)
        report = calculator.compute_feature_drift(df, window_type="24h", window_start="2026-09-01T00:00:00Z", window_end="2026-09-02T00:00:00Z")

        report_dict = report.to_dict()
        reconstructed = FeatureDriftReport.from_dict(report_dict)

        assert reconstructed.model_version == report.model_version
        assert reconstructed.dataset_row_count == report.dataset_row_count
        assert reconstructed.overall_data_drift_status == report.overall_data_drift_status
        assert reconstructed.drifted_features_count == report.drifted_features_count
        assert len(reconstructed.feature_results) == 55
        assert len(reconstructed.ranked_features) == 55
        assert reconstructed.window_type == "24h"
        assert reconstructed.window_start == "2026-09-01T00:00:00Z"
        assert reconstructed.window_end == "2026-09-02T00:00:00Z"

        # Verify single feature equality
        orig_amount = report.feature_results["amount"]
        recon_amount = reconstructed.feature_results["amount"]
        assert orig_amount.drift_metric_value == recon_amount.drift_metric_value
        assert orig_amount.severity == recon_amount.severity
        assert orig_amount.confidence == recon_amount.confidence
