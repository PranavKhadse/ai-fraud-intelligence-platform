"""
Feature Drift Engine for Phase 13 ML & Model Monitoring.

Evaluates data & feature drift across all 55 predictive platform features:
- Numerical features: 10-decile PSI and two-sample KS test with asymptotic p-value.
- Categorical features: Jensen-Shannon Divergence (JSD), unseen categories, and frequency shifts.
- Data health: Missing-value rate deltas, sample-size confidence guardrails, and ranked drift reporting.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, CATEGORICAL_PREDICTORS
from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
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
    calculate_two_sample_ks,
    calculate_unseen_category_rate,
)


class FeatureDriftCalculator:
    """
    Evaluator for feature-level and aggregate dataset drift against frozen baseline artifacts.
    """

    def __init__(
        self,
        baseline_profile: Optional[FeatureBaselineProfile] = None,
        config: Optional[MonitoringConfig] = None,
    ) -> None:
        self.config = config or default_monitoring_config
        if baseline_profile is not None:
            self.baseline_profile = baseline_profile
        else:
            self.baseline_profile = FeatureBaselineProfile.load(self.config.feature_profile_path)

        # Precompute baseline expected bin proportions for all numerical features from ref_samples
        self._expected_numerical_proportions: Dict[str, List[float]] = {}
        self._init_expected_numerical_proportions()

    def _init_expected_numerical_proportions(self) -> None:
        """Precompute expected decile proportions for fast subsequent drift evaluations."""
        for feat_name, profile in self.baseline_profile.features.items():
            if isinstance(profile, NumericalFeatureProfile):
                edges = np.asarray(profile.bin_edges, dtype=np.float64)
                num_bins = len(edges) - 1
                cutoffs = edges[1:-1]
                if len(profile.ref_samples) > 0 and len(cutoffs) > 0:
                    ref_arr = np.asarray(profile.ref_samples, dtype=np.float64)
                    ref_idx = np.digitize(ref_arr, cutoffs, right=False)
                    ref_counts = np.bincount(ref_idx, minlength=num_bins)
                    self._expected_numerical_proportions[feat_name] = (ref_counts / len(ref_arr)).tolist()
                else:
                    self._expected_numerical_proportions[feat_name] = [1.0 / num_bins] * num_bins

    def evaluate_numerical_feature(
        self,
        series: pd.Series,
        feature_name: str,
        profile: NumericalFeatureProfile,
    ) -> SingleFeatureDriftResult:
        """
        Evaluate drift metrics and alert status for a continuous/numerical feature.
        """
        n_total = len(series)
        missing_count = int(series.isna().sum())
        observed_missing_rate = float(missing_count / n_total) if n_total > 0 else 0.0
        missing_delta = calculate_missing_rate_delta(observed_missing_rate, profile.missing_rate)

        # Check sample size constraint
        if n_total < self.config.min_feature_samples:
            return SingleFeatureDriftResult(
                feature_name=feature_name,
                feature_type="numerical",
                drift_metric_name="PSI",
                drift_metric_value=0.0,
                ks_statistic=None,
                ks_pvalue=None,
                observed_missing_rate=observed_missing_rate,
                baseline_missing_rate=profile.missing_rate,
                missing_rate_delta=missing_delta,
                observed_sample_count=n_total,
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=MetricConfidence.INSUFFICIENT_DATA,
                baseline_bin_edges=profile.bin_edges,
                observed_proportions=None,
                expected_proportions=self._expected_numerical_proportions.get(feature_name),
                alert_message=f"Sample size {n_total} below required minimum {self.config.min_feature_samples}",
            )

        valid_series = series.dropna().to_numpy(dtype=np.float64)
        if len(valid_series) == 0:
            # Feature is 100% missing in this window
            return SingleFeatureDriftResult(
                feature_name=feature_name,
                feature_type="numerical",
                drift_metric_name="PSI",
                drift_metric_value=0.0,
                ks_statistic=None,
                ks_pvalue=None,
                observed_missing_rate=1.0,
                baseline_missing_rate=profile.missing_rate,
                missing_rate_delta=missing_delta,
                observed_sample_count=n_total,
                severity=DriftSeverity.CRITICAL,
                confidence=MetricConfidence.NORMAL_CONFIDENCE,
                baseline_bin_edges=profile.bin_edges,
                observed_proportions=None,
                expected_proportions=self._expected_numerical_proportions.get(feature_name),
                alert_message="All observed values are missing/NaN in the observation window.",
            )

        expected_props = self._expected_numerical_proportions.get(feature_name)
        psi_val, obs_props, exp_props = calculate_numerical_psi(
            observed_values=valid_series,
            baseline_bin_edges=profile.bin_edges,
            baseline_expected_proportions=expected_props,
            epsilon=self.config.laplace_epsilon,
        )

        ks_stat, ks_pval = calculate_two_sample_ks(
            observed_values=valid_series,
            reference_samples=profile.ref_samples,
            method="asymp",
        )

        # Evaluate configurable severity classification
        is_critical = (
            psi_val >= self.config.feature_psi_critical
            or ks_pval <= self.config.feature_ks_pvalue_critical
            or missing_delta > self.config.missing_rate_delta_critical
        )
        is_warning = (
            psi_val >= self.config.feature_psi_warning
            or ks_pval <= self.config.feature_ks_pvalue_warning
            or missing_delta > self.config.missing_rate_delta_warning
        )

        if is_critical:
            severity = DriftSeverity.CRITICAL
            reasons = []
            if psi_val >= self.config.feature_psi_critical:
                reasons.append(f"PSI={psi_val:.4f} >= {self.config.feature_psi_critical}")
            if ks_pval <= self.config.feature_ks_pvalue_critical:
                reasons.append(f"KS p-val={ks_pval:.2e} <= {self.config.feature_ks_pvalue_critical}")
            if missing_delta > self.config.missing_rate_delta_critical:
                reasons.append(f"Missing delta={missing_delta:.4f} > {self.config.missing_rate_delta_critical}")
            alert_msg = f"Critical numerical drift: {', '.join(reasons)}"
        elif is_warning:
            severity = DriftSeverity.WARNING
            reasons = []
            if psi_val >= self.config.feature_psi_warning:
                reasons.append(f"PSI={psi_val:.4f} >= {self.config.feature_psi_warning}")
            if ks_pval <= self.config.feature_ks_pvalue_warning:
                reasons.append(f"KS p-val={ks_pval:.2e} <= {self.config.feature_ks_pvalue_warning}")
            if missing_delta > self.config.missing_rate_delta_warning:
                reasons.append(f"Missing delta={missing_delta:.4f} > {self.config.missing_rate_delta_warning}")
            alert_msg = f"Warning numerical drift: {', '.join(reasons)}"
        else:
            severity = DriftSeverity.NORMAL
            alert_msg = None

        return SingleFeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            drift_metric_name="PSI",
            drift_metric_value=float(psi_val),
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            observed_missing_rate=observed_missing_rate,
            baseline_missing_rate=profile.missing_rate,
            missing_rate_delta=missing_delta,
            unseen_category_rate=None,
            unseen_categories=None,
            observed_sample_count=n_total,
            severity=severity,
            confidence=MetricConfidence.NORMAL_CONFIDENCE,
            baseline_bin_edges=profile.bin_edges,
            observed_proportions=obs_props,
            expected_proportions=exp_props,
            alert_message=alert_msg,
        )

    def evaluate_categorical_feature(
        self,
        series: pd.Series,
        feature_name: str,
        profile: CategoricalFeatureProfile,
    ) -> SingleFeatureDriftResult:
        """
        Evaluate drift metrics and alert status for a categorical discrete feature.
        """
        n_total = len(series)
        missing_count = int(series.isna().sum())
        observed_missing_rate = float(missing_count / n_total) if n_total > 0 else 0.0
        missing_delta = calculate_missing_rate_delta(observed_missing_rate, profile.missing_rate)

        if n_total < self.config.min_feature_samples:
            return SingleFeatureDriftResult(
                feature_name=feature_name,
                feature_type="categorical",
                drift_metric_name="JSD",
                drift_metric_value=0.0,
                ks_statistic=None,
                ks_pvalue=None,
                observed_missing_rate=observed_missing_rate,
                baseline_missing_rate=profile.missing_rate,
                missing_rate_delta=missing_delta,
                unseen_category_rate=0.0,
                unseen_categories=[],
                observed_sample_count=n_total,
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=MetricConfidence.INSUFFICIENT_DATA,
                baseline_bin_edges=None,
                observed_proportions=None,
                expected_proportions=None,
                alert_message=f"Sample size {n_total} below required minimum {self.config.min_feature_samples}",
            )

        valid_series = series.dropna().astype(str)
        if len(valid_series) == 0:
            return SingleFeatureDriftResult(
                feature_name=feature_name,
                feature_type="categorical",
                drift_metric_name="JSD",
                drift_metric_value=0.0,
                ks_statistic=None,
                ks_pvalue=None,
                observed_missing_rate=1.0,
                baseline_missing_rate=profile.missing_rate,
                missing_rate_delta=missing_delta,
                unseen_category_rate=0.0,
                unseen_categories=[],
                observed_sample_count=n_total,
                severity=DriftSeverity.CRITICAL,
                confidence=MetricConfidence.NORMAL_CONFIDENCE,
                baseline_bin_edges=None,
                observed_proportions=None,
                expected_proportions=None,
                alert_message="All observed categorical values are missing/NaN.",
            )

        unseen_rate, unseen_cats = calculate_unseen_category_rate(valid_series, profile.vocabulary)
        obs_counts = valid_series.value_counts(normalize=True).to_dict()
        jsd_val = calculate_jsd(profile.probabilities, obs_counts)

        is_critical = (
            jsd_val >= self.config.tier_jsd_critical
            or unseen_rate > self.config.unseen_category_rate_critical
            or missing_delta > self.config.missing_rate_delta_critical
        )
        is_warning = (
            jsd_val >= self.config.tier_jsd_warning
            or unseen_rate > self.config.unseen_category_rate_warning
            or missing_delta > self.config.missing_rate_delta_warning
        )

        if is_critical:
            severity = DriftSeverity.CRITICAL
            reasons = []
            if jsd_val >= self.config.tier_jsd_critical:
                reasons.append(f"JSD={jsd_val:.4f} >= {self.config.tier_jsd_critical}")
            if unseen_rate > self.config.unseen_category_rate_critical:
                reasons.append(f"Unseen rate={unseen_rate:.4f} > {self.config.unseen_category_rate_critical} ({len(unseen_cats)} new)")
            if missing_delta > self.config.missing_rate_delta_critical:
                reasons.append(f"Missing delta={missing_delta:.4f} > {self.config.missing_rate_delta_critical}")
            alert_msg = f"Critical categorical drift: {', '.join(reasons)}"
        elif is_warning:
            severity = DriftSeverity.WARNING
            reasons = []
            if jsd_val >= self.config.tier_jsd_warning:
                reasons.append(f"JSD={jsd_val:.4f} >= {self.config.tier_jsd_warning}")
            if unseen_rate > self.config.unseen_category_rate_warning:
                reasons.append(f"Unseen rate={unseen_rate:.4f} > {self.config.unseen_category_rate_warning} ({len(unseen_cats)} new)")
            if missing_delta > self.config.missing_rate_delta_warning:
                reasons.append(f"Missing delta={missing_delta:.4f} > {self.config.missing_rate_delta_warning}")
            alert_msg = f"Warning categorical drift: {', '.join(reasons)}"
        else:
            severity = DriftSeverity.NORMAL
            alert_msg = None

        return SingleFeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            drift_metric_name="JSD",
            drift_metric_value=float(jsd_val),
            ks_statistic=None,
            ks_pvalue=None,
            observed_missing_rate=observed_missing_rate,
            baseline_missing_rate=profile.missing_rate,
            missing_rate_delta=missing_delta,
            unseen_category_rate=float(unseen_rate),
            unseen_categories=unseen_cats,
            observed_sample_count=n_total,
            severity=severity,
            confidence=MetricConfidence.NORMAL_CONFIDENCE,
            baseline_bin_edges=None,
            observed_proportions=None,
            expected_proportions=None,
            alert_message=alert_msg,
        )

    def compute_single_feature_drift(
        self,
        series: pd.Series,
        feature_name: str,
    ) -> SingleFeatureDriftResult:
        """
        Evaluate drift for a single feature series by name.
        """
        if feature_name not in self.baseline_profile.features:
            raise KeyError(f"Feature '{feature_name}' not found in baseline profile.")

        profile = self.baseline_profile.features[feature_name]
        if isinstance(profile, NumericalFeatureProfile):
            return self.evaluate_numerical_feature(series, feature_name, profile)
        elif isinstance(profile, CategoricalFeatureProfile):
            return self.evaluate_categorical_feature(series, feature_name, profile)
        else:
            raise TypeError(f"Unknown profile type for feature '{feature_name}': {type(profile)}")

    def compute_feature_drift(
        self,
        df: pd.DataFrame,
        window_type: Optional[str] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> FeatureDriftReport:
        """
        Evaluate feature drift across all 55 baseline features for an observation window DataFrame.

        Args:
            df: DataFrame containing observation feature columns.
            window_type: Optional window type label (e.g. '1h', '24h', '7d', '30d', 'custom').
            window_start: Optional ISO start timestamp string.
            window_end: Optional ISO end timestamp string.

        Returns:
            FeatureDriftReport containing detailed per-feature and aggregate drift metrics.
        """
        n_rows = len(df)
        feature_results: Dict[str, SingleFeatureDriftResult] = {}

        # Iterate over all features registered in baseline profile (55 features)
        for feat_name, profile in self.baseline_profile.features.items():
            if feat_name in df.columns:
                series = df[feat_name]
            else:
                # Column is completely missing from input dataframe
                series = pd.Series([np.nan] * max(1, n_rows), name=feat_name)

            if isinstance(profile, NumericalFeatureProfile):
                res = self.evaluate_numerical_feature(series, feat_name, profile)
            else:
                res = self.evaluate_categorical_feature(series, feat_name, profile)

            feature_results[feat_name] = res

        # Aggregate counts
        crit_count = sum(1 for r in feature_results.values() if r.severity == DriftSeverity.CRITICAL)
        warn_count = sum(1 for r in feature_results.values() if r.severity == DriftSeverity.WARNING)
        norm_count = sum(1 for r in feature_results.values() if r.severity == DriftSeverity.NORMAL)
        insuf_count = sum(1 for r in feature_results.values() if r.severity == DriftSeverity.INSUFFICIENT_DATA)
        drifted_count = crit_count + warn_count

        # Overall Status
        if n_rows < self.config.min_feature_samples or insuf_count == len(feature_results):
            overall_status = DriftSeverity.INSUFFICIENT_DATA
        elif crit_count > 0:
            overall_status = DriftSeverity.CRITICAL
        elif warn_count > 0:
            overall_status = DriftSeverity.WARNING
        else:
            overall_status = DriftSeverity.NORMAL

        # Ranking logic:
        # 1. Severity: CRITICAL (0), WARNING (1), NORMAL (2), INSUFFICIENT_DATA (3)
        # 2. Primary drift metric value descending
        def severity_rank(sev: DriftSeverity) -> int:
            if sev == DriftSeverity.CRITICAL:
                return 0
            if sev == DriftSeverity.WARNING:
                return 1
            if sev == DriftSeverity.NORMAL:
                return 2
            return 3

        ranked = sorted(
            feature_results.values(),
            key=lambda r: (
                severity_rank(r.severity),
                -r.drift_metric_value,
                -r.missing_rate_delta,
            ),
        )

        created_at_iso = datetime.now(timezone.utc).isoformat()

        return FeatureDriftReport(
            model_version=self.baseline_profile.model_version,
            dataset_row_count=n_rows,
            created_at=created_at_iso,
            overall_data_drift_status=overall_status,
            drifted_features_count=drifted_count,
            critical_features_count=crit_count,
            warning_features_count=warn_count,
            normal_features_count=norm_count,
            insufficient_data_features_count=insuf_count,
            feature_results=feature_results,
            ranked_features=ranked,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
        )
