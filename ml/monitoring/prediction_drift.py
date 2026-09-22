"""
Prediction & Risk Score Drift Engine for Phase 13 ML & Model Monitoring.

Evaluates output distribution shifts:
- Continuous gradient-boosted ranking scores (model_score in [0.0, 1.0]) via 10 equal-width bins PSI.
- Normalized integer risk scores (risk_score in [0, 100]) via 10 standard histogram buckets PSI.
- Categorical risk tiers (LOW, MEDIUM, HIGH, CRITICAL) via JSD.
- Operational decision actions (APPROVE, REVIEW, BLOCK) via JSD.
- Rule engine override rate (is_overridden) volume shifts.
- Sample-size confidence guardrails (N < 50 -> INSUFFICIENT_DATA).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
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
from ml.monitoring.stats import calculate_jsd, calculate_psi


class PredictionDriftCalculator:
    """
    Evaluator for prediction outputs, risk scores, tier distributions, and operational decision drift.
    """

    def __init__(
        self,
        baseline_profile: Optional[PredictionBaselineProfile] = None,
        config: Optional[MonitoringConfig] = None,
    ) -> None:
        self.config = config or default_monitoring_config
        if baseline_profile is not None:
            self.baseline_profile = baseline_profile
        else:
            self.baseline_profile = PredictionBaselineProfile.load(self.config.prediction_profile_path)
        self._evaluator = None

    def _get_evaluator(self) -> Any:
        """Lazy loader for in-memory RiskEvaluator if raw feature DataFrames need scoring."""
        if self._evaluator is None:
            from ml.risk_engine.evaluator import RiskEvaluator
            from ml.risk_engine.rules import RuleEngine, get_standard_rule_catalog
            self._evaluator = RiskEvaluator(rule_engine=RuleEngine(get_standard_rule_catalog()))
        return self._evaluator

    def evaluate_model_score_drift(self, model_scores: Sequence[float]) -> ScoreDriftResult:
        """
        Evaluate continuous model score distribution shift across 10 equal-width bins [0.0, 0.1), ..., [0.9, 1.0].
        """
        n_total = len(model_scores)
        if n_total < self.config.min_prediction_samples:
            return ScoreDriftResult(
                psi_value=0.0,
                observed_bins=[],
                expected_bins=self.baseline_profile.model_score_bins,
                observed_mean=0.0,
                baseline_mean=self.baseline_profile.mean_model_score,
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=MetricConfidence.INSUFFICIENT_DATA,
                alert_message=f"Sample size {n_total} below required minimum {self.config.min_prediction_samples}",
            )

        arr = np.asarray(model_scores, dtype=np.float64)
        clean_scores = arr[~np.isnan(arr) & ~np.isinf(arr)]
        if len(clean_scores) == 0:
            return ScoreDriftResult(
                psi_value=0.0,
                observed_bins=[],
                expected_bins=self.baseline_profile.model_score_bins,
                observed_mean=0.0,
                baseline_mean=self.baseline_profile.mean_model_score,
                severity=DriftSeverity.CRITICAL,
                confidence=MetricConfidence.NORMAL_CONFIDENCE,
                alert_message="All observed model scores are NaN or non-finite.",
            )

        # 10 equal-width bins: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        cutoffs = np.linspace(0.1, 0.9, 9)
        bin_indices = np.digitize(clean_scores, cutoffs, right=False)
        obs_counts = np.bincount(bin_indices, minlength=10)
        obs_props = obs_counts / len(clean_scores)

        observed_bins = [
            ScoreHistogramBin(
                bin_index=i,
                lower_bound=round(i * 0.1, 1),
                upper_bound=round((i + 1) * 0.1, 1),
                count=int(obs_counts[i]),
                proportion=float(obs_props[i]),
            )
            for i in range(10)
        ]

        exp_props = [b.proportion for b in self.baseline_profile.model_score_bins]
        psi_val = calculate_psi(exp_props, obs_props, epsilon=self.config.laplace_epsilon)
        observed_mean = float(np.mean(clean_scores))

        if psi_val >= self.config.prediction_psi_critical:
            sev = DriftSeverity.CRITICAL
            alert = f"Critical model score drift: PSI={psi_val:.4f} >= {self.config.prediction_psi_critical}"
        elif psi_val >= self.config.prediction_psi_warning:
            sev = DriftSeverity.WARNING
            alert = f"Warning model score drift: PSI={psi_val:.4f} >= {self.config.prediction_psi_warning}"
        else:
            sev = DriftSeverity.NORMAL
            alert = None

        return ScoreDriftResult(
            psi_value=float(psi_val),
            observed_bins=observed_bins,
            expected_bins=self.baseline_profile.model_score_bins,
            observed_mean=observed_mean,
            baseline_mean=self.baseline_profile.mean_model_score,
            severity=sev,
            confidence=MetricConfidence.NORMAL_CONFIDENCE,
            alert_message=alert,
        )

    def evaluate_risk_score_drift(self, risk_scores: Sequence[Union[int, float]]) -> RiskScoreDriftResult:
        """
        Evaluate normalized integer risk score distribution shift across 10 buckets 0-9, 10-19, ..., 90-100.
        """
        n_total = len(risk_scores)
        if n_total < self.config.min_prediction_samples:
            return RiskScoreDriftResult(
                psi_value=0.0,
                observed_buckets=[],
                expected_buckets=self.baseline_profile.risk_score_buckets,
                observed_mean=0.0,
                baseline_mean=self.baseline_profile.mean_risk_score,
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=MetricConfidence.INSUFFICIENT_DATA,
                alert_message=f"Sample size {n_total} below required minimum {self.config.min_prediction_samples}",
            )

        arr = np.asarray(risk_scores, dtype=np.float64)
        clean_scores = arr[~np.isnan(arr) & ~np.isinf(arr)]
        if len(clean_scores) == 0:
            return RiskScoreDriftResult(
                psi_value=0.0,
                observed_buckets=[],
                expected_buckets=self.baseline_profile.risk_score_buckets,
                observed_mean=0.0,
                baseline_mean=self.baseline_profile.mean_risk_score,
                severity=DriftSeverity.CRITICAL,
                confidence=MetricConfidence.NORMAL_CONFIDENCE,
                alert_message="All observed risk scores are NaN or non-finite.",
            )

        # 10 buckets: 0-9, 10-19, ..., 80-89, 90-100
        bucket_indices = np.clip(np.floor(clean_scores / 10.0).astype(int), 0, 9)
        obs_counts = np.bincount(bucket_indices, minlength=10)
        obs_props = obs_counts / len(clean_scores)

        bucket_labels = [
            ("0-9", 0, 9),
            ("10-19", 10, 19),
            ("20-29", 20, 29),
            ("30-39", 30, 39),
            ("40-49", 40, 49),
            ("50-59", 50, 59),
            ("60-69", 60, 69),
            ("70-79", 70, 79),
            ("80-89", 80, 89),
            ("90-100", 90, 100),
        ]

        observed_buckets = [
            RiskScoreBucket(
                bucket_index=i,
                label=bucket_labels[i][0],
                lower_bound=bucket_labels[i][1],
                upper_bound=bucket_labels[i][2],
                count=int(obs_counts[i]),
                proportion=float(obs_props[i]),
            )
            for i in range(10)
        ]

        exp_props = [b.proportion for b in self.baseline_profile.risk_score_buckets]
        psi_val = calculate_psi(exp_props, obs_props, epsilon=self.config.laplace_epsilon)
        observed_mean = float(np.mean(clean_scores))

        if psi_val >= self.config.prediction_psi_critical:
            sev = DriftSeverity.CRITICAL
            alert = f"Critical risk score drift: PSI={psi_val:.4f} >= {self.config.prediction_psi_critical}"
        elif psi_val >= self.config.prediction_psi_warning:
            sev = DriftSeverity.WARNING
            alert = f"Warning risk score drift: PSI={psi_val:.4f} >= {self.config.prediction_psi_warning}"
        else:
            sev = DriftSeverity.NORMAL
            alert = None

        return RiskScoreDriftResult(
            psi_value=float(psi_val),
            observed_buckets=observed_buckets,
            expected_buckets=self.baseline_profile.risk_score_buckets,
            observed_mean=observed_mean,
            baseline_mean=self.baseline_profile.mean_risk_score,
            severity=sev,
            confidence=MetricConfidence.NORMAL_CONFIDENCE,
            alert_message=alert,
        )

    def evaluate_tier_drift(self, risk_tiers: Sequence[Any]) -> CategoricalDistributionDriftResult:
        """
        Evaluate risk tier distribution shift across LOW, MEDIUM, HIGH, CRITICAL using JSD.
        """
        n_total = len(risk_tiers)
        if n_total < self.config.min_prediction_samples:
            return CategoricalDistributionDriftResult(
                metric_name="JSD",
                jsd_value=0.0,
                observed_proportions={},
                expected_proportions=self.baseline_profile.tier_proportions,
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=MetricConfidence.INSUFFICIENT_DATA,
                alert_message=f"Sample size {n_total} below required minimum {self.config.min_prediction_samples}",
            )

        standard_tiers = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        clean_tiers = [
            getattr(t, "value", str(t)).upper().strip()
            for t in risk_tiers
            if t is not None and not (isinstance(t, float) and np.isnan(t))
        ]

        if len(clean_tiers) == 0:
            return CategoricalDistributionDriftResult(
                metric_name="JSD",
                jsd_value=1.0,
                observed_proportions={},
                expected_proportions=self.baseline_profile.tier_proportions,
                severity=DriftSeverity.CRITICAL,
                confidence=MetricConfidence.NORMAL_CONFIDENCE,
                alert_message="All observed risk tiers are missing or invalid.",
            )

        obs_props = {t: float(sum(1 for x in clean_tiers if x == t) / len(clean_tiers)) for t in standard_tiers}
        jsd_val = calculate_jsd(self.baseline_profile.tier_proportions, obs_props)

        if jsd_val >= self.config.tier_jsd_critical:
            sev = DriftSeverity.CRITICAL
            alert = f"Critical risk tier drift: JSD={jsd_val:.4f} >= {self.config.tier_jsd_critical}"
        elif jsd_val >= self.config.tier_jsd_warning:
            sev = DriftSeverity.WARNING
            alert = f"Warning risk tier drift: JSD={jsd_val:.4f} >= {self.config.tier_jsd_warning}"
        else:
            sev = DriftSeverity.NORMAL
            alert = None

        return CategoricalDistributionDriftResult(
            metric_name="JSD",
            jsd_value=float(jsd_val),
            observed_proportions=obs_props,
            expected_proportions=self.baseline_profile.tier_proportions,
            severity=sev,
            confidence=MetricConfidence.NORMAL_CONFIDENCE,
            alert_message=alert,
        )

    def evaluate_action_drift(self, actions: Sequence[Any]) -> CategoricalDistributionDriftResult:
        """
        Evaluate operational decision action distribution shift across APPROVE, REVIEW, BLOCK using JSD.
        """
        n_total = len(actions)
        if n_total < self.config.min_prediction_samples:
            return CategoricalDistributionDriftResult(
                metric_name="JSD",
                jsd_value=0.0,
                observed_proportions={},
                expected_proportions=self.baseline_profile.action_proportions,
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=MetricConfidence.INSUFFICIENT_DATA,
                alert_message=f"Sample size {n_total} below required minimum {self.config.min_prediction_samples}",
            )

        standard_actions = ["APPROVE", "REVIEW", "BLOCK"]
        clean_actions = [
            getattr(a, "value", str(a)).upper().strip()
            for a in actions
            if a is not None and not (isinstance(a, float) and np.isnan(a))
        ]

        if len(clean_actions) == 0:
            return CategoricalDistributionDriftResult(
                metric_name="JSD",
                jsd_value=1.0,
                observed_proportions={},
                expected_proportions=self.baseline_profile.action_proportions,
                severity=DriftSeverity.CRITICAL,
                confidence=MetricConfidence.NORMAL_CONFIDENCE,
                alert_message="All observed actions are missing or invalid.",
            )

        obs_props = {a: float(sum(1 for x in clean_actions if x == a) / len(clean_actions)) for a in standard_actions}
        jsd_val = calculate_jsd(self.baseline_profile.action_proportions, obs_props)

        if jsd_val >= self.config.action_jsd_critical:
            sev = DriftSeverity.CRITICAL
            alert = f"Critical decision action drift: JSD={jsd_val:.4f} >= {self.config.action_jsd_critical}"
        elif jsd_val >= self.config.action_jsd_warning:
            sev = DriftSeverity.WARNING
            alert = f"Warning decision action drift: JSD={jsd_val:.4f} >= {self.config.action_jsd_warning}"
        else:
            sev = DriftSeverity.NORMAL
            alert = None

        return CategoricalDistributionDriftResult(
            metric_name="JSD",
            jsd_value=float(jsd_val),
            observed_proportions=obs_props,
            expected_proportions=self.baseline_profile.action_proportions,
            severity=sev,
            confidence=MetricConfidence.NORMAL_CONFIDENCE,
            alert_message=alert,
        )

    def evaluate_override_drift(self, is_overridden: Sequence[Any]) -> OverrideRateDriftResult:
        """
        Evaluate the frequency of business rule overrides vs baseline rate.
        """
        n_total = len(is_overridden)
        if n_total == 0:
            return OverrideRateDriftResult(
                observed_rate=0.0,
                baseline_rate=self.baseline_profile.is_overridden_rate,
                rate_delta=0.0,
                overridden_count=0,
                total_count=0,
            )

        overridden_count = sum(1 for x in is_overridden if bool(x) is True)
        obs_rate = float(overridden_count / n_total)
        rate_delta = float(abs(obs_rate - self.baseline_profile.is_overridden_rate))

        return OverrideRateDriftResult(
            observed_rate=obs_rate,
            baseline_rate=self.baseline_profile.is_overridden_rate,
            rate_delta=rate_delta,
            overridden_count=overridden_count,
            total_count=n_total,
        )

    def compute_prediction_drift_from_arrays(
        self,
        model_scores: Sequence[float],
        risk_scores: Sequence[Union[int, float]],
        risk_tiers: Sequence[Any],
        actions: Sequence[Any],
        is_overridden: Sequence[Any],
        window_type: Optional[str] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> PredictionDriftReport:
        """
        Compute prediction drift report directly from raw arrays/sequences.
        """
        n_total = len(model_scores)
        score_res = self.evaluate_model_score_drift(model_scores)
        risk_res = self.evaluate_risk_score_drift(risk_scores)
        tier_res = self.evaluate_tier_drift(risk_tiers)
        action_res = self.evaluate_action_drift(actions)
        override_res = self.evaluate_override_drift(is_overridden)

        # Overall Status Precedence:
        # INSUFFICIENT_DATA (if N < 50) > CRITICAL (if any critical) > WARNING (if any warning) > NORMAL
        if n_total < self.config.min_prediction_samples:
            overall_status = DriftSeverity.INSUFFICIENT_DATA
        elif (
            score_res.severity == DriftSeverity.CRITICAL
            or risk_res.severity == DriftSeverity.CRITICAL
            or tier_res.severity == DriftSeverity.CRITICAL
            or action_res.severity == DriftSeverity.CRITICAL
        ):
            overall_status = DriftSeverity.CRITICAL
        elif (
            score_res.severity == DriftSeverity.WARNING
            or risk_res.severity == DriftSeverity.WARNING
            or tier_res.severity == DriftSeverity.WARNING
            or action_res.severity == DriftSeverity.WARNING
        ):
            overall_status = DriftSeverity.WARNING
        else:
            overall_status = DriftSeverity.NORMAL

        created_at_iso = datetime.now(timezone.utc).isoformat()

        return PredictionDriftReport(
            model_version=self.baseline_profile.model_version,
            dataset_row_count=n_total,
            created_at=created_at_iso,
            overall_prediction_drift_status=overall_status,
            model_score_drift=score_res,
            risk_score_drift=risk_res,
            tier_drift=tier_res,
            action_drift=action_res,
            override_drift=override_res,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
        )

    def compute_prediction_drift(
        self,
        data: Union[pd.DataFrame, Sequence[Any]],
        window_type: Optional[str] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> PredictionDriftReport:
        """
        Evaluate prediction drift for a batch DataFrame, list of DecisionResults, or feature DataFrame.

        If input DataFrame contains 55 features without prediction columns, evaluates them
        in-memory via RiskEvaluator with zero persistence side effects.
        """
        if data is None or len(data) == 0:
            return self.compute_prediction_drift_from_arrays(
                model_scores=[],
                risk_scores=[],
                risk_tiers=[],
                actions=[],
                is_overridden=[],
                window_type=window_type,
                window_start=window_start,
                window_end=window_end,
            )

        if isinstance(data, pd.DataFrame):
            df = data
            if "model_score" in df.columns:
                model_scores = df["model_score"].tolist()
                risk_scores = df["risk_score"].tolist() if "risk_score" in df.columns else [int(round(s * 100)) for s in model_scores]
                risk_tiers = df["risk_tier"].tolist() if "risk_tier" in df.columns else (df["tier"].tolist() if "tier" in df.columns else ["LOW"] * len(df))
                actions = df["action"].tolist() if "action" in df.columns else (df["decision_action"].tolist() if "decision_action" in df.columns else ["APPROVE"] * len(df))
                is_overridden = df["is_overridden"].tolist() if "is_overridden" in df.columns else [False] * len(df)
            else:
                # Raw feature DataFrame: evaluate in-memory with frozen RiskEvaluator
                evaluator = self._get_evaluator()
                results = evaluator.evaluate_dataframe(df)
                model_scores = [r.model_score for r in results]
                risk_scores = [r.risk_score for r in results]
                risk_tiers = [r.risk_tier.value for r in results]
                actions = [r.action.value for r in results]
                is_overridden = [r.is_overridden for r in results]

            return self.compute_prediction_drift_from_arrays(
                model_scores=model_scores,
                risk_scores=risk_scores,
                risk_tiers=risk_tiers,
                actions=actions,
                is_overridden=is_overridden,
                window_type=window_type,
                window_start=window_start,
                window_end=window_end,
            )

        # List of DecisionResult or dicts
        model_scores = []
        risk_scores = []
        risk_tiers = []
        actions = []
        is_overridden = []

        for item in data:
            if hasattr(item, "model_score"):
                model_scores.append(float(item.model_score))
                risk_scores.append(int(item.risk_score))
                tier_val = getattr(item, "risk_tier", "LOW")
                risk_tiers.append(getattr(tier_val, "value", str(tier_val)))
                act_val = getattr(item, "decision_action", getattr(item, "action", "APPROVE"))
                actions.append(getattr(act_val, "value", str(act_val)))
                is_overridden.append(bool(getattr(item, "is_overridden", False)))
            elif isinstance(item, dict):
                model_scores.append(item.get("model_score", 0.0))
                risk_scores.append(item.get("risk_score", 0))
                risk_tiers.append(item.get("risk_tier", item.get("tier", "LOW")))
                actions.append(item.get("action", item.get("decision_action", "APPROVE")))
                is_overridden.append(bool(item.get("is_overridden", False)))

        return self.compute_prediction_drift_from_arrays(
            model_scores=model_scores,
            risk_scores=risk_scores,
            risk_tiers=risk_tiers,
            actions=actions,
            is_overridden=is_overridden,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
        )
