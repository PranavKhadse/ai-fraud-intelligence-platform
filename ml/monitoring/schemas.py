"""
Data Contracts, Schemas, and Serialization Helpers for Phase 13 ML & Model Monitoring.
"""

from dataclasses import dataclass, asdict, field
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


def compute_content_sha256(content: str) -> str:
    """Compute SHA-256 hex digest for a JSON content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class NumericalFeatureProfile:
    """Statistical distribution profile for a continuous/numerical predictor."""
    feature_name: str
    data_type: str
    count: int
    missing_count: int
    missing_rate: float
    mean: float
    std: float
    min: float
    max: float
    bin_edges: List[float]
    ref_samples: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NumericalFeatureProfile":
        return cls(**data)


@dataclass
class CategoricalFeatureProfile:
    """Statistical discrete frequency profile for a categorical predictor."""
    feature_name: str
    data_type: str
    count: int
    missing_count: int
    missing_rate: float
    vocabulary: List[str]
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoricalFeatureProfile":
        return cls(**data)


@dataclass
class FeatureBaselineProfile:
    """Comprehensive baseline profile for all 55 features."""
    model_version: str
    dataset_name: str
    dataset_row_count: int
    created_at: str
    random_seed: int
    num_features: int
    features: Dict[str, Union[NumericalFeatureProfile, CategoricalFeatureProfile]]
    sha256_checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        feat_dict = {}
        for name, profile in self.features.items():
            if isinstance(profile, (NumericalFeatureProfile, CategoricalFeatureProfile)):
                feat_dict[name] = profile.to_dict()
            else:
                feat_dict[name] = profile
        return {
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_row_count": self.dataset_row_count,
            "created_at": self.created_at,
            "random_seed": self.random_seed,
            "num_features": self.num_features,
            "features": feat_dict,
            "sha256_checksum": self.sha256_checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureBaselineProfile":
        features_parsed = {}
        for name, f_data in data.get("features", {}).items():
            if "bin_edges" in f_data:
                features_parsed[name] = NumericalFeatureProfile.from_dict(f_data)
            else:
                features_parsed[name] = CategoricalFeatureProfile.from_dict(f_data)
        return cls(
            model_version=data["model_version"],
            dataset_name=data["dataset_name"],
            dataset_row_count=data["dataset_row_count"],
            created_at=data["created_at"],
            random_seed=data["random_seed"],
            num_features=data["num_features"],
            features=features_parsed,
            sha256_checksum=data.get("sha256_checksum"),
        )

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Compute checksum on payload without sha256_checksum field
        d = self.to_dict()
        d["sha256_checksum"] = None
        raw_json = json.dumps(d, indent=2, sort_keys=True)
        d["sha256_checksum"] = compute_content_sha256(raw_json)
        self.sha256_checksum = d["sha256_checksum"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "FeatureBaselineProfile":
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ScoreHistogramBin:
    """Single bin for model score probability distribution."""
    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    proportion: float


@dataclass
class RiskScoreBucket:
    """Single 10-point bucket for normalized integer risk score [0, 100]."""
    bucket_index: int
    label: str
    lower_bound: int
    upper_bound: int
    count: int
    proportion: float


@dataclass
class PredictionBaselineProfile:
    """Baseline profile for model scores, risk scores, tiers, and decision actions."""
    model_version: str
    dataset_name: str
    dataset_row_count: int
    created_at: str
    model_score_bins: List[ScoreHistogramBin]
    risk_score_buckets: List[RiskScoreBucket]
    tier_proportions: Dict[str, float]
    action_proportions: Dict[str, float]
    is_overridden_rate: float
    mean_model_score: float
    mean_risk_score: float
    sha256_checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_row_count": self.dataset_row_count,
            "created_at": self.created_at,
            "model_score_bins": [asdict(b) if isinstance(b, ScoreHistogramBin) else b for b in self.model_score_bins],
            "risk_score_buckets": [asdict(b) if isinstance(b, RiskScoreBucket) else b for b in self.risk_score_buckets],
            "tier_proportions": dict(self.tier_proportions),
            "action_proportions": dict(self.action_proportions),
            "is_overridden_rate": self.is_overridden_rate,
            "mean_model_score": self.mean_model_score,
            "mean_risk_score": self.mean_risk_score,
            "sha256_checksum": self.sha256_checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionBaselineProfile":
        bins = [ScoreHistogramBin(**b) if isinstance(b, dict) else b for b in data.get("model_score_bins", [])]
        buckets = [RiskScoreBucket(**b) if isinstance(b, dict) else b for b in data.get("risk_score_buckets", [])]
        return cls(
            model_version=data["model_version"],
            dataset_name=data["dataset_name"],
            dataset_row_count=data["dataset_row_count"],
            created_at=data["created_at"],
            model_score_bins=bins,
            risk_score_buckets=buckets,
            tier_proportions=data["tier_proportions"],
            action_proportions=data["action_proportions"],
            is_overridden_rate=data["is_overridden_rate"],
            mean_model_score=data["mean_model_score"],
            mean_risk_score=data["mean_risk_score"],
            sha256_checksum=data.get("sha256_checksum"),
        )

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        d["sha256_checksum"] = None
        raw_json = json.dumps(d, indent=2, sort_keys=True)
        d["sha256_checksum"] = compute_content_sha256(raw_json)
        self.sha256_checksum = d["sha256_checksum"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PredictionBaselineProfile":
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ConfusionMatrixData:
    """Confusion matrix counts."""
    tp: int
    fp: int
    fn: int
    tn: int
    total: int


@dataclass
class ThresholdPerformanceMetrics:
    """Performance metrics evaluated at a specific decision threshold."""
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr: float
    tpr: float
    confusion_matrix: ConfusionMatrixData


@dataclass
class OperationalDecisionMetrics:
    """Operational metrics for hybrid decisions and human review queue."""
    decision_precision_block: float
    decision_recall_intervention: float
    review_queue_purity: float
    total_reviews_count: int
    fraud_in_review_count: int
    total_blocks_count: int
    fraud_in_block_count: int


@dataclass
class PerformanceBaselineProfile:
    """Authoritative baseline performance benchmarks for degradation monitoring."""
    model_version: str
    dataset_name: str
    dataset_row_count: int
    total_frauds: int
    total_legitimate: int
    created_at: str
    default_operating_threshold: float
    pr_auc: float
    roc_auc: float
    primary_operating_metrics: ThresholdPerformanceMetrics
    comparison_metrics: Optional[ThresholdPerformanceMetrics]
    operational_decision_metrics: OperationalDecisionMetrics
    sha256_checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        prim_cm = asdict(self.primary_operating_metrics.confusion_matrix) if isinstance(self.primary_operating_metrics.confusion_matrix, ConfusionMatrixData) else self.primary_operating_metrics.confusion_matrix
        prim_d = asdict(self.primary_operating_metrics)
        prim_d["confusion_matrix"] = prim_cm

        comp_d = None
        if self.comparison_metrics is not None:
            comp_cm = asdict(self.comparison_metrics.confusion_matrix) if isinstance(self.comparison_metrics.confusion_matrix, ConfusionMatrixData) else self.comparison_metrics.confusion_matrix
            comp_d = asdict(self.comparison_metrics)
            comp_d["confusion_matrix"] = comp_cm

        ops_d = asdict(self.operational_decision_metrics) if isinstance(self.operational_decision_metrics, OperationalDecisionMetrics) else self.operational_decision_metrics

        return {
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_row_count": self.dataset_row_count,
            "total_frauds": self.total_frauds,
            "total_legitimate": self.total_legitimate,
            "created_at": self.created_at,
            "default_operating_threshold": self.default_operating_threshold,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "primary_operating_metrics": prim_d,
            "comparison_metrics": comp_d,
            "operational_decision_metrics": ops_d,
            "sha256_checksum": self.sha256_checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceBaselineProfile":
        prim_raw = data["primary_operating_metrics"]
        prim_cm = ConfusionMatrixData(**prim_raw["confusion_matrix"])
        prim_metrics = ThresholdPerformanceMetrics(
            threshold=prim_raw["threshold"],
            precision=prim_raw["precision"],
            recall=prim_raw["recall"],
            f1=prim_raw["f1"],
            accuracy=prim_raw["accuracy"],
            fpr=prim_raw["fpr"],
            tpr=prim_raw["tpr"],
            confusion_matrix=prim_cm,
        )

        comp_metrics = None
        if data.get("comparison_metrics") is not None:
            comp_raw = data["comparison_metrics"]
            comp_cm = ConfusionMatrixData(**comp_raw["confusion_matrix"])
            comp_metrics = ThresholdPerformanceMetrics(
                threshold=comp_raw["threshold"],
                precision=comp_raw["precision"],
                recall=comp_raw["recall"],
                f1=comp_raw["f1"],
                accuracy=comp_raw["accuracy"],
                fpr=comp_raw["fpr"],
                tpr=comp_raw["tpr"],
                confusion_matrix=comp_cm,
            )

        ops_metrics = OperationalDecisionMetrics(**data["operational_decision_metrics"])

        return cls(
            model_version=data["model_version"],
            dataset_name=data["dataset_name"],
            dataset_row_count=data["dataset_row_count"],
            total_frauds=data["total_frauds"],
            total_legitimate=data["total_legitimate"],
            created_at=data["created_at"],
            default_operating_threshold=data["default_operating_threshold"],
            pr_auc=data["pr_auc"],
            roc_auc=data["roc_auc"],
            primary_operating_metrics=prim_metrics,
            comparison_metrics=comp_metrics,
            operational_decision_metrics=ops_metrics,
            sha256_checksum=data.get("sha256_checksum"),
        )

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        d["sha256_checksum"] = None
        raw_json = json.dumps(d, indent=2, sort_keys=True)
        d["sha256_checksum"] = compute_content_sha256(raw_json)
        self.sha256_checksum = d["sha256_checksum"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PerformanceBaselineProfile":
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


from ml.monitoring.config import DriftSeverity, MetricConfidence


@dataclass
class SingleFeatureDriftResult:
    """Drift metrics and severity evaluation for an individual feature."""
    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    drift_metric_name: str  # "PSI" or "JSD"
    drift_metric_value: float
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    observed_missing_rate: float = 0.0
    baseline_missing_rate: float = 0.0
    missing_rate_delta: float = 0.0
    unseen_category_rate: Optional[float] = None
    unseen_categories: Optional[List[str]] = None
    observed_sample_count: int = 0
    severity: DriftSeverity = DriftSeverity.NORMAL
    confidence: MetricConfidence = MetricConfidence.NORMAL_CONFIDENCE
    baseline_bin_edges: Optional[List[float]] = None
    observed_proportions: Optional[List[float]] = None
    expected_proportions: Optional[List[float]] = None
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value if isinstance(self.severity, DriftSeverity) else self.severity
        d["confidence"] = self.confidence.value if isinstance(self.confidence, MetricConfidence) else self.confidence
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleFeatureDriftResult":
        d = dict(data)
        if isinstance(d.get("severity"), str):
            d["severity"] = DriftSeverity(d["severity"])
        if isinstance(d.get("confidence"), str):
            d["confidence"] = MetricConfidence(d["confidence"])
        return cls(**d)


@dataclass
class FeatureDriftReport:
    """Comprehensive 55-feature data drift evaluation report."""
    model_version: str
    dataset_row_count: int
    created_at: str
    overall_data_drift_status: DriftSeverity
    drifted_features_count: int
    critical_features_count: int
    warning_features_count: int
    normal_features_count: int
    insufficient_data_features_count: int
    feature_results: Dict[str, SingleFeatureDriftResult]
    ranked_features: List[SingleFeatureDriftResult]
    window_type: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_row_count": self.dataset_row_count,
            "created_at": self.created_at,
            "overall_data_drift_status": self.overall_data_drift_status.value if isinstance(self.overall_data_drift_status, DriftSeverity) else self.overall_data_drift_status,
            "drifted_features_count": self.drifted_features_count,
            "critical_features_count": self.critical_features_count,
            "warning_features_count": self.warning_features_count,
            "normal_features_count": self.normal_features_count,
            "insufficient_data_features_count": self.insufficient_data_features_count,
            "window_type": self.window_type,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "feature_results": {k: v.to_dict() if isinstance(v, SingleFeatureDriftResult) else v for k, v in self.feature_results.items()},
            "ranked_features": [f.to_dict() if isinstance(f, SingleFeatureDriftResult) else f for f in self.ranked_features],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDriftReport":
        feat_res = {
            k: SingleFeatureDriftResult.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.get("feature_results", {}).items()
        }
        ranked = [
            SingleFeatureDriftResult.from_dict(f) if isinstance(f, dict) else f
            for f in data.get("ranked_features", [])
        ]
        status = data.get("overall_data_drift_status")
        if isinstance(status, str):
            status = DriftSeverity(status)
        return cls(
            model_version=data["model_version"],
            dataset_row_count=data["dataset_row_count"],
            created_at=data["created_at"],
            overall_data_drift_status=status,
            drifted_features_count=data["drifted_features_count"],
            critical_features_count=data["critical_features_count"],
            warning_features_count=data["warning_features_count"],
            normal_features_count=data["normal_features_count"],
            insufficient_data_features_count=data["insufficient_data_features_count"],
            feature_results=feat_res,
            ranked_features=ranked,
            window_type=data.get("window_type"),
            window_start=data.get("window_start"),
            window_end=data.get("window_end"),
        )


@dataclass
class ScoreDriftResult:
    """Drift metrics for continuous gradient-boosted model ranking scores."""
    psi_value: float
    observed_bins: List[ScoreHistogramBin]
    expected_bins: List[ScoreHistogramBin]
    observed_mean: float
    baseline_mean: float
    severity: DriftSeverity = DriftSeverity.NORMAL
    confidence: MetricConfidence = MetricConfidence.NORMAL_CONFIDENCE
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psi_value": self.psi_value,
            "observed_bins": [asdict(b) if isinstance(b, ScoreHistogramBin) else b for b in self.observed_bins],
            "expected_bins": [asdict(b) if isinstance(b, ScoreHistogramBin) else b for b in self.expected_bins],
            "observed_mean": self.observed_mean,
            "baseline_mean": self.baseline_mean,
            "severity": self.severity.value if isinstance(self.severity, DriftSeverity) else self.severity,
            "confidence": self.confidence.value if isinstance(self.confidence, MetricConfidence) else self.confidence,
            "alert_message": self.alert_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreDriftResult":
        obs_bins = [ScoreHistogramBin(**b) if isinstance(b, dict) else b for b in data.get("observed_bins", [])]
        exp_bins = [ScoreHistogramBin(**b) if isinstance(b, dict) else b for b in data.get("expected_bins", [])]
        sev = data.get("severity", DriftSeverity.NORMAL)
        if isinstance(sev, str):
            sev = DriftSeverity(sev)
        conf = data.get("confidence", MetricConfidence.NORMAL_CONFIDENCE)
        if isinstance(conf, str):
            conf = MetricConfidence(conf)
        return cls(
            psi_value=data["psi_value"],
            observed_bins=obs_bins,
            expected_bins=exp_bins,
            observed_mean=data["observed_mean"],
            baseline_mean=data["baseline_mean"],
            severity=sev,
            confidence=conf,
            alert_message=data.get("alert_message"),
        )


@dataclass
class RiskScoreDriftResult:
    """Drift metrics for normalized integer risk scores [0, 100]."""
    psi_value: float
    observed_buckets: List[RiskScoreBucket]
    expected_buckets: List[RiskScoreBucket]
    observed_mean: float
    baseline_mean: float
    severity: DriftSeverity = DriftSeverity.NORMAL
    confidence: MetricConfidence = MetricConfidence.NORMAL_CONFIDENCE
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psi_value": self.psi_value,
            "observed_buckets": [asdict(b) if isinstance(b, RiskScoreBucket) else b for b in self.observed_buckets],
            "expected_buckets": [asdict(b) if isinstance(b, RiskScoreBucket) else b for b in self.expected_buckets],
            "observed_mean": self.observed_mean,
            "baseline_mean": self.baseline_mean,
            "severity": self.severity.value if isinstance(self.severity, DriftSeverity) else self.severity,
            "confidence": self.confidence.value if isinstance(self.confidence, MetricConfidence) else self.confidence,
            "alert_message": self.alert_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskScoreDriftResult":
        obs_b = [RiskScoreBucket(**b) if isinstance(b, dict) else b for b in data.get("observed_buckets", [])]
        exp_b = [RiskScoreBucket(**b) if isinstance(b, dict) else b for b in data.get("expected_buckets", [])]
        sev = data.get("severity", DriftSeverity.NORMAL)
        if isinstance(sev, str):
            sev = DriftSeverity(sev)
        conf = data.get("confidence", MetricConfidence.NORMAL_CONFIDENCE)
        if isinstance(conf, str):
            conf = MetricConfidence(conf)
        return cls(
            psi_value=data["psi_value"],
            observed_buckets=obs_b,
            expected_buckets=exp_b,
            observed_mean=data["observed_mean"],
            baseline_mean=data["baseline_mean"],
            severity=sev,
            confidence=conf,
            alert_message=data.get("alert_message"),
        )


@dataclass
class CategoricalDistributionDriftResult:
    """Drift metrics for discrete categorical prediction outputs (Risk Tiers, Decision Actions)."""
    metric_name: str  # "JSD"
    jsd_value: float
    observed_proportions: Dict[str, float]
    expected_proportions: Dict[str, float]
    severity: DriftSeverity = DriftSeverity.NORMAL
    confidence: MetricConfidence = MetricConfidence.NORMAL_CONFIDENCE
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "jsd_value": self.jsd_value,
            "observed_proportions": dict(self.observed_proportions),
            "expected_proportions": dict(self.expected_proportions),
            "severity": self.severity.value if isinstance(self.severity, DriftSeverity) else self.severity,
            "confidence": self.confidence.value if isinstance(self.confidence, MetricConfidence) else self.confidence,
            "alert_message": self.alert_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoricalDistributionDriftResult":
        sev = data.get("severity", DriftSeverity.NORMAL)
        if isinstance(sev, str):
            sev = DriftSeverity(sev)
        conf = data.get("confidence", MetricConfidence.NORMAL_CONFIDENCE)
        if isinstance(conf, str):
            conf = MetricConfidence(conf)
        return cls(
            metric_name=data.get("metric_name", "JSD"),
            jsd_value=data["jsd_value"],
            observed_proportions=data["observed_proportions"],
            expected_proportions=data["expected_proportions"],
            severity=sev,
            confidence=conf,
            alert_message=data.get("alert_message"),
        )


@dataclass
class OverrideRateDriftResult:
    """Rate and volume tracking for deterministic rule-engine overrides."""
    observed_rate: float
    baseline_rate: float
    rate_delta: float
    overridden_count: int
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverrideRateDriftResult":
        return cls(**data)


@dataclass
class PredictionDriftReport:
    """Comprehensive prediction, risk score, tier, and action drift evaluation report."""
    model_version: str
    dataset_row_count: int
    created_at: str
    overall_prediction_drift_status: DriftSeverity
    model_score_drift: ScoreDriftResult
    risk_score_drift: RiskScoreDriftResult
    tier_drift: CategoricalDistributionDriftResult
    action_drift: CategoricalDistributionDriftResult
    override_drift: OverrideRateDriftResult
    window_type: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_row_count": self.dataset_row_count,
            "created_at": self.created_at,
            "overall_prediction_drift_status": self.overall_prediction_drift_status.value if isinstance(self.overall_prediction_drift_status, DriftSeverity) else self.overall_prediction_drift_status,
            "model_score_drift": self.model_score_drift.to_dict() if isinstance(self.model_score_drift, ScoreDriftResult) else self.model_score_drift,
            "risk_score_drift": self.risk_score_drift.to_dict() if isinstance(self.risk_score_drift, RiskScoreDriftResult) else self.risk_score_drift,
            "tier_drift": self.tier_drift.to_dict() if isinstance(self.tier_drift, CategoricalDistributionDriftResult) else self.tier_drift,
            "action_drift": self.action_drift.to_dict() if isinstance(self.action_drift, CategoricalDistributionDriftResult) else self.action_drift,
            "override_drift": self.override_drift.to_dict() if isinstance(self.override_drift, OverrideRateDriftResult) else self.override_drift,
            "window_type": self.window_type,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionDriftReport":
        sev = data.get("overall_prediction_drift_status")
        if isinstance(sev, str):
            sev = DriftSeverity(sev)
        return cls(
            model_version=data["model_version"],
            dataset_row_count=data["dataset_row_count"],
            created_at=data["created_at"],
            overall_prediction_drift_status=sev,
            model_score_drift=ScoreDriftResult.from_dict(data["model_score_drift"]) if isinstance(data["model_score_drift"], dict) else data["model_score_drift"],
            risk_score_drift=RiskScoreDriftResult.from_dict(data["risk_score_drift"]) if isinstance(data["risk_score_drift"], dict) else data["risk_score_drift"],
            tier_drift=CategoricalDistributionDriftResult.from_dict(data["tier_drift"]) if isinstance(data["tier_drift"], dict) else data["tier_drift"],
            action_drift=CategoricalDistributionDriftResult.from_dict(data["action_drift"]) if isinstance(data["action_drift"], dict) else data["action_drift"],
            override_drift=OverrideRateDriftResult.from_dict(data["override_drift"]) if isinstance(data["override_drift"], dict) else data["override_drift"],
            window_type=data.get("window_type"),
            window_start=data.get("window_start"),
            window_end=data.get("window_end"),
        )


@dataclass
class MetricDegradationResult:
    """Degradation metric evaluation comparing observed performance against baseline benchmark."""
    metric_name: str
    observed_value: float
    baseline_value: float
    relative_delta: float
    absolute_delta: float
    severity: DriftSeverity = DriftSeverity.NORMAL
    confidence: MetricConfidence = MetricConfidence.NORMAL_CONFIDENCE
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "observed_value": self.observed_value,
            "baseline_value": self.baseline_value,
            "relative_delta": self.relative_delta,
            "absolute_delta": self.absolute_delta,
            "severity": self.severity.value if isinstance(self.severity, DriftSeverity) else self.severity,
            "confidence": self.confidence.value if isinstance(self.confidence, MetricConfidence) else self.confidence,
            "alert_message": self.alert_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricDegradationResult":
        sev = data.get("severity", DriftSeverity.NORMAL)
        if isinstance(sev, str):
            sev = DriftSeverity(sev)
        conf = data.get("confidence", MetricConfidence.NORMAL_CONFIDENCE)
        if isinstance(conf, str):
            conf = MetricConfidence(conf)
        return cls(
            metric_name=data["metric_name"],
            observed_value=data["observed_value"],
            baseline_value=data["baseline_value"],
            relative_delta=data["relative_delta"],
            absolute_delta=data["absolute_delta"],
            severity=sev,
            confidence=conf,
            alert_message=data.get("alert_message"),
        )


@dataclass
class PerformanceReport:
    """Comprehensive performance monitoring evaluation against delayed ground truth."""
    model_version: str
    dataset_row_count: int
    labeled_sample_count: int
    fraud_cases_count: int
    legitimate_cases_count: int
    suspicious_resolved_count: int
    created_at: str
    overall_performance_status: DriftSeverity
    confidence: MetricConfidence
    operating_threshold: float
    primary_metrics: ThresholdPerformanceMetrics
    comparison_metrics: Optional[ThresholdPerformanceMetrics]
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    operational_metrics: OperationalDecisionMetrics
    degradation_results: Dict[str, MetricDegradationResult]
    active_degradation_alerts: List[MetricDegradationResult]
    window_type: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        prim_d = asdict(self.primary_metrics) if isinstance(self.primary_metrics, ThresholdPerformanceMetrics) else self.primary_metrics
        if isinstance(self.primary_metrics, ThresholdPerformanceMetrics) and isinstance(self.primary_metrics.confusion_matrix, ConfusionMatrixData):
            prim_d["confusion_matrix"] = asdict(self.primary_metrics.confusion_matrix)

        comp_d = None
        if self.comparison_metrics is not None:
            comp_d = asdict(self.comparison_metrics) if isinstance(self.comparison_metrics, ThresholdPerformanceMetrics) else self.comparison_metrics
            if isinstance(self.comparison_metrics, ThresholdPerformanceMetrics) and isinstance(self.comparison_metrics.confusion_matrix, ConfusionMatrixData):
                comp_d["confusion_matrix"] = asdict(self.comparison_metrics.confusion_matrix)

        ops_d = asdict(self.operational_metrics) if isinstance(self.operational_metrics, OperationalDecisionMetrics) else self.operational_metrics

        return {
            "model_version": self.model_version,
            "dataset_row_count": self.dataset_row_count,
            "labeled_sample_count": self.labeled_sample_count,
            "fraud_cases_count": self.fraud_cases_count,
            "legitimate_cases_count": self.legitimate_cases_count,
            "suspicious_resolved_count": self.suspicious_resolved_count,
            "created_at": self.created_at,
            "overall_performance_status": self.overall_performance_status.value if isinstance(self.overall_performance_status, DriftSeverity) else self.overall_performance_status,
            "confidence": self.confidence.value if isinstance(self.confidence, MetricConfidence) else self.confidence,
            "operating_threshold": self.operating_threshold,
            "primary_metrics": prim_d,
            "comparison_metrics": comp_d,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "operational_metrics": ops_d,
            "degradation_results": {k: v.to_dict() if isinstance(v, MetricDegradationResult) else v for k, v in self.degradation_results.items()},
            "active_degradation_alerts": [a.to_dict() if isinstance(a, MetricDegradationResult) else a for a in self.active_degradation_alerts],
            "window_type": self.window_type,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceReport":
        prim_raw = data["primary_metrics"]
        prim_cm = ConfusionMatrixData(**prim_raw["confusion_matrix"]) if isinstance(prim_raw.get("confusion_matrix"), dict) else prim_raw.get("confusion_matrix")
        prim_metrics = ThresholdPerformanceMetrics(
            threshold=prim_raw["threshold"],
            precision=prim_raw["precision"],
            recall=prim_raw["recall"],
            f1=prim_raw["f1"],
            accuracy=prim_raw["accuracy"],
            fpr=prim_raw["fpr"],
            tpr=prim_raw["tpr"],
            confusion_matrix=prim_cm,
        )

        comp_metrics = None
        if data.get("comparison_metrics") is not None:
            comp_raw = data["comparison_metrics"]
            comp_cm = ConfusionMatrixData(**comp_raw["confusion_matrix"]) if isinstance(comp_raw.get("confusion_matrix"), dict) else comp_raw.get("confusion_matrix")
            comp_metrics = ThresholdPerformanceMetrics(
                threshold=comp_raw["threshold"],
                precision=comp_raw["precision"],
                recall=comp_raw["recall"],
                f1=comp_raw["f1"],
                accuracy=comp_raw["accuracy"],
                fpr=comp_raw["fpr"],
                tpr=comp_raw["tpr"],
                confusion_matrix=comp_cm,
            )

        ops_metrics = OperationalDecisionMetrics(**data["operational_metrics"]) if isinstance(data["operational_metrics"], dict) else data["operational_metrics"]
        degradations = {
            k: MetricDegradationResult.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.get("degradation_results", {}).items()
        }
        alerts = [
            MetricDegradationResult.from_dict(a) if isinstance(a, dict) else a
            for a in data.get("active_degradation_alerts", [])
        ]

        status = data.get("overall_performance_status")
        if isinstance(status, str):
            status = DriftSeverity(status)
        conf = data.get("confidence")
        if isinstance(conf, str):
            conf = MetricConfidence(conf)

        return cls(
            model_version=data["model_version"],
            dataset_row_count=data["dataset_row_count"],
            labeled_sample_count=data["labeled_sample_count"],
            fraud_cases_count=data["fraud_cases_count"],
            legitimate_cases_count=data["legitimate_cases_count"],
            suspicious_resolved_count=data["suspicious_resolved_count"],
            created_at=data["created_at"],
            overall_performance_status=status,
            confidence=conf,
            operating_threshold=data["operating_threshold"],
            primary_metrics=prim_metrics,
            comparison_metrics=comp_metrics,
            pr_auc=data.get("pr_auc"),
            roc_auc=data.get("roc_auc"),
            operational_metrics=ops_metrics,
            degradation_results=degradations,
            active_degradation_alerts=alerts,
            window_type=data.get("window_type"),
            window_start=data.get("window_start"),
            window_end=data.get("window_end"),
        )



