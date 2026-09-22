"""
Monitoring Pydantic Schemas for Phase 13: ML & Model Monitoring.

Defines strongly typed, validated request and response contracts for:
- Monitoring health and component breakdown
- 55-Feature drift summaries and detailed single-feature diagnostics
- Prediction score and risk score distribution drift
- 3-Tier model performance and human review queue purity
- Persisted monitoring snapshots and rollup listings
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class MonitoringAlertItem(BaseModel):
    """Derived monitoring alert for a warning or critical drift/degradation condition."""
    model_config = ConfigDict(extra="ignore")

    component: str = Field(..., description="Monitoring component ('data_drift', 'prediction_drift', 'performance').")
    feature_name: Optional[str] = Field(None, description="Feature name if alert is feature-specific.")
    severity: str = Field(..., description="Alert severity ('WARNING', 'CRITICAL').")
    metric_name: str = Field(..., description="Name of the metric triggering the alert.")
    observed_value: float = Field(..., description="Observed metric value in the monitoring window.")
    threshold_value: Optional[float] = Field(None, description="Benchmark or threshold value triggering alert.")
    message: str = Field(..., description="Human-readable alert explanation.")


class MonitoringHealthResponse(BaseModel):
    """High-level operational health and component status breakdown."""
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Evaluated model release version (e.g. '1.0.0').")
    window_type: str = Field(..., description="Time window type ('1h', '24h', '7d', '30d', 'custom', 'HOURLY', 'DAILY').")
    window_start: str = Field(..., description="UTC start timestamp of the evaluation window.")
    window_end: str = Field(..., description="UTC end timestamp of the evaluation window.")
    sample_count: int = Field(..., ge=0, description="Total transactions evaluated in the window.")
    labeled_count: int = Field(..., ge=0, description="Total resolved ground-truth cases evaluated.")
    overall_status: str = Field(..., description="Overall health status ('NORMAL', 'WARNING', 'CRITICAL', 'INSUFFICIENT_DATA').")
    data_drift_status: str = Field(..., description="Feature data drift component status.")
    prediction_drift_status: str = Field(..., description="Prediction & risk score drift component status.")
    performance_status: str = Field(..., description="Ground-truth performance component status.")
    active_alert_count: int = Field(..., ge=0, description="Count of active WARNING and CRITICAL alerts.")
    active_alerts: List[MonitoringAlertItem] = Field(default_factory=list, description="List of active alerts.")
    created_at: str = Field(..., description="Timestamp of evaluation.")


class FeatureDriftItemResponse(BaseModel):
    """Summary metrics for a single feature in the feature drift report."""
    model_config = ConfigDict(extra="ignore")

    feature_name: str = Field(..., description="Name of canonical feature.")
    feature_type: str = Field(..., description="Feature type ('numerical', 'categorical').")
    category: str = Field(..., description="Domain group (e.g. 'amount', 'velocity', 'geographic', 'demographic').")
    status: str = Field(..., description="Drift severity status ('NORMAL', 'WARNING', 'CRITICAL', 'INSUFFICIENT_DATA').")
    psi: float = Field(..., description="Population Stability Index value.")
    ks_statistic: Optional[float] = Field(None, description="Two-sample Kolmogorov-Smirnov statistic for numerical features.")
    ks_p_value: Optional[float] = Field(None, description="Asymptotic p-value for KS statistic.")
    js_divergence: Optional[float] = Field(None, description="Jensen-Shannon Divergence for categorical features.")
    missing_rate_current: float = Field(..., description="Observed missing value rate.")
    missing_rate_delta: float = Field(..., description="Delta from baseline missing rate.")
    unseen_category_rate: Optional[float] = Field(None, description="Proportion of unseen categories observed.")


class FeatureDriftListResponse(BaseModel):
    """Ranked 55-feature drift report response."""
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Model version.")
    window_type: str = Field(..., description="Window type.")
    window_start: str = Field(..., description="Window start.")
    window_end: str = Field(..., description="Window end.")
    sample_count: int = Field(..., ge=0, description="Evaluated transaction count.")
    overall_status: str = Field(..., description="Overall feature drift status.")
    total_features_count: int = Field(..., ge=0, description="Total features monitored (55).")
    drifted_features_count: int = Field(..., ge=0, description="Count of features with WARNING or CRITICAL drift.")
    critical_features_count: int = Field(..., ge=0, description="Count of features with CRITICAL drift.")
    warning_features_count: int = Field(..., ge=0, description="Count of features with WARNING drift.")
    items: List[FeatureDriftItemResponse] = Field(default_factory=list, description="List of feature drift items.")


class FeatureDriftDetailResponse(BaseModel):
    """Comprehensive single-feature drift breakdown with baseline vs observed distributions."""
    model_config = ConfigDict(extra="ignore")

    feature_name: str = Field(..., description="Feature name.")
    feature_type: str = Field(..., description="Feature type ('numerical', 'categorical').")
    category: str = Field(..., description="Feature group category.")
    status: str = Field(..., description="Drift severity status.")
    psi: float = Field(..., description="PSI value.")
    ks_statistic: Optional[float] = Field(None, description="KS statistic.")
    ks_p_value: Optional[float] = Field(None, description="KS asymptotic p-value.")
    js_divergence: Optional[float] = Field(None, description="JSD value.")
    missing_rate_baseline: float = Field(..., description="Baseline missing value rate.")
    missing_rate_current: float = Field(..., description="Current missing value rate.")
    missing_rate_delta: float = Field(..., description="Missing rate delta.")
    unseen_category_rate: Optional[float] = Field(None, description="Unseen category rate.")
    unseen_categories: List[str] = Field(default_factory=list, description="List of newly observed unseen categories.")
    baseline_distribution: Dict[str, float] = Field(default_factory=dict, description="Baseline bin probabilities.")
    current_distribution: Dict[str, float] = Field(default_factory=dict, description="Observed bin probabilities.")
    bin_edges: Optional[List[float]] = Field(None, description="Numerical bin edges if applicable.")


class PredictionDriftResponse(BaseModel):
    """Prediction score, 10-bucket risk score, risk tier, and action distribution drift."""
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Model release version.")
    window_type: str = Field(..., description="Window type.")
    window_start: str = Field(..., description="Window start.")
    window_end: str = Field(..., description="Window end.")
    sample_count: int = Field(..., ge=0, description="Sample count.")
    overall_status: str = Field(..., description="Overall prediction drift status.")
    model_score_psi: float = Field(..., description="PSI on continuous model scores.")
    model_score_status: str = Field(..., description="Model score drift status.")
    model_score_mean: float = Field(..., description="Mean model score.")
    model_score_std: float = Field(..., description="Standard deviation of model scores.")
    risk_score_psi: float = Field(..., description="PSI across the 10 standardized risk score buckets.")
    risk_score_status: str = Field(..., description="Risk score bucket drift status.")
    risk_tier_jsd: float = Field(..., description="JSD across risk tiers (LOW, MEDIUM, HIGH, CRITICAL).")
    risk_tier_status: str = Field(..., description="Risk tier drift status.")
    action_jsd: float = Field(..., description="JSD across decision actions (APPROVE, REVIEW, BLOCK).")
    action_status: str = Field(..., description="Decision action drift status.")
    override_rate_current: float = Field(..., description="Proportion of rule-overridden decisions.")
    override_rate_baseline: float = Field(..., description="Baseline override rate.")
    override_rate_delta: float = Field(..., description="Override rate delta.")
    model_score_distribution: Dict[str, float] = Field(default_factory=dict, description="Observed model score 10-bin distribution.")
    risk_score_buckets_distribution: Dict[str, float] = Field(default_factory=dict, description="Observed 10-bucket risk score distribution.")
    risk_tier_distribution: Dict[str, float] = Field(default_factory=dict, description="Observed risk tier distribution.")
    action_distribution: Dict[str, float] = Field(default_factory=dict, description="Observed action distribution.")
    active_alerts: List[MonitoringAlertItem] = Field(default_factory=list, description="Active prediction alerts.")


class ConfusionMatrixResponse(BaseModel):
    """Confusion matrix counts."""
    model_config = ConfigDict(extra="ignore")

    tp: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)
    tn: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


class ThresholdMetricsResponse(BaseModel):
    """Metrics evaluated at a specific threshold operating point."""
    model_config = ConfigDict(extra="ignore")

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr: float
    tpr: float
    confusion_matrix: ConfusionMatrixResponse


class OperationalMetricsResponse(BaseModel):
    """Operational hybrid decision and review queue metrics."""
    model_config = ConfigDict(extra="ignore")

    decision_precision_block: float
    decision_recall_intervention: float
    review_queue_purity: float
    total_reviews_count: int
    fraud_in_review_count: int
    total_blocks_count: int
    fraud_in_block_count: int


class MetricDegradationItemResponse(BaseModel):
    """Individual metric degradation evaluation result."""
    model_config = ConfigDict(extra="ignore")

    metric_name: str
    observed_value: float
    baseline_value: float
    relative_delta: float
    absolute_delta: float
    severity: str
    confidence: str
    alert_message: Optional[str] = None


class ModelPerformanceResponse(BaseModel):
    """Ground-truth model, operational decision, and review queue performance report."""
    model_config = ConfigDict(extra="ignore")

    model_version: str
    window_type: str
    window_start: str
    window_end: str
    dataset_row_count: int
    labeled_sample_count: int
    fraud_cases_count: int
    legitimate_cases_count: int
    suspicious_resolved_count: int
    overall_performance_status: str
    confidence: str
    operating_threshold: float
    primary_metrics: ThresholdMetricsResponse
    comparison_metrics: ThresholdMetricsResponse
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    operational_metrics: OperationalMetricsResponse
    degradation_results: Dict[str, MetricDegradationItemResponse] = Field(default_factory=dict)
    active_degradation_alerts: List[MonitoringAlertItem] = Field(default_factory=list)


class MonitoringSnapshotItemResponse(BaseModel):
    """Summary representation of a persisted monitoring snapshot."""
    model_config = ConfigDict(extra="ignore")

    id: str
    model_version: str
    window_type: str
    window_start: str
    window_end: str
    sample_count: int
    labeled_count: int
    overall_status: str
    data_drift_status: str
    prediction_drift_status: str
    performance_status: str
    created_at: str


class MonitoringSnapshotListResponse(BaseModel):
    """Paginated listing of persisted monitoring snapshots."""
    model_config = ConfigDict(extra="ignore")

    total_count: int
    limit: int
    offset: int
    items: List[MonitoringSnapshotItemResponse] = Field(default_factory=list)
