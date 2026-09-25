"""
Evaluation Schemas and Data Models for Phase 14 Candidate Model Evaluation & Threshold Analysis.

Defines strongly typed schemas for:
- Evaluation scopes (Validation, OOT Holdout, Custom).
- Threshold sweep points and deterministic selection outcomes.
- Operational decision metrics via existing RiskEvaluator and DecisionPolicyEngine.
- Standalone candidate evaluation outcomes with metrics.
- Full-pipeline evaluation result (Validation -> Threshold tau* -> Frozen OOT).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from ml.lifecycle.schemas import EvaluationMetricsSummary


class EvaluationScope(str, Enum):
    """Dataset partition and governance scope for candidate evaluation."""
    VALIDATION = "VALIDATION"
    OOT_HOLDOUT = "OOT_HOLDOUT"
    CUSTOM = "CUSTOM"


class ThresholdOptimizationObjective(str, Enum):
    """Objective criterion for candidate decision threshold selection."""
    MIN_EXPECTED_COST = "MIN_EXPECTED_COST"
    MAX_F1 = "MAX_F1"
    TARGET_RECALL_80 = "TARGET_RECALL_80"
    TARGET_RECALL_90 = "TARGET_RECALL_90"


class ThresholdSweepPoint(BaseModel):
    """Metrics evaluated at an individual candidate threshold step."""
    model_config = ConfigDict(frozen=True)

    threshold: float = Field(..., gt=0.0, lt=1.0)
    expected_cost: float = Field(..., ge=0.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    fpr: float = Field(..., ge=0.0, le=1.0)
    tp: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)
    tn: int = Field(..., ge=0)


class ThresholdSelectionResult(BaseModel):
    """Outcome of validation threshold sweep and deterministic threshold selection."""
    model_config = ConfigDict(frozen=True)

    selected_threshold: float = Field(..., gt=0.0, lt=1.0, description="Selected optimal operating threshold tau*")
    optimization_objective: ThresholdOptimizationObjective = Field(
        default=ThresholdOptimizationObjective.MIN_EXPECTED_COST,
        description="Objective criterion used for threshold selection",
    )
    best_cost: float = Field(..., ge=0.0, description="Expected financial cost at selected threshold")
    best_f1: float = Field(..., ge=0.0, le=1.0, description="F1-score at selected threshold")
    best_recall: float = Field(..., ge=0.0, le=1.0, description="Recall at selected threshold")
    best_precision: float = Field(..., ge=0.0, le=1.0, description="Precision at selected threshold")
    sweep_table: List[Dict[str, Any]] = Field(default_factory=list, description="Summary table of full threshold sweep")
    tie_breaking_notes: Optional[str] = Field(default=None, description="Notes on tie-breaking applied during selection")
    evaluated_samples: int = Field(..., ge=0, description="Total samples in validation sweep")
    positive_fraud_count: int = Field(..., ge=0, description="Total fraud positive samples in validation sweep")


class OperationalDecisionMetrics(BaseModel):
    """
    Operational routing and business metrics extracted via existing RiskEvaluator,
    DecisionPolicyEngine, and RuleEngine.
    """
    model_config = ConfigDict(frozen=True)

    action_counts: Dict[str, int] = Field(..., description="Counts per decision action: APPROVE, REVIEW, BLOCK")
    action_percentages: Dict[str, float] = Field(..., description="Percentages per decision action")
    risk_tier_counts: Dict[str, int] = Field(..., description="Counts per risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    risk_tier_percentages: Dict[str, float] = Field(..., description="Percentages per risk tier")
    model_score_stats: Dict[str, float] = Field(..., description="Summary statistics (min, mean, max) of raw scores")
    risk_score_stats: Dict[str, float] = Field(..., description="Summary statistics (min, mean, max) of risk scores")
    policy_mode: str = Field(..., description="Operating policy mode (TRI_TIER or BINARY_AUTO)")
    operating_threshold: float = Field(..., gt=0.0, lt=1.0, description="Operating threshold applied")
    
    # Ground-truth cross-referenced routing metrics
    fraud_in_block_count: int = Field(default=0, ge=0, description="Positive frauds routed to auto-BLOCK")
    fraud_in_review_count: int = Field(default=0, ge=0, description="Positive frauds routed to manual REVIEW")
    fraud_in_approve_count: int = Field(default=0, ge=0, description="Positive frauds erroneously APPROVED (FN)")
    legit_in_block_count: int = Field(default=0, ge=0, description="Legitimate transactions auto-BLOCKED (FP)")
    legit_in_review_count: int = Field(default=0, ge=0, description="Legitimate transactions routed to REVIEW")
    legit_in_approve_count: int = Field(default=0, ge=0, description="Legitimate transactions APPROVED (TN)")
    
    review_queue_purity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision of manual review queue")
    block_precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision of hard auto-block tier")
    rule_overridden_count: int = Field(default=0, ge=0, description="Transactions where a rule overrode ML policy action")
    rule_trigger_counts: Dict[str, int] = Field(default_factory=dict, description="Trigger counts per business rule")


class GateResult(BaseModel):
    """
    Individual evaluation gate verification outcome (for standalone gate checks if invoked).
    """
    model_config = ConfigDict(frozen=True)

    gate_name: str = Field(..., description="Unique descriptive identifier for the evaluated gate")
    metric_name: str = Field(..., description="Target metric key evaluated")
    operator: str = Field(..., description="Comparison operator: '>=', '<=', '>', '<', '=='")
    target_value: float = Field(..., description="Configured gate passing threshold")
    actual_value: float = Field(..., description="Empirical candidate model metric achieved")
    passed: bool = Field(..., description="True if actual_value satisfies operator against target_value")
    description: str = Field(..., description="Human-readable explanation of the gate criteria")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra diagnostic details")


class CandidateEvaluationResult(BaseModel):
    """
    Outcome of a Candidate Model Evaluation run on a single dataset partition.
    """
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Semantic version string of the evaluated candidate")
    model_family: str = Field(default="xgboost", description="Model algorithm family (strictly 'xgboost')")
    scope: EvaluationScope = Field(default=EvaluationScope.VALIDATION, description="Evaluation dataset scope")
    operating_threshold: float = Field(..., gt=0.0, lt=1.0, description="Probability decision threshold applied")
    
    metrics: EvaluationMetricsSummary = Field(..., description="Comprehensive evaluation metrics summary")
    gate_results: List[GateResult] = Field(default_factory=list, description="Individual gate results")
    overall_passed: bool = Field(default=True, description="True if evaluated gates passed")
    
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of evaluation",
    )
    evaluated_by: Optional[str] = Field(default="candidate_evaluator", description="System/analyst author of evaluation")
    
    bundle_directory: Optional[str] = Field(default=None, description="Filesystem directory of the evaluated bundle")
    dataset_path: Optional[str] = Field(default=None, description="Path to evaluation partition dataset")
    sample_count: int = Field(..., ge=0, description="Total number of evaluated samples")
    fraud_count: int = Field(..., ge=0, description="Number of positive ground truth fraud samples")
    
    latency_stats: Optional[Dict[str, float]] = Field(
        default=None,
        description="Inference latency distribution metrics (mean_ms, p50_ms, p95_ms, p99_ms)",
    )


class FullCandidateEvaluationResult(BaseModel):
    """
    Complete, end-to-end outcome of Phase 14.3 Full-Pipeline Candidate Evaluation:
    Validation Inference -> Threshold Sweep -> Select & Freeze tau* -> Protected OOT Evaluation.
    """
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Semantic version string of evaluated candidate")
    model_family: str = Field(default="xgboost", description="Model algorithm family (strictly 'xgboost')")
    selected_operating_threshold: float = Field(..., gt=0.0, lt=1.0, description="Frozen operating threshold tau*")
    
    threshold_selection: ThresholdSelectionResult = Field(
        ..., description="Full validation threshold sweep and selection result"
    )
    validation_metrics: EvaluationMetricsSummary = Field(
        ..., description="Performance and cost metrics on validation partition evaluated at tau*"
    )
    validation_operational_metrics: Optional[OperationalDecisionMetrics] = Field(
        default=None, description="Operational routing metrics on validation partition via RiskEvaluator"
    )
    oot_metrics: EvaluationMetricsSummary = Field(
        ..., description="Performance and cost metrics on protected OOT partition evaluated strictly at frozen tau*"
    )
    oot_operational_metrics: Optional[OperationalDecisionMetrics] = Field(
        default=None, description="Operational routing metrics on protected OOT partition via RiskEvaluator at tau*"
    )
    
    latency_stats: Optional[Dict[str, float]] = Field(
        default=None, description="Validation inference latency distribution percentiles"
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of full evaluation pipeline execution",
    )
    evaluated_by: Optional[str] = Field(
        default="full_candidate_evaluator", description="Actor or service authoring evaluation"
    )
    bundle_directory: Optional[str] = Field(default=None, description="Path to evaluated candidate bundle")
    validation_dataset_path: Optional[str] = Field(default=None, description="Path to validation dataset partition")
    oot_dataset_path: Optional[str] = Field(default=None, description="Path to protected OOT dataset partition")
    validation_sample_count: int = Field(..., ge=0, description="Total samples in validation partition")
    validation_fraud_count: int = Field(..., ge=0, description="Total frauds in validation partition")
    oot_sample_count: int = Field(..., ge=0, description="Total samples in protected OOT partition")
    oot_fraud_count: int = Field(..., ge=0, description="Total frauds in protected OOT partition")
