"""
Evaluation Schemas and Data Models for Phase 14 Candidate Model Evaluation.

Defines strongly typed schemas for:
- Evaluation scopes (Validation, OOT Holdout, Custom).
- Individual evaluation gate validation results.
- End-to-end candidate evaluation outcomes with metrics and gate status.
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


class GateResult(BaseModel):
    """
    Individual evaluation gate verification outcome.
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
    Complete, strongly typed outcome of a Candidate Model Evaluation run.
    """
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Semantic version string of the evaluated candidate")
    model_family: str = Field(..., description="Model algorithm family (strictly 'xgboost')")
    scope: EvaluationScope = Field(default=EvaluationScope.VALIDATION, description="Evaluation dataset scope")
    operating_threshold: float = Field(..., gt=0.0, lt=1.0, description="Probability decision threshold applied")
    
    metrics: EvaluationMetricsSummary = Field(..., description="Comprehensive evaluation metrics summary")
    gate_results: List[GateResult] = Field(default_factory=list, description="Individual gate results")
    overall_passed: bool = Field(..., description="True if ALL evaluated gates passed")
    
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
