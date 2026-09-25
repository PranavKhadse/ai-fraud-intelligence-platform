"""
Comparison Schemas and Data Models for Phase 14.4 Champion vs. Challenger Comparative Evaluation.

Provides strongly typed, measurement-only data models for:
- Neutral mathematical metric deltas (POSITIVE, NEGATIVE, ZERO).
- Side-by-side classification performance comparisons across Validation and OOT.
- Operational decision routing deltas (APPROVE, REVIEW, BLOCK, Queue Purities).
- Inference latency distribution comparisons under standardized timing methodology.
- Immutable, persistent Champion-vs-Candidate comparison results.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from ml.lifecycle.schemas import EvaluationMetricsSummary
from ml.lifecycle.evaluation.schemas import OperationalDecisionMetrics


class DeltaSign(str, Enum):
    """
    Neutral mathematical sign indicator for metric deltas (Candidate minus Champion).
    Strictly measurement-only: no evaluative, ranking, or qualitative connotations.
    """
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    ZERO = "ZERO"


class MetricDelta(BaseModel):
    """
    Neutral mathematical difference between Candidate and Champion for a single metric.
    
    Formula:
        absolute_delta = candidate_value - champion_value
        percentage_point_delta = (candidate_value - champion_value) * 100  (for rates in [0, 1])
        relative_change_pct = ((candidate_value - champion_value) / champion_value) * 100
    """
    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., description="Name of the evaluated metric")
    champion_value: float = Field(..., description="Empirical value from Champion baseline")
    candidate_value: float = Field(..., description="Empirical value from Candidate model")
    absolute_delta: float = Field(..., description="Arithmetic difference (Candidate - Champion)")
    percentage_point_delta: Optional[float] = Field(
        None, description="Percentage-point difference for rate metrics in [0.0, 1.0]"
    )
    relative_change_pct: Optional[float] = Field(
        None, description="Relative percentage change relative to Champion baseline"
    )
    delta_sign: DeltaSign = Field(
        ..., description="Mathematical sign of the delta: POSITIVE (> 0), NEGATIVE (< 0), ZERO (== 0)"
    )


def compute_metric_delta(
    metric_name: str,
    champion_val: Union[float, int],
    candidate_val: Union[float, int],
    is_rate: bool = False,
    epsilon: float = 1e-6,
) -> MetricDelta:
    """
    Compute a strictly neutral MetricDelta container.
    """
    c_val = float(champion_val)
    cand_val = float(candidate_val)
    abs_delta = cand_val - c_val

    # Sign determination with tolerance epsilon
    if abs_delta > epsilon:
        sign = DeltaSign.POSITIVE
    elif abs_delta < -epsilon:
        sign = DeltaSign.NEGATIVE
    else:
        sign = DeltaSign.ZERO

    # Percentage point delta for rates in [0, 1]
    pp_delta: Optional[float] = round(abs_delta * 100.0, 5) if is_rate else None

    # Relative change percentage
    rel_change: Optional[float] = None
    if abs(c_val) > epsilon:
        rel_change = round((abs_delta / abs(c_val)) * 100.0, 5)

    return MetricDelta(
        metric_name=metric_name,
        champion_value=round(c_val, 6),
        candidate_value=round(cand_val, 6),
        absolute_delta=round(abs_delta, 6),
        percentage_point_delta=pp_delta,
        relative_change_pct=rel_change,
        delta_sign=sign,
    )


class ClassificationComparison(BaseModel):
    """
    Side-by-side classification performance and financial cost comparison.
    """
    model_config = ConfigDict(frozen=True)

    champion_metrics: EvaluationMetricsSummary = Field(..., description="Champion evaluation metrics")
    candidate_metrics: EvaluationMetricsSummary = Field(..., description="Candidate evaluation metrics")
    deltas: Dict[str, MetricDelta] = Field(..., description="Neutral metric deltas keyed by metric name")


class OperationalComparison(BaseModel):
    """
    Side-by-side operational decision routing and queue purity comparison.
    """
    model_config = ConfigDict(frozen=True)

    champion_operational: OperationalDecisionMetrics = Field(..., description="Champion operational metrics")
    candidate_operational: OperationalDecisionMetrics = Field(..., description="Candidate operational metrics")
    action_count_deltas: Dict[str, int] = Field(..., description="Count deltas per action (APPROVE, REVIEW, BLOCK)")
    action_percentage_deltas: Dict[str, float] = Field(..., description="Percentage deltas per action")
    fraud_routing_deltas: Dict[str, int] = Field(..., description="Deltas for fraud counts across queues")
    legit_routing_deltas: Dict[str, int] = Field(..., description="Deltas for legitimate counts across queues")
    queue_purity_deltas: Dict[str, MetricDelta] = Field(..., description="Deltas for BLOCK precision & Review purity")
    rule_overridden_delta: int = Field(default=0, description="Delta in business rule overrides")


class LatencyComparison(BaseModel):
    """
    Side-by-side inference latency distribution comparison under identical test conditions.
    """
    model_config = ConfigDict(frozen=True)

    champion_latency: Dict[str, float] = Field(..., description="Champion latency quartiles (mean_ms, p50, p95, p99)")
    candidate_latency: Dict[str, float] = Field(..., description="Candidate latency quartiles (mean_ms, p50, p95, p99)")
    deltas: Dict[str, MetricDelta] = Field(..., description="Neutral latency deltas (Candidate - Champion)")
    warmup_iterations: int = Field(default=5, ge=0, description="Number of unmeasured warm-up iterations")
    measured_samples: int = Field(default=50, ge=0, description="Number of measured single-row inference samples")


class ChampionChallengerComparisonResult(BaseModel):
    """
    Complete, self-contained comparative evaluation artifact for Phase 14.4.
    """
    model_config = ConfigDict(extra="ignore")

    champion_version: str = Field(..., description="Champion semantic version string")
    candidate_version: str = Field(..., description="Candidate semantic version string")
    champion_operating_threshold: float = Field(..., gt=0.0, lt=1.0, description="Authoritative Champion threshold")
    candidate_operating_threshold: float = Field(..., gt=0.0, lt=1.0, description="Frozen Candidate operating threshold")

    validation_comparison: ClassificationComparison = Field(..., description="Validation partition comparison")
    oot_comparison: ClassificationComparison = Field(..., description="Protected OOT partition comparison")
    validation_operational_comparison: OperationalComparison = Field(
        ..., description="Validation operational routing comparison"
    )
    oot_operational_comparison: OperationalComparison = Field(..., description="OOT operational routing comparison")
    latency_comparison: LatencyComparison = Field(..., description="Inference latency distribution comparison")

    dataset_metadata: Dict[str, Any] = Field(..., description="Dataset row counts, features, and partition dates")
    champion_sha256_checksums: Dict[str, str] = Field(..., description="Cryptographic SHA-256 hashes of Champion artifacts")
    candidate_sha256_checksums: Dict[str, str] = Field(..., description="Cryptographic SHA-256 hashes of Candidate artifacts")

    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of comparison execution",
    )
    evaluated_by: str = Field(
        default="champion_challenger_comparator",
        description="System author identifier for the comparative evaluation",
    )

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist comparison result to a JSON artifact file."""
        target_path = Path(filepath).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ChampionChallengerComparisonResult":
        """Load and validate comparison result from a JSON artifact file."""
        target_path = Path(filepath).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Comparison artifact not found at: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
