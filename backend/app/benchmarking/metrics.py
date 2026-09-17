"""
Mathematical Latency and Throughput Metrics Engine for Phase 10 Benchmarking.

Provides exact percentile calculations (NumPy linear interpolation), comprehensive
statistical aggregations (sample standard deviation, IQR, min, mean, max), throughput
accounting (offered vs completed vs successful TPS), seven-stage component breakdown,
and configurable engineering SLA compliance validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from backend.app.benchmarking.config import TargetSLA


class PipelineStage(str, Enum):
    """
    Seven distinct lifecycle stages of the real-time fraud detection pipeline.
    """
    REQUEST_VALIDATION = "request_validation"
    FEATURE_PREPARATION = "feature_preparation"
    ML_INFERENCE = "ml_inference"
    TREESHAP_EXPLAINABILITY = "treeshap_explainability"
    RULE_ENGINE_POLICY = "rule_engine_policy"
    PERSISTENCE_MAPPING = "persistence_mapping"
    POSTGRES_PERSISTENCE = "postgres_persistence"


DEFAULT_PERCENTILES: Tuple[float, ...] = (0.0, 50.0, 90.0, 95.0, 99.0, 99.9, 100.0)


def calculate_percentiles(
    latencies: Sequence[float],
    percentiles: Sequence[float] = (0.0, 50.0, 90.0, 95.0, 99.0, 99.9, 100.0),
    method: str = "linear",
) -> Dict[float, float]:
    """
    Calculate exact percentiles for a sequence of latency measurements using NumPy.

    Args:
        latencies: Sequence of numeric latency values in milliseconds.
        percentiles: Sequence of percentile points in [0.0, 100.0].
        method: Interpolation method for NumPy percentile ('linear', 'lower', 'higher', 'midpoint', 'nearest').

    Returns:
        Dict mapping each requested percentile (float) to its calculated latency value (float).

    Raises:
        ValueError: If latencies sequence is empty or if any percentile is outside [0.0, 100.0].
        TypeError: If latencies contains non-numeric values.
    """
    if latencies is None or len(latencies) == 0:
        raise ValueError("Cannot calculate percentiles on an empty or None latency sequence.")

    # Validate percentiles
    validated_percentiles: List[float] = []
    for p in percentiles:
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError(f"Percentile must be a numeric float or int, got {type(p).__name__}")
        p_float = float(p)
        if p_float < 0.0 or p_float > 100.0:
            raise ValueError(f"Percentile value {p_float} is out of valid range [0.0, 100.0].")
        validated_percentiles.append(p_float)

    # Convert latencies to NumPy float array
    try:
        arr = np.asarray(latencies, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(f"All latency elements must be numeric: {e}") from e

    if arr.size == 0:
        raise ValueError("Cannot calculate percentiles on an empty latency array.")

    # Calculate percentiles with explicit linear interpolation
    computed_vals = np.percentile(arr, validated_percentiles, method=method)

    if np.isscalar(computed_vals):
        return {validated_percentiles[0]: round(float(computed_vals), 6)}

    return {
        p: round(float(val), 6)
        for p, val in zip(validated_percentiles, computed_vals)
    }


@dataclass(frozen=True)
class LatencyMetrics:
    """
    Comprehensive statistical metrics for measured latency distributions.

    Standard Deviation Convention:
    Uses Bessel's correction for sample standard deviation (`ddof=1`) when sample count > 1.
    For count == 1, sample standard deviation is defined as 0.0.
    """
    count: int
    min_ms: float
    mean_ms: float
    median_ms: float
    max_ms: float
    std_ms: float
    iqr_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p99_9_ms: float

    @classmethod
    def from_latencies(
        cls,
        latencies: Sequence[float],
        method: str = "linear",
    ) -> "LatencyMetrics":
        """
        Construct LatencyMetrics from an empirical sequence of latency measurements.

        Args:
            latencies: Sequence of numeric latency samples in milliseconds.
            method: Percentile calculation interpolation method (default: 'linear').
        """
        if latencies is None or len(latencies) == 0:
            raise ValueError("Cannot construct LatencyMetrics from an empty latency sequence.")

        arr = np.asarray(latencies, dtype=np.float64)
        n = len(arr)

        pcts = calculate_percentiles(
            arr,
            percentiles=[25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9],
            method=method,
        )

        min_val = round(float(np.min(arr)), 6)
        max_val = round(float(np.max(arr)), 6)
        mean_val = round(float(np.mean(arr)), 6)
        median_val = pcts[50.0]
        std_val = round(float(np.std(arr, ddof=1)), 6) if n > 1 else 0.0
        iqr_val = round(float(pcts[75.0] - pcts[25.0]), 6)

        return cls(
            count=n,
            min_ms=min_val,
            mean_ms=mean_val,
            median_ms=median_val,
            max_ms=max_val,
            std_ms=std_val,
            iqr_ms=iqr_val,
            p50_ms=pcts[50.0],
            p90_ms=pcts[90.0],
            p95_ms=pcts[95.0],
            p99_ms=pcts[99.0],
            p99_9_ms=pcts[99.9],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "count": self.count,
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "max_ms": self.max_ms,
            "std_ms": self.std_ms,
            "iqr_ms": self.iqr_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "p99_9_ms": self.p99_9_ms,
        }


@dataclass(frozen=True)
class ThroughputMetrics:
    """
    Throughput accounting metrics distinguishing offered rate, completed throughput, and successful throughput.

    Definitions:
    - offered_tps: total_requests / elapsed_seconds (Rate of offered traffic)
    - completed_tps: (successful_requests + failed_requests) / elapsed_seconds (Total completed traffic rate)
    - successful_tps: successful_requests / elapsed_seconds (Error-free throughput rate)
    - error_rate_percent: (failed_requests / total_requests) * 100.0
    """
    total_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_seconds: float
    offered_tps: float
    completed_tps: float
    successful_tps: float
    error_rate_percent: float

    @classmethod
    def from_counts(
        cls,
        total_requests: int,
        successful_requests: int,
        failed_requests: int,
        elapsed_seconds: float,
    ) -> "ThroughputMetrics":
        """
        Construct ThroughputMetrics with strict zero-division guards.
        """
        if total_requests < 0:
            raise ValueError(f"total_requests cannot be negative, got {total_requests}")
        if successful_requests < 0:
            raise ValueError(f"successful_requests cannot be negative, got {successful_requests}")
        if failed_requests < 0:
            raise ValueError(f"failed_requests cannot be negative, got {failed_requests}")
        if elapsed_seconds < 0.0:
            raise ValueError(f"elapsed_seconds cannot be negative, got {elapsed_seconds}")

        completed = successful_requests + failed_requests

        if elapsed_seconds > 0.0:
            offered_tps = round(total_requests / elapsed_seconds, 2)
            completed_tps = round(completed / elapsed_seconds, 2)
            successful_tps = round(successful_requests / elapsed_seconds, 2)
        else:
            offered_tps = 0.0
            completed_tps = 0.0
            successful_tps = 0.0

        if total_requests > 0:
            error_rate = round((failed_requests / total_requests) * 100.0, 4)
        else:
            error_rate = 0.0

        return cls(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            elapsed_seconds=round(elapsed_seconds, 4),
            offered_tps=offered_tps,
            completed_tps=completed_tps,
            successful_tps=successful_tps,
            error_rate_percent=error_rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert throughput metrics to a JSON-serializable dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "elapsed_seconds": self.elapsed_seconds,
            "offered_tps": self.offered_tps,
            "completed_tps": self.completed_tps,
            "successful_tps": self.successful_tps,
            "error_rate_percent": self.error_rate_percent,
        }


@dataclass(frozen=True)
class ComponentLatencyBreakdown:
    """
    Seven-stage component latency breakdown and contribution analysis.

    Note on Architectural Decomposition:
    Component-level isolated measurements may differ slightly from end-to-end request latency
    due to micro-profiling overhead, async context switching, HTTP framing, and network serialization.
    """
    stage_metrics: Dict[Union[PipelineStage, str], LatencyMetrics]
    stage_percentages: Dict[Union[PipelineStage, str], float]
    total_component_mean_ms: float

    @classmethod
    def from_stage_latencies(
        cls,
        stage_latencies: Mapping[Union[PipelineStage, str], Sequence[float]],
    ) -> "ComponentLatencyBreakdown":
        """
        Aggregate individual component latency distributions into a unified breakdown.
        """
        metrics_dict: Dict[Union[PipelineStage, str], LatencyMetrics] = {}

        for raw_stage, samples in stage_latencies.items():
            try:
                stage_key: Union[PipelineStage, str] = (
                    PipelineStage(raw_stage) if not isinstance(raw_stage, PipelineStage) else raw_stage
                )
            except ValueError:
                stage_key = str(raw_stage)

            if samples and len(samples) > 0:
                metrics_dict[stage_key] = LatencyMetrics.from_latencies(samples)

        total_mean = sum(m.mean_ms for m in metrics_dict.values())
        percentages: Dict[Union[PipelineStage, str], float] = {}

        for stage_key, m in metrics_dict.items():
            if total_mean > 0.0:
                percentages[stage_key] = round((m.mean_ms / total_mean) * 100.0, 2)
            else:
                percentages[stage_key] = 0.0

        return cls(
            stage_metrics=metrics_dict,
            stage_percentages=percentages,
            total_component_mean_ms=round(total_mean, 4),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert component breakdown to a JSON-serializable dictionary."""
        return {
            "total_component_mean_ms": self.total_component_mean_ms,
            "stages": {
                (stage.value if isinstance(stage, PipelineStage) else str(stage)): {
                    "metrics": metrics.to_dict(),
                    "percentage_of_total": self.stage_percentages.get(stage, 0.0),
                }
                for stage, metrics in self.stage_metrics.items()
            },
        }


@dataclass(frozen=True)
class SLACompliance:
    """
    Evaluates measured latency distributions against configurable reference engineering targets.
    """
    target_sla: TargetSLA
    p50_compliant: Optional[bool]
    p95_compliant: Optional[bool]
    p99_compliant: Optional[bool]
    overall_compliant: bool
    under_15ms_pct: float
    under_25ms_pct: float
    under_50ms_pct: float
    under_100ms_pct: float

    @classmethod
    def from_latency_metrics(
        cls,
        metrics: LatencyMetrics,
        latencies: Sequence[float],
        target_sla: TargetSLA,
    ) -> "SLACompliance":
        """
        Evaluate compliance against target SLA thresholds.
        """
        arr = np.asarray(latencies, dtype=np.float64)
        n = len(arr)

        if n > 0:
            under_15 = round(float(np.sum(arr <= 15.0) / n) * 100.0, 2)
            under_25 = round(float(np.sum(arr <= 25.0) / n) * 100.0, 2)
            under_50 = round(float(np.sum(arr <= 50.0) / n) * 100.0, 2)
            under_100 = round(float(np.sum(arr <= 100.0) / n) * 100.0, 2)
        else:
            under_15 = under_25 = under_50 = under_100 = 0.0

        p50_ok = (metrics.p50_ms <= target_sla.target_p50_ms) if target_sla.target_p50_ms is not None else None
        p95_ok = (metrics.p95_ms <= target_sla.target_p95_ms) if target_sla.target_p95_ms is not None else None
        p99_ok = (metrics.p99_ms <= target_sla.target_p99_ms) if target_sla.target_p99_ms is not None else None

        active_checks = [c for c in (p50_ok, p95_ok, p99_ok) if c is not None]
        overall = all(active_checks) if active_checks else True

        return cls(
            target_sla=target_sla,
            p50_compliant=p50_ok,
            p95_compliant=p95_ok,
            p99_compliant=p99_ok,
            overall_compliant=overall,
            under_15ms_pct=under_15,
            under_25ms_pct=under_25,
            under_50ms_pct=under_50,
            under_100ms_pct=under_100,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert SLA compliance results to a JSON-serializable dictionary."""
        return {
            "sla_name": self.target_sla.sla_name,
            "overall_compliant": self.overall_compliant,
            "p50_compliant": self.p50_compliant,
            "p95_compliant": self.p95_compliant,
            "p99_compliant": self.p99_compliant,
            "target_thresholds": {
                "target_p50_ms": self.target_sla.target_p50_ms,
                "target_p95_ms": self.target_sla.target_p95_ms,
                "target_p99_ms": self.target_sla.target_p99_ms,
            },
            "distribution_compliance_pct": {
                "under_15ms": self.under_15ms_pct,
                "under_25ms": self.under_25ms_pct,
                "under_50ms": self.under_50ms_pct,
                "under_100ms": self.under_100ms_pct,
            },
        }
