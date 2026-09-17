"""
High-Resolution Metric Collector for Phase 10 Benchmarking.

Provides thread-safe and async-safe collection of per-request timing samples, HTTP status
codes, error traces, and component-level micro-timings using nanosecond monotonic clocks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from backend.app.benchmarking.config import TargetSLA
from backend.app.benchmarking.metrics import (
    ComponentLatencyBreakdown,
    LatencyMetrics,
    PipelineStage,
    SLACompliance,
    ThroughputMetrics,
)


@dataclass(frozen=True)
class RequestMetricSample:
    """
    Individual request telemetry sample recorded during a benchmark run.
    """
    request_id: str
    start_time_ns: int
    end_time_ns: int
    latency_ms: float
    status_code: int = 200
    success: bool = True
    error_message: Optional[str] = None
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    timestamp_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate sample values."""
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id cannot be empty or whitespace.")
        if self.latency_ms < 0.0:
            raise ValueError(f"latency_ms cannot be negative, got {self.latency_ms}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to a JSON-serializable dictionary."""
        return {
            "request_id": self.request_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "success": self.success,
            "error_message": self.error_message,
            "stage_latencies_ms": dict(self.stage_latencies_ms),
            "timestamp_iso": self.timestamp_iso,
        }


class MetricCollector:
    """
    Thread-safe and async-safe high-resolution metric collector.
    """

    def __init__(self) -> None:
        """Initialize empty collector with synchronization lock."""
        self._lock = threading.Lock()
        self._samples: List[RequestMetricSample] = []

    @staticmethod
    def start_timer() -> int:
        """
        Capture high-resolution monotonic start timestamp in nanoseconds.
        """
        return time.perf_counter_ns()

    @staticmethod
    def stop_timer(start_ns: int) -> float:
        """
        Calculate elapsed time in milliseconds from a starting nanosecond timestamp.

        Returns:
            Elapsed time in milliseconds, rounded to 4 decimal places (never negative).
        """
        now_ns = time.perf_counter_ns()
        elapsed_ns = max(0, now_ns - start_ns)
        return round(elapsed_ns / 1_000_000.0, 4)

    def record_sample(self, sample: RequestMetricSample) -> None:
        """
        Append a pre-constructed RequestMetricSample in a thread-safe manner.
        """
        if not isinstance(sample, RequestMetricSample):
            raise TypeError(f"Expected RequestMetricSample, got {type(sample).__name__}")
        with self._lock:
            self._samples.append(sample)

    def record_request(
        self,
        request_id: str,
        latency_ms: float,
        status_code: int = 200,
        success: bool = True,
        error_message: Optional[str] = None,
        stage_latencies_ms: Optional[Mapping[str, float]] = None,
    ) -> RequestMetricSample:
        """
        Record a completed transaction request sample.
        """
        now_ns = time.perf_counter_ns()
        latency_clamped = max(0.0, float(latency_ms))
        duration_ns = int(latency_clamped * 1_000_000)
        start_ns = now_ns - duration_ns

        sample = RequestMetricSample(
            request_id=request_id,
            start_time_ns=start_ns,
            end_time_ns=now_ns,
            latency_ms=round(latency_clamped, 4),
            status_code=status_code,
            success=success,
            error_message=error_message,
            stage_latencies_ms=dict(stage_latencies_ms) if stage_latencies_ms else {},
        )
        with self._lock:
            self._samples.append(sample)
        return sample

    def record_error(
        self,
        request_id: str,
        latency_ms: float,
        status_code: int = 500,
        error_message: str = "Internal error",
    ) -> RequestMetricSample:
        """
        Record a failed request with error details.
        """
        return self.record_request(
            request_id=request_id,
            latency_ms=latency_ms,
            status_code=status_code,
            success=False,
            error_message=error_message,
        )

    def get_samples(self) -> List[RequestMetricSample]:
        """
        Retrieve a shallow copy of all recorded samples.
        """
        with self._lock:
            return list(self._samples)

    def get_latencies(self, successful_only: bool = True) -> List[float]:
        """
        Retrieve all collected latencies in milliseconds.
        """
        with self._lock:
            if successful_only:
                return [s.latency_ms for s in self._samples if s.success]
            return [s.latency_ms for s in self._samples]

    def get_stage_latencies(self) -> Dict[str, List[float]]:
        """
        Retrieve collected stage latencies organized by stage name.
        """
        stage_map: Dict[str, List[float]] = {}
        with self._lock:
            for s in self._samples:
                if s.success:
                    for stage, lat in s.stage_latencies_ms.items():
                        stage_map.setdefault(stage, []).append(lat)
        return stage_map

    def count(self) -> int:
        """Total number of recorded samples."""
        with self._lock:
            return len(self._samples)

    def successful_count(self) -> int:
        """Total number of successful requests."""
        with self._lock:
            return sum(1 for s in self._samples if s.success)

    def failed_count(self) -> int:
        """Total number of failed requests."""
        with self._lock:
            return sum(1 for s in self._samples if not s.success)

    def clear(self) -> None:
        """Clear all recorded samples."""
        with self._lock:
            self._samples.clear()

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def compute_summary(
        self,
        elapsed_seconds: Optional[float] = None,
        target_sla: Optional[TargetSLA] = None,
    ) -> Dict[str, Any]:
        """
        Compute a comprehensive summary dictionary containing latency metrics, throughput,
        component breakdowns, and SLA compliance.
        """
        with self._lock:
            samples_copy = list(self._samples)

        total_reqs = len(samples_copy)
        if total_reqs == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "message": "No samples recorded.",
            }

        successful_samples = [s for s in samples_copy if s.success]
        failed_samples = [s for s in samples_copy if not s.success]
        successful_latencies = [s.latency_ms for s in successful_samples]

        # 1. Latency metrics (successful requests)
        latency_metrics: Optional[LatencyMetrics] = None
        if successful_latencies:
            latency_metrics = LatencyMetrics.from_latencies(successful_latencies)

        # 2. Elapsed time calculation
        if elapsed_seconds is None:
            if samples_copy:
                min_start = min(s.start_time_ns for s in samples_copy)
                max_end = max(s.end_time_ns for s in samples_copy)
                elapsed_seconds = max(0.0001, (max_end - min_start) / 1_000_000_000.0)
            else:
                elapsed_seconds = 0.0

        # 3. Throughput metrics
        throughput = ThroughputMetrics.from_counts(
            total_requests=total_reqs,
            successful_requests=len(successful_samples),
            failed_requests=len(failed_samples),
            elapsed_seconds=elapsed_seconds,
        )

        # 4. Component breakdown
        stage_map: Dict[str, List[float]] = {}
        for s in successful_samples:
            for stg, lat in s.stage_latencies_ms.items():
                stage_map.setdefault(stg, []).append(lat)

        component_breakdown: Optional[ComponentLatencyBreakdown] = None
        if stage_map:
            component_breakdown = ComponentLatencyBreakdown.from_stage_latencies(stage_map)

        # 5. SLA Compliance
        sla_results: Optional[SLACompliance] = None
        if latency_metrics and target_sla:
            sla_results = SLACompliance.from_latency_metrics(
                metrics=latency_metrics,
                latencies=successful_latencies,
                target_sla=target_sla,
            )

        return {
            "total_requests": total_reqs,
            "successful_requests": len(successful_samples),
            "failed_requests": len(failed_samples),
            "elapsed_seconds": throughput.elapsed_seconds,
            "latency": latency_metrics.to_dict() if latency_metrics else None,
            "throughput": throughput.to_dict(),
            "component_breakdown": component_breakdown.to_dict() if component_breakdown else None,
            "sla_compliance": sla_results.to_dict() if sla_results else None,
        }
