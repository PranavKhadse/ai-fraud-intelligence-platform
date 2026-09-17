"""
Benchmarking Configuration Module for Phase 10: Real-Time Detection & Benchmarking.

Defines strongly typed, validated configuration parameters for latency profiling,
throughput evaluation, concurrency sweeps, and reference engineering SLA targets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class TargetSLA:
    """
    Configurable engineering reference target for latency SLA validation.

    Note on Governance:
    The platform specification (PROJECT_SPEC.md) does not define a mandatory contractual SLA.
    This structure represents configurable engineering / reference targets for benchmarking
    evaluations and performance diagnostics.
    """
    sla_name: str = "Reference Engineering Target"
    target_p50_ms: Optional[float] = None
    target_p95_ms: Optional[float] = None
    target_p99_ms: Optional[float] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate target latency thresholds."""
        if not self.sla_name or not self.sla_name.strip():
            raise ValueError("sla_name cannot be empty or whitespace.")

        for name, val in [
            ("target_p50_ms", self.target_p50_ms),
            ("target_p95_ms", self.target_p95_ms),
            ("target_p99_ms", self.target_p99_ms),
        ]:
            if val is not None:
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise TypeError(f"{name} must be a numeric float or int, got {type(val).__name__}")
                if val <= 0.0:
                    raise ValueError(f"{name} must be strictly positive (> 0.0), got {val}")

        # Validate monotonic ordering if multiple targets are specified
        if self.target_p50_ms is not None and self.target_p95_ms is not None:
            if self.target_p50_ms > self.target_p95_ms:
                raise ValueError(
                    f"target_p50_ms ({self.target_p50_ms} ms) cannot exceed target_p95_ms ({self.target_p95_ms} ms)."
                )
        if self.target_p95_ms is not None and self.target_p99_ms is not None:
            if self.target_p95_ms > self.target_p99_ms:
                raise ValueError(
                    f"target_p95_ms ({self.target_p95_ms} ms) cannot exceed target_p99_ms ({self.target_p99_ms} ms)."
                )

    @classmethod
    def standard_gateway_target(cls) -> "TargetSLA":
        """
        Standard real-time payment gateway engineering reference target:
        P50 <= 15.0 ms, P95 <= 35.0 ms, P99 <= 60.0 ms.
        """
        return cls(
            sla_name="Standard Payment Gateway Reference Target",
            target_p50_ms=15.0,
            target_p95_ms=35.0,
            target_p99_ms=60.0,
            description="Typical real-time payment gateway sub-100ms authorization target.",
        )

    @classmethod
    def ultra_low_latency_target(cls) -> "TargetSLA":
        """
        Aggressive ultra-low-latency engineering reference target:
        P50 <= 10.0 ms, P95 <= 25.0 ms, P99 <= 50.0 ms.
        """
        return cls(
            sla_name="Ultra-Low Latency Reference Target",
            target_p50_ms=10.0,
            target_p95_ms=25.0,
            target_p99_ms=50.0,
            description="Aggressive low-latency financial risk scoring target.",
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Strongly typed configuration for reproducible fraud detection benchmarks.

    Attributes:
        num_requests: Total measured requests per concurrency evaluation.
        concurrency_levels: Tuple of concurrent workers / clients to evaluate (e.g. (1, 2, 4, 8, 16)).
        warmup_requests: Number of initial warm-up requests executed before measurement window.
        dataset_path: Path to the Parquet dataset partition containing evaluated feature vectors.
        target_rate: Optional offered request rate limiter in transactions per second (TPS).
        include_persistence: Whether database persistence is included in the benchmarked request flow.
        random_seed: Seed for deterministic sampling of transaction feature records.
    """
    num_requests: int = 500
    concurrency_levels: Tuple[int, ...] = (1, 2, 4, 8, 16)
    warmup_requests: int = 50
    dataset_path: Union[str, Path] = "data/processed/features/test_features.parquet"
    target_rate: Optional[float] = None
    include_persistence: bool = True
    random_seed: Optional[int] = 42

    def __post_init__(self) -> None:
        """Validate benchmarking configuration boundaries."""
        # 1. Validate num_requests
        if not isinstance(self.num_requests, int) or isinstance(self.num_requests, bool):
            raise TypeError(f"num_requests must be an integer, got {type(self.num_requests).__name__}")
        if self.num_requests <= 0:
            raise ValueError(f"num_requests must be strictly positive (> 0), got {self.num_requests}")

        # 2. Validate warmup_requests
        if not isinstance(self.warmup_requests, int) or isinstance(self.warmup_requests, bool):
            raise TypeError(f"warmup_requests must be an integer, got {type(self.warmup_requests).__name__}")
        if self.warmup_requests < 0:
            raise ValueError(f"warmup_requests must be non-negative (>= 0), got {self.warmup_requests}")

        # 3. Validate concurrency_levels
        if isinstance(self.concurrency_levels, str) or not isinstance(self.concurrency_levels, (tuple, list, Sequence)):
            raise TypeError(f"concurrency_levels must be a sequence of ints, got {type(self.concurrency_levels).__name__}")
        if len(self.concurrency_levels) == 0:
            raise ValueError("concurrency_levels sequence cannot be empty.")

        cleaned_concurrency: list[int] = []
        for c in self.concurrency_levels:
            if not isinstance(c, int) or isinstance(c, bool):
                raise TypeError(f"All concurrency values must be integers, got {type(c).__name__}")
            if c <= 0:
                raise ValueError(f"Every concurrency level must be positive (> 0), got {c}")
            cleaned_concurrency.append(c)

        # Enforce tuple immutability
        object.__setattr__(self, "concurrency_levels", tuple(cleaned_concurrency))

        # 4. Validate dataset_path
        if not isinstance(self.dataset_path, (str, Path)):
            raise TypeError(f"dataset_path must be a string or Path, got {type(self.dataset_path).__name__}")
        cleaned_path = Path(self.dataset_path)
        object.__setattr__(self, "dataset_path", cleaned_path)

        # 5. Validate target_rate
        if self.target_rate is not None:
            if not isinstance(self.target_rate, (int, float)) or isinstance(self.target_rate, bool):
                raise TypeError(f"target_rate must be a numeric float or int, got {type(self.target_rate).__name__}")
            if self.target_rate <= 0.0:
                raise ValueError(f"target_rate must be strictly positive (> 0.0), got {self.target_rate}")

        # 6. Validate include_persistence
        if not isinstance(self.include_persistence, bool):
            raise TypeError(f"include_persistence must be a boolean, got {type(self.include_persistence).__name__}")

        # 7. Validate random_seed
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
                raise TypeError(f"random_seed must be an integer, got {type(self.random_seed).__name__}")
