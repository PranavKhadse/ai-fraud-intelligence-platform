"""
Benchmark Suite & Multi-Scenario Orchestrator for Phase 10.

Coordinates full empirical performance benchmarking across:
1. Cold-start latency measurement and unmeasured warm-up isolation.
2. Multi-concurrency matrix sweeps across C in {1, 2, 4, 8, 16}.
3. Multi-tier persistence ablation (Full API + DB vs In-Memory vs Idempotent Replay).
4. Seven-stage component latency decomposition and bottleneck diagnostics.
5. Reference engineering target SLA compliance verification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

from httpx import AsyncClient

from backend.app.benchmarking.collector import MetricCollector
from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.metrics import (
    ComponentLatencyBreakdown,
    LatencyMetrics,
    PipelineStage,
    SLACompliance,
    ThroughputMetrics,
)
from backend.app.benchmarking.profiler import PipelineProfiler, ProfileResult
from backend.app.benchmarking.runner import (
    BenchmarkRunner,
    BenchmarkRunResult,
    IdempotencyReplayResult,
)
from backend.app.services.persistence_service import FraudPersistenceService
from backend.app.services.risk_service import RiskService, get_risk_service

logger = logging.getLogger("fraud_api.benchmark_suite")


@dataclass(frozen=True)
class ConcurrencySweepResult:
    """
    Structured outcome of a multi-concurrency sweep across multiple worker levels.
    """
    concurrency_results: Dict[int, BenchmarkRunResult]
    baseline_concurrency: int
    baseline_tps: float
    peak_tps: float
    peak_concurrency: int
    scaling_ratios: Dict[int, float]
    scaling_efficiencies: Dict[int, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert sweep result to a JSON-serializable dictionary."""
        return {
            "baseline_concurrency": self.baseline_concurrency,
            "baseline_tps": round(self.baseline_tps, 2),
            "peak_tps": round(self.peak_tps, 2),
            "peak_concurrency": self.peak_concurrency,
            "scaling_ratios": {str(k): round(v, 3) for k, v in self.scaling_ratios.items()},
            "scaling_efficiencies": {str(k): round(v, 2) for k, v in self.scaling_efficiencies.items()},
            "concurrency_results": {
                str(k): v.to_dict() for k, v in self.concurrency_results.items()
            },
        }


@dataclass(frozen=True)
class AblationResult:
    """
    Quantitative outcome of multi-tier persistence ablation analysis.

    Compares:
    - Mode A: Full API + PostgreSQL Persistence
    - Mode B: In-Memory ML Pipeline (Zero DB)
    - Mode C: Idempotent Database Replay Fast Path
    """
    mode_a_full_api: LatencyMetrics
    mode_b_in_memory: LatencyMetrics
    mode_c_idempotent_replay: LatencyMetrics
    persistence_overhead_ms: float
    persistence_overhead_pct: float
    replay_speedup_factor: float
    replay_latency_reduction_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert ablation result to a JSON-serializable dictionary."""
        return {
            "mode_a_full_api": self.mode_a_full_api.to_dict(),
            "mode_b_in_memory": self.mode_b_in_memory.to_dict(),
            "mode_c_idempotent_replay": self.mode_c_idempotent_replay.to_dict(),
            "persistence_overhead_ms": round(self.persistence_overhead_ms, 3),
            "persistence_overhead_pct": round(self.persistence_overhead_pct, 2),
            "replay_speedup_factor": round(self.replay_speedup_factor, 2),
            "replay_latency_reduction_pct": round(self.replay_latency_reduction_pct, 2),
        }


@dataclass(frozen=True)
class BottleneckAnalysisResult:
    """
    Detailed diagnostic ranking and breakdown of pipeline latency constraints.
    """
    component_breakdown: ComponentLatencyBreakdown
    primary_bottleneck_stage: str
    primary_bottleneck_mean_ms: float
    primary_bottleneck_pct: float
    secondary_bottleneck_stage: str
    secondary_bottleneck_mean_ms: float
    secondary_bottleneck_pct: float
    diagnostic_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert bottleneck analysis to a JSON-serializable dictionary."""
        return {
            "component_breakdown": self.component_breakdown.to_dict(),
            "primary_bottleneck": {
                "stage": self.primary_bottleneck_stage,
                "mean_ms": round(self.primary_bottleneck_mean_ms, 3),
                "contribution_pct": round(self.primary_bottleneck_pct, 2),
            },
            "secondary_bottleneck": {
                "stage": self.secondary_bottleneck_stage,
                "mean_ms": round(self.secondary_bottleneck_mean_ms, 3),
                "contribution_pct": round(self.secondary_bottleneck_pct, 2),
            },
            "diagnostic_summary": self.diagnostic_summary,
        }


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    """
    Complete composite output of a multi-scenario benchmark execution.
    """
    suite_id: str
    config: BenchmarkConfig
    cold_start_latency_ms: Optional[float]
    warmup_requests_executed: int
    warmup_latency: Optional[LatencyMetrics]
    concurrency_sweep: Optional[ConcurrencySweepResult]
    ablation: Optional[AblationResult]
    bottleneck_analysis: Optional[BottleneckAnalysisResult]
    target_sla_evaluated: Optional[TargetSLA]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert composite suite result to a JSON-serializable dictionary."""
        return {
            "suite_id": self.suite_id,
            "timestamp_utc": self.timestamp_utc,
            "cold_start_latency_ms": (
                round(self.cold_start_latency_ms, 3) if self.cold_start_latency_ms is not None else None
            ),
            "warmup_requests_executed": self.warmup_requests_executed,
            "warmup_latency": self.warmup_latency.to_dict() if self.warmup_latency else None,
            "concurrency_sweep": self.concurrency_sweep.to_dict() if self.concurrency_sweep else None,
            "ablation": self.ablation.to_dict() if self.ablation else None,
            "bottleneck_analysis": self.bottleneck_analysis.to_dict() if self.bottleneck_analysis else None,
            "target_sla_name": self.target_sla_evaluated.sla_name if self.target_sla_evaluated else None,
        }


class BenchmarkSuite:
    """
    Benchmark Suite Orchestrator coordinating full multi-scenario evaluations.

    Responsibilities:
    - Measure cold-start latency on initial request.
    - Prime system with unmeasured warm-up requests.
    - Execute multi-concurrency matrix sweeps (C in {1, 2, 4, 8, 16}).
    - Execute persistence ablation (Full API vs In-Memory vs Idempotent Replay).
    - Execute 7-stage component micro-profiling and bottleneck diagnostics.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        runner: Optional[BenchmarkRunner] = None,
        profiler: Optional[PipelineProfiler] = None,
        target_sla: Optional[TargetSLA] = None,
    ) -> None:
        """
        Initialize the BenchmarkSuite with configuration and component runners.

        Args:
            config: BenchmarkConfig instance. Defaults to BenchmarkConfig().
            runner: BenchmarkRunner instance. If None, constructed from config.
            profiler: PipelineProfiler instance. If None, constructed with default services.
            target_sla: Optional TargetSLA reference targets.
        """
        self.config = config or BenchmarkConfig()
        self.runner = runner or BenchmarkRunner(config=self.config)
        self.profiler = profiler or PipelineProfiler()
        self.target_sla = target_sla

    async def measure_cold_start(
        self,
        client: Optional[AsyncClient] = None,
        prefix: str = "cold_start",
    ) -> float:
        """
        Execute and measure the latency of a single cold-start request.

        Returns:
            Cold-start latency in milliseconds.
        """
        logger.info("Measuring cold-start transaction request latency...")
        cold_payload = self.runner.generate_payloads(count=1, prefix=prefix)
        result = await self.runner.run(
            num_requests=1,
            concurrency=1,
            client=client,
            payloads=cold_payload,
            prefix=prefix,
        )
        cold_ms = result.latency.mean_ms
        logger.info(f"Cold-start transaction latency: {cold_ms:.3f} ms")
        return cold_ms

    async def execute_warmup(
        self,
        warmup_requests: Optional[int] = None,
        concurrency: int = 1,
        client: Optional[AsyncClient] = None,
        prefix: str = "warmup",
    ) -> Optional[LatencyMetrics]:
        """
        Execute unmeasured warm-up requests to prime OpenMP, XGBoost, and DB connections.

        Args:
            warmup_requests: Number of warm-up requests. Defaults to config.warmup_requests.
            concurrency: Concurrency for warm-up execution. Defaults to 1.
            client: Optional AsyncClient instance.
            prefix: Prefix for warm-up transaction IDs.

        Returns:
            LatencyMetrics for the warm-up batch (for diagnostic tracking).
        """
        count = warmup_requests if warmup_requests is not None else self.config.warmup_requests
        if count <= 0:
            logger.info("Warm-up requests configured to 0; skipping warm-up.")
            return None

        logger.info(f"Executing {count} unmeasured warm-up requests...")
        warmup_payloads = self.runner.generate_payloads(count=count, prefix=prefix)
        warmup_result = await self.runner.run(
            num_requests=count,
            concurrency=concurrency,
            client=client,
            payloads=warmup_payloads,
            prefix=prefix,
        )
        logger.info(
            f"Warm-up completed: {warmup_result.successful_requests}/{count} successful, "
            f"mean latency: {warmup_result.latency.mean_ms:.3f} ms"
        )
        return warmup_result.latency

    async def run_concurrency_sweep(
        self,
        concurrency_levels: Optional[Sequence[int]] = None,
        num_requests: Optional[int] = None,
        target_rate: Optional[float] = None,
        target_sla: Optional[TargetSLA] = None,
        client: Optional[AsyncClient] = None,
        prefix: str = "sweep",
    ) -> ConcurrencySweepResult:
        """
        Execute multi-concurrency matrix sweep across configured worker levels.

        Args:
            concurrency_levels: Sequence of concurrency levels (e.g. (1, 2, 4, 8, 16)).
            num_requests: Requests to evaluate per concurrency level.
            target_rate: Optional offered rate limit.
            target_sla: Optional TargetSLA reference target.
            client: Optional AsyncClient instance.
            prefix: Prefix for transaction IDs.

        Returns:
            ConcurrencySweepResult containing per-concurrency metrics and scaling ratios.
        """
        levels = list(concurrency_levels or self.config.concurrency_levels)
        req_count = num_requests if num_requests is not None else self.config.num_requests
        sla = target_sla or self.target_sla

        if not levels:
            raise ValueError("concurrency_levels cannot be empty.")

        logger.info(f"Starting concurrency sweep across levels: {levels} with {req_count} requests each...")

        concurrency_results: Dict[int, BenchmarkRunResult] = {}

        for c in levels:
            logger.info(f"Running concurrency level C={c} ({req_count} requests)...")
            run_result = await self.runner.run(
                num_requests=req_count,
                concurrency=c,
                target_rate=target_rate,
                target_sla=sla,
                client=client,
                prefix=f"{prefix}_c{c}",
            )
            concurrency_results[c] = run_result

        # Calculate scaling metrics relative to baseline (first concurrency level)
        baseline_c = levels[0]
        baseline_res = concurrency_results[baseline_c]
        baseline_tps = max(0.0001, baseline_res.throughput.completed_tps)

        peak_tps = 0.0
        peak_c = baseline_c
        scaling_ratios: Dict[int, float] = {}
        scaling_efficiencies: Dict[int, float] = {}

        for c, res in concurrency_results.items():
            tps = res.throughput.completed_tps
            if tps > peak_tps:
                peak_tps = tps
                peak_c = c

            ratio = tps / baseline_tps
            scaling_ratios[c] = ratio

            # Scaling efficiency = (Speedup / C) * 100%
            c_factor = c / baseline_c if baseline_c > 0 else c
            efficiency = (ratio / c_factor) * 100.0 if c_factor > 0 else 0.0
            scaling_efficiencies[c] = efficiency

        return ConcurrencySweepResult(
            concurrency_results=concurrency_results,
            baseline_concurrency=baseline_c,
            baseline_tps=baseline_tps,
            peak_tps=peak_tps,
            peak_concurrency=peak_c,
            scaling_ratios=scaling_ratios,
            scaling_efficiencies=scaling_efficiencies,
        )

    async def run_persistence_ablation(
        self,
        num_requests: int = 100,
        concurrency: int = 1,
        client: Optional[AsyncClient] = None,
        prefix: str = "ablation",
    ) -> AblationResult:
        """
        Execute multi-tier persistence ablation benchmarking.

        Compares:
        - Mode A (Full API + PostgreSQL Persistence)
        - Mode B (In-Memory Pipeline Stages 1-6)
        - Mode C (Idempotent Replay Fast Path)

        Returns:
            AblationResult with quantified persistence overhead and replay speedups.
        """
        if num_requests <= 0:
            raise ValueError(f"num_requests must be strictly positive, got {num_requests}")

        logger.info(f"Starting Multi-Tier Persistence Ablation ({num_requests} requests)...")

        # 1. Generate N identical test payloads
        payloads = self.runner.generate_payloads(count=num_requests, prefix=prefix)

        # 2. Mode A: Full API + PostgreSQL Persistence
        logger.info("Executing Mode A (Full API + PostgreSQL Persistence)...")
        mode_a_res = await self.runner.run(
            num_requests=num_requests,
            concurrency=concurrency,
            client=client,
            payloads=payloads,
            prefix=f"{prefix}_mode_a",
        )
        mode_a_latency = mode_a_res.latency

        # 3. Mode B: In-Memory ML Pipeline (Stages 1 through 6)
        logger.info("Executing Mode B (In-Memory ML Pipeline Stages 1-6)...")
        in_memory_latencies: List[float] = []
        for p in payloads:
            profile_res = self.profiler.profile_transaction(p)
            in_memory_latencies.append(profile_res.total_profiled_latency_ms)

        if in_memory_latencies:
            mode_b_latency = LatencyMetrics.from_latencies(in_memory_latencies)
        else:
            mode_b_latency = mode_a_latency

        # 4. Mode C: Idempotent Replay Fast Path
        # Re-submitting the exact same payloads against the API exercises the idempotency replay cache
        logger.info("Executing Mode C (Idempotent Replay Fast Path)...")
        mode_c_res = await self.runner.run(
            num_requests=num_requests,
            concurrency=concurrency,
            client=client,
            payloads=payloads,
            prefix=f"{prefix}_mode_a",  # same IDs to trigger idempotency replay
        )
        mode_c_latency = mode_c_res.latency

        # 5. Compute Comparative Metrics
        # Persistence Overhead = Mode A mean - Mode B mean
        overhead_ms = max(0.0, mode_a_latency.mean_ms - mode_b_latency.mean_ms)
        if mode_a_latency.mean_ms > 0.0:
            overhead_pct = (overhead_ms / mode_a_latency.mean_ms) * 100.0
        else:
            overhead_pct = 0.0

        # Replay speedup vs Mode A
        if mode_c_latency.mean_ms > 0.0:
            replay_speedup = mode_a_latency.mean_ms / mode_c_latency.mean_ms
        else:
            replay_speedup = 1.0

        if mode_a_latency.mean_ms > 0.0:
            replay_reduction_pct = (
                (mode_a_latency.mean_ms - mode_c_latency.mean_ms) / mode_a_latency.mean_ms
            ) * 100.0
        else:
            replay_reduction_pct = 0.0

        return AblationResult(
            mode_a_full_api=mode_a_latency,
            mode_b_in_memory=mode_b_latency,
            mode_c_idempotent_replay=mode_c_latency,
            persistence_overhead_ms=overhead_ms,
            persistence_overhead_pct=overhead_pct,
            replay_speedup_factor=replay_speedup,
            replay_latency_reduction_pct=replay_reduction_pct,
        )

    async def run_bottleneck_analysis(
        self,
        num_samples: int = 50,
        prefix: str = "bottleneck",
    ) -> BottleneckAnalysisResult:
        """
        Execute micro-profiling across 7 lifecycle stages to identify primary bottlenecks.

        Args:
            num_samples: Number of sample transactions to profile.
            prefix: Identifier prefix for test payloads.

        Returns:
            BottleneckAnalysisResult ranking stages and providing diagnostic insights.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be strictly positive, got {num_samples}")

        logger.info(f"Executing 7-stage bottleneck micro-profiling across {num_samples} samples...")
        payloads = self.runner.generate_payloads(count=num_samples, prefix=prefix)

        stage_timings: Dict[PipelineStage, List[float]] = {
            stage: [] for stage in PipelineStage
        }

        for p in payloads:
            # Use async profiling to include PostgreSQL persistence stage 7 if configured
            profile_res = await self.profiler.profile_transaction_async(
                payload=p,
                skip_persistence=not self.config.include_persistence,
            )
            for stage, lat in profile_res.stage_latencies_ms.items():
                stage_timings[stage].append(lat)

        breakdown = ComponentLatencyBreakdown.from_stage_latencies(stage_timings)

        # Rank stages by mean latency descending
        sorted_stages = sorted(
            breakdown.stage_metrics.items(),
            key=lambda item: item[1].mean_ms,
            reverse=True,
        )

        primary_stage, primary_metrics = sorted_stages[0] if sorted_stages else ("None", None)
        secondary_stage, secondary_metrics = sorted_stages[1] if len(sorted_stages) > 1 else ("None", None)

        primary_mean = primary_metrics.mean_ms if primary_metrics else 0.0
        primary_pct = breakdown.stage_percentages.get(primary_stage, 0.0) if primary_metrics else 0.0
        secondary_mean = secondary_metrics.mean_ms if secondary_metrics else 0.0
        secondary_pct = breakdown.stage_percentages.get(secondary_stage, 0.0) if secondary_metrics else 0.0


        diagnostic = (
            f"Primary latency bottleneck is '{primary_stage}' contributing {primary_pct:.1f}% "
            f"({primary_mean:.3f} ms mean). Secondary constraint is '{secondary_stage}' contributing "
            f"{secondary_pct:.1f}% ({secondary_mean:.3f} ms mean)."
        )

        return BottleneckAnalysisResult(
            component_breakdown=breakdown,
            primary_bottleneck_stage=str(primary_stage),
            primary_bottleneck_mean_ms=primary_mean,
            primary_bottleneck_pct=primary_pct,
            secondary_bottleneck_stage=str(secondary_stage),
            secondary_bottleneck_mean_ms=secondary_mean,
            secondary_bottleneck_pct=secondary_pct,
            diagnostic_summary=diagnostic,
        )

    async def run_full_suite(
        self,
        client: Optional[AsyncClient] = None,
        include_ablation: bool = True,
        include_bottleneck: bool = True,
    ) -> BenchmarkSuiteResult:
        """
        Execute the full composite benchmarking suite.

        Workflow:
        1. Measure cold-start latency.
        2. Execute warm-up requests.
        3. Execute multi-concurrency sweep.
        4. Execute multi-tier persistence ablation.
        5. Execute 7-stage bottleneck diagnostics.

        Returns:
            BenchmarkSuiteResult aggregating all benchmark results.
        """
        suite_id = f"suite_{uuid.uuid4().hex[:8]}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        logger.info(f"Starting Full Benchmark Suite [{suite_id}] at {timestamp_utc}...")

        # 1. Cold-start
        cold_ms = await self.measure_cold_start(client=client, prefix=f"{suite_id}_cold")

        # 2. Warm-up
        warmup_latency = await self.execute_warmup(
            warmup_requests=self.config.warmup_requests,
            client=client,
            prefix=f"{suite_id}_warmup",
        )

        # 3. Concurrency Sweep
        sweep_res = await self.run_concurrency_sweep(
            concurrency_levels=self.config.concurrency_levels,
            num_requests=self.config.num_requests,
            target_rate=self.config.target_rate,
            target_sla=self.target_sla,
            client=client,
            prefix=f"{suite_id}_sweep",
        )

        # 4. Persistence Ablation
        ablation_res: Optional[AblationResult] = None
        if include_ablation:
            ablation_res = await self.run_persistence_ablation(
                num_requests=min(self.config.num_requests, 100),
                client=client,
                prefix=f"{suite_id}_ablation",
            )

        # 5. Bottleneck Analysis
        bottleneck_res: Optional[BottleneckAnalysisResult] = None
        if include_bottleneck:
            bottleneck_res = await self.run_bottleneck_analysis(
                num_samples=min(self.config.num_requests, 50),
                prefix=f"{suite_id}_bottleneck",
            )

        logger.info(f"Full Benchmark Suite [{suite_id}] successfully completed.")

        return BenchmarkSuiteResult(
            suite_id=suite_id,
            config=self.config,
            cold_start_latency_ms=cold_ms,
            warmup_requests_executed=self.config.warmup_requests,
            warmup_latency=warmup_latency,
            concurrency_sweep=sweep_res,
            ablation=ablation_res,
            bottleneck_analysis=bottleneck_res,
            target_sla_evaluated=self.target_sla,
            timestamp_utc=timestamp_utc,
        )
