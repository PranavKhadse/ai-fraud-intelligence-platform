"""
Asynchronous Multi-Concurrency Load Generator & Benchmark Runner for Phase 10.

Provides high-performance, asynchronous workload generation against the fraud detection
prediction API with configurable concurrency, rate limiting, dataset pre-loading,
high-resolution latency collection, and dedicated idempotency replay benchmarking.
"""

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

from httpx import ASGITransport, AsyncClient
import numpy as np
import pandas as pd

from backend.app.benchmarking.collector import MetricCollector, RequestMetricSample
from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.metrics import (
    calculate_percentiles,
    LatencyMetrics,
    SLACompliance,
    ThroughputMetrics,
)
from backend.app.main import app as default_fastapi_app
from backend.app.schemas.predict import TransactionPredictRequest
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS

logger = logging.getLogger("fraud_api.benchmark_runner")


@dataclass(frozen=True)
class BenchmarkRunResult:
    """
    Structured outcome of a single benchmark execution run at a fixed concurrency level.
    """
    benchmark_id: str
    mode: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_seconds: float
    throughput: ThroughputMetrics
    latency: LatencyMetrics
    status_code_counts: Dict[int, int]
    error_counts: Dict[str, int]
    sla_compliance: Optional[SLACompliance] = None
    samples: List[RequestMetricSample] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput": self.throughput.to_dict(),
            "latency": self.latency.to_dict(),
            "status_code_counts": {str(k): v for k, v in self.status_code_counts.items()},
            "error_counts": dict(self.error_counts),
            "sla_compliance": self.sla_compliance.to_dict() if self.sla_compliance else None,
        }


@dataclass(frozen=True)
class IdempotencyReplayResult:
    """
    Comparative evaluation outcome measuring fresh ML scoring vs idempotent database replay.
    """
    num_requests: int
    fresh_evaluation_latency: LatencyMetrics
    replay_evaluation_latency: LatencyMetrics
    latency_reduction_pct: float
    speedup_factor: float
    verified_idempotent: bool
    first_pass_status_codes: Dict[int, int]
    second_pass_status_codes: Dict[int, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert replay comparison result to a JSON-serializable dictionary."""
        return {
            "num_requests": self.num_requests,
            "fresh_evaluation_latency": self.fresh_evaluation_latency.to_dict(),
            "replay_evaluation_latency": self.replay_evaluation_latency.to_dict(),
            "latency_reduction_pct": self.latency_reduction_pct,
            "speedup_factor": self.speedup_factor,
            "verified_idempotent": self.verified_idempotent,
            "first_pass_status_codes": {str(k): v for k, v in self.first_pass_status_codes.items()},
            "second_pass_status_codes": {str(k): v for k, v in self.second_pass_status_codes.items()},
        }


class BenchmarkRunner:
    """
    Asynchronous load generator executing concurrent requests against the FastAPI /predict API.

    Guarantees:
    - Zero Disk I/O during timing: Dataset rows are pre-loaded and cached into memory before timing.
    - Idempotency Safety: Fresh requests receive unique 128-char external_transaction_id values.
    - Concurrency Bounding: Uses asyncio.Semaphore to enforce strictly bounded concurrency.
    - High-Resolution Timing: Monotonic timing around each HTTP request lifecycle.
    - Transport Flexibility: Supports in-process ASGI client and external network URLs.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        target_url: Optional[str] = None,
        endpoint_path: str = "/predict",
        app_instance: Optional[Any] = None,
    ) -> None:
        """
        Initialize the BenchmarkRunner with configuration and transport parameters.

        Args:
            config: BenchmarkConfig instance. Defaults to BenchmarkConfig().
            target_url: Optional base URL (e.g. 'http://localhost:8000') for network HTTP mode.
            endpoint_path: Endpoint path to benchmark (defaults to '/predict').
            app_instance: Optional FastAPI application instance for ASGI mode.
        """
        self.config = config or BenchmarkConfig()
        self.target_url = target_url.rstrip("/") if target_url else None
        self.endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        self.app = app_instance or default_fastapi_app
        self._cached_records: Optional[List[Dict[str, Any]]] = None

    def preload_dataset(self) -> List[Dict[str, Any]]:
        """
        Load and cache transaction feature records into memory before the benchmark run.

        Returns:
            List of raw transaction dictionaries conforming to 55-feature schema.
        """
        if self._cached_records is not None:
            return self._cached_records

        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Benchmark dataset not found at '{dataset_path.resolve()}'. "
                "Ensure Parquet features are generated."
            )

        logger.info(f"Pre-loading benchmark dataset from '{dataset_path}' into memory...")
        df = pd.read_parquet(dataset_path)

        if len(df) == 0:
            raise ValueError(f"Benchmark dataset at '{dataset_path}' contains zero rows.")

        # Deterministic sampling / shuffling if random_seed is set
        if self.config.random_seed is not None:
            df = df.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)

        # Convert DataFrame rows to plain Python dicts
        records = df.to_dict(orient="records")
        logger.info(f"Pre-loaded {len(records)} transaction records into memory cache.")
        self._cached_records = records
        return self._cached_records

    def generate_payloads(
        self,
        count: int,
        prefix: str = "bench",
        fixed_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a list of transaction payloads with unique or designated external transaction IDs.

        Args:
            count: Number of payloads to produce.
            prefix: Identifier prefix for generated external transaction IDs.
            fixed_ids: Optional sequence of pre-defined IDs (e.g. for replay testing).

        Returns:
            List of JSON-serializable request payload dictionaries.
        """
        records = self.preload_dataset()
        num_available = len(records)
        payloads: List[Dict[str, Any]] = []

        batch_id = uuid.uuid4().hex[:8]

        for i in range(count):
            # Deterministically cycle records if count exceeds available rows
            base_record = dict(records[i % num_available])

            if fixed_ids is not None and i < len(fixed_ids):
                tx_id = fixed_ids[i]
            else:
                # Generate unique 128-char safe transaction ID
                tx_id = f"{prefix}_{batch_id}_{i}_{uuid.uuid4().hex[:12]}"
                if len(tx_id) > 128:
                    tx_id = tx_id[:128]

            base_record["transaction_id"] = tx_id
            if "account_id" not in base_record or base_record["account_id"] is None:
                base_record["account_id"] = f"acc_{prefix}_{i % 1000}"
            if "timestamp" not in base_record or base_record["timestamp"] is None:
                base_record["timestamp"] = "2026-09-17T12:00:00Z"

            # Clean any NumPy/Pandas types to standard JSON-serializable Python types
            clean_payload: Dict[str, Any] = {}
            for k, v in base_record.items():
                if pd.isna(v):
                    clean_payload[k] = None
                elif isinstance(v, (np.integer, int)):
                    clean_payload[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_payload[k] = float(v)
                elif isinstance(v, (np.bool_, bool)):
                    clean_payload[k] = bool(v)
                elif hasattr(v, "isoformat"):
                    clean_payload[k] = v.isoformat()
                elif isinstance(v, pd.Timestamp):
                    clean_payload[k] = str(v)
                else:
                    clean_payload[k] = v

            payloads.append(clean_payload)

        return payloads

    async def _execute_single_request(
        self,
        client: AsyncClient,
        payload: Dict[str, Any],
        collector: MetricCollector,
        semaphore: asyncio.Semaphore,
        rate_limiter_delay: float = 0.0,
    ) -> None:
        """
        Execute a single transaction prediction request under concurrency and rate controls.
        """
        if rate_limiter_delay > 0.0:
            await asyncio.sleep(rate_limiter_delay)

        tx_id = str(payload.get("transaction_id", "unknown_tx"))

        async with semaphore:
            start_ns = MetricCollector.start_timer()
            status_code = 500
            success = False
            error_msg: Optional[str] = None

            try:
                response = await client.post(
                    self.endpoint_path,
                    json=payload,
                    headers={"X-Correlation-ID": f"corr_{tx_id}"},
                )
                status_code = response.status_code
                latency_ms = MetricCollector.stop_timer(start_ns)

                if status_code in (200, 201):
                    success = True
                else:
                    success = False
                    error_msg = f"HTTP {status_code}: {response.text[:200]}"
            except Exception as e:
                latency_ms = MetricCollector.stop_timer(start_ns)
                status_code = 500
                success = False
                error_msg = f"Client exception: {str(e)}"

            if success:
                collector.record_request(
                    request_id=tx_id,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    success=True,
                )
            else:
                collector.record_error(
                    request_id=tx_id,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    error_message=error_msg or "Unknown error",
                )

    async def run(
        self,
        num_requests: Optional[int] = None,
        concurrency: Optional[int] = None,
        target_rate: Optional[float] = None,
        target_sla: Optional[TargetSLA] = None,
        client: Optional[AsyncClient] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
        prefix: str = "bench",
    ) -> BenchmarkRunResult:
        """
        Execute an asynchronous load benchmark run at a configured concurrency level.

        Args:
            num_requests: Number of requests to execute. Defaults to config.num_requests.
            concurrency: Active concurrency limit. Defaults to config.concurrency_levels[0].
            target_rate: Optional offered rate limit in TPS. Defaults to config.target_rate.
            target_sla: Optional TargetSLA reference target for compliance evaluation.
            client: Optional AsyncClient instance (e.g. pre-configured ASGI test client).
            payloads: Optional pre-constructed list of payloads.
            prefix: Identifier prefix for auto-generated payloads.

        Returns:
            BenchmarkRunResult containing comprehensive throughput, latency, and SLA metrics.
        """
        req_count = num_requests if num_requests is not None else self.config.num_requests
        conc = concurrency if concurrency is not None else (
            self.config.concurrency_levels[0] if self.config.concurrency_levels else 1
        )
        rate = target_rate if target_rate is not None else self.config.target_rate

        if req_count <= 0:
            raise ValueError(f"num_requests must be strictly positive, got {req_count}")
        if conc <= 0:
            raise ValueError(f"concurrency must be strictly positive, got {conc}")

        # 1. Pre-load payloads in memory
        if payloads is None:
            active_payloads = self.generate_payloads(count=req_count, prefix=prefix)
        else:
            active_payloads = payloads[:req_count]

        collector = MetricCollector()
        semaphore = asyncio.Semaphore(conc)
        benchmark_id = f"run_{conc}c_{req_count}r_{uuid.uuid4().hex[:6]}"

        # Calculate optional inter-request rate pacing
        rate_delay = (1.0 / rate) if (rate is not None and rate > 0.0) else 0.0

        # Determine transport mode
        transport_mode = "Network HTTP" if self.target_url else "In-process ASGI"
        owns_client = False

        if client is not None:
            active_client = client
        elif self.target_url:
            active_client = AsyncClient(base_url=self.target_url, timeout=60.0)
            owns_client = True
        else:
            transport = ASGITransport(app=self.app)
            active_client = AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0)
            owns_client = True

        benchmark_start_ns = MetricCollector.start_timer()
        try:
            tasks = [
                self._execute_single_request(
                    client=active_client,
                    payload=p,
                    collector=collector,
                    semaphore=semaphore,
                    rate_limiter_delay=rate_delay * i if rate_delay > 0.0 else 0.0,
                )
                for i, p in enumerate(active_payloads)
            ]
            await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            if owns_client:
                await active_client.aclose()

        elapsed_ms = MetricCollector.stop_timer(benchmark_start_ns)
        elapsed_seconds = max(0.0001, elapsed_ms / 1000.0)

        # 2. Extract metrics from collector
        samples = collector.get_samples()
        successful_samples = [s for s in samples if s.success]
        failed_samples = [s for s in samples if not s.success]
        successful_latencies = [s.latency_ms for s in successful_samples]

        # Status code and error counting
        status_counts: Dict[int, int] = {}
        error_counts: Dict[str, int] = {}
        for s in samples:
            status_counts[s.status_code] = status_counts.get(s.status_code, 0) + 1
            if not s.success and s.error_message:
                err_key = s.error_message[:100]
                error_counts[err_key] = error_counts.get(err_key, 0) + 1

        # Throughput accounting
        throughput = ThroughputMetrics.from_counts(
            total_requests=len(samples),
            successful_requests=len(successful_samples),
            failed_requests=len(failed_samples),
            elapsed_seconds=elapsed_seconds,
        )

        # Latency statistics
        if successful_latencies:
            latency_metrics = LatencyMetrics.from_latencies(successful_latencies)
        else:
            # Fallback if 100% failed
            latency_metrics = LatencyMetrics(
                count=0,
                min_ms=0.0,
                mean_ms=0.0,
                median_ms=0.0,
                max_ms=0.0,
                std_ms=0.0,
                iqr_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                p99_9_ms=0.0,
            )

        # SLA Compliance
        sla_comp: Optional[SLACompliance] = None
        if target_sla and successful_latencies:
            sla_comp = SLACompliance.from_latency_metrics(
                metrics=latency_metrics,
                latencies=successful_latencies,
                target_sla=target_sla,
            )

        return BenchmarkRunResult(
            benchmark_id=benchmark_id,
            mode=transport_mode,
            concurrency=conc,
            total_requests=len(samples),
            successful_requests=len(successful_samples),
            failed_requests=len(failed_samples),
            elapsed_seconds=round(elapsed_seconds, 4),
            throughput=throughput,
            latency=latency_metrics,
            status_code_counts=status_counts,
            error_counts=error_counts,
            sla_compliance=sla_comp,
            samples=samples,
        )

    async def run_idempotency_replay_benchmark(
        self,
        num_requests: int = 50,
        concurrency: int = 1,
        client: Optional[AsyncClient] = None,
        prefix: str = "replay",
    ) -> IdempotencyReplayResult:
        """
        Execute a dedicated two-pass benchmark evaluating idempotent replay performance.

        Pass 1: Submits N unique transactions (Fresh ML inference + DB persistence).
        Pass 2: Re-submits the EXACT SAME N transactions (Fast-path DB replay without ML inference).

        Returns:
            IdempotencyReplayResult comparing fresh vs replay latency distributions.
        """
        if num_requests <= 0:
            raise ValueError(f"num_requests must be positive, got {num_requests}")

        # 1. Generate N identical payloads with deterministic unique IDs
        fixed_payloads = self.generate_payloads(count=num_requests, prefix=prefix)

        logger.info(f"Starting Idempotency Replay Benchmark with {num_requests} transactions...")

        # Pass 1: Fresh Evaluation
        first_pass_result = await self.run(
            num_requests=num_requests,
            concurrency=concurrency,
            client=client,
            payloads=fixed_payloads,
            prefix=prefix,
        )

        # Pass 2: Idempotent Replay
        second_pass_result = await self.run(
            num_requests=num_requests,
            concurrency=concurrency,
            client=client,
            payloads=fixed_payloads,
            prefix=prefix,
        )

        m1 = first_pass_result.latency
        m2 = second_pass_result.latency

        # Calculate latency reduction and speedup
        if m1.mean_ms > 0.0:
            reduction_pct = round(((m1.mean_ms - m2.mean_ms) / m1.mean_ms) * 100.0, 2)
        else:
            reduction_pct = 0.0

        if m2.mean_ms > 0.0:
            speedup = round(m1.mean_ms / m2.mean_ms, 2)
        else:
            speedup = 1.0

        # Verify idempotency: all second-pass requests should return HTTP 200
        verified = (
            second_pass_result.successful_requests == num_requests
            and second_pass_result.status_code_counts.get(200, 0) == num_requests
        )

        return IdempotencyReplayResult(
            num_requests=num_requests,
            fresh_evaluation_latency=m1,
            replay_evaluation_latency=m2,
            latency_reduction_pct=reduction_pct,
            speedup_factor=speedup,
            verified_idempotent=verified,
            first_pass_status_codes=first_pass_result.status_code_counts,
            second_pass_status_codes=second_pass_result.status_code_counts,
        )
