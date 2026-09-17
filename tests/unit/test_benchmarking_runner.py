"""
Unit tests for BenchmarkRunner dataset loading, payload generation, and accounting (Phase 10.2).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.metrics import LatencyMetrics, ThroughputMetrics
from backend.app.benchmarking.runner import (
    BenchmarkRunner,
    BenchmarkRunResult,
    IdempotencyReplayResult,
)
from ml.models.config import VAL_FEATURES_PATH


class TestBenchmarkRunnerUnit:
    """Unit tests for BenchmarkRunner data preparation and payload generation."""

    def test_runner_initialization_defaults(self) -> None:
        """Test default runner initialization."""
        runner = BenchmarkRunner()
        assert runner.config is not None
        assert runner.target_url is None
        assert runner.endpoint_path == "/predict"
        assert runner.app is not None

    def test_runner_custom_config(self) -> None:
        """Test runner initialization with custom config and endpoint."""
        cfg = BenchmarkConfig(num_requests=100, concurrency_levels=(4,))
        runner = BenchmarkRunner(config=cfg, target_url="http://localhost:8000/", endpoint_path="api/v1/predict")
        assert runner.config.num_requests == 100
        assert runner.target_url == "http://localhost:8000"
        assert runner.endpoint_path == "/api/v1/predict"

    def test_preload_dataset_success(self) -> None:
        """Test pre-loading dataset from valid Parquet file."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, random_seed=42)
        runner = BenchmarkRunner(config=cfg)

        records = runner.preload_dataset()
        assert isinstance(records, list)
        assert len(records) > 0
        assert isinstance(records[0], dict)
        assert "amount" in records[0]

        # Verify caching
        records_2 = runner.preload_dataset()
        assert records is records_2

    def test_preload_nonexistent_dataset_raises_file_not_found(self) -> None:
        """Test FileNotFoundError when dataset path does not exist."""
        cfg = BenchmarkConfig(dataset_path="non_existent/path/data.parquet")
        runner = BenchmarkRunner(config=cfg)
        with pytest.raises(FileNotFoundError, match="Benchmark dataset not found"):
            runner.preload_dataset()

    def test_generate_payloads_unique_ids(self) -> None:
        """Test generated payloads contain unique external transaction IDs."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, random_seed=42)
        runner = BenchmarkRunner(config=cfg)

        payloads = runner.generate_payloads(count=50, prefix="test_bench")
        assert len(payloads) == 50

        ids = [p["transaction_id"] for p in payloads]
        # Verify 100% uniqueness
        assert len(set(ids)) == 50

        # Verify ID length constraint (<= 128 chars)
        for tx_id in ids:
            assert len(tx_id) <= 128
            assert tx_id.startswith("test_bench_")

    def test_generate_payloads_with_fixed_ids(self) -> None:
        """Test payload generation with fixed IDs for replay testing."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH)
        runner = BenchmarkRunner(config=cfg)

        fixed_ids = ["fixed_001", "fixed_002", "fixed_003"]
        payloads = runner.generate_payloads(count=3, fixed_ids=fixed_ids)

        assert len(payloads) == 3
        assert payloads[0]["transaction_id"] == "fixed_001"
        assert payloads[1]["transaction_id"] == "fixed_002"
        assert payloads[2]["transaction_id"] == "fixed_003"

    def test_generate_payloads_cycling_exceeds_available(self) -> None:
        """Test requesting more rows than available in dataset cycles safely."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH)
        runner = BenchmarkRunner(config=cfg)
        # Mock dataset cache with 3 rows
        runner._cached_records = [
            {"amount": 10.0, "merchant_category": "food", "job_category": "eng"},
            {"amount": 20.0, "merchant_category": "gas", "job_category": "doc"},
            {"amount": 30.0, "merchant_category": "retail", "job_category": "law"},
        ]

        payloads = runner.generate_payloads(count=7, prefix="cycle")
        assert len(payloads) == 7
        assert payloads[0]["amount"] == 10.0
        assert payloads[3]["amount"] == 10.0  # Cycled
        assert payloads[6]["amount"] == 10.0  # Cycled
        assert len(set(p["transaction_id"] for p in payloads)) == 7

    def test_benchmark_run_result_to_dict(self) -> None:
        """Test serialization of BenchmarkRunResult."""
        throughput = ThroughputMetrics.from_counts(10, 10, 0, 1.0)
        latency = LatencyMetrics.from_latencies([10.0, 20.0])
        res = BenchmarkRunResult(
            benchmark_id="test_run_01",
            mode="In-process ASGI",
            concurrency=2,
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            elapsed_seconds=1.0,
            throughput=throughput,
            latency=latency,
            status_code_counts={200: 10},
            error_counts={},
        )
        d = res.to_dict()
        assert d["benchmark_id"] == "test_run_01"
        assert d["concurrency"] == 2
        assert d["status_code_counts"] == {"200": 10}

    def test_idempotency_replay_result_to_dict(self) -> None:
        """Test serialization of IdempotencyReplayResult."""
        m1 = LatencyMetrics.from_latencies([20.0, 30.0])
        m2 = LatencyMetrics.from_latencies([5.0, 10.0])
        res = IdempotencyReplayResult(
            num_requests=2,
            fresh_evaluation_latency=m1,
            replay_evaluation_latency=m2,
            latency_reduction_pct=70.0,
            speedup_factor=3.33,
            verified_idempotent=True,
            first_pass_status_codes={200: 2},
            second_pass_status_codes={200: 2},
        )
        d = res.to_dict()
        assert d["num_requests"] == 2
        assert d["speedup_factor"] == 3.33
        assert d["verified_idempotent"] is True
        assert d["first_pass_status_codes"] == {"200": 2}
