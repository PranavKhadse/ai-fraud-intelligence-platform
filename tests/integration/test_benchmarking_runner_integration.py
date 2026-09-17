"""
Integration tests for BenchmarkRunner asynchronous load generator and idempotency replay harness (Phase 10.2).
"""

from typing import AsyncGenerator, Dict, Any
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.runner import BenchmarkRunner
from backend.app.db.session import get_db_session
from backend.app.main import app
from ml.models.config import VAL_FEATURES_PATH

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="function")
async def benchmark_api_client(
    db_session: AsyncSession, pg_engine: AsyncEngine
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous httpx client with request-scoped database sessions."""
    session_factory = async_sessionmaker(
        bind=pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async def override_get_db_session():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_db_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


class TestBenchmarkRunnerIntegration:
    """Integration test suite for BenchmarkRunner multi-concurrency load and replay benchmarking."""

    async def test_1_run_in_process_asgi_single_concurrency(
        self, benchmark_api_client: AsyncClient
    ) -> None:
        """Test in-process ASGI benchmark with concurrency=1 and 10 requests."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=10, concurrency_levels=(1,))
        runner = BenchmarkRunner(config=cfg)

        result = await runner.run(
            num_requests=10,
            concurrency=1,
            client=benchmark_api_client,
            prefix="test_int_1c",
        )

        assert result.total_requests == 10
        assert result.successful_requests == 10
        assert result.failed_requests == 0
        assert result.concurrency == 1
        assert result.mode == "In-process ASGI"
        assert result.status_code_counts.get(200) == 10
        assert result.latency.count == 10
        assert result.latency.min_ms > 0.0
        assert result.latency.p50_ms > 0.0
        assert result.throughput.completed_tps > 0.0
        assert result.throughput.error_rate_percent == 0.0

    async def test_2_run_multi_concurrency_execution(
        self, benchmark_api_client: AsyncClient
    ) -> None:
        """Test concurrent execution with concurrency=4 and 20 requests."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=20, concurrency_levels=(4,))
        runner = BenchmarkRunner(config=cfg)

        result = await runner.run(
            num_requests=20,
            concurrency=4,
            client=benchmark_api_client,
            prefix="test_int_4c",
        )

        assert result.total_requests == 20
        assert result.successful_requests == 20
        assert result.failed_requests == 0
        assert result.concurrency == 4
        assert len(result.samples) == 20
        assert result.status_code_counts.get(200) == 20

    async def test_3_run_with_target_sla_compliance(
        self, benchmark_api_client: AsyncClient
    ) -> None:
        """Test SLA compliance calculation attached to benchmark run."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=5)
        runner = BenchmarkRunner(config=cfg)
        sla = TargetSLA.standard_gateway_target()  # P50<=15, P95<=35, P99<=60

        result = await runner.run(
            num_requests=5,
            concurrency=1,
            target_sla=sla,
            client=benchmark_api_client,
            prefix="test_sla",
        )

        assert result.sla_compliance is not None
        assert result.sla_compliance.target_sla.sla_name == sla.sla_name
        assert isinstance(result.sla_compliance.overall_compliant, bool)

    async def test_4_run_with_failed_requests_error_capture(
        self, benchmark_api_client: AsyncClient
    ) -> None:
        """Test that invalid payloads trigger 422 and are recorded as failed requests without crashing."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH)
        runner = BenchmarkRunner(config=cfg)

        # Build 3 invalid payloads (missing required amount)
        invalid_payloads = [
            {"transaction_id": f"invalid_{i}", "cardholder_lat": 40.0}
            for i in range(3)
        ]

        result = await runner.run(
            num_requests=3,
            concurrency=1,
            client=benchmark_api_client,
            payloads=invalid_payloads,
            prefix="err_test",
        )

        assert result.total_requests == 3
        assert result.successful_requests == 0
        assert result.failed_requests == 3
        assert result.status_code_counts.get(422) == 3
        assert result.throughput.error_rate_percent == 100.0
        assert len(result.error_counts) > 0

    async def test_5_run_with_target_rate_limiting(
        self, benchmark_api_client: AsyncClient
    ) -> None:
        """Test controlled rate limiting pacing."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, target_rate=50.0)
        runner = BenchmarkRunner(config=cfg)

        result = await runner.run(
            num_requests=5,
            concurrency=2,
            target_rate=50.0,
            client=benchmark_api_client,
            prefix="rate_test",
        )

        assert result.total_requests == 5
        assert result.successful_requests == 5
        assert result.failed_requests == 0

    async def test_6_idempotency_replay_benchmark_harness(
        self, benchmark_api_client: AsyncClient
    ) -> None:
        """Test dedicated two-pass idempotency replay benchmark flow."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH)
        runner = BenchmarkRunner(config=cfg)

        replay_result = await runner.run_idempotency_replay_benchmark(
            num_requests=5,
            concurrency=1,
            client=benchmark_api_client,
            prefix="replay_flow",
        )

        assert replay_result.num_requests == 5
        assert replay_result.verified_idempotent is True
        assert replay_result.first_pass_status_codes.get(200) == 5
        assert replay_result.second_pass_status_codes.get(200) == 5
        assert replay_result.fresh_evaluation_latency.count == 5
        assert replay_result.replay_evaluation_latency.count == 5
        # Replay must be non-negative and valid
        assert replay_result.replay_evaluation_latency.min_ms >= 0.0
        assert replay_result.speedup_factor >= 0.0
