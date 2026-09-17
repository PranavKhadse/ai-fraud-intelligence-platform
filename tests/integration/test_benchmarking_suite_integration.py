"""
Integration tests for BenchmarkSuite (Increment 10.3) against PostgreSQL and FastAPI ASGI app.
"""

from typing import AsyncGenerator
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.profiler import PipelineProfiler
from backend.app.benchmarking.suite import (
    AblationResult,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    BottleneckAnalysisResult,
    ConcurrencySweepResult,
)
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.services.persistence_service import FraudPersistenceService
from backend.app.services.unit_of_work import FraudPersistenceUnitOfWork
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
    async with AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0) as client:
        yield client
    app.dependency_overrides.clear()


class TestBenchmarkSuiteIntegration:
    """Integration tests validating BenchmarkSuite multi-scenario execution."""

    async def test_1_suite_cold_start_and_warmup(
        self,
        benchmark_api_client: AsyncClient,
    ) -> None:
        """Verify cold-start latency measurement and unmeasured warm-up priming."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=5, warmup_requests=2)
        suite = BenchmarkSuite(config=cfg)

        # 1. Measure cold-start
        cold_lat_ms = await suite.measure_cold_start(
            client=benchmark_api_client,
            prefix="suite_cold",
        )
        assert isinstance(cold_lat_ms, float)
        assert cold_lat_ms >= 0.0

        # 2. Execute unmeasured warm-up
        warmup_lat = await suite.execute_warmup(
            warmup_requests=2,
            client=benchmark_api_client,
            prefix="suite_warm",
        )
        assert warmup_lat is not None
        assert warmup_lat.count == 2
        assert warmup_lat.mean_ms >= 0.0

    async def test_2_suite_concurrency_sweep_execution(
        self,
        benchmark_api_client: AsyncClient,
    ) -> None:
        """Verify multi-concurrency matrix sweep execution across C in (1, 2)."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=5, concurrency_levels=(1, 2))
        suite = BenchmarkSuite(config=cfg)

        sweep_res = await suite.run_concurrency_sweep(
            concurrency_levels=(1, 2),
            num_requests=5,
            client=benchmark_api_client,
            prefix="suite_sweep",
        )

        assert isinstance(sweep_res, ConcurrencySweepResult)
        assert sweep_res.baseline_concurrency == 1
        assert 1 in sweep_res.concurrency_results
        assert 2 in sweep_res.concurrency_results

        res_1 = sweep_res.concurrency_results[1]
        res_2 = sweep_res.concurrency_results[2]

        assert res_1.total_requests == 5
        assert res_1.successful_requests == 5
        assert res_2.total_requests == 5
        assert res_2.successful_requests == 5

        assert sweep_res.scaling_ratios[1] == pytest.approx(1.0)
        assert sweep_res.scaling_ratios[2] > 0.0
        assert sweep_res.scaling_efficiencies[1] == pytest.approx(100.0)
        assert sweep_res.scaling_efficiencies[2] > 0.0

        d = sweep_res.to_dict()
        assert "concurrency_results" in d
        assert "1" in d["concurrency_results"]
        assert "2" in d["concurrency_results"]

    async def test_3_suite_persistence_ablation_execution(
        self,
        benchmark_api_client: AsyncClient,
    ) -> None:
        """Verify multi-tier persistence ablation comparing Mode A vs Mode B vs Mode C."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=5, concurrency_levels=(1,))
        profiler = PipelineProfiler()
        suite = BenchmarkSuite(config=cfg, profiler=profiler)

        ablation_res = await suite.run_persistence_ablation(
            num_requests=5,
            concurrency=1,
            client=benchmark_api_client,
            prefix="suite_ablation",
        )

        assert isinstance(ablation_res, AblationResult)
        assert ablation_res.mode_a_full_api.count == 5
        assert ablation_res.mode_b_in_memory.count == 5
        assert ablation_res.mode_c_idempotent_replay.count == 5

        # Quantitative checks
        assert ablation_res.persistence_overhead_ms >= 0.0
        assert ablation_res.persistence_overhead_pct >= 0.0
        assert ablation_res.replay_speedup_factor > 0.0

        d = ablation_res.to_dict()
        assert "mode_a_full_api" in d
        assert "mode_b_in_memory" in d
        assert "mode_c_idempotent_replay" in d

    async def test_4_suite_bottleneck_analysis_execution(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Verify 7-stage bottleneck micro-profiling and diagnostic ranking."""
        cfg = BenchmarkConfig(dataset_path=VAL_FEATURES_PATH, num_requests=5, include_persistence=True)
        uow = FraudPersistenceUnitOfWork(db_session)
        persistence_svc = FraudPersistenceService(uow=uow)
        profiler = PipelineProfiler(persistence_service=persistence_svc)
        suite = BenchmarkSuite(config=cfg, profiler=profiler)

        result = await suite.run_bottleneck_analysis(
            num_samples=5,
            prefix="suite_bottleneck",
        )

        assert isinstance(result, BottleneckAnalysisResult)
        assert result.primary_bottleneck_stage != "None"
        assert result.primary_bottleneck_mean_ms >= 0.0
        assert result.primary_bottleneck_pct >= 0.0
        assert "Primary latency bottleneck" in result.diagnostic_summary

        d = result.to_dict()
        assert "primary_bottleneck" in d
        assert "secondary_bottleneck" in d
        assert "diagnostic_summary" in d

    async def test_5_suite_run_full_suite_execution(
        self,
        benchmark_api_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Verify end-to-end full composite benchmark suite execution."""
        cfg = BenchmarkConfig(
            dataset_path=VAL_FEATURES_PATH,
            num_requests=5,
            concurrency_levels=(1, 2),
            warmup_requests=2,
        )
        uow = FraudPersistenceUnitOfWork(db_session)
        persistence_svc = FraudPersistenceService(uow=uow)
        profiler = PipelineProfiler(persistence_service=persistence_svc)
        sla = TargetSLA.standard_gateway_target()

        suite = BenchmarkSuite(
            config=cfg,
            profiler=profiler,
            target_sla=sla,
        )

        full_res = await suite.run_full_suite(
            client=benchmark_api_client,
            include_ablation=True,
            include_bottleneck=True,
        )

        assert isinstance(full_res, BenchmarkSuiteResult)
        assert full_res.cold_start_latency_ms is not None
        assert full_res.warmup_requests_executed == 2
        assert full_res.concurrency_sweep is not None
        assert full_res.ablation is not None
        assert full_res.bottleneck_analysis is not None
        assert full_res.target_sla_evaluated == sla

        d = full_res.to_dict()
        assert "suite_id" in d
        assert "timestamp_utc" in d
        assert "concurrency_sweep" in d
        assert "ablation" in d
        assert "bottleneck_analysis" in d
        assert d["target_sla_name"] == "Standard Payment Gateway Reference Target"
