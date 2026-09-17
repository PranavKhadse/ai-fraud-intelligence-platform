"""
Unit tests for BenchmarkSuite and related data models (Increment 10.3).
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.metrics import (
    ComponentLatencyBreakdown,
    LatencyMetrics,
    PipelineStage,
    ThroughputMetrics,
)
from backend.app.benchmarking.runner import BenchmarkRunner, BenchmarkRunResult
from backend.app.benchmarking.suite import (
    AblationResult,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    BottleneckAnalysisResult,
    ConcurrencySweepResult,
)


class TestBenchmarkSuiteUnit:
    """Unit test suite for BenchmarkSuite and composite result dataclasses."""

    def test_suite_initialization_defaults(self) -> None:
        """Verify BenchmarkSuite initializes with default components."""
        suite = BenchmarkSuite()
        assert isinstance(suite.config, BenchmarkConfig)
        assert isinstance(suite.runner, BenchmarkRunner)
        assert suite.target_sla is None

    def test_suite_initialization_custom(self) -> None:
        """Verify BenchmarkSuite initializes with custom config and SLA targets."""
        cfg = BenchmarkConfig(num_requests=100, concurrency_levels=(1, 2, 4))
        sla = TargetSLA.standard_gateway_target()
        suite = BenchmarkSuite(config=cfg, target_sla=sla)

        assert suite.config.num_requests == 100
        assert suite.config.concurrency_levels == (1, 2, 4)
        assert suite.target_sla == sla

    def test_concurrency_sweep_result_to_dict(self) -> None:
        """Verify ConcurrencySweepResult serialization and dictionary formatting."""
        mock_throughput_1 = ThroughputMetrics(10, 10, 0, 1.0, 10.0, 10.0, 10.0, 0.0)
        mock_latency_1 = LatencyMetrics(10, 5.0, 10.0, 9.5, 20.0, 2.0, 3.0, 9.5, 15.0, 18.0, 20.0, 20.0)
        run_res_1 = BenchmarkRunResult(
            benchmark_id="run_1c",
            mode="In-process ASGI",
            concurrency=1,
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            elapsed_seconds=1.0,
            throughput=mock_throughput_1,
            latency=mock_latency_1,
            status_code_counts={200: 10},
            error_counts={},
        )

        sweep_res = ConcurrencySweepResult(
            concurrency_results={1: run_res_1},
            baseline_concurrency=1,
            baseline_tps=10.0,
            peak_tps=10.0,
            peak_concurrency=1,
            scaling_ratios={1: 1.0},
            scaling_efficiencies={1: 100.0},
        )

        d = sweep_res.to_dict()
        assert d["baseline_concurrency"] == 1
        assert d["baseline_tps"] == 10.0
        assert d["peak_tps"] == 10.0
        assert d["scaling_ratios"]["1"] == 1.0
        assert d["scaling_efficiencies"]["1"] == 100.0
        assert "1" in d["concurrency_results"]

    def test_ablation_result_to_dict(self) -> None:
        """Verify AblationResult calculation serialization."""
        m_a = LatencyMetrics(10, 5.0, 20.0, 19.0, 30.0, 2.0, 3.0, 19.0, 25.0, 28.0, 30.0, 30.0)
        m_b = LatencyMetrics(10, 2.0, 8.0, 7.5, 12.0, 1.0, 1.5, 7.5, 10.0, 11.5, 12.0, 12.0)
        m_c = LatencyMetrics(10, 1.0, 4.0, 3.8, 6.0, 0.5, 0.8, 3.8, 5.0, 5.8, 6.0, 6.0)

        ablation = AblationResult(
            mode_a_full_api=m_a,
            mode_b_in_memory=m_b,
            mode_c_idempotent_replay=m_c,
            persistence_overhead_ms=12.0,
            persistence_overhead_pct=60.0,
            replay_speedup_factor=5.0,
            replay_latency_reduction_pct=80.0,
        )

        d = ablation.to_dict()
        assert d["persistence_overhead_ms"] == 12.0
        assert d["persistence_overhead_pct"] == 60.0
        assert d["replay_speedup_factor"] == 5.0
        assert d["replay_latency_reduction_pct"] == 80.0
        assert "mode_a_full_api" in d
        assert "mode_b_in_memory" in d
        assert "mode_c_idempotent_replay" in d

    def test_bottleneck_analysis_result_to_dict(self) -> None:
        """Verify BottleneckAnalysisResult data model and serialization."""
        lat_tree = LatencyMetrics(10, 4.0, 6.5, 6.2, 8.0, 1.0, 1.2, 6.2, 7.5, 7.8, 8.0, 8.0)
        lat_pg = LatencyMetrics(10, 2.0, 3.5, 3.2, 5.0, 0.5, 0.8, 3.2, 4.5, 4.8, 5.0, 5.0)

        stage_metrics = {
            PipelineStage.TREESHAP_EXPLAINABILITY: lat_tree,
            PipelineStage.POSTGRES_PERSISTENCE: lat_pg,
        }
        percentages = {
            PipelineStage.TREESHAP_EXPLAINABILITY: 65.0,
            PipelineStage.POSTGRES_PERSISTENCE: 35.0,
        }
        breakdown = ComponentLatencyBreakdown(
            stage_metrics=stage_metrics,
            stage_percentages=percentages,
            total_component_mean_ms=10.0,
        )

        result = BottleneckAnalysisResult(
            component_breakdown=breakdown,
            primary_bottleneck_stage="treeshap_explainability",
            primary_bottleneck_mean_ms=6.5,
            primary_bottleneck_pct=65.0,
            secondary_bottleneck_stage="postgres_persistence",
            secondary_bottleneck_mean_ms=3.5,
            secondary_bottleneck_pct=35.0,
            diagnostic_summary="TreeSHAP is the primary bottleneck.",
        )

        d = result.to_dict()
        assert d["primary_bottleneck"]["stage"] == "treeshap_explainability"
        assert d["primary_bottleneck"]["contribution_pct"] == 65.0
        assert d["secondary_bottleneck"]["stage"] == "postgres_persistence"
        assert d["diagnostic_summary"] == "TreeSHAP is the primary bottleneck."

    def test_benchmark_suite_result_to_dict(self) -> None:
        """Verify BenchmarkSuiteResult composite serialization."""
        cfg = BenchmarkConfig(num_requests=50)
        sla = TargetSLA.standard_gateway_target()
        suite_res = BenchmarkSuiteResult(
            suite_id="suite_test_123",
            config=cfg,
            cold_start_latency_ms=45.2,
            warmup_requests_executed=10,
            warmup_latency=None,
            concurrency_sweep=None,
            ablation=None,
            bottleneck_analysis=None,
            target_sla_evaluated=sla,
            timestamp_utc="2026-09-17T12:00:00Z",
        )

        d = suite_res.to_dict()
        assert d["suite_id"] == "suite_test_123"
        assert d["cold_start_latency_ms"] == 45.2
        assert d["warmup_requests_executed"] == 10
        assert d["target_sla_name"] == "Standard Payment Gateway Reference Target"
        assert d["concurrency_sweep"] is None

    @pytest.mark.asyncio
    async def test_suite_concurrency_sweep_scaling_math(self) -> None:
        """Verify mathematical scaling and efficiency calculations in run_concurrency_sweep."""
        runner_mock = MagicMock(spec=BenchmarkRunner)

        def make_mock_result(concurrency: int, tps: float) -> BenchmarkRunResult:
            throughput = ThroughputMetrics(100, 100, 0, 100.0 / tps, tps, tps, tps, 0.0)
            latency = LatencyMetrics(100, 5.0, 10.0, 9.5, 20.0, 2.0, 3.0, 9.5, 15.0, 18.0, 20.0, 20.0)
            return BenchmarkRunResult(
                benchmark_id=f"run_{concurrency}c",
                mode="In-process ASGI",
                concurrency=concurrency,
                total_requests=100,
                successful_requests=100,
                failed_requests=0,
                elapsed_seconds=100.0 / tps,
                throughput=throughput,
                latency=latency,
                status_code_counts={200: 100},
                error_counts={},
            )

        # Mock runs: C=1 (50 TPS), C=2 (90 TPS), C=4 (160 TPS)
        runner_mock.run = AsyncMock(side_effect=[
            make_mock_result(1, 50.0),
            make_mock_result(2, 90.0),
            make_mock_result(4, 160.0),
        ])

        cfg = BenchmarkConfig(concurrency_levels=(1, 2, 4), num_requests=100)
        suite = BenchmarkSuite(config=cfg, runner=runner_mock)

        sweep = await suite.run_concurrency_sweep()

        assert sweep.baseline_concurrency == 1
        assert sweep.baseline_tps == 50.0
        assert sweep.peak_tps == 160.0
        assert sweep.peak_concurrency == 4

        # Scaling ratio: C=1 -> 1.0, C=2 -> 90/50 = 1.8, C=4 -> 160/50 = 3.2
        assert sweep.scaling_ratios[1] == pytest.approx(1.0)
        assert sweep.scaling_ratios[2] == pytest.approx(1.8)
        assert sweep.scaling_ratios[4] == pytest.approx(3.2)

        # Scaling efficiency: C=1 -> 100%, C=2 -> (1.8/2)*100 = 90%, C=4 -> (3.2/4)*100 = 80%
        assert sweep.scaling_efficiencies[1] == pytest.approx(100.0)
        assert sweep.scaling_efficiencies[2] == pytest.approx(90.0)
        assert sweep.scaling_efficiencies[4] == pytest.approx(80.0)

    @pytest.mark.asyncio
    async def test_suite_ablation_math_calculations(self) -> None:
        """Verify mathematical formulas in run_persistence_ablation."""
        runner_mock = MagicMock(spec=BenchmarkRunner)
        runner_mock.generate_payloads = MagicMock(return_value=[{"transaction_id": "tx_1"}])

        # Mode A (Full API): 20ms mean
        lat_a = LatencyMetrics(10, 5.0, 20.0, 19.0, 30.0, 2.0, 3.0, 19.0, 25.0, 28.0, 30.0, 30.0)
        res_a = BenchmarkRunResult(
            benchmark_id="a", mode="ASGI", concurrency=1, total_requests=10, successful_requests=10,
            failed_requests=0, elapsed_seconds=0.2,
            throughput=ThroughputMetrics(10, 10, 0, 0.2, 50.0, 50.0, 50.0, 0.0),
            latency=lat_a, status_code_counts={200: 10}, error_counts={},
        )

        # Mode C (Replay): 5ms mean
        lat_c = LatencyMetrics(10, 1.0, 5.0, 4.8, 8.0, 0.5, 0.8, 4.8, 6.0, 7.0, 8.0, 8.0)
        res_c = BenchmarkRunResult(
            benchmark_id="c", mode="ASGI", concurrency=1, total_requests=10, successful_requests=10,
            failed_requests=0, elapsed_seconds=0.05,
            throughput=ThroughputMetrics(10, 10, 0, 0.05, 200.0, 200.0, 200.0, 0.0),
            latency=lat_c, status_code_counts={200: 10}, error_counts={},
        )

        runner_mock.run = AsyncMock(side_effect=[res_a, res_c])

        # Mode B (In-Memory): Profiler returns 8ms
        profiler_mock = MagicMock()
        mock_profile_res = MagicMock()
        mock_profile_res.total_profiled_latency_ms = 8.0
        profiler_mock.profile_transaction = MagicMock(return_value=mock_profile_res)

        suite = BenchmarkSuite(runner=runner_mock, profiler=profiler_mock)

        ablation = await suite.run_persistence_ablation(num_requests=1)

        # Overhead = Mode A (20ms) - Mode B (8ms) = 12ms
        assert ablation.persistence_overhead_ms == pytest.approx(12.0)
        # Overhead % = (12/20)*100 = 60%
        assert ablation.persistence_overhead_pct == pytest.approx(60.0)
        # Replay speedup = 20/5 = 4.0
        assert ablation.replay_speedup_factor == pytest.approx(4.0)
        # Replay reduction % = ((20-5)/20)*100 = 75%
        assert ablation.replay_latency_reduction_pct == pytest.approx(75.0)
