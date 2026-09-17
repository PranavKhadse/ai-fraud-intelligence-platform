"""
Unit tests for BenchmarkReporter formatting, JSON/CSV exports, and metadata extraction (Increment 10.4).
"""

import csv
import json
from pathlib import Path
import pytest

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.metrics import (
    ComponentLatencyBreakdown,
    LatencyMetrics,
    PipelineStage,
    ThroughputMetrics,
)
from backend.app.benchmarking.reporter import BenchmarkReporter
from backend.app.benchmarking.runner import BenchmarkRunResult
from backend.app.benchmarking.suite import (
    AblationResult,
    BenchmarkSuiteResult,
    BottleneckAnalysisResult,
    ConcurrencySweepResult,
)


@pytest.fixture
def sample_suite_result() -> BenchmarkSuiteResult:
    """Construct a complete synthetic BenchmarkSuiteResult for testing reporting and exports."""
    cfg = BenchmarkConfig(num_requests=50, concurrency_levels=(1, 2), random_seed=42)
    sla = TargetSLA.standard_gateway_target()

    mock_tp_1 = ThroughputMetrics(50, 50, 0, 1.0, 50.0, 50.0, 50.0, 0.0)
    mock_lat_1 = LatencyMetrics(50, 4.0, 12.0, 11.5, 25.0, 2.0, 3.0, 11.5, 18.0, 22.0, 25.0, 25.0)
    run_1 = BenchmarkRunResult(
        benchmark_id="run_1c",
        mode="In-process ASGI",
        concurrency=1,
        total_requests=50,
        successful_requests=50,
        failed_requests=0,
        elapsed_seconds=1.0,
        throughput=mock_tp_1,
        latency=mock_lat_1,
        status_code_counts={200: 50},
        error_counts={},
    )

    mock_tp_2 = ThroughputMetrics(50, 50, 0, 0.55, 90.0, 90.0, 90.0, 0.0)
    mock_lat_2 = LatencyMetrics(50, 4.5, 13.5, 13.0, 28.0, 2.5, 3.5, 13.0, 20.0, 25.0, 28.0, 28.0)
    run_2 = BenchmarkRunResult(
        benchmark_id="run_2c",
        mode="In-process ASGI",
        concurrency=2,
        total_requests=50,
        successful_requests=50,
        failed_requests=0,
        elapsed_seconds=0.55,
        throughput=mock_tp_2,
        latency=mock_lat_2,
        status_code_counts={200: 50},
        error_counts={},
    )

    sweep = ConcurrencySweepResult(
        concurrency_results={1: run_1, 2: run_2},
        baseline_concurrency=1,
        baseline_tps=50.0,
        peak_tps=90.0,
        peak_concurrency=2,
        scaling_ratios={1: 1.0, 2: 1.8},
        scaling_efficiencies={1: 100.0, 2: 90.0},
    )

    ablation = AblationResult(
        mode_a_full_api=mock_lat_1,
        mode_b_in_memory=LatencyMetrics(50, 2.0, 6.0, 5.8, 10.0, 1.0, 1.2, 5.8, 8.0, 9.5, 10.0, 10.0),
        mode_c_idempotent_replay=LatencyMetrics(50, 1.0, 3.0, 2.8, 5.0, 0.5, 0.6, 2.8, 4.0, 4.8, 5.0, 5.0),
        persistence_overhead_ms=6.0,
        persistence_overhead_pct=50.0,
        replay_speedup_factor=4.0,
        replay_latency_reduction_pct=75.0,
    )

    stage_lat = LatencyMetrics(50, 2.0, 5.0, 4.8, 8.0, 1.0, 1.2, 4.8, 6.5, 7.5, 8.0, 8.0)
    breakdown = ComponentLatencyBreakdown(
        stage_metrics={PipelineStage.TREESHAP_EXPLAINABILITY: stage_lat},
        stage_percentages={PipelineStage.TREESHAP_EXPLAINABILITY: 100.0},
        total_component_mean_ms=5.0,
    )

    bottleneck = BottleneckAnalysisResult(
        component_breakdown=breakdown,
        primary_bottleneck_stage="treeshap_explainability",
        primary_bottleneck_mean_ms=5.0,
        primary_bottleneck_pct=100.0,
        secondary_bottleneck_stage="none",
        secondary_bottleneck_mean_ms=0.0,
        secondary_bottleneck_pct=0.0,
        diagnostic_summary="TreeSHAP is the primary bottleneck.",
    )

    return BenchmarkSuiteResult(
        suite_id="suite_test_456",
        config=cfg,
        cold_start_latency_ms=35.0,
        warmup_requests_executed=10,
        warmup_latency=mock_lat_1,
        concurrency_sweep=sweep,
        ablation=ablation,
        bottleneck_analysis=bottleneck,
        target_sla_evaluated=sla,
        timestamp_utc="2026-09-17T12:00:00Z",
    )


class TestBenchmarkReporterUnit:
    """Unit tests for BenchmarkReporter formatting and export mechanisms."""

    def test_get_system_metadata(self) -> None:
        """Verify system metadata extraction contains necessary hardware & OS details."""
        meta = BenchmarkReporter.get_system_metadata()
        assert "platform" in meta
        assert "os_name" in meta
        assert "cpu_cores_logical" in meta
        assert "python_version" in meta
        assert meta["cpu_cores_logical"] >= 1

    def test_format_console_summary_contains_all_sections(
        self, sample_suite_result: BenchmarkSuiteResult
    ) -> None:
        """Verify console ASCII formatting renders all 5 evaluation sections."""
        summary = BenchmarkReporter.format_console_summary(sample_suite_result)

        assert "AI-POWERED FRAUD DETECTION & RISK INTELLIGENCE PLATFORM" in summary
        assert "1. COLD-START & WARM-UP LATENCY ISOLATION" in summary
        assert "2. MULTI-CONCURRENCY SCALABILITY MATRIX" in summary
        assert "3. MULTI-TIER PERSISTENCE ABLATION" in summary
        assert "4. SEVEN-STAGE COMPONENT LATENCY BREAKDOWN" in summary
        assert "5. ENGINEERING REFERENCE TARGET EVALUATION" in summary
        assert "NON-CONTRACTUAL" in summary
        assert "C=1" in summary
        assert "C=2" in summary

    def test_export_json_and_sensitive_data_exclusion(
        self, sample_suite_result: BenchmarkSuiteResult, tmp_path: Path
    ) -> None:
        """Verify JSON export contains structured metrics and excludes sensitive payload PII."""
        out_file = tmp_path / "benchmark_report.json"
        saved_path = BenchmarkReporter.export_json(sample_suite_result, out_file)

        assert saved_path.exists()

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "benchmark_config" in data
        assert "suite_results" in data
        assert data["suite_results"]["suite_id"] == "suite_test_456"

        # Verify sensitive transaction data / PII exclusion
        json_str = json.dumps(data)
        assert "credit_card" not in json_str.lower()
        assert "password" not in json_str.lower()
        assert "ssn" not in json_str.lower()

    def test_export_csv_sections(
        self, sample_suite_result: BenchmarkSuiteResult, tmp_path: Path
    ) -> None:
        """Verify CSV export formats tabular data for sweep, components, and ablation."""
        out_file = tmp_path / "benchmark_report.csv"
        saved_path = BenchmarkReporter.export_csv(sample_suite_result, out_file)

        assert saved_path.exists()

        with open(saved_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "=== MULTI-CONCURRENCY SCALABILITY MATRIX ===" in content
        assert "=== SEVEN-STAGE COMPONENT LATENCY BREAKDOWN ===" in content
        assert "=== MULTI-TIER PERSISTENCE ABLATION ===" in content
        assert "Completed_TPS" in content
        assert "Mode_A_Full_API_PostgreSQL" in content

    def test_format_markdown_tables(
        self, sample_suite_result: BenchmarkSuiteResult
    ) -> None:
        """Verify Markdown table generation produces valid GFM tables."""
        md = BenchmarkReporter.format_markdown_tables(sample_suite_result)

        assert "### Concurrency Scalability Matrix" in md
        assert "### Multi-Tier Persistence Ablation" in md
        assert "### Seven-Stage Component Latency Breakdown" in md
        assert "| **C = 1** |" in md
        assert "| **C = 2** |" in md
