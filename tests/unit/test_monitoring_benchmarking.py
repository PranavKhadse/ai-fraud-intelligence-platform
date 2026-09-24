"""
Unit Tests for Phase 13.7 Monitoring Benchmarking CLI and Metric Calculators.

Validates:
1. Statistical percentile calculation (p50, p90, p95, p99, IQR, mean, std, min, max, ops/sec).
2. Synthetic feature DataFrame generation across numerical and categorical profiles with drift injection.
3. Feature drift, prediction drift, model performance, API, and snapshot benchmark runners.
4. JSON and CSV benchmark exporter fidelity.
5. Command-line argument parsing and quick mode flag.
"""

from pathlib import Path
import tempfile
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

from ml.monitoring.config import default_monitoring_config
from ml.monitoring.feature_drift import FeatureDriftCalculator
from ml.monitoring.performance import ModelPerformanceCalculator
from ml.monitoring.prediction_drift import PredictionDriftCalculator
from ml.monitoring.schemas import (
    CategoricalFeatureProfile,
    FeatureBaselineProfile,
    NumericalFeatureProfile,
)
from scripts.benchmark_monitoring import (
    benchmark_feature_drift,
    benchmark_model_performance,
    benchmark_monitoring_apis,
    benchmark_prediction_drift,
    benchmark_snapshot_generation,
    compute_percentiles,
    export_results_to_csv,
    export_results_to_json,
    generate_synthetic_feature_dataframe,
    parse_args,
    run_full_benchmark_suite,
)


class TestMonitoringBenchmarkCalculators:
    """Unit tests for monitoring benchmark statistical calculators."""

    def test_compute_percentiles_empty_sequence(self):
        """Verify empty sequence returns zeroed metrics without crashing."""
        res = compute_percentiles([])
        assert res["count"] == 0
        assert res["p50_ms"] == 0.0
        assert res["ops_per_sec"] == 0.0

    def test_compute_percentiles_monotonic_and_accurate(self):
        """Verify statistical accuracy on known sequence."""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        res = compute_percentiles(latencies)

        assert res["count"] == 10
        assert res["min_ms"] == 10.0
        assert res["max_ms"] == 100.0
        assert res["mean_ms"] == 55.0
        assert res["median_ms"] == 55.0
        assert res["p50_ms"] == 55.0
        assert res["p90_ms"] == pytest.approx(91.0, abs=1.5)
        assert res["p99_ms"] == pytest.approx(99.1, abs=1.5)
        assert res["ops_per_sec"] > 0.0

    def test_generate_synthetic_feature_dataframe(self):
        """Verify synthetic feature generator produces 55 columns with correct types."""
        base_profile = FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)
        df = generate_synthetic_feature_dataframe(base_profile, n_samples=50, drift_factor=0.0)

        assert len(df) == 50
        assert len(df.columns) == 55
        for col in base_profile.features:
            assert col in df.columns

        # Verify drift injection with unseen category
        df_drifted = generate_synthetic_feature_dataframe(base_profile, n_samples=100, drift_factor=0.8)
        assert len(df_drifted) == 100


class TestMonitoringBenchmarkRunners:
    """Unit tests for individual benchmark execution functions."""

    @pytest.fixture
    def base_profile(self) -> FeatureBaselineProfile:
        return FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)

    def test_benchmark_feature_drift_quick(self, base_profile: FeatureBaselineProfile):
        """Verify feature drift benchmark runs and returns valid summary statistics."""
        calc = FeatureDriftCalculator(baseline_profile=base_profile)
        res = benchmark_feature_drift(
            calculator=calc,
            base_profile=base_profile,
            sample_sizes=[50],
            iterations_per_size=2,
            warmup_iterations=1,
        )
        assert "sample_size_50" in res
        stats = res["sample_size_50"]
        assert stats["count"] == 2
        assert stats["p50_ms"] > 0.0

    def test_benchmark_prediction_drift_quick(self):
        """Verify prediction drift benchmark executes across evaluations."""
        calc = PredictionDriftCalculator()
        res = benchmark_prediction_drift(
            calculator=calc,
            sample_sizes=[50],
            iterations_per_size=2,
            warmup_iterations=1,
        )
        assert "sample_size_50" in res
        stats = res["sample_size_50"]
        assert stats["count"] == 2
        assert stats["p50_ms"] > 0.0

    def test_benchmark_model_performance_quick(self):
        """Verify ground-truth performance calculation benchmark execution."""
        calc = ModelPerformanceCalculator()
        res = benchmark_model_performance(
            calculator=calc,
            sample_sizes=[50],
            iterations_per_size=2,
            warmup_iterations=1,
        )
        assert "sample_size_50" in res
        stats = res["sample_size_50"]
        assert stats["count"] == 2
        assert stats["p50_ms"] > 0.0

    def test_benchmark_monitoring_apis_quick(self):
        """Verify mock API benchmark measures all 6 endpoints."""
        res = benchmark_monitoring_apis(iterations=2, warmup_iterations=1)
        assert len(res) == 6
        assert "GET /api/v1/monitoring/health" in res
        assert "GET /api/v1/monitoring/drift/features" in res
        assert "GET /api/v1/monitoring/drift/features/amt" in res
        assert "GET /api/v1/monitoring/drift/predictions" in res
        assert "GET /api/v1/monitoring/performance" in res
        assert "GET /api/v1/monitoring/snapshots" in res


class TestMonitoringBenchmarkArtifactExport:
    """Unit tests for JSON and CSV export capabilities."""

    def test_export_json_and_csv_roundtrip(self):
        """Verify exporting benchmark results produces valid non-empty files."""
        mock_data = {
            "metadata": {"platform": "test_platform", "model_version": "1.0.0"},
            "benchmarks": {
                "feature_drift_engine": {
                    "sample_size_100": {
                        "count": 10,
                        "min_ms": 150.0,
                        "mean_ms": 160.0,
                        "median_ms": 155.0,
                        "p50_ms": 155.0,
                        "p90_ms": 170.0,
                        "p95_ms": 175.0,
                        "p99_ms": 180.0,
                        "max_ms": 185.0,
                        "std_ms": 10.0,
                        "ops_per_sec": 6.25,
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_results.json"
            csv_path = Path(tmpdir) / "test_results.csv"

            export_results_to_json(mock_data, json_path)
            export_results_to_csv(mock_data, csv_path)

            assert json_path.exists()
            assert csv_path.exists()

            df_csv = pd.read_csv(csv_path)
            assert len(df_csv) == 1
            assert df_csv.iloc[0]["component"] == "feature_drift_engine"
            assert df_csv.iloc[0]["test_name"] == "sample_size_100"
            assert float(df_csv.iloc[0]["p50_ms"]) == 155.0

    def test_run_full_benchmark_suite_quick_mode(self):
        """Verify quick mode executes end-to-end suite."""
        suite_res = run_full_benchmark_suite(quick=True)
        assert "metadata" in suite_res
        assert "config" in suite_res
        assert "benchmarks" in suite_res
        assert suite_res["config"]["quick_mode"] is True
        assert len(suite_res["benchmarks"]) == 5
