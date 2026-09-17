"""
Unit tests for Latency and Throughput Metrics calculation engine (Phase 10.1).
"""

import numpy as np
import pytest

from backend.app.benchmarking.config import TargetSLA
from backend.app.benchmarking.metrics import (
    calculate_percentiles,
    ComponentLatencyBreakdown,
    LatencyMetrics,
    PipelineStage,
    SLACompliance,
    ThroughputMetrics,
)


class TestCalculatePercentiles:
    """Tests for calculate_percentiles function using NumPy linear interpolation."""

    def test_known_uniform_distribution(self) -> None:
        """Test percentiles on 101 points from 0 to 100."""
        data = list(range(101))  # 0, 1, 2, ..., 100
        pcts = calculate_percentiles(data, percentiles=[0.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0])
        assert pcts[0.0] == 0.0
        assert pcts[25.0] == 25.0
        assert pcts[50.0] == 50.0
        assert pcts[75.0] == 75.0
        assert pcts[90.0] == 90.0
        assert pcts[95.0] == 95.0
        assert pcts[99.0] == 99.0
        assert pcts[100.0] == 100.0

    def test_single_element_sample(self) -> None:
        """Test all percentiles of a single element return that exact element."""
        data = [42.5]
        pcts = calculate_percentiles(data, percentiles=[0.0, 50.0, 95.0, 99.0, 100.0])
        for p, val in pcts.items():
            assert val == 42.5

    def test_repeated_values(self) -> None:
        """Test constant vector has identical percentile values."""
        data = [10.0] * 50
        pcts = calculate_percentiles(data, percentiles=[0.0, 50.0, 90.0, 99.0, 100.0])
        for p, val in pcts.items():
            assert val == 10.0

    def test_two_elements(self) -> None:
        """Test two-element interpolation."""
        data = [10.0, 20.0]
        pcts = calculate_percentiles(data, percentiles=[0.0, 50.0, 100.0])
        assert pcts[0.0] == 10.0
        assert pcts[50.0] == 15.0
        assert pcts[100.0] == 20.0

    def test_empty_input_raises_value_error(self) -> None:
        """Test empty latency list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate percentiles on an empty"):
            calculate_percentiles([])

        with pytest.raises(ValueError, match="Cannot calculate percentiles on an empty"):
            calculate_percentiles(None)  # type: ignore

    def test_invalid_percentile_value_raises_value_error(self) -> None:
        """Test percentile outside [0.0, 100.0] raises ValueError."""
        with pytest.raises(ValueError, match="out of valid range"):
            calculate_percentiles([1.0, 2.0, 3.0], percentiles=[-1.0])

        with pytest.raises(ValueError, match="out of valid range"):
            calculate_percentiles([1.0, 2.0, 3.0], percentiles=[101.0])

    def test_non_numeric_percentile_raises_type_error(self) -> None:
        """Test non-numeric percentile raises TypeError."""
        with pytest.raises(TypeError, match="Percentile must be a numeric float or int"):
            calculate_percentiles([1.0, 2.0], percentiles=["50"])  # type: ignore

        with pytest.raises(TypeError, match="Percentile must be a numeric float or int"):
            calculate_percentiles([1.0, 2.0], percentiles=[True])  # type: ignore

    def test_non_numeric_latency_raises_type_error(self) -> None:
        """Test non-numeric element in latencies raises TypeError."""
        with pytest.raises(TypeError, match="All latency elements must be numeric"):
            calculate_percentiles(["invalid", 1.0])  # type: ignore


class TestLatencyMetrics:
    """Tests for LatencyMetrics statistical aggregator."""

    def test_from_latencies_deterministic_vector(self) -> None:
        """Test statistical aggregations on known vector."""
        # 1, 2, 3, 4, 5
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = LatencyMetrics.from_latencies(data)

        assert m.count == 5
        assert m.min_ms == 1.0
        assert m.max_ms == 5.0
        assert m.mean_ms == 3.0
        assert m.median_ms == 3.0
        assert m.p50_ms == 3.0
        # NumPy linear percentile for [1,2,3,4,5]: 25th = 2.0, 75th = 4.0 => IQR = 2.0
        assert m.iqr_ms == 2.0
        # Sample std for [1,2,3,4,5] (ddof=1) is sqrt(2.5) ~ 1.581139
        assert round(m.std_ms, 4) == 1.5811

    def test_single_element_latency_metrics(self) -> None:
        """Test LatencyMetrics with 1 element."""
        m = LatencyMetrics.from_latencies([15.2])
        assert m.count == 1
        assert m.min_ms == 15.2
        assert m.max_ms == 15.2
        assert m.mean_ms == 15.2
        assert m.median_ms == 15.2
        assert m.std_ms == 0.0
        assert m.iqr_ms == 0.0

    def test_empty_latencies_raises_value_error(self) -> None:
        """Test empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Cannot construct LatencyMetrics from an empty"):
            LatencyMetrics.from_latencies([])

    def test_to_dict_serialization(self) -> None:
        """Test to_dict produces expected structure."""
        m = LatencyMetrics.from_latencies([10.0, 20.0, 30.0])
        d = m.to_dict()
        assert d["count"] == 3
        assert d["min_ms"] == 10.0
        assert d["max_ms"] == 30.0
        assert "p50_ms" in d
        assert "p95_ms" in d
        assert "p99_ms" in d
        assert "std_ms" in d


class TestThroughputMetrics:
    """Tests for ThroughputMetrics accounting and formulas."""

    def test_normal_throughput_calculation(self) -> None:
        """Test normal throughput numbers."""
        # 100 requests in 2.0 seconds: 90 successful, 10 failed
        tm = ThroughputMetrics.from_counts(
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
            elapsed_seconds=2.0,
        )
        assert tm.total_requests == 100
        assert tm.successful_requests == 90
        assert tm.failed_requests == 10
        assert tm.elapsed_seconds == 2.0
        assert tm.offered_tps == 50.0
        assert tm.completed_tps == 50.0
        assert tm.successful_tps == 45.0
        assert tm.error_rate_percent == 10.0

    def test_zero_elapsed_time(self) -> None:
        """Test zero elapsed time does not divide by zero or raise exception."""
        tm = ThroughputMetrics.from_counts(
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            elapsed_seconds=0.0,
        )
        assert tm.offered_tps == 0.0
        assert tm.completed_tps == 0.0
        assert tm.successful_tps == 0.0
        assert tm.error_rate_percent == 0.0

    def test_zero_total_requests(self) -> None:
        """Test 0 requests handling."""
        tm = ThroughputMetrics.from_counts(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            elapsed_seconds=1.0,
        )
        assert tm.offered_tps == 0.0
        assert tm.error_rate_percent == 0.0

    def test_negative_counts_raise_value_error(self) -> None:
        """Test negative counts raise ValueError."""
        with pytest.raises(ValueError, match="total_requests cannot be negative"):
            ThroughputMetrics.from_counts(-1, 0, 0, 1.0)

        with pytest.raises(ValueError, match="successful_requests cannot be negative"):
            ThroughputMetrics.from_counts(10, -1, 0, 1.0)

        with pytest.raises(ValueError, match="failed_requests cannot be negative"):
            ThroughputMetrics.from_counts(10, 10, -1, 1.0)

        with pytest.raises(ValueError, match="elapsed_seconds cannot be negative"):
            ThroughputMetrics.from_counts(10, 10, 0, -1.0)


class TestComponentLatencyBreakdown:
    """Tests for ComponentLatencyBreakdown aggregation across the 7 stages."""

    def test_from_stage_latencies(self) -> None:
        """Test stage latency aggregation and percentage contribution."""
        stage_data = {
            PipelineStage.REQUEST_VALIDATION: [1.0, 1.0, 1.0],
            PipelineStage.FEATURE_PREPARATION: [2.0, 2.0, 2.0],
            PipelineStage.ML_INFERENCE: [3.0, 3.0, 3.0],
            PipelineStage.TREESHAP_EXPLAINABILITY: [4.0, 4.0, 4.0],
            PipelineStage.RULE_ENGINE_POLICY: [1.0, 1.0, 1.0],
            PipelineStage.PERSISTENCE_MAPPING: [1.0, 1.0, 1.0],
            PipelineStage.POSTGRES_PERSISTENCE: [8.0, 8.0, 8.0],
        }
        # Total sum of means = 1 + 2 + 3 + 4 + 1 + 1 + 8 = 20.0
        breakdown = ComponentLatencyBreakdown.from_stage_latencies(stage_data)

        assert breakdown.total_component_mean_ms == 20.0
        assert breakdown.stage_percentages[PipelineStage.REQUEST_VALIDATION] == 5.0
        assert breakdown.stage_percentages[PipelineStage.FEATURE_PREPARATION] == 10.0
        assert breakdown.stage_percentages[PipelineStage.ML_INFERENCE] == 15.0
        assert breakdown.stage_percentages[PipelineStage.TREESHAP_EXPLAINABILITY] == 20.0
        assert breakdown.stage_percentages[PipelineStage.POSTGRES_PERSISTENCE] == 40.0

        d = breakdown.to_dict()
        assert d["total_component_mean_ms"] == 20.0
        assert "request_validation" in d["stages"]
        assert "postgres_persistence" in d["stages"]


class TestSLACompliance:
    """Tests for SLACompliance evaluator against TargetSLA."""

    def test_compliant_scenario(self) -> None:
        """Test dataset that satisfies standard target."""
        # 100 requests all <= 10.0 ms
        latencies = [5.0] * 50 + [10.0] * 50
        metrics = LatencyMetrics.from_latencies(latencies)
        sla = TargetSLA.standard_gateway_target()  # P50<=15, P95<=35, P99<=60

        compliance = SLACompliance.from_latency_metrics(metrics, latencies, sla)
        assert compliance.p50_compliant is True
        assert compliance.p95_compliant is True
        assert compliance.p99_compliant is True
        assert compliance.overall_compliant is True
        assert compliance.under_15ms_pct == 100.0
        assert compliance.under_25ms_pct == 100.0

    def test_non_compliant_scenario(self) -> None:
        """Test dataset that violates target."""
        # 100 requests all 40.0 ms
        latencies = [40.0] * 100
        metrics = LatencyMetrics.from_latencies(latencies)
        sla = TargetSLA.standard_gateway_target()  # P50<=15, P95<=35, P99<=60

        compliance = SLACompliance.from_latency_metrics(metrics, latencies, sla)
        assert compliance.p50_compliant is False
        assert compliance.p95_compliant is False
        assert compliance.p99_compliant is True  # 40 <= 60
        assert compliance.overall_compliant is False
        assert compliance.under_15ms_pct == 0.0
        assert compliance.under_50ms_pct == 100.0
