"""
Unit tests for MetricCollector and RequestMetricSample (Phase 10.1).
"""

import asyncio
import pytest

from backend.app.benchmarking.collector import MetricCollector, RequestMetricSample
from backend.app.benchmarking.config import TargetSLA


class TestRequestMetricSample:
    """Tests for RequestMetricSample dataclass validation."""

    def test_valid_sample_creation(self) -> None:
        """Test valid sample creation and properties."""
        s = RequestMetricSample(
            request_id="req_001",
            start_time_ns=1000,
            end_time_ns=2000,
            latency_ms=1.0,
            status_code=200,
            success=True,
            stage_latencies_ms={"request_validation": 0.2, "ml_inference": 0.8},
        )
        assert s.request_id == "req_001"
        assert s.latency_ms == 1.0
        assert s.status_code == 200
        assert s.success is True
        assert s.error_message is None
        assert s.stage_latencies_ms["ml_inference"] == 0.8
        assert s.timestamp_iso is not None

    def test_empty_request_id_raises_value_error(self) -> None:
        """Test empty request_id raises ValueError."""
        with pytest.raises(ValueError, match="request_id cannot be empty"):
            RequestMetricSample(
                request_id="",
                start_time_ns=0,
                end_time_ns=100,
                latency_ms=0.1,
            )

    def test_negative_latency_raises_value_error(self) -> None:
        """Test negative latency raises ValueError."""
        with pytest.raises(ValueError, match="latency_ms cannot be negative"):
            RequestMetricSample(
                request_id="req_001",
                start_time_ns=0,
                end_time_ns=100,
                latency_ms=-0.5,
            )

    def test_sample_to_dict(self) -> None:
        """Test to_dict produces clean dictionary."""
        s = RequestMetricSample(
            request_id="req_001",
            start_time_ns=1000,
            end_time_ns=2000,
            latency_ms=1.0,
        )
        d = s.to_dict()
        assert d["request_id"] == "req_001"
        assert d["latency_ms"] == 1.0
        assert d["success"] is True


class TestMetricCollector:
    """Tests for MetricCollector thread safety, recording, and summaries."""

    def test_empty_collector_state(self) -> None:
        """Test initial state of an empty collector."""
        c = MetricCollector()
        assert c.count() == 0
        assert c.successful_count() == 0
        assert c.failed_count() == 0
        assert c.get_samples() == []
        assert c.get_latencies() == []
        assert c.get_stage_latencies() == {}

        summary = c.compute_summary()
        assert summary["total_requests"] == 0
        assert summary["successful_requests"] == 0

    def test_record_request(self) -> None:
        """Test record_request appends valid sample."""
        c = MetricCollector()
        sample = c.record_request(
            request_id="tx_1",
            latency_ms=12.5,
            status_code=200,
            success=True,
            stage_latencies_ms={"ml_inference": 4.5},
        )
        assert c.count() == 1
        assert c.successful_count() == 1
        assert c.failed_count() == 0
        assert sample.request_id == "tx_1"
        assert sample.latency_ms == 12.5

    def test_record_error(self) -> None:
        """Test record_error appends failed sample."""
        c = MetricCollector()
        sample = c.record_error(
            request_id="tx_err",
            latency_ms=2.0,
            status_code=422,
            error_message="Validation error",
        )
        assert c.count() == 1
        assert c.successful_count() == 0
        assert c.failed_count() == 1
        assert sample.success is False
        assert sample.status_code == 422
        assert sample.error_message == "Validation error"

    def test_get_latencies_filtering(self) -> None:
        """Test successful_only filter in get_latencies."""
        c = MetricCollector()
        c.record_request("tx_1", 10.0, success=True)
        c.record_request("tx_2", 20.0, success=True)
        c.record_error("tx_err", 5.0, status_code=500)

        assert c.get_latencies(successful_only=True) == [10.0, 20.0]
        assert c.get_latencies(successful_only=False) == [10.0, 20.0, 5.0]

    def test_get_stage_latencies_aggregation(self) -> None:
        """Test stage latency organization."""
        c = MetricCollector()
        c.record_request("tx_1", 10.0, stage_latencies_ms={"val": 1.0, "ml": 5.0})
        c.record_request("tx_2", 12.0, stage_latencies_ms={"val": 1.2, "ml": 5.5})

        stage_map = c.get_stage_latencies()
        assert stage_map["val"] == [1.0, 1.2]
        assert stage_map["ml"] == [5.0, 5.5]

    def test_clear_and_reset(self) -> None:
        """Test clear resets state."""
        c = MetricCollector()
        c.record_request("tx_1", 10.0)
        assert c.count() == 1
        c.clear()
        assert c.count() == 0

        c.record_request("tx_2", 15.0)
        assert c.count() == 1
        c.reset()
        assert c.count() == 0

    def test_high_res_timers(self) -> None:
        """Test start_timer and stop_timer non-negativity."""
        t0 = MetricCollector.start_timer()
        assert isinstance(t0, int)
        assert t0 > 0

        lat = MetricCollector.stop_timer(t0)
        assert isinstance(lat, float)
        assert lat >= 0.0

    def test_compute_summary_full(self) -> None:
        """Test summary computation with multiple samples and SLA target."""
        c = MetricCollector()
        c.record_request("tx_1", 10.0, stage_latencies_ms={"ml": 5.0})
        c.record_request("tx_2", 20.0, stage_latencies_ms={"ml": 8.0})
        c.record_request("tx_3", 30.0, stage_latencies_ms={"ml": 12.0})
        c.record_error("tx_4", 5.0, status_code=500)

        sla = TargetSLA.standard_gateway_target()
        summary = c.compute_summary(elapsed_seconds=1.0, target_sla=sla)

        assert summary["total_requests"] == 4
        assert summary["successful_requests"] == 3
        assert summary["failed_requests"] == 1
        assert summary["elapsed_seconds"] == 1.0

        assert summary["latency"]["count"] == 3
        assert summary["latency"]["min_ms"] == 10.0
        assert summary["latency"]["max_ms"] == 30.0
        assert summary["latency"]["mean_ms"] == 20.0

        assert summary["throughput"]["offered_tps"] == 4.0
        assert summary["throughput"]["successful_tps"] == 3.0
        assert summary["throughput"]["error_rate_percent"] == 25.0

        assert "component_breakdown" in summary
        assert "sla_compliance" in summary

    @pytest.mark.asyncio
    async def test_async_safe_concurrent_recording(self) -> None:
        """Test concurrent recording across async tasks."""
        c = MetricCollector()

        async def worker(worker_id: int) -> None:
            for i in range(50):
                c.record_request(
                    request_id=f"w{worker_id}_{i}",
                    latency_ms=10.0 + i,
                    stage_latencies_ms={"stage_a": 1.0},
                )

        tasks = [worker(w) for w in range(10)]
        await asyncio.gather(*tasks)

        assert c.count() == 500
        assert c.successful_count() == 500
        assert len(c.get_latencies()) == 500
