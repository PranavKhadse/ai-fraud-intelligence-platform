"""
Unit tests for Benchmarking Configuration and TargetSLA module (Phase 10.1).
"""

from pathlib import Path
import pytest

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass and boundary validation."""

    def test_default_config_instantiation(self) -> None:
        """Test default values of BenchmarkConfig."""
        config = BenchmarkConfig()
        assert config.num_requests == 500
        assert config.concurrency_levels == (1, 2, 4, 8, 16)
        assert config.warmup_requests == 50
        assert config.dataset_path == Path("data/processed/features/test_features.parquet")
        assert config.target_rate is None
        assert config.include_persistence is True
        assert config.random_seed == 42

    def test_custom_valid_config(self) -> None:
        """Test custom valid values."""
        config = BenchmarkConfig(
            num_requests=1000,
            concurrency_levels=(2, 4),
            warmup_requests=10,
            dataset_path="data/processed/features/val_features.parquet",
            target_rate=150.0,
            include_persistence=False,
            random_seed=123,
        )
        assert config.num_requests == 1000
        assert config.concurrency_levels == (2, 4)
        assert config.warmup_requests == 10
        assert config.dataset_path == Path("data/processed/features/val_features.parquet")
        assert config.target_rate == 150.0
        assert config.include_persistence is False
        assert config.random_seed == 123

    def test_invalid_num_requests_type(self) -> None:
        """Test non-integer num_requests raises TypeError."""
        with pytest.raises(TypeError, match="num_requests must be an integer"):
            BenchmarkConfig(num_requests="500")  # type: ignore

        with pytest.raises(TypeError, match="num_requests must be an integer"):
            BenchmarkConfig(num_requests=True)  # type: ignore

    def test_invalid_num_requests_value(self) -> None:
        """Test zero or negative num_requests raises ValueError."""
        with pytest.raises(ValueError, match="num_requests must be strictly positive"):
            BenchmarkConfig(num_requests=0)

        with pytest.raises(ValueError, match="num_requests must be strictly positive"):
            BenchmarkConfig(num_requests=-10)

    def test_invalid_warmup_requests_type(self) -> None:
        """Test non-integer warmup_requests raises TypeError."""
        with pytest.raises(TypeError, match="warmup_requests must be an integer"):
            BenchmarkConfig(warmup_requests="10")  # type: ignore

        with pytest.raises(TypeError, match="warmup_requests must be an integer"):
            BenchmarkConfig(warmup_requests=False)  # type: ignore

    def test_invalid_warmup_requests_value(self) -> None:
        """Test negative warmup_requests raises ValueError."""
        with pytest.raises(ValueError, match="warmup_requests must be non-negative"):
            BenchmarkConfig(warmup_requests=-5)

    def test_zero_warmup_requests_is_allowed(self) -> None:
        """Test that warmup_requests=0 is valid."""
        config = BenchmarkConfig(warmup_requests=0)
        assert config.warmup_requests == 0

    def test_invalid_concurrency_levels_type(self) -> None:
        """Test non-sequence concurrency_levels raises TypeError."""
        with pytest.raises(TypeError, match="concurrency_levels must be a sequence"):
            BenchmarkConfig(concurrency_levels="1,2,4")  # type: ignore

    def test_empty_concurrency_levels_raises_value_error(self) -> None:
        """Test empty concurrency_levels sequence raises ValueError."""
        with pytest.raises(ValueError, match="concurrency_levels sequence cannot be empty"):
            BenchmarkConfig(concurrency_levels=())

    def test_invalid_concurrency_element_type(self) -> None:
        """Test non-integer element in concurrency_levels raises TypeError."""
        with pytest.raises(TypeError, match="All concurrency values must be integers"):
            BenchmarkConfig(concurrency_levels=(1, "2", 4))  # type: ignore

        with pytest.raises(TypeError, match="All concurrency values must be integers"):
            BenchmarkConfig(concurrency_levels=(1, True, 4))  # type: ignore

    def test_invalid_concurrency_element_value(self) -> None:
        """Test non-positive element in concurrency_levels raises ValueError."""
        with pytest.raises(ValueError, match="Every concurrency level must be positive"):
            BenchmarkConfig(concurrency_levels=(1, 0, 4))

        with pytest.raises(ValueError, match="Every concurrency level must be positive"):
            BenchmarkConfig(concurrency_levels=(1, -2, 4))

    def test_invalid_dataset_path_type(self) -> None:
        """Test non-string/non-Path dataset_path raises TypeError."""
        with pytest.raises(TypeError, match="dataset_path must be a string or Path"):
            BenchmarkConfig(dataset_path=123)  # type: ignore

    def test_invalid_target_rate_type(self) -> None:
        """Test non-numeric target_rate raises TypeError."""
        with pytest.raises(TypeError, match="target_rate must be a numeric float or int"):
            BenchmarkConfig(target_rate="100")  # type: ignore

        with pytest.raises(TypeError, match="target_rate must be a numeric float or int"):
            BenchmarkConfig(target_rate=True)  # type: ignore

    def test_invalid_target_rate_value(self) -> None:
        """Test non-positive target_rate raises ValueError."""
        with pytest.raises(ValueError, match="target_rate must be strictly positive"):
            BenchmarkConfig(target_rate=0.0)

        with pytest.raises(ValueError, match="target_rate must be strictly positive"):
            BenchmarkConfig(target_rate=-15.0)

    def test_invalid_include_persistence_type(self) -> None:
        """Test non-boolean include_persistence raises TypeError."""
        with pytest.raises(TypeError, match="include_persistence must be a boolean"):
            BenchmarkConfig(include_persistence="true")  # type: ignore

    def test_invalid_random_seed_type(self) -> None:
        """Test non-integer random_seed raises TypeError."""
        with pytest.raises(TypeError, match="random_seed must be an integer"):
            BenchmarkConfig(random_seed="42")  # type: ignore

        with pytest.raises(TypeError, match="random_seed must be an integer"):
            BenchmarkConfig(random_seed=True)  # type: ignore


class TestTargetSLA:
    """Tests for TargetSLA reference target configuration."""

    def test_default_target_sla(self) -> None:
        """Test default TargetSLA attributes."""
        sla = TargetSLA()
        assert sla.sla_name == "Reference Engineering Target"
        assert sla.target_p50_ms is None
        assert sla.target_p95_ms is None
        assert sla.target_p99_ms is None

    def test_standard_gateway_target(self) -> None:
        """Test standard_gateway_target factory method."""
        sla = TargetSLA.standard_gateway_target()
        assert sla.target_p50_ms == 15.0
        assert sla.target_p95_ms == 35.0
        assert sla.target_p99_ms == 60.0
        assert "Payment Gateway" in sla.sla_name

    def test_ultra_low_latency_target(self) -> None:
        """Test ultra_low_latency_target factory method."""
        sla = TargetSLA.ultra_low_latency_target()
        assert sla.target_p50_ms == 10.0
        assert sla.target_p95_ms == 25.0
        assert sla.target_p99_ms == 50.0
        assert "Ultra-Low Latency" in sla.sla_name

    def test_empty_sla_name_raises_value_error(self) -> None:
        """Test empty SLA name raises ValueError."""
        with pytest.raises(ValueError, match="sla_name cannot be empty"):
            TargetSLA(sla_name="")

        with pytest.raises(ValueError, match="sla_name cannot be empty"):
            TargetSLA(sla_name="   ")

    def test_invalid_threshold_type(self) -> None:
        """Test non-numeric threshold raises TypeError."""
        with pytest.raises(TypeError, match="target_p50_ms must be a numeric float or int"):
            TargetSLA(target_p50_ms="15.0")  # type: ignore

        with pytest.raises(TypeError, match="target_p95_ms must be a numeric float or int"):
            TargetSLA(target_p95_ms=True)  # type: ignore

    def test_non_positive_threshold_value(self) -> None:
        """Test zero or negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="target_p50_ms must be strictly positive"):
            TargetSLA(target_p50_ms=0.0)

        with pytest.raises(ValueError, match="target_p99_ms must be strictly positive"):
            TargetSLA(target_p99_ms=-5.0)

    def test_non_monotonic_thresholds_raise_value_error(self) -> None:
        """Test P50 > P95 or P95 > P99 raises ValueError."""
        with pytest.raises(ValueError, match="target_p50_ms .* cannot exceed target_p95_ms"):
            TargetSLA(target_p50_ms=40.0, target_p95_ms=30.0)

        with pytest.raises(ValueError, match="target_p95_ms .* cannot exceed target_p99_ms"):
            TargetSLA(target_p95_ms=70.0, target_p99_ms=60.0)
