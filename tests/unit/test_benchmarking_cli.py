"""
Unit tests for benchmark CLI argument parsing, configuration building, and error handling (Increment 10.4).
"""

from pathlib import Path
import pytest

from scripts.benchmark import build_benchmark_config, parse_args


class TestBenchmarkCLIUnit:
    """Unit test suite for scripts/benchmark.py command-line argument parsing."""

    def test_parse_args_defaults(self) -> None:
        """Verify CLI default argument parsing."""
        args = parse_args([])
        assert args.requests == 500
        assert args.concurrency == "1,2,4,8,16"
        assert args.warmup == 50
        assert args.transport == "asgi"
        assert args.seed == 42
        assert args.reference_target == "standard"
        assert not args.no_ablation
        assert not args.no_bottleneck
        assert not args.quiet

    def test_parse_args_custom(self) -> None:
        """Verify CLI custom argument values."""
        custom_args = [
            "--requests", "100",
            "--concurrency", "1,4,8",
            "--warmup", "10",
            "--transport", "http",
            "--target-url", "http://localhost:9000",
            "--seed", "123",
            "--reference-target", "ultra_low",
            "--no-ablation",
            "--quiet",
        ]
        args = parse_args(custom_args)
        assert args.requests == 100
        assert args.concurrency == "1,4,8"
        assert args.warmup == 10
        assert args.transport == "http"
        assert args.target_url == "http://localhost:9000"
        assert args.seed == 123
        assert args.reference_target == "ultra_low"
        assert args.no_ablation
        assert args.quiet

    def test_build_benchmark_config_standard(self) -> None:
        """Verify construction of BenchmarkConfig and TargetSLA from parsed args."""
        args = parse_args(["--requests", "250", "--concurrency", "2,4,8", "--seed", "99"])
        config, sla = build_benchmark_config(args)

        assert config.num_requests == 250
        assert config.concurrency_levels == (2, 4, 8)
        assert config.random_seed == 99
        assert sla is not None
        assert sla.sla_name == "Standard Payment Gateway Reference Target"
        assert sla.target_p50_ms == 15.0

    def test_build_benchmark_config_ultra_low(self) -> None:
        """Verify construction of TargetSLA with ultra_low target."""
        args = parse_args(["--reference-target", "ultra_low"])
        _, sla = build_benchmark_config(args)

        assert sla is not None
        assert sla.sla_name == "Ultra-Low Latency Reference Target"
        assert sla.target_p50_ms == 10.0

    def test_build_benchmark_config_no_reference(self) -> None:
        """Verify construction with reference-target='none' returns None for SLA."""
        args = parse_args(["--reference-target", "none"])
        _, sla = build_benchmark_config(args)
        assert sla is None

    def test_build_benchmark_config_invalid_concurrency_raises_error(self) -> None:
        """Verify invalid non-integer concurrency string raises ValueError."""
        args = parse_args(["--concurrency", "1,two,3"])
        with pytest.raises(ValueError, match="Invalid concurrency sequence"):
            build_benchmark_config(args)

    def test_build_benchmark_config_empty_concurrency_raises_error(self) -> None:
        """Verify empty concurrency string raises ValueError."""
        args = parse_args(["--concurrency", ",,,"])
        with pytest.raises(ValueError, match="Concurrency levels list cannot be empty"):
            build_benchmark_config(args)
