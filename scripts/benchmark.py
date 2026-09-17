#!/usr/bin/env python
"""
Reproducible Benchmark CLI for Phase 10: Real-Time Detection & Benchmarking.

Provides a unified command-line entrypoint to execute end-to-end performance benchmarks,
multi-concurrency scalability sweeps, persistence ablation, bottleneck profiling,
and reference engineering target evaluation against the AI-Powered Fraud Detection Platform.

Usage Examples:
    # Standard in-process ASGI benchmark sweep across C in (1, 2, 4, 8, 16)
    python scripts/benchmark.py --requests 500 --concurrency 1,2,4,8,16 --output docs/benchmark_results.json

    # Fast validation run with JSON and CSV export
    python scripts/benchmark.py --requests 50 --concurrency 1,2,4 --output docs/benchmark_results.json --csv docs/benchmark_results.csv

    # HTTP network mode against a running server
    python scripts/benchmark.py --transport http --target-url http://127.0.0.1:8000 --requests 200
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.benchmarking.config import BenchmarkConfig, TargetSLA
from backend.app.benchmarking.profiler import PipelineProfiler
from backend.app.benchmarking.reporter import BenchmarkReporter
from backend.app.benchmarking.runner import BenchmarkRunner
from backend.app.benchmarking.suite import BenchmarkSuite, BenchmarkSuiteResult

logger = logging.getLogger("fraud_api.benchmark_cli")


def find_default_dataset() -> Path:
    """Locate the best available processed transaction dataset partition."""
    candidates = [
        project_root / "data" / "processed" / "features" / "test_features.parquet",
        project_root / "data" / "processed" / "features" / "val_features.parquet",
        project_root / "data" / "processed" / "benchmark_dataset.parquet",
        project_root / "data" / "processed" / "features_test.parquet",
        project_root / "data" / "processed" / "features_sample_50k.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]



def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Execute reproducible performance benchmarks and latency profiling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(find_default_dataset()),
        help="Path to the Parquet dataset partition containing evaluated feature vectors.",
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=500,
        help="Total measured requests per concurrency evaluation level.",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated sequence of concurrency levels (e.g. '1,2,4,8,16').",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=50,
        help="Number of unmeasured warm-up requests executed before measurement.",
    )
    parser.add_argument(
        "--target-rate",
        "-r",
        type=float,
        default=None,
        help="Optional offered request rate limit in transactions per second (TPS).",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["asgi", "http"],
        default="asgi",
        help="Transport mode: 'asgi' for direct in-process FastAPI or 'http' for network requests.",
    )
    parser.add_argument(
        "--target-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base server URL for network HTTP transport mode.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/predict",
        help="API prediction endpoint path.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for deterministic transaction record sampling.",
    )
    parser.add_argument(
        "--output",
        "-o",
        "--json",
        type=str,
        default=None,
        help="Output file path for machine-readable JSON benchmark report.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output file path for tabular CSV benchmark export.",
    )
    parser.add_argument(
        "--reference-target",
        choices=["standard", "ultra_low", "none"],
        default="standard",
        help="Engineering reference target profile (non-contractual).",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip multi-tier persistence ablation benchmarking.",
    )
    parser.add_argument(
        "--no-bottleneck",
        action="store_true",
        help="Skip seven-stage bottleneck micro-profiling.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress human-readable console ASCII table output.",
    )

    return parser.parse_args(args)


def build_benchmark_config(args: argparse.Namespace) -> Tuple[BenchmarkConfig, Optional[TargetSLA]]:
    """
    Construct validated BenchmarkConfig and TargetSLA from parsed CLI arguments.
    """
    # 1. Parse concurrency levels
    try:
        concurrency_levels = tuple(
            int(x.strip()) for x in str(args.concurrency).split(",") if x.strip()
        )
    except Exception as e:
        raise ValueError(f"Invalid concurrency sequence '{args.concurrency}': {e}")

    if not concurrency_levels:
        raise ValueError("Concurrency levels list cannot be empty.")

    # 2. Select Reference Target
    target_sla: Optional[TargetSLA] = None
    if args.reference_target == "standard":
        target_sla = TargetSLA.standard_gateway_target()
    elif args.reference_target == "ultra_low":
        target_sla = TargetSLA.ultra_low_latency_target()

    # 3. Construct BenchmarkConfig
    config = BenchmarkConfig(
        num_requests=args.requests,
        concurrency_levels=concurrency_levels,
        warmup_requests=args.warmup,
        dataset_path=args.dataset,
        target_rate=args.target_rate,
        include_persistence=True,
        random_seed=args.seed,
    )

    return config, target_sla


async def run_benchmark_cli(args: argparse.Namespace) -> int:
    """
    Execute full benchmarking workflow based on CLI configuration.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    try:
        config, target_sla = build_benchmark_config(args)

        # Validate dataset exists
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            print(f"[ERROR] Dataset file not found at '{dataset_path.resolve()}'.", file=sys.stderr)
            return 1

        target_url = args.target_url if args.transport == "http" else None

        runner = BenchmarkRunner(
            config=config,
            target_url=target_url,
            endpoint_path=args.endpoint,
        )

        profiler = PipelineProfiler()

        suite = BenchmarkSuite(
            config=config,
            runner=runner,
            profiler=profiler,
            target_sla=target_sla,
        )

        if not args.quiet:
            print("\n============================================================")
            print(" Starting Empirical Benchmark Execution...")
            print(f" Transport:   {'Network HTTP (' + str(target_url) + ')' if target_url else 'In-process ASGI'}")
            print(f" Requests:    {config.num_requests} per concurrency level")
            print(f" Concurrency: {config.concurrency_levels}")
            print(f" Warm-up:     {config.warmup_requests} unmeasured requests")
            print("============================================================\n")

        # Execute full suite
        result: BenchmarkSuiteResult = await suite.run_full_suite(
            include_ablation=not args.no_ablation,
            include_bottleneck=not args.no_bottleneck,
        )

        # Display console summary
        if not args.quiet:
            summary_text = BenchmarkReporter.format_console_summary(result)
            print(summary_text)

        # Export JSON
        if args.output:
            json_path = BenchmarkReporter.export_json(result, args.output)
            if not args.quiet:
                print(f"[SUCCESS] JSON report saved to: {json_path.resolve()}")

        # Export CSV
        if args.csv:
            csv_path = BenchmarkReporter.export_csv(result, args.csv)
            if not args.quiet:
                print(f"[SUCCESS] CSV report saved to: {csv_path.resolve()}")

        return 0

    except Exception as exc:
        print(f"[FATAL] Benchmark execution failed: {exc}", file=sys.stderr)
        logger.exception("Benchmark CLI error")
        return 1


def main() -> None:
    """CLI entrypoint function."""
    args = parse_args()
    exit_code = asyncio.run(run_benchmark_cli(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
