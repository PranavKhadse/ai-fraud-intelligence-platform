"""
Benchmark Results Reporter & Export Engine for Phase 10.

Provides comprehensive formatting and export capabilities:
1. Formatted console ASCII tables for interactive inspection.
2. Machine-readable JSON serialization with environment and hardware metadata.
3. Tabular CSV serialization for spreadsheet/graphing analysis.
4. GitHub-flavored Markdown tables for documentation.
"""

import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from backend.app.benchmarking.metrics import PipelineStage
from backend.app.benchmarking.suite import (
    AblationResult,
    BenchmarkSuiteResult,
    BottleneckAnalysisResult,
    ConcurrencySweepResult,
)

logger = logging.getLogger("fraud_api.benchmark_reporter")


class BenchmarkReporter:
    """
    Formats, displays, and exports benchmark suite execution results.
    """

    @staticmethod
    def get_system_metadata() -> Dict[str, Any]:
        """
        Capture hardware, OS, and runtime platform metadata for reproducibility.
        """
        return {
            "platform": platform.platform(),
            "os_name": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "AMD64 / x86_64",
            "cpu_cores_logical": os.cpu_count() or 16,
            "python_version": sys.version.split()[0],
            "python_compiler": platform.python_compiler(),
            "python_implementation": platform.python_implementation(),
        }

    @classmethod
    def format_console_summary(cls, suite_result: BenchmarkSuiteResult) -> str:
        """
        Generate a comprehensive, human-readable ASCII console summary.
        """
        lines: List[str] = []
        meta = cls.get_system_metadata()

        lines.append("=" * 80)
        lines.append(" AI-POWERED FRAUD DETECTION & RISK INTELLIGENCE PLATFORM")
        lines.append(" REAL-TIME INFERENCE & PERSISTENCE BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Suite ID:           {suite_result.suite_id}")
        lines.append(f"Timestamp (UTC):    {suite_result.timestamp_utc}")
        lines.append(f"Environment:        {meta['os_name']} ({meta['architecture']}) | Python {meta['python_version']}")
        lines.append(f"Logical CPU Cores:  {meta['cpu_cores_logical']}")
        lines.append(f"Dataset Path:       {suite_result.config.dataset_path}")
        lines.append(f"Random Seed:        {suite_result.config.random_seed}")
        lines.append("-" * 80)

        # 1. Cold-start & Warm-up Section
        lines.append("\n1. COLD-START & WARM-UP LATENCY ISOLATION")
        lines.append("-" * 80)
        cold_val = (
            f"{suite_result.cold_start_latency_ms:.3f} ms"
            if suite_result.cold_start_latency_ms is not None
            else "N/A"
        )
        lines.append(f"Cold-Start Latency (initial transaction):   {cold_val}")
        lines.append(f"Unmeasured Warm-up Requests Executed:       {suite_result.warmup_requests_executed}")
        if suite_result.warmup_latency:
            lines.append(
                f"Warm-up Mean Latency (isolated from run):  {suite_result.warmup_latency.mean_ms:.3f} ms "
                f"(P50: {suite_result.warmup_latency.p50_ms:.3f} ms, P95: {suite_result.warmup_latency.p95_ms:.3f} ms)"
            )

        # 2. Concurrency Sweep Section
        if suite_result.concurrency_sweep:
            sweep = suite_result.concurrency_sweep
            lines.append("\n2. MULTI-CONCURRENCY SCALABILITY MATRIX")
            lines.append("-" * 80)
            lines.append(
                f"{'Concurrency':<12} | {'Reqs':<6} | {'TPS (comp)':<12} | {'TPS (succ)':<12} | "
                f"{'P50 (ms)':<9} | {'P90 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | "
                f"{'Speedup':<8} | {'Eff %':<6}"
            )
            lines.append("-" * 105)

            for c, res in sorted(sweep.concurrency_results.items()):
                tps_comp = f"{res.throughput.completed_tps:.1f}"
                tps_succ = f"{res.throughput.successful_tps:.1f}"
                p50 = f"{res.latency.p50_ms:.2f}"
                p90 = f"{res.latency.p90_ms:.2f}"
                p95 = f"{res.latency.p95_ms:.2f}"
                p99 = f"{res.latency.p99_ms:.2f}"
                speedup = f"{sweep.scaling_ratios.get(c, 1.0):.2f}x"
                eff = f"{sweep.scaling_efficiencies.get(c, 100.0):.1f}%"

                lines.append(
                    f"{'C=' + str(c):<12} | {res.total_requests:<6} | {tps_comp:<12} | {tps_succ:<12} | "
                    f"{p50:<9} | {p90:<9} | {p95:<9} | {p99:<9} | "
                    f"{speedup:<8} | {eff:<6}"
                )

            lines.append(
                f"\nPeak Throughput: {sweep.peak_tps:.1f} TPS at Concurrency C={sweep.peak_concurrency} "
                f"(Baseline C={sweep.baseline_concurrency}: {sweep.baseline_tps:.1f} TPS)"
            )

        # 3. Persistence Ablation Section
        if suite_result.ablation:
            abl = suite_result.ablation
            lines.append("\n3. MULTI-TIER PERSISTENCE ABLATION (OVERHEAD DECOMPOSITION)")
            lines.append("-" * 80)
            lines.append(
                f"{'Operational Mode':<35} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10}"
            )
            lines.append("-" * 85)
            lines.append(
                f"{'Mode A: Full API + PostgreSQL':<35} | {abl.mode_a_full_api.mean_ms:<10.3f} | "
                f"{abl.mode_a_full_api.p50_ms:<10.3f} | {abl.mode_a_full_api.p95_ms:<10.3f} | {abl.mode_a_full_api.p99_ms:<10.3f}"
            )
            lines.append(
                f"{'Mode B: In-Memory Pipeline (No DB)':<35} | {abl.mode_b_in_memory.mean_ms:<10.3f} | "
                f"{abl.mode_b_in_memory.p50_ms:<10.3f} | {abl.mode_b_in_memory.p95_ms:<10.3f} | {abl.mode_b_in_memory.p99_ms:<10.3f}"
            )
            lines.append(
                f"{'Mode C: Idempotent DB Replay Fast Path':<35} | {abl.mode_c_idempotent_replay.mean_ms:<10.3f} | "
                f"{abl.mode_c_idempotent_replay.p50_ms:<10.3f} | {abl.mode_c_idempotent_replay.p95_ms:<10.3f} | {abl.mode_c_idempotent_replay.p99_ms:<10.3f}"
            )
            lines.append("-" * 85)
            lines.append(
                f"PostgreSQL Persistence Overhead:  {abl.persistence_overhead_ms:.3f} ms "
                f"({abl.persistence_overhead_pct:.1f}% of end-to-end latency)"
            )
            lines.append(
                f"Idempotent Replay Speedup:        {abl.replay_speedup_factor:.2f}x faster than fresh evaluation "
                f"({abl.replay_latency_reduction_pct:.1f}% latency reduction)"
            )

        # 4. Bottleneck & 7-Stage Component Breakdown Section
        if suite_result.bottleneck_analysis:
            bot = suite_result.bottleneck_analysis
            lines.append("\n4. SEVEN-STAGE COMPONENT LATENCY BREAKDOWN")
            lines.append("-" * 80)
            lines.append(
                f"{'Pipeline Stage':<32} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'% of Total':<10}"
            )
            lines.append("-" * 85)

            for stage_key, metrics in bot.component_breakdown.stage_metrics.items():
                s_name = stage_key.value if hasattr(stage_key, "value") else str(stage_key)
                pct = bot.component_breakdown.stage_percentages.get(stage_key, 0.0)
                lines.append(
                    f"{s_name:<32} | {metrics.mean_ms:<10.3f} | {metrics.p50_ms:<10.3f} | "
                    f"{metrics.p95_ms:<10.3f} | {pct:<9.1f}%"
                )

            lines.append("-" * 85)
            lines.append(
                f"Total Isolated Component Mean:    {bot.component_breakdown.total_component_mean_ms:.3f} ms"
            )
            lines.append(f"Diagnostic Summary:               {bot.diagnostic_summary}")

        # 5. Reference Target Compliance
        if suite_result.target_sla_evaluated:
            sla = suite_result.target_sla_evaluated
            lines.append("\n5. ENGINEERING REFERENCE TARGET EVALUATION (NON-CONTRACTUAL)")
            lines.append("-" * 80)
            lines.append(f"Target Reference Profile: {sla.sla_name}")
            if sla.description:
                lines.append(f"Description:              {sla.description}")

            lines.append(
                f"{'Percentile':<12} | {'Target (ms)':<15} | {'Measured (C=1)':<15} | {'Status':<10}"
            )
            lines.append("-" * 60)

            c1_res = (
                suite_result.concurrency_sweep.concurrency_results.get(1)
                if suite_result.concurrency_sweep
                else None
            )

            if c1_res:
                lat = c1_res.latency
                for name, target, actual in [
                    ("P50", sla.target_p50_ms, lat.p50_ms),
                    ("P95", sla.target_p95_ms, lat.p95_ms),
                    ("P99", sla.target_p99_ms, lat.p99_ms),
                ]:
                    if target is not None:
                        status = "PASS" if actual <= target else "EXCEEDED"
                        lines.append(
                            f"{name:<12} | {'<= ' + str(target) + ' ms':<15} | {str(actual) + ' ms':<15} | {status:<10}"
                        )
            lines.append(
                "\n* Note: Engineering reference targets represent low-latency architectural benchmarks "
                "and are explicitly non-contractual."
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    @classmethod
    def export_json(
        cls,
        suite_result: BenchmarkSuiteResult,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export complete, sanitized benchmark suite result to a JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "metadata": cls.get_system_metadata(),
            "benchmark_config": {
                "num_requests": suite_result.config.num_requests,
                "concurrency_levels": list(suite_result.config.concurrency_levels),
                "warmup_requests": suite_result.config.warmup_requests,
                "dataset_path": str(suite_result.config.dataset_path),
                "target_rate": suite_result.config.target_rate,
                "include_persistence": suite_result.config.include_persistence,
                "random_seed": suite_result.config.random_seed,
            },
            "suite_results": suite_result.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Exported benchmark JSON report to '{path.resolve()}'.")
        return path

    @classmethod
    def export_csv(
        cls,
        suite_result: BenchmarkSuiteResult,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export concurrency sweep matrix and component breakdown to a structured CSV file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Section 1: Concurrency Sweep
            if suite_result.concurrency_sweep:
                writer.writerow(["=== MULTI-CONCURRENCY SCALABILITY MATRIX ==="])
                writer.writerow([
                    "Concurrency",
                    "Total_Requests",
                    "Successful_Requests",
                    "Failed_Requests",
                    "Completed_TPS",
                    "Successful_TPS",
                    "Min_ms",
                    "Mean_ms",
                    "Median_ms",
                    "P50_ms",
                    "P90_ms",
                    "P95_ms",
                    "P99_ms",
                    "Max_ms",
                    "Speedup_Ratio",
                    "Scaling_Efficiency_Pct",
                ])

                sweep = suite_result.concurrency_sweep
                for c, res in sorted(sweep.concurrency_results.items()):
                    lat = res.latency
                    tp = res.throughput
                    writer.writerow([
                        c,
                        res.total_requests,
                        res.successful_requests,
                        res.failed_requests,
                        tp.completed_tps,
                        tp.successful_tps,
                        lat.min_ms,
                        lat.mean_ms,
                        lat.median_ms,
                        lat.p50_ms,
                        lat.p90_ms,
                        lat.p95_ms,
                        lat.p99_ms,
                        lat.max_ms,
                        sweep.scaling_ratios.get(c, 1.0),
                        sweep.scaling_efficiencies.get(c, 100.0),
                    ])
                writer.writerow([])

            # Section 2: Component Breakdown
            if suite_result.bottleneck_analysis:
                writer.writerow(["=== SEVEN-STAGE COMPONENT LATENCY BREAKDOWN ==="])
                writer.writerow([
                    "Stage_Name",
                    "Mean_ms",
                    "Median_ms",
                    "P50_ms",
                    "P90_ms",
                    "P95_ms",
                    "P99_ms",
                    "Percentage_Of_Total",
                ])

                bot = suite_result.bottleneck_analysis
                for stage_key, m in bot.component_breakdown.stage_metrics.items():
                    s_name = stage_key.value if hasattr(stage_key, "value") else str(stage_key)
                    pct = bot.component_breakdown.stage_percentages.get(stage_key, 0.0)
                    writer.writerow([
                        s_name,
                        m.mean_ms,
                        m.median_ms,
                        m.p50_ms,
                        m.p90_ms,
                        m.p95_ms,
                        m.p99_ms,
                        pct,
                    ])
                writer.writerow([])

            # Section 3: Persistence Ablation
            if suite_result.ablation:
                writer.writerow(["=== MULTI-TIER PERSISTENCE ABLATION ==="])
                writer.writerow(["Mode", "Mean_ms", "Median_ms", "P50_ms", "P95_ms", "P99_ms"])
                abl = suite_result.ablation
                writer.writerow([
                    "Mode_A_Full_API_PostgreSQL",
                    abl.mode_a_full_api.mean_ms,
                    abl.mode_a_full_api.median_ms,
                    abl.mode_a_full_api.p50_ms,
                    abl.mode_a_full_api.p95_ms,
                    abl.mode_a_full_api.p99_ms,
                ])
                writer.writerow([
                    "Mode_B_In_Memory_Pipeline",
                    abl.mode_b_in_memory.mean_ms,
                    abl.mode_b_in_memory.median_ms,
                    abl.mode_b_in_memory.p50_ms,
                    abl.mode_b_in_memory.p95_ms,
                    abl.mode_b_in_memory.p99_ms,
                ])
                writer.writerow([
                    "Mode_C_Idempotent_Replay",
                    abl.mode_c_idempotent_replay.mean_ms,
                    abl.mode_c_idempotent_replay.median_ms,
                    abl.mode_c_idempotent_replay.p50_ms,
                    abl.mode_c_idempotent_replay.p95_ms,
                    abl.mode_c_idempotent_replay.p99_ms,
                ])

        logger.info(f"Exported benchmark CSV report to '{path.resolve()}'.")
        return path

    @classmethod
    def format_markdown_tables(cls, suite_result: BenchmarkSuiteResult) -> str:
        """
        Generate GitHub-flavored Markdown tables for embedding into documentation.
        """
        md_lines: List[str] = []

        # 1. Concurrency Table
        if suite_result.concurrency_sweep:
            sweep = suite_result.concurrency_sweep
            md_lines.append("### Concurrency Scalability Matrix")
            md_lines.append("")
            md_lines.append(
                "| Concurrency | Requests | Completed TPS | Successful TPS | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Speedup | Scaling Efficiency |"
            )
            md_lines.append(
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            )
            for c, res in sorted(sweep.concurrency_results.items()):
                lat = res.latency
                tp = res.throughput
                speedup = f"{sweep.scaling_ratios.get(c, 1.0):.2f}x"
                eff = f"{sweep.scaling_efficiencies.get(c, 100.0):.1f}%"
                md_lines.append(
                    f"| **C = {c}** | {res.total_requests} | {tp.completed_tps:.1f} | {tp.successful_tps:.1f} | "
                    f"{lat.p50_ms:.2f} | {lat.p90_ms:.2f} | {lat.p95_ms:.2f} | {lat.p99_ms:.2f} | "
                    f"{speedup} | {eff} |"
                )
            md_lines.append("")

        # 2. Ablation Table
        if suite_result.ablation:
            abl = suite_result.ablation
            md_lines.append("### Multi-Tier Persistence Ablation")
            md_lines.append("")
            md_lines.append(
                "| Operational Mode | Mean (ms) | Median (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Overhead vs In-Memory |"
            )
            md_lines.append(
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            )
            md_lines.append(
                f"| **Mode A: Full API + PostgreSQL** | {abl.mode_a_full_api.mean_ms:.2f} | "
                f"{abl.mode_a_full_api.median_ms:.2f} | {abl.mode_a_full_api.p50_ms:.2f} | "
                f"{abl.mode_a_full_api.p95_ms:.2f} | {abl.mode_a_full_api.p99_ms:.2f} | +{abl.persistence_overhead_ms:.2f} ms ({abl.persistence_overhead_pct:.1f}%) |"
            )
            md_lines.append(
                f"| **Mode B: In-Memory Pipeline (No DB)** | {abl.mode_b_in_memory.mean_ms:.2f} | "
                f"{abl.mode_b_in_memory.median_ms:.2f} | {abl.mode_b_in_memory.p50_ms:.2f} | "
                f"{abl.mode_b_in_memory.p95_ms:.2f} | {abl.mode_b_in_memory.p99_ms:.2f} | Baseline (0.0 ms) |"
            )
            md_lines.append(
                f"| **Mode C: Idempotent DB Replay Fast Path** | {abl.mode_c_idempotent_replay.mean_ms:.2f} | "
                f"{abl.mode_c_idempotent_replay.median_ms:.2f} | {abl.mode_c_idempotent_replay.p50_ms:.2f} | "
                f"{abl.mode_c_idempotent_replay.p95_ms:.2f} | {abl.mode_c_idempotent_replay.p99_ms:.2f} | {abl.replay_speedup_factor:.2f}x Speedup |"
            )
            md_lines.append("")

        # 3. Component Breakdown Table
        if suite_result.bottleneck_analysis:
            bot = suite_result.bottleneck_analysis
            md_lines.append("### Seven-Stage Component Latency Breakdown")
            md_lines.append("")
            md_lines.append(
                "| Pipeline Stage | Mean (ms) | P50 (ms) | P95 (ms) | % Contribution |"
            )
            md_lines.append(
                "| :--- | :--- | :--- | :--- | :--- |"
            )
            for stage_key, m in bot.component_breakdown.stage_metrics.items():
                s_name = stage_key.value if hasattr(stage_key, "value") else str(stage_key)
                pct = bot.component_breakdown.stage_percentages.get(stage_key, 0.0)
                md_lines.append(
                    f"| `{s_name}` | {m.mean_ms:.3f} | {m.p50_ms:.3f} | {m.p95_ms:.3f} | **{pct:.1f}%** |"
                )
            md_lines.append("")

        return "\n".join(md_lines)
