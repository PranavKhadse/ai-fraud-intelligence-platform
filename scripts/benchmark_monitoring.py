#!/usr/bin/env python
"""
Reproducible Performance Benchmarking CLI for Phase 13: Model Monitoring & Drift Detection.

Measures latency, throughput, and computational overhead for:
1. Feature Drift Engine (55 features: 49 numerical KS/PSI + 6 categorical PSI/JSD/unseen category)
2. Prediction Drift Engine (Score PSI 10-bin, Tier JSD, Action JSD, Override Rate Drift)
3. Ground-Truth Model Performance Engine (Confusion Matrix, Precision, Recall, Specificity, F1, FPR, ROC-AUC, PR-AUC, Accuracy, Degradation)
4. Monitoring REST APIs (FastAPI TestClient in-memory ASGI)
5. Snapshot Generation Engine (Data Extraction + Computation + Persistence Serialization)

Outputs rich statistical percentiles (p50, p90, p95, p99, min, mean, max, std) and
exports structured artifacts to JSON and CSV.

Usage Examples:
    # Full benchmark execution across sample sweeps
    python scripts/benchmark_monitoring.py --output-json docs/monitoring_benchmark_results.json --output-csv docs/monitoring_benchmark_results.csv

    # Fast validation mode for CI/automated testing
    python scripts/benchmark_monitoring.py --quick
"""

import argparse
import asyncio
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.models.enums import DecisionAction, PolicyMode, RiskTier
from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.main import app
from backend.app.services.monitoring_service import MonitoringService
from ml.monitoring.config import MonitoringConfig, default_monitoring_config
from ml.monitoring.feature_drift import FeatureDriftCalculator
from ml.monitoring.performance import ModelPerformanceCalculator
from ml.monitoring.prediction_drift import PredictionDriftCalculator
from ml.monitoring.schemas import (
    CategoricalFeatureProfile,
    FeatureBaselineProfile,
    NumericalFeatureProfile,
    PerformanceBaselineProfile,
    PredictionBaselineProfile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitoring_benchmark_cli")


def compute_percentiles(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Compute comprehensive statistical metrics from a sequence of latency values in ms."""
    if not latencies_ms:
        return {
            "count": 0,
            "min_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "ops_per_sec": 0.0,
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    total_time_s = np.sum(arr) / 1000.0
    ops_per_sec = len(arr) / total_time_s if total_time_s > 0 else 0.0

    return {
        "count": len(arr),
        "min_ms": float(np.min(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p50_ms": float(np.percentile(arr, 50.0)),
        "p90_ms": float(np.percentile(arr, 90.0)),
        "p95_ms": float(np.percentile(arr, 95.0)),
        "p99_ms": float(np.percentile(arr, 99.0)),
        "ops_per_sec": float(ops_per_sec),
    }


def generate_synthetic_feature_dataframe(
    base_profile: FeatureBaselineProfile,
    n_samples: int,
    drift_factor: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic pandas DataFrame matching the 55 canonical features.
    drift_factor: 0.0 generates in-distribution samples, > 0.0 introduces drift.
    """
    rng = np.random.default_rng(random_seed)
    data: Dict[str, Any] = {}

    for name, feat_meta in base_profile.features.items():
        if isinstance(feat_meta, NumericalFeatureProfile):
            if feat_meta.ref_samples and len(feat_meta.ref_samples) >= n_samples:
                base_vals = np.array(feat_meta.ref_samples[:n_samples], dtype=np.float64)
            elif feat_meta.ref_samples:
                base_vals = np.array(rng.choice(feat_meta.ref_samples, size=n_samples, replace=True), dtype=np.float64)
            else:
                base_vals = rng.normal(loc=feat_meta.mean, scale=feat_meta.std if feat_meta.std > 1e-6 else 1.0, size=n_samples)

            if drift_factor > 0.0:
                shift = drift_factor * (feat_meta.std if feat_meta.std > 1e-6 else 1.0)
                vals = base_vals + shift
            else:
                vals = base_vals

            if feat_meta.min >= 0:
                vals = np.clip(vals, 0.0, None)
            data[name] = vals.tolist()
        elif isinstance(feat_meta, CategoricalFeatureProfile):
            vocab = feat_meta.vocabulary or ["UNKNOWN"]
            probs = [feat_meta.probabilities.get(v, 1.0 / len(vocab)) for v in vocab]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            if drift_factor > 0.5:
                # Add unseen category
                vocab_extended = vocab + ["UNSEEN_TEST_CATEGORY"]
                probs_extended = [0.8 * p for p in probs] + [0.2]
                chosen = rng.choice(vocab_extended, size=n_samples, p=probs_extended)
            else:
                chosen = rng.choice(vocab, size=n_samples, p=probs)
            data[name] = chosen.tolist()
        else:
            data[name] = [0.0] * n_samples

    return pd.DataFrame(data)


def benchmark_feature_drift(
    calculator: FeatureDriftCalculator,
    base_profile: FeatureBaselineProfile,
    sample_sizes: Sequence[int],
    iterations_per_size: int = 20,
    warmup_iterations: int = 3,
) -> Dict[str, Any]:
    """Benchmark Feature Drift Engine across multiple sample sizes."""
    results: Dict[str, Any] = {}
    logger.info("Executing Feature Drift Engine Benchmarks...")

    for n in sample_sizes:
        df = generate_synthetic_feature_dataframe(base_profile, n_samples=n, drift_factor=0.1)

        # Warmup
        for _ in range(warmup_iterations):
            _ = calculator.compute_feature_drift(df)

        latencies: List[float] = []
        for _ in range(iterations_per_size):
            t0 = time.perf_counter()
            _ = calculator.compute_feature_drift(df)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        stats = compute_percentiles(latencies)
        results[f"sample_size_{n}"] = stats
        logger.info(
            f"Feature Drift (N={n:5d}, 55 feats): p50={stats['p50_ms']:6.2f}ms | "
            f"p95={stats['p95_ms']:6.2f}ms | mean={stats['mean_ms']:6.2f}ms | throughput={stats['ops_per_sec']:6.2f} ops/s"
        )

    return results


def benchmark_prediction_drift(
    calculator: PredictionDriftCalculator,
    sample_sizes: Sequence[int],
    iterations_per_size: int = 50,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """Benchmark Prediction & Concept Drift Engine across multiple evaluation sample sizes."""
    results: Dict[str, Any] = {}
    logger.info("Executing Prediction Drift Engine Benchmarks...")
    rng = np.random.default_rng(42)

    for n in sample_sizes:
        # Create synthetic risk evaluations
        scores = rng.beta(0.5, 5.0, size=n)  # realistic right-skewed fraud score distribution
        evaluations: List[RiskEvaluation] = []
        for s in scores:
            score_dec = Decimal(str(round(float(s), 4)))
            if s >= 0.85:
                tier = RiskTier.CRITICAL
                action = DecisionAction.BLOCK
            elif s >= 0.60:
                tier = RiskTier.HIGH
                action = DecisionAction.REVIEW
            elif s >= 0.30:
                tier = RiskTier.MEDIUM
                action = DecisionAction.REVIEW
            else:
                tier = RiskTier.LOW
                action = DecisionAction.APPROVE

            ev = RiskEvaluation(
                id=uuid.uuid4(),
                transaction_id=uuid.uuid4(),
                model_score=score_dec,
                risk_score=int(round(float(s) * 100)),
                decision_action=action,
                baseline_action=action,
                decision_reason="Benchmark evaluation",
                risk_tier=tier,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                evaluated_at=datetime.now(timezone.utc),
            )
            evaluations.append(ev)

        # Warmup
        for _ in range(warmup_iterations):
            _ = calculator.compute_prediction_drift(evaluations)

        latencies: List[float] = []
        for _ in range(iterations_per_size):
            t0 = time.perf_counter()
            _ = calculator.compute_prediction_drift(evaluations)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        stats = compute_percentiles(latencies)
        results[f"sample_size_{n}"] = stats
        logger.info(
            f"Prediction Drift (N={n:5d} scores): p50={stats['p50_ms']:6.3f}ms | "
            f"p95={stats['p95_ms']:6.3f}ms | mean={stats['mean_ms']:6.3f}ms | throughput={stats['ops_per_sec']:6.2f} ops/s"
        )

    return results


def benchmark_model_performance(
    calculator: ModelPerformanceCalculator,
    sample_sizes: Sequence[int],
    iterations_per_size: int = 50,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """Benchmark Ground-Truth Model Performance Calculation Engine across sample sizes."""
    results: Dict[str, Any] = {}
    logger.info("Executing Model Performance Engine Benchmarks...")
    rng = np.random.default_rng(42)

    for n in sample_sizes:
        # Generate synthetic ground truth labels (5% fraud) and predicted scores
        y_true = (rng.uniform(0, 1, size=n) < 0.05).astype(int).tolist()
        y_score = rng.beta(0.5, 5.0, size=n).tolist()
        # Boost score for true fraud
        for idx in range(n):
            if y_true[idx] == 1:
                y_score[idx] = min(1.0, y_score[idx] + 0.5)
        y_action = [
            "BLOCK" if s >= 0.78 else ("REVIEW" if s >= 0.40 else "APPROVE")
            for s in y_score
        ]

        # Warmup
        for _ in range(warmup_iterations):
            _ = calculator.compute_performance_from_labels(
                y_true=y_true,
                model_scores=y_score,
                actions=y_action,
                total_raw_count=n,
                operating_threshold=0.78,
            )

        latencies: List[float] = []
        for _ in range(iterations_per_size):
            t0 = time.perf_counter()
            _ = calculator.compute_performance_from_labels(
                y_true=y_true,
                model_scores=y_score,
                actions=y_action,
                total_raw_count=n,
                operating_threshold=0.78,
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        stats = compute_percentiles(latencies)
        results[f"sample_size_{n}"] = stats
        logger.info(
            f"Performance Engine (N={n:5d} cases): p50={stats['p50_ms']:6.3f}ms | "
            f"p95={stats['p95_ms']:6.3f}ms | mean={stats['mean_ms']:6.3f}ms | throughput={stats['ops_per_sec']:6.2f} ops/s"
        )

    return results


def benchmark_monitoring_apis(
    iterations: int = 30,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """Benchmark REST API endpoints under /api/v1/monitoring."""
    from unittest.mock import AsyncMock, MagicMock
    from backend.app.schemas.monitoring import (
        ConfusionMatrixResponse,
        FeatureDriftDetailResponse,
        FeatureDriftItemResponse,
        FeatureDriftListResponse,
        MetricDegradationItemResponse,
        ModelPerformanceResponse,
        MonitoringHealthResponse,
        MonitoringSnapshotItemResponse,
        MonitoringSnapshotListResponse,
        OperationalMetricsResponse,
        PredictionDriftResponse,
        ThresholdMetricsResponse,
    )
    from backend.app.db.session import get_db_session
    from backend.app.services.monitoring_service import get_monitoring_service

    results: Dict[str, Any] = {}
    logger.info("Executing Monitoring REST API Latency Benchmarks...")

    # Set up representative mock response objects
    mock_service = MagicMock(spec=MonitoringService)
    mock_service.get_health_overview = AsyncMock(
        return_value=MonitoringHealthResponse(
            model_version="1.0.0",
            window_type="DAILY",
            window_start="2026-09-23T00:00:00Z",
            window_end="2026-09-24T00:00:00Z",
            sample_count=5000,
            labeled_count=500,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            active_alert_count=0,
            active_alerts=[],
            created_at="2026-09-24T00:00:00Z",
        )
    )

    sample_items = [
        FeatureDriftItemResponse(
            feature_name=f"feature_{i}",
            feature_type="numerical",
            category="amount" if i < 10 else "velocity",
            status="NORMAL",
            psi=0.02,
            ks_statistic=0.03,
            ks_p_value=0.45,
            js_divergence=0.0,
            missing_rate_current=0.0,
            missing_rate_delta=0.0,
            unseen_category_rate=0.0,
        )
        for i in range(55)
    ]
    mock_service.get_feature_drift = AsyncMock(
        return_value=FeatureDriftListResponse(
            model_version="1.0.0",
            window_type="DAILY",
            window_start="2026-09-23T00:00:00Z",
            window_end="2026-09-24T00:00:00Z",
            sample_count=5000,
            overall_status="NORMAL",
            total_features_count=55,
            drifted_features_count=0,
            critical_features_count=0,
            warning_features_count=0,
            items=sample_items,
        )
    )

    mock_service.get_feature_drift_detail = AsyncMock(
        return_value=FeatureDriftDetailResponse(
            feature_name="amt",
            feature_type="numerical",
            category="amount",
            status="NORMAL",
            psi=0.02,
            ks_statistic=0.03,
            ks_p_value=0.45,
            js_divergence=0.0,
            missing_rate_baseline=0.0,
            missing_rate_current=0.0,
            missing_rate_delta=0.0,
            unseen_category_rate=0.0,
            unseen_categories=[],
            baseline_distribution={"0-10": 0.2, "10-50": 0.2, "50-100": 0.2, "100-500": 0.2, "500-1000": 0.2},
            current_distribution={"0-10": 0.2, "10-50": 0.2, "50-100": 0.2, "100-500": 0.2, "500-1000": 0.2},
            bin_edges=[0.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        )
    )

    mock_service.get_prediction_drift = AsyncMock(
        return_value=PredictionDriftResponse(
            model_version="1.0.0",
            window_type="DAILY",
            window_start="2026-09-23T00:00:00Z",
            window_end="2026-09-24T00:00:00Z",
            sample_count=5000,
            overall_status="NORMAL",
            model_score_psi=0.03,
            model_score_status="NORMAL",
            model_score_mean=0.25,
            model_score_std=0.15,
            risk_score_psi=0.03,
            risk_score_status="NORMAL",
            risk_tier_jsd=0.01,
            risk_tier_status="NORMAL",
            action_jsd=0.01,
            action_status="NORMAL",
            override_rate_current=0.05,
            override_rate_baseline=0.05,
            override_rate_delta=0.0,
            model_score_distribution={"0.0-0.1": 0.4, "0.1-0.2": 0.3},
            risk_score_buckets_distribution={"0-10": 0.4, "10-20": 0.3},
            risk_tier_distribution={"LOW": 0.7, "MEDIUM": 0.2, "HIGH": 0.08, "CRITICAL": 0.02},
            action_distribution={"APPROVE": 0.8, "REVIEW": 0.15, "BLOCK": 0.05},
            active_alerts=[],
        )
    )

    mock_service.get_performance = AsyncMock(
        return_value=ModelPerformanceResponse(
            model_version="1.0.0",
            window_type="DAILY",
            window_start="2026-09-23T00:00:00Z",
            window_end="2026-09-24T00:00:00Z",
            dataset_row_count=5000,
            labeled_sample_count=500,
            fraud_cases_count=25,
            legitimate_cases_count=475,
            suspicious_resolved_count=0,
            overall_performance_status="NORMAL",
            confidence="NORMAL_CONFIDENCE",
            operating_threshold=0.78,
            primary_metrics=ThresholdMetricsResponse(
                threshold=0.78,
                precision=0.833,
                recall=0.833,
                f1=0.833,
                accuracy=0.980,
                fpr=0.011,
                tpr=0.833,
                confusion_matrix=ConfusionMatrixResponse(tp=25, fp=5, tn=465, fn=5, total=500),
            ),
            comparison_metrics=ThresholdMetricsResponse(
                threshold=0.50,
                precision=0.651,
                recall=0.933,
                f1=0.767,
                accuracy=0.966,
                fpr=0.032,
                tpr=0.933,
                confusion_matrix=ConfusionMatrixResponse(tp=28, fp=15, tn=455, fn=2, total=500),
            ),
            operational_metrics=OperationalMetricsResponse(
                decision_precision_block=0.90,
                decision_recall_intervention=0.95,
                review_queue_purity=0.60,
                total_reviews_count=50,
                fraud_in_review_count=30,
                total_blocks_count=30,
                fraud_in_block_count=27,
            ),
            pr_auc=0.88,
            roc_auc=0.96,
            degradation_results={
                "model_precision": MetricDegradationItemResponse(
                    metric_name="model_precision",
                    observed_value=0.833,
                    baseline_value=0.850,
                    relative_delta=-0.02,
                    absolute_delta=-0.017,
                    severity="NORMAL",
                    confidence="NORMAL_CONFIDENCE",
                    alert_message=None,
                )
            },
            active_degradation_alerts=[],
        )
    )

    mock_service.list_snapshots = AsyncMock(
        return_value=MonitoringSnapshotListResponse(
            total_count=10,
            limit=20,
            offset=0,
            items=[
                MonitoringSnapshotItemResponse(
                    id=str(uuid.uuid4()),
                    model_version="1.0.0",
                    window_type="DAILY",
                    window_start="2026-09-23T00:00:00Z",
                    window_end="2026-09-24T00:00:00Z",
                    sample_count=5000,
                    labeled_count=500,
                    overall_status="NORMAL",
                    data_drift_status="NORMAL",
                    prediction_drift_status="NORMAL",
                    performance_status="NORMAL",
                    created_at="2026-09-24T00:00:00Z",
                )
            ],
        )
    )

    app.dependency_overrides[get_monitoring_service] = lambda: mock_service
    app.dependency_overrides[get_db_session] = lambda: AsyncMock(spec=AsyncSession)

    endpoints = [
        ("GET /api/v1/monitoring/health", "/api/v1/monitoring/health?window=24h"),
        ("GET /api/v1/monitoring/drift/features", "/api/v1/monitoring/drift/features?window=24h&limit=55"),
        ("GET /api/v1/monitoring/drift/features/amt", "/api/v1/monitoring/drift/features/amt?window=24h"),
        ("GET /api/v1/monitoring/drift/predictions", "/api/v1/monitoring/drift/predictions?window=24h"),
        ("GET /api/v1/monitoring/performance", "/api/v1/monitoring/performance?window=24h"),
        ("GET /api/v1/monitoring/snapshots", "/api/v1/monitoring/snapshots?limit=20"),
    ]

    try:
        with TestClient(app) as test_client:
            for name, url in endpoints:
                # Warmup
                for _ in range(warmup_iterations):
                    _ = test_client.get(url)

                latencies: List[float] = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    resp = test_client.get(url)
                    t1 = time.perf_counter()
                    assert resp.status_code == 200, f"Endpoint {name} returned {resp.status_code}: {resp.text}"
                    latencies.append((t1 - t0) * 1000.0)

                stats = compute_percentiles(latencies)
                results[name] = stats
                logger.info(
                    f"API {name:40s}: p50={stats['p50_ms']:6.2f}ms | "
                    f"p95={stats['p95_ms']:6.2f}ms | mean={stats['mean_ms']:6.2f}ms | throughput={stats['ops_per_sec']:6.2f} req/s"
                )
    finally:
        app.dependency_overrides.clear()

    return results


def benchmark_snapshot_generation(
    service: MonitoringService,
    base_profile: FeatureBaselineProfile,
    sample_sizes: Sequence[int],
    iterations_per_size: int = 15,
) -> Dict[str, Any]:
    """Benchmark Snapshot Assembly and Serialization Pipeline."""
    results: Dict[str, Any] = {}
    logger.info("Executing Snapshot Assembly & Serialization Benchmarks...")

    for n in sample_sizes:
        df = generate_synthetic_feature_dataframe(base_profile, n_samples=n, drift_factor=0.05)
        rng = np.random.default_rng(42)
        scores = rng.beta(0.5, 5.0, size=n)
        evaluations: List[RiskEvaluation] = []
        for s in scores:
            ev = RiskEvaluation(
                id=uuid.uuid4(),
                transaction_id=uuid.uuid4(),
                model_score=Decimal(str(round(float(s), 4))),
                risk_score=int(round(float(s) * 100)),
                decision_action=DecisionAction.BLOCK if s >= 0.85 else DecisionAction.APPROVE,
                baseline_action=DecisionAction.BLOCK if s >= 0.85 else DecisionAction.APPROVE,
                decision_reason="Snapshot benchmark",
                risk_tier=RiskTier.CRITICAL if s >= 0.85 else RiskTier.LOW,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                evaluated_at=datetime.now(timezone.utc),
            )
            evaluations.append(ev)

        latencies: List[float] = []
        for _ in range(iterations_per_size):
            t0 = time.perf_counter()
            # 1. Feature drift
            f_rep = service.feature_calculator.compute_feature_drift(df)
            # 2. Prediction drift
            p_rep = service.prediction_calculator.compute_prediction_drift(evaluations)
            # 3. Snapshot JSON serialization
            snap_dict = {
                "id": str(uuid.uuid4()),
                "model_version": "1.0.0",
                "window_type": "DAILY",
                "sample_count": n,
                "overall_status": "NORMAL",
                "data_drift_status": f_rep.overall_data_drift_status.value,
                "prediction_drift_status": p_rep.overall_prediction_drift_status.value,
                "feature_drift_summary": f_rep.to_dict(),
                "prediction_drift_summary": p_rep.to_dict(),
            }
            json_str = json.dumps(snap_dict)
            _ = len(json_str)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        stats = compute_percentiles(latencies)
        results[f"sample_size_{n}"] = stats
        logger.info(
            f"Snapshot Pipeline (N={n:5d}): p50={stats['p50_ms']:6.2f}ms | "
            f"p95={stats['p95_ms']:6.2f}ms | mean={stats['mean_ms']:6.2f}ms | throughput={stats['ops_per_sec']:6.2f} ops/s"
        )

    return results


def export_results_to_csv(benchmark_data: Dict[str, Any], csv_path: Path) -> None:
    """Export benchmark summary metrics to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for component, measurements in benchmark_data.get("benchmarks", {}).items():
        for test_name, stats in measurements.items():
            if isinstance(stats, dict) and "p50_ms" in stats:
                rows.append({
                    "component": component,
                    "test_name": test_name,
                    "count": stats["count"],
                    "min_ms": f"{stats['min_ms']:.3f}",
                    "mean_ms": f"{stats['mean_ms']:.3f}",
                    "median_ms": f"{stats['median_ms']:.3f}",
                    "p50_ms": f"{stats['p50_ms']:.3f}",
                    "p90_ms": f"{stats['p90_ms']:.3f}",
                    "p95_ms": f"{stats['p95_ms']:.3f}",
                    "p99_ms": f"{stats['p99_ms']:.3f}",
                    "max_ms": f"{stats['max_ms']:.3f}",
                    "std_ms": f"{stats['std_ms']:.3f}",
                    "ops_per_sec": f"{stats['ops_per_sec']:.2f}",
                })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved benchmark CSV summary to: {csv_path}")


def export_results_to_json(benchmark_data: Dict[str, Any], json_path: Path) -> None:
    """Export detailed benchmark dictionary to JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(f"Saved benchmark JSON details to: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Phase 13 Model Monitoring & Drift Detection Performance."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast benchmark with fewer iterations (ideal for automated testing/CI).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/monitoring_benchmark_results.json",
        help="Target output path for JSON benchmark results.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="docs/monitoring_benchmark_results.csv",
        help="Target output path for CSV benchmark results.",
    )
    return parser.parse_args()


def run_full_benchmark_suite(quick: bool = False) -> Dict[str, Any]:
    """Execute complete Phase 13 performance benchmark suite."""
    logger.info("Initializing Phase 13 Model Monitoring Benchmark Suite...")

    # Load baseline profile
    config = default_monitoring_config
    base_profile = FeatureBaselineProfile.load(config.feature_profile_path)
    service = MonitoringService(config=config)
    client = TestClient(app)

    if quick:
        sample_sizes = [100, 500]
        iterations = 10
        api_iterations = 10
    else:
        sample_sizes = [100, 500, 1000, 5000]
        iterations = 25
        api_iterations = 30

    system_metadata = {
        "platform": platform.platform(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0",
        "features_count": base_profile.num_features,
    }

    benchmarks: Dict[str, Any] = {
        "feature_drift_engine": benchmark_feature_drift(
            calculator=service.feature_calculator,
            base_profile=base_profile,
            sample_sizes=sample_sizes,
            iterations_per_size=iterations,
        ),
        "prediction_drift_engine": benchmark_prediction_drift(
            calculator=service.prediction_calculator,
            sample_sizes=sample_sizes,
            iterations_per_size=iterations,
        ),
        "model_performance_engine": benchmark_model_performance(
            calculator=service.performance_calculator,
            sample_sizes=sample_sizes,
            iterations_per_size=iterations,
        ),
        "monitoring_rest_apis": benchmark_monitoring_apis(
            iterations=api_iterations,
        ),
        "snapshot_generation_pipeline": benchmark_snapshot_generation(
            service=service,
            base_profile=base_profile,
            sample_sizes=sample_sizes,
            iterations_per_size=iterations,
        ),
    }

    return {
        "metadata": system_metadata,
        "config": {
            "quick_mode": quick,
            "sample_sizes": sample_sizes,
            "iterations_per_size": iterations,
            "feature_profile_path": str(config.feature_profile_path),
            "prediction_profile_path": str(config.prediction_profile_path),
            "performance_profile_path": str(config.performance_profile_path),
        },
        "benchmarks": benchmarks,
    }


def main() -> None:
    args = parse_args()
    results = run_full_benchmark_suite(quick=args.quick)

    if args.output_json:
        export_results_to_json(results, Path(args.output_json))

    if args.output_csv:
        export_results_to_csv(results, Path(args.output_csv))

    print("\n" + "=" * 80)
    print("PHASE 13 MODEL MONITORING BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    for comp, metrics in results["benchmarks"].items():
        print(f"\n--- {comp.upper()} ---")
        for test_key, stats in metrics.items():
            print(
                f"  {test_key:35s} | p50={stats['p50_ms']:6.2f}ms | p95={stats['p95_ms']:6.2f}ms | "
                f"mean={stats['mean_ms']:6.2f}ms | throughput={stats['ops_per_sec']:6.2f} ops/s"
            )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
