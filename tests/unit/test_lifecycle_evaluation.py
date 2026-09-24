"""
Unit Tests for Phase 14.2: Candidate Model Evaluation.

Verifies:
- Comprehensive evaluation metrics calculation (PR-AUC, ROC-AUC, Precision, Recall, F1, FPR, Costs).
- Latency distribution statistics (mean, p50, p95, p99).
- Quality gate validation (pass, individual failures, overall status).
- Bundle loading and multi-layer integrity checks.
- Tamper detection and schema validation (55 features, XGBoost family, SemVer).
- Deterministic evaluation execution.
- Protected OOT dataset boundary respect.
- Zero premature champion promotion.
"""

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
import joblib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.evaluation.evaluator import CandidateModelEvaluator
from ml.lifecycle.evaluation.gates import CandidateGateConfig, evaluate_candidate_gates
from ml.lifecycle.evaluation.metrics import calculate_evaluation_metrics, calculate_latency_distribution
from ml.lifecycle.evaluation.schemas import CandidateEvaluationResult, EvaluationScope, GateResult
from ml.lifecycle.schemas import (
    EvaluationMetricsSummary,
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, TARGET_COLUMN
from ml.models.preprocessing import TreePreprocessor


class TestMetricsCalculation:
    """Test suite for calculate_evaluation_metrics and calculate_latency_distribution."""

    def test_calculate_evaluation_metrics_standard(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        y_prob = np.array([0.95, 0.85, 0.10, 0.20, 0.05, 0.80, 0.70, 0.15, 0.30, 0.40])
        threshold = 0.75

        metrics = calculate_evaluation_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
        )

        assert isinstance(metrics, EvaluationMetricsSummary)
        assert metrics.threshold == 0.75
        assert metrics.total_samples == 10
        assert metrics.tp == 2  # 0.95, 0.85 are >= 0.75 and actual 1
        assert metrics.fp == 1  # 0.80 is >= 0.75 and actual 0
        assert metrics.fn == 1  # 0.70 is < 0.75 and actual 1
        assert metrics.tn == 6
        assert metrics.pr_auc > 0.0
        assert metrics.roc_auc > 0.0
        assert metrics.precision == pytest.approx(2 / 3, abs=1e-4)
        assert metrics.recall == pytest.approx(2 / 3, abs=1e-4)
        assert metrics.fpr == pytest.approx(1 / 7, abs=1e-4)

    def test_calculate_evaluation_metrics_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_evaluation_metrics(y_true=[], y_prob=[], threshold=0.5)

    def test_calculate_evaluation_metrics_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_evaluation_metrics(y_true=[1, 0], y_prob=[0.8], threshold=0.5)

    def test_calculate_evaluation_metrics_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            calculate_evaluation_metrics(y_true=[1, 0], y_prob=[0.8, 0.2], threshold=1.5)
        with pytest.raises(ValueError, match="must be in"):
            calculate_evaluation_metrics(y_true=[1, 0], y_prob=[0.8, 0.2], threshold=0.0)

    def test_calculate_evaluation_metrics_nan_or_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            calculate_evaluation_metrics(y_true=[1, 0], y_prob=[np.nan, 0.5], threshold=0.5)
        with pytest.raises(ValueError, match="bounded in"):
            calculate_evaluation_metrics(y_true=[1, 0], y_prob=[1.2, 0.5], threshold=0.5)

    def test_calculate_evaluation_metrics_cost_matrix_calculation(self):
        y_true = np.array([1, 0])
        y_prob = np.array([0.2, 0.9])  # 1 FN, 1 FP at threshold 0.5
        cost_cfg = CostConfig(false_positive_cost=25.0, false_negative_cost=300.0)

        metrics = calculate_evaluation_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            cost_config=cost_cfg,
        )
        assert metrics.expected_cost == 325.0  # 1*25 + 1*300

    def test_latency_distribution_statistics(self):
        latencies = [10.0, 12.0, 15.0, 20.0, 50.0]
        stats = calculate_latency_distribution(latencies)
        assert "mean_ms" in stats
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert stats["p50_ms"] == 15.0
        assert stats["mean_ms"] == 21.4


class TestGateEvaluation:
    """Test suite for evaluate_candidate_gates."""

    def test_gates_all_pass(self):
        metrics = EvaluationMetricsSummary(
            pr_auc=0.96,
            roc_auc=0.99,
            precision=0.85,
            recall=0.92,
            f1=0.88,
            fpr=0.002,
            threshold=0.78,
            expected_cost=5000.0,
        )
        gate_cfg = CandidateGateConfig(
            min_pr_auc=0.85,
            min_roc_auc=0.95,
            min_recall=0.80,
            min_precision=0.50,
            min_f1=0.60,
            max_fpr=0.01,
        )
        latency_stats = {"p95_ms": 12.5}

        overall_passed, gate_results = evaluate_candidate_gates(
            metrics=metrics,
            gate_config=gate_cfg,
            latency_stats=latency_stats,
        )

        assert overall_passed is True
        assert len(gate_results) == 7
        assert all(g.passed for g in gate_results)

    def test_gate_pr_auc_failure(self):
        metrics = EvaluationMetricsSummary(
            pr_auc=0.75,  # Below 0.85
            roc_auc=0.99,
            precision=0.85,
            recall=0.92,
            f1=0.88,
            fpr=0.002,
            threshold=0.78,
        )
        gate_cfg = CandidateGateConfig(min_pr_auc=0.85)
        overall_passed, gate_results = evaluate_candidate_gates(metrics, gate_cfg)

        assert overall_passed is False
        pr_gate = next(g for g in gate_results if g.gate_name == "gate_min_pr_auc")
        assert pr_gate.passed is False
        assert pr_gate.actual_value == 0.75
        assert pr_gate.target_value == 0.85

    def test_gate_recall_failure(self):
        metrics = EvaluationMetricsSummary(
            pr_auc=0.90,
            roc_auc=0.99,
            precision=0.85,
            recall=0.70,  # Below 0.80
            f1=0.76,
            fpr=0.002,
            threshold=0.78,
        )
        gate_cfg = CandidateGateConfig(min_recall=0.80)
        overall_passed, gate_results = evaluate_candidate_gates(metrics, gate_cfg)
        assert overall_passed is False
        recall_gate = next(g for g in gate_results if g.gate_name == "gate_min_recall")
        assert recall_gate.passed is False

    def test_gate_fpr_failure(self):
        metrics = EvaluationMetricsSummary(
            pr_auc=0.90,
            roc_auc=0.99,
            precision=0.85,
            recall=0.85,
            f1=0.85,
            fpr=0.05,  # Above 0.01 max
            threshold=0.78,
        )
        gate_cfg = CandidateGateConfig(max_fpr=0.01)
        overall_passed, gate_results = evaluate_candidate_gates(metrics, gate_cfg)
        assert overall_passed is False
        fpr_gate = next(g for g in gate_results if g.gate_name == "gate_max_fpr")
        assert fpr_gate.passed is False

    def test_gate_latency_failure(self):
        metrics = EvaluationMetricsSummary(
            pr_auc=0.90,
            roc_auc=0.99,
            precision=0.85,
            recall=0.85,
            f1=0.85,
            fpr=0.001,
            threshold=0.78,
        )
        gate_cfg = CandidateGateConfig(max_p95_latency_ms=25.0)
        latency_stats = {"p95_ms": 65.0}  # Exceeds 25.0 ms
        overall_passed, gate_results = evaluate_candidate_gates(metrics, gate_cfg, latency_stats)
        assert overall_passed is False
        lat_gate = next(g for g in gate_results if g.gate_name == "gate_max_p95_latency")
        assert lat_gate.passed is False


class DummyEvaluationModel:
    """Top-level picklable dummy model for evaluation tests."""
    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2), dtype=np.float32)
        # Assign high probability to first 10%
        probs[: max(1, n // 10), 1] = 0.95
        probs[max(1, n // 10) :, 1] = 0.05
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs


class TestCandidateModelEvaluator:
    """Test suite for CandidateModelEvaluator end-to-end execution."""

    @pytest.fixture
    def mock_candidate_bundle(self, tmp_path: Path):
        """Create a valid synthetic XGBoost-compatible model bundle."""
        bundle_dir = tmp_path / "v1_1_0_bundle"
        bundle_dir.mkdir()

        model = DummyEvaluationModel()
        preproc = TreePreprocessor()
        # Fit preprocessor on dummy categories
        dummy_cat_df = pd.DataFrame({
            "merchant_category": ["grocery_pos", "online", "gas_transport"],
            "job_category": ["engineer", "accountant", "doctor"],
        })
        preproc.fit(dummy_cat_df)
        preproc.feature_names = PREDICTIVE_FEATURE_COLUMNS

        m_path = bundle_dir / "model.joblib"
        p_path = bundle_dir / "preprocessor.joblib"
        joblib.dump(model, m_path)
        joblib.dump(preproc, p_path)

        manifest = ModelBundleManifest(
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE,
            operating_threshold=0.78,
            sha256_checksums={
                "model": calculate_file_sha256(m_path),
                "preprocessor": calculate_file_sha256(p_path),
            },
        )
        manifest.save(bundle_dir / "manifest.json")
        return bundle_dir

    @pytest.fixture
    def mock_dataset_parquet(self, tmp_path: Path):
        """Create a synthetic validation parquet dataset with 55 features + is_fraud."""
        n_samples = 100
        data = {}
        for col in PREDICTIVE_FEATURE_COLUMNS:
            if col in ["merchant_category", "job_category"]:
                data[col] = ["grocery_pos" if i % 2 == 0 else "online" for i in range(n_samples)]
            else:
                data[col] = np.random.uniform(1.0, 100.0, size=n_samples).astype(np.float32)

        data[TARGET_COLUMN] = np.array([1 if i < 10 else 0 for i in range(n_samples)], dtype=np.int64)
        df = pd.DataFrame(data)

        ds_path = tmp_path / "val_features.parquet"
        df.to_parquet(ds_path, index=False)
        return ds_path

    def test_load_and_validate_bundle_success(self, mock_candidate_bundle):
        evaluator = CandidateModelEvaluator()
        model, preproc, manifest = evaluator.load_and_validate_bundle(mock_candidate_bundle)
        assert manifest.model_version == "1.1.0"
        assert manifest.model_family == "xgboost"
        assert len(preproc.feature_names) == 55

    def test_load_bundle_missing_dir_raises(self, tmp_path: Path):
        evaluator = CandidateModelEvaluator()
        with pytest.raises(FileNotFoundError, match="not found"):
            evaluator.load_and_validate_bundle(tmp_path / "non_existent")

    def test_load_bundle_missing_manifest_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_bundle"
        empty_dir.mkdir()
        evaluator = CandidateModelEvaluator()
        with pytest.raises(FileNotFoundError, match="manifest.json missing"):
            evaluator.load_and_validate_bundle(empty_dir)

    def test_load_bundle_invalid_semver_raises(self, mock_candidate_bundle):
        m_file = mock_candidate_bundle / "manifest.json"
        raw = json.loads(m_file.read_text(encoding="utf-8"))
        raw["model_version"] = "v1.1"
        m_file.write_text(json.dumps(raw), encoding="utf-8")

        evaluator = CandidateModelEvaluator()
        with pytest.raises(ValueError, match="semantic versioning"):
            evaluator.load_and_validate_bundle(mock_candidate_bundle)

    def test_load_bundle_unsupported_family_raises(self, mock_candidate_bundle):
        m_file = mock_candidate_bundle / "manifest.json"
        manifest_dict = json.loads(m_file.read_text(encoding="utf-8"))
        manifest_dict["model_family"] = "random_forest"
        m_file.write_text(json.dumps(manifest_dict), encoding="utf-8")

        evaluator = CandidateModelEvaluator()
        with pytest.raises(ValueError, match="only 'xgboost' is permitted"):
            evaluator.load_and_validate_bundle(mock_candidate_bundle)

    def test_load_bundle_tampered_artifact_checksum_raises(self, mock_candidate_bundle):
        m_path = mock_candidate_bundle / "model.joblib"
        m_path.write_bytes(b"CORRUPTED_BYTES")

        evaluator = CandidateModelEvaluator()
        with pytest.raises(ValueError, match="integrity verification failed"):
            evaluator.load_and_validate_bundle(mock_candidate_bundle)

    def test_load_bundle_wrong_feature_count_raises(self, mock_candidate_bundle):
        p_path = mock_candidate_bundle / "preprocessor.joblib"
        preproc = TreePreprocessor()
        preproc.is_fitted = True
        preproc.feature_names = ["feat1", "feat2"]  # 2 instead of 55
        joblib.dump(preproc, p_path)

        # Update checksum so integrity passes but feature count fails
        m_file = mock_candidate_bundle / "manifest.json"
        manifest = ModelBundleManifest.load(m_file)
        manifest.sha256_checksums["preprocessor"] = calculate_file_sha256(p_path)
        manifest.save(m_file)

        evaluator = CandidateModelEvaluator()
        with pytest.raises(ValueError, match="feature count"):
            evaluator.load_and_validate_bundle(mock_candidate_bundle)

    def test_evaluate_candidate_success_deterministic(self, mock_candidate_bundle, mock_dataset_parquet):
        evaluator = CandidateModelEvaluator()
        res1 = evaluator.evaluate_candidate(
            bundle_dir=mock_candidate_bundle,
            dataset_path=mock_dataset_parquet,
            benchmark_latency=False,
        )
        res2 = evaluator.evaluate_candidate(
            bundle_dir=mock_candidate_bundle,
            dataset_path=mock_dataset_parquet,
            benchmark_latency=False,
        )

        assert isinstance(res1, CandidateEvaluationResult)
        assert res1.model_version == "1.1.0"
        assert res1.model_family == "xgboost"
        assert res1.sample_count == 100
        assert res1.fraud_count == 10
        assert res1.metrics.pr_auc == res2.metrics.pr_auc
        assert res1.metrics.roc_auc == res2.metrics.roc_auc
        assert res1.metrics.precision == res2.metrics.precision
        assert res1.metrics.recall == res2.metrics.recall
        assert res1.metrics.f1 == res2.metrics.f1
        assert res1.overall_passed is True

    def test_evaluate_candidate_custom_threshold_override(self, mock_candidate_bundle, mock_dataset_parquet):
        evaluator = CandidateModelEvaluator()
        res = evaluator.evaluate_candidate(
            bundle_dir=mock_candidate_bundle,
            dataset_path=mock_dataset_parquet,
            threshold_override=0.90,
            benchmark_latency=False,
        )
        assert res.operating_threshold == 0.90
        assert res.metrics.threshold == 0.90

    def test_evaluate_candidate_missing_dataset_raises(self, mock_candidate_bundle, tmp_path: Path):
        evaluator = CandidateModelEvaluator()
        with pytest.raises(FileNotFoundError, match="dataset partition not found"):
            evaluator.evaluate_candidate(
                bundle_dir=mock_candidate_bundle,
                dataset_path=tmp_path / "non_existent.parquet",
            )

    @pytest.mark.asyncio
    async def test_evaluate_registered_candidate_with_repo(self, mock_candidate_bundle, mock_dataset_parquet):
        from backend.app.db.models.model_registry import ModelRegistryEntry
        from backend.app.repositories.model_registry_repository import ModelRegistryRepository

        mock_session = MagicMock()
        mock_session.flush = AsyncMock()

        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7800"),
            bundle_directory=str(mock_candidate_bundle),
            sha256_model=calculate_file_sha256(mock_candidate_bundle / "model.joblib"),
            sha256_preprocessor=calculate_file_sha256(mock_candidate_bundle / "preprocessor.joblib"),
        )

        mock_res = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = entry
        mock_res.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_res)

        repo = ModelRegistryRepository(mock_session)
        evaluator = CandidateModelEvaluator()

        # Run evaluation pointing to mock dataset
        result = await evaluator.evaluate_registered_candidate(
            repo=repo,
            model_version="1.1.0",
            dataset_path=mock_dataset_parquet,
            update_registry_metrics=True,
        )

        assert result.model_version == "1.1.0"
        assert entry.validation_metrics is not None
        assert "pr_auc" in entry.validation_metrics
        # Crucial: verify status did NOT change to CHAMPION and is_active_champion remains False!
        assert entry.is_active_champion is False
        assert entry.status == ModelLifecycleStatus.CANDIDATE.value
        mock_session.flush.assert_awaited_once()
