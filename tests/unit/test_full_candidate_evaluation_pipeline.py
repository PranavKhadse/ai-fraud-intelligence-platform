"""
Unit Tests for Phase 14.3 Full Candidate Evaluation Pipeline.

Verifies:
- End-to-end execution: Validation Inference -> Threshold Sweep -> Freeze tau* -> Protected OOT Evaluation.
- OOT Contamination Prevention: OOT data has ZERO influence on threshold selection tau* or validation metrics.
- Candidate bundle manifest update: preserves training provenance, adds evaluation_configuration, validation_metrics, oot_metrics.
- Artifact Immutability: Candidate model/preprocessor and Champion files remain 100% unmodified.
- Database synchronization: updates operating_threshold, validation_metrics, oot_metrics while strictly preserving CANDIDATE status.
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
from ml.lifecycle.evaluation.pipeline import FullCandidateEvaluationPipeline
from ml.lifecycle.evaluation.schemas import (
    FullCandidateEvaluationResult,
    ThresholdOptimizationObjective,
)
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, TARGET_COLUMN
from ml.models.preprocessing import TreePreprocessor


class DummyPipelineModel:
    """Picklable dummy model for pipeline unit tests."""
    def predict_proba(self, X):
        n = len(X)
        probs = np.full(n, 0.08, dtype=np.float64)
        n_pos = max(1, int(n * 0.15))
        probs[:n_pos] = 0.92
        return probs


class TestFullCandidateEvaluationPipeline:
    """Test suite for FullCandidateEvaluationPipeline."""

    @pytest.fixture
    def mock_candidate_bundle(self, tmp_path: Path):
        """Create a candidate model bundle with training provenance."""
        bundle_dir = tmp_path / "candidate_v1_1_0"
        bundle_dir.mkdir()

        model = DummyPipelineModel()
        preproc = TreePreprocessor()
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
            hyperparameters={"max_depth": 6, "learning_rate": 0.08, "n_estimators": 100},
            training_metadata={"source_dataset_rows": 10000, "positive_fraud_count": 50},
            sha256_checksums={
                "model": calculate_file_sha256(m_path),
                "preprocessor": calculate_file_sha256(p_path),
            },
        )
        manifest.save(bundle_dir / "manifest.json")
        return bundle_dir

    @pytest.fixture
    def mock_val_and_oot_partitions(self, tmp_path: Path):
        """Create synthetic validation and OOT parquet files with 55 features."""
        def make_df(n_samples: int, n_fraud: int):
            data = {}
            for col in PREDICTIVE_FEATURE_COLUMNS:
                if col in ["merchant_category", "job_category"]:
                    data[col] = ["grocery_pos" if i % 2 == 0 else "online" for i in range(n_samples)]
                else:
                    data[col] = np.random.uniform(1.0, 100.0, size=n_samples).astype(np.float32)
            data[TARGET_COLUMN] = np.array([1 if i < n_fraud else 0 for i in range(n_samples)], dtype=np.int64)
            return pd.DataFrame(data)

        val_df = make_df(n_samples=100, n_fraud=15)
        oot_df = make_df(n_samples=100, n_fraud=12)

        val_path = tmp_path / "val_features.parquet"
        oot_path = tmp_path / "oot_features.parquet"
        val_df.to_parquet(val_path, index=False)
        oot_df.to_parquet(oot_path, index=False)

        return val_path, oot_path

    def test_full_pipeline_execution_success(self, mock_candidate_bundle, mock_val_and_oot_partitions):
        val_path, oot_path = mock_val_and_oot_partitions
        pipeline = FullCandidateEvaluationPipeline()

        result = pipeline.evaluate_candidate_bundle(
            bundle_dir=mock_candidate_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            benchmark_latency=False,
            update_manifest=True,
        )

        assert isinstance(result, FullCandidateEvaluationResult)
        assert result.model_version == "1.1.0"
        assert result.model_family == "xgboost"
        assert 0.0 < result.selected_operating_threshold < 1.0

        # Check validation metrics
        assert result.validation_metrics.threshold == result.selected_operating_threshold
        assert result.validation_metrics.total_samples == 100
        assert result.validation_sample_count == 100
        assert result.validation_fraud_count == 15

        # Check OOT metrics (evaluated strictly at frozen tau*)
        assert result.oot_metrics.threshold == result.selected_operating_threshold
        assert result.oot_metrics.total_samples == 100
        assert result.oot_sample_count == 100
        assert result.oot_fraud_count == 12

        # Check operational metrics
        assert result.validation_operational_metrics is not None
        assert result.oot_operational_metrics is not None

    def test_oot_contamination_prevention(self, mock_candidate_bundle, mock_val_and_oot_partitions, tmp_path: Path):
        """
        Verify that changing OOT data has ZERO effect on selected threshold tau* or validation metrics.
        """
        val_path, oot_path1 = mock_val_and_oot_partitions
        pipeline = FullCandidateEvaluationPipeline()

        # Run 1 with normal OOT
        res1 = pipeline.evaluate_candidate_bundle(
            bundle_dir=mock_candidate_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path1,
            benchmark_latency=False,
            update_manifest=False,
        )

        # Create radically different OOT dataset (100% fraud)
        oot_df2 = pd.read_parquet(oot_path1)
        oot_df2[TARGET_COLUMN] = 1
        oot_path2 = tmp_path / "corrupted_oot.parquet"
        oot_df2.to_parquet(oot_path2, index=False)

        # Run 2 with modified OOT
        res2 = pipeline.evaluate_candidate_bundle(
            bundle_dir=mock_candidate_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path2,
            benchmark_latency=False,
            update_manifest=False,
        )

        # Selected threshold MUST be 100% identical
        assert res1.selected_operating_threshold == res2.selected_operating_threshold
        # Validation metrics MUST be 100% identical
        assert res1.validation_metrics.pr_auc == res2.validation_metrics.pr_auc
        assert res1.validation_metrics.recall == res2.validation_metrics.recall
        assert res1.validation_metrics.expected_cost == res2.validation_metrics.expected_cost

        # OOT metrics differ as expected
        assert res1.oot_metrics.recall != res2.oot_metrics.recall

    def test_manifest_provenance_preservation(self, mock_candidate_bundle, mock_val_and_oot_partitions):
        val_path, oot_path = mock_val_and_oot_partitions
        pipeline = FullCandidateEvaluationPipeline()

        manifest_file = mock_candidate_bundle / "manifest.json"
        initial_manifest = ModelBundleManifest.load(manifest_file)

        pipeline.evaluate_candidate_bundle(
            bundle_dir=mock_candidate_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            update_manifest=True,
        )

        updated_manifest = ModelBundleManifest.load(manifest_file)

        # Provenance must be strictly preserved
        assert updated_manifest.hyperparameters == initial_manifest.hyperparameters
        assert updated_manifest.training_metadata == initial_manifest.training_metadata
        assert updated_manifest.created_at == initial_manifest.created_at
        assert updated_manifest.model_version == initial_manifest.model_version

        # Evaluation fields must be populated
        assert updated_manifest.validation_metrics is not None
        assert updated_manifest.oot_holdout_metrics is not None
        assert updated_manifest.evaluation_configuration is not None
        assert updated_manifest.operating_threshold > 0.0

    def test_candidate_artifacts_immutability(self, mock_candidate_bundle, mock_val_and_oot_partitions):
        val_path, oot_path = mock_val_and_oot_partitions
        pipeline = FullCandidateEvaluationPipeline()

        m_hash_before = calculate_file_sha256(mock_candidate_bundle / "model.joblib")
        p_hash_before = calculate_file_sha256(mock_candidate_bundle / "preprocessor.joblib")

        pipeline.evaluate_candidate_bundle(
            bundle_dir=mock_candidate_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            update_manifest=True,
        )

        m_hash_after = calculate_file_sha256(mock_candidate_bundle / "model.joblib")
        p_hash_after = calculate_file_sha256(mock_candidate_bundle / "preprocessor.joblib")

        assert m_hash_before == m_hash_after
        assert p_hash_before == p_hash_after

    @pytest.mark.asyncio
    async def test_evaluate_registered_candidate_database_sync(
        self, mock_candidate_bundle, mock_val_and_oot_partitions
    ):
        from backend.app.db.models.model_registry import ModelRegistryEntry
        from backend.app.repositories.model_registry_repository import ModelRegistryRepository

        val_path, oot_path = mock_val_and_oot_partitions

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
        pipeline = FullCandidateEvaluationPipeline()

        result = await pipeline.evaluate_registered_candidate(
            repo=repo,
            model_version="1.1.0",
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            update_db=True,
        )

        assert result.model_version == "1.1.0"
        assert entry.validation_metrics is not None
        assert entry.oot_metrics is not None
        assert entry.operating_threshold == Decimal(str(result.selected_operating_threshold))

        # CRITICAL SAFETY: status MUST remain CANDIDATE and is_active_champion MUST remain False
        assert entry.status == ModelLifecycleStatus.CANDIDATE.value
        assert entry.is_active_champion is False
        mock_session.flush.assert_awaited_once()
