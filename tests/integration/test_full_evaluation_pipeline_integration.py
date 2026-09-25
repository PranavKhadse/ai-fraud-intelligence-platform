"""
Integration Tests for Phase 14.3 Full Candidate Evaluation Pipeline.

Validates end-to-end integration:
1. Validation inference & threshold optimization across actual 277,859-row validation partition (`val_features.parquet`).
2. Protected OOT evaluation across actual 277,860-row test partition (`test_features.parquet`) strictly at frozen tau*.
3. Operational routing evaluation using existing RiskEvaluator on real partitions.
4. Active Champion v1.0.0 artifact immutability verification (hashes unmodified).
5. PostgreSQL database registry integration: updates candidate entry while strictly preserving CANDIDATE status.
"""

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
import joblib
import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.model_registry import ModelRegistryEntry
from backend.app.repositories.model_registry_repository import ModelRegistryRepository
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.evaluation.pipeline import FullCandidateEvaluationPipeline
from ml.lifecycle.evaluation.schemas import FullCandidateEvaluationResult
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.models.config import TEST_FEATURES_PATH, VAL_FEATURES_PATH

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestFullEvaluationPipelineIntegration:
    """End-to-end integration tests on real datasets and PostgreSQL database."""

    @pytest.fixture
    def real_candidate_bundle(self, tmp_path: Path):
        """Create a valid candidate bundle copying real model artifacts."""
        bundle_dir = tmp_path / "candidate_v1_1_0"
        bundle_dir.mkdir()

        m_src = Path("ml/models/artifacts/champion_model.joblib")
        p_src = Path("ml/models/artifacts/champion_preprocessor.joblib")

        m_dest = bundle_dir / "model.joblib"
        p_dest = bundle_dir / "preprocessor.joblib"

        m_dest.write_bytes(m_src.read_bytes())
        p_dest.write_bytes(p_src.read_bytes())

        manifest = ModelBundleManifest(
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE,
            operating_threshold=0.78,
            hyperparameters={"max_depth": 6, "learning_rate": 0.08},
            training_metadata={"source_dataset_rows": 648336, "positive_fraud_count": 3120},
            sha256_checksums={
                "model": calculate_file_sha256(m_dest),
                "preprocessor": calculate_file_sha256(p_dest),
            },
        )
        manifest.save(bundle_dir / "manifest.json")
        return bundle_dir

    async def test_end_to_end_evaluation_real_validation_and_oot(self, real_candidate_bundle):
        """
        Verify full evaluation pipeline on real 277,859-row validation partition
        and real 277,860-row protected OOT partition.
        """
        # 1. Snapshot Champion hashes to assert immutability
        champion_model_p = Path("ml/models/artifacts/champion_model.joblib")
        champion_prep_p = Path("ml/models/artifacts/champion_preprocessor.joblib")
        champion_meta_p = Path("ml/models/artifacts/model_metadata.json")

        champ_m_hash = calculate_file_sha256(champion_model_p)
        champ_p_hash = calculate_file_sha256(champion_prep_p)
        champ_meta_hash = calculate_file_sha256(champion_meta_p) if champion_meta_p.exists() else None

        pipeline = FullCandidateEvaluationPipeline()

        # 2. Execute full pipeline evaluation
        result = pipeline.evaluate_candidate_bundle(
            bundle_dir=real_candidate_bundle,
            val_dataset_path=VAL_FEATURES_PATH,
            oot_dataset_path=TEST_FEATURES_PATH,
            benchmark_latency=True,
            update_manifest=True,
        )

        assert isinstance(result, FullCandidateEvaluationResult)
        assert result.model_version == "1.1.0"
        assert result.model_family == "xgboost"

        # Validate partition sizes and fraud counts
        assert result.validation_sample_count == 277859
        assert result.validation_fraud_count == 1221
        assert result.oot_sample_count == 277860
        assert result.oot_fraud_count == 924

        # Validate optimal threshold tau*
        assert 0.0 < result.selected_operating_threshold < 1.0
        tau_star = result.selected_operating_threshold

        # Validate performance metrics on validation set
        assert result.validation_metrics.pr_auc > 0.90
        assert result.validation_metrics.roc_auc > 0.99
        assert result.validation_metrics.threshold == tau_star
        assert result.validation_metrics.tp > 0
        assert result.validation_metrics.expected_cost is not None
        assert result.validation_metrics.expected_cost > 0.0

        # Validate performance metrics on protected OOT set (evaluated at frozen tau*)
        assert result.oot_metrics.pr_auc > 0.90
        assert result.oot_metrics.roc_auc > 0.99
        assert result.oot_metrics.threshold == tau_star
        assert result.oot_metrics.tp > 0
        assert result.oot_metrics.expected_cost is not None
        assert result.oot_metrics.expected_cost > 0.0

        # Validate operational metrics
        assert result.validation_operational_metrics is not None
        assert result.validation_operational_metrics.operating_threshold == tau_star
        assert sum(result.validation_operational_metrics.action_counts.values()) == 277859
        assert result.validation_operational_metrics.fraud_in_block_count > 0

        assert result.oot_operational_metrics is not None
        assert result.oot_operational_metrics.operating_threshold == tau_star
        assert sum(result.oot_operational_metrics.action_counts.values()) == 277860

        # Validate latency distribution
        assert result.latency_stats is not None
        assert result.latency_stats["p95_ms"] > 0.0

        # 3. CRITICAL: Champion immutability assertion
        assert calculate_file_sha256(champion_model_p) == champ_m_hash
        assert calculate_file_sha256(champion_prep_p) == champ_p_hash
        if champ_meta_hash:
            assert calculate_file_sha256(champion_meta_p) == champ_meta_hash

        # 4. Validate updated manifest.json
        manifest = ModelBundleManifest.load(real_candidate_bundle / "manifest.json")
        assert manifest.operating_threshold == tau_star
        assert manifest.validation_metrics is not None
        assert manifest.oot_holdout_metrics is not None
        assert manifest.evaluation_configuration is not None
        assert manifest.status == ModelLifecycleStatus.CANDIDATE

    async def test_evaluate_registered_candidate_with_database(
        self, db_session: AsyncSession, real_candidate_bundle
    ):
        """
        Verify database candidate evaluation updates validation and OOT metrics
        in PostgreSQL while strictly preserving CANDIDATE lifecycle status.
        """
        repo = ModelRegistryRepository(db_session)
        entry_id = uuid.uuid4()

        candidate_entry = ModelRegistryEntry(
            id=entry_id,
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7800"),
            bundle_directory=str(real_candidate_bundle),
            sha256_model=calculate_file_sha256(real_candidate_bundle / "model.joblib"),
            sha256_preprocessor=calculate_file_sha256(real_candidate_bundle / "preprocessor.joblib"),
        )
        await repo.create_entry(candidate_entry)
        await db_session.commit()

        pipeline = FullCandidateEvaluationPipeline()
        result = await pipeline.evaluate_registered_candidate(
            repo=repo,
            model_version="1.1.0",
            val_dataset_path=VAL_FEATURES_PATH,
            oot_dataset_path=TEST_FEATURES_PATH,
            update_db=True,
        )
        await db_session.commit()

        assert result.model_version == "1.1.0"

        # Query database entry to verify state invariants
        fetched = await repo.get_by_version("1.1.0")
        assert fetched is not None
        assert fetched.operating_threshold == Decimal(str(result.selected_operating_threshold))
        assert fetched.validation_metrics is not None
        assert fetched.validation_metrics["pr_auc"] > 0.90
        assert fetched.oot_metrics is not None
        assert fetched.oot_metrics["pr_auc"] > 0.90

        # CRITICAL SAFETY INVARIANTS:
        # 1. Candidate must NOT be promoted to Champion
        assert fetched.is_active_champion is False
        # 2. Status remains strictly CANDIDATE
        assert fetched.status == ModelLifecycleStatus.CANDIDATE.value
        # 3. Promotion fields remain null
        assert fetched.promoted_at is None
        assert fetched.promoted_by is None
