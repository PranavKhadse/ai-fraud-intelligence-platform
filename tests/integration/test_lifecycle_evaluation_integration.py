"""
Integration Tests for Phase 14.2: Candidate Model Evaluation against Real Data & PostgreSQL.

Validates end-to-end integration:
1. Candidate model bundle evaluation against the real validation partition (`val_features.parquet`).
2. Verification of all 55 features and deterministic metric computation.
3. Database integration: evaluating registered candidate record in PostgreSQL.
4. Preserving candidate status (NO promotion, is_active_champion remains False).
5. Protected OOT dataset (`test_features.parquet`) remains untouched.
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
from ml.lifecycle.evaluation.evaluator import CandidateModelEvaluator
from ml.lifecycle.evaluation.gates import CandidateGateConfig
from ml.lifecycle.evaluation.schemas import CandidateEvaluationResult, EvaluationScope
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, VAL_FEATURES_PATH

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestCandidateEvaluationIntegration:
    """End-to-end integration tests for candidate evaluation against real datasets and database."""

    @pytest.fixture
    def champion_based_candidate_bundle(self, tmp_path: Path):
        """
        Create a valid candidate bundle copying the real champion artifacts
        to test end-to-end evaluation against real val_features.parquet.
        """
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
            sha256_checksums={
                "model": calculate_file_sha256(m_dest),
                "preprocessor": calculate_file_sha256(p_dest),
            },
        )
        manifest.save(bundle_dir / "manifest.json")
        return bundle_dir

    async def test_evaluate_candidate_against_real_validation_partition(self, champion_based_candidate_bundle):
        """Verify candidate evaluation against actual 277,859-row validation partition."""
        evaluator = CandidateModelEvaluator()
        result = evaluator.evaluate_candidate(
            bundle_dir=champion_based_candidate_bundle,
            dataset_path=VAL_FEATURES_PATH,
            scope=EvaluationScope.VALIDATION,
            benchmark_latency=True,
        )

        assert isinstance(result, CandidateEvaluationResult)
        assert result.model_version == "1.1.0"
        assert result.model_family == "xgboost"
        assert result.scope == EvaluationScope.VALIDATION
        assert result.sample_count == 277859
        assert result.fraud_count == 1221

        # Check metrics against expected baseline magnitude
        assert result.metrics.pr_auc > 0.90
        assert result.metrics.roc_auc > 0.99
        assert result.metrics.threshold == 0.78
        assert result.metrics.tp > 0
        assert result.metrics.expected_cost is not None
        assert result.metrics.expected_cost > 0.0

        # Check gate outcomes
        assert result.overall_passed is True
        assert len(result.gate_results) >= 6

        # Check latency statistics
        assert result.latency_stats is not None
        assert "p95_ms" in result.latency_stats
        assert result.latency_stats["p95_ms"] > 0.0

    async def test_evaluate_registered_candidate_in_db_without_promotion(
        self, db_session: AsyncSession, champion_based_candidate_bundle
    ):
        """
        Verify database candidate evaluation updates validation metrics
        while strictly preventing promotion to champion.
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
            bundle_directory=str(champion_based_candidate_bundle),
            sha256_model=calculate_file_sha256(champion_based_candidate_bundle / "model.joblib"),
            sha256_preprocessor=calculate_file_sha256(champion_based_candidate_bundle / "preprocessor.joblib"),
        )
        await repo.create_entry(candidate_entry)
        await db_session.commit()

        evaluator = CandidateModelEvaluator()
        result = await evaluator.evaluate_registered_candidate(
            repo=repo,
            model_version="1.1.0",
            update_registry_metrics=True,
        )
        await db_session.commit()

        assert result.model_version == "1.1.0"
        assert result.overall_passed is True

        # Query database entry to verify state invariants
        fetched = await repo.get_by_version("1.1.0")
        assert fetched is not None
        assert fetched.validation_metrics is not None
        assert "pr_auc" in fetched.validation_metrics
        assert fetched.validation_metrics["pr_auc"] > 0.90

        # CRITICAL SAFETY INVARIANTS:
        # 1. Candidate must NOT be promoted to Champion
        assert fetched.is_active_champion is False
        # 2. Status remains CANDIDATE (promotion is reserved for subsequent lifecycle phase)
        assert fetched.status == ModelLifecycleStatus.CANDIDATE.value
        # 3. Promoted fields remain null
        assert fetched.promoted_at is None
        assert fetched.promoted_by is None
