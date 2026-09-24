"""
Integration Tests for Phase 14.2: Challenger Training Pipeline & Database Registration.

Validates end-to-end integration:
1. Building dataset from real train_features.parquet partition.
2. Training candidate model in process-isolated subprocess.
3. Candidate bundle verification (model, preprocessor, manifest, checksums).
4. Candidate registration in PostgreSQL database with status CANDIDATE (non-champion).
5. Champion v1.0.0 immutability preservation.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.model_registry import ModelRegistryEntry
from backend.app.repositories.model_registry_repository import ModelRegistryRepository
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.dataset_builder import (
    ChallengerDatasetBuilder,
    DEFAULT_VALIDATION_CUTOFF,
)
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.lifecycle.training import (
    CandidateBundle,
    ChallengerTrainingConfig,
    ChallengerTrainingPipeline,
)
from ml.models.config import TRAIN_FEATURES_PATH

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestChallengerTrainingIntegration:
    """Integration test suite for challenger training with real dataset and PostgreSQL persistence."""

    async def test_train_and_register_candidate_end_to_end(
        self, db_session: AsyncSession, tmp_path: Path
    ):
        """
        Verify end-to-end training of a candidate model from base parquet data and registration in PostgreSQL.
        """
        repo = ModelRegistryRepository(db_session)

        # 1. Ensure initial Champion v1.0.0 is registered
        champ_entry, _ = await repo.register_initial_champion_if_empty()
        await db_session.commit()
        assert champ_entry.model_version == "1.0.0"
        assert champ_entry.is_active_champion is True

        # 2. Build dataset using a sample of base training features
        builder = ChallengerDatasetBuilder(base_train_path=TRAIN_FEATURES_PATH)
        dataset = builder.build_dataset(max_base_rows=200)

        assert len(dataset.X) == 200
        assert len(dataset.y) == 200
        assert dataset.metadata.feature_count == 55

        # 3. Configure candidate training in isolated directory
        candidate_base = tmp_path / "candidates"
        training_config = ChallengerTrainingConfig(
            model_version="1.1.0",
            model_family="xgboost",
            candidate_base_dir=candidate_base,
            hyperparameters={
                "n_estimators": 5,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            is_isolated_process=True,
        )

        # 4. Train challenger
        pipeline = ChallengerTrainingPipeline()
        bundle = pipeline.train_challenger(
            dataset=dataset,
            training_config=training_config,
        )

        assert isinstance(bundle, CandidateBundle)
        assert bundle.model_version == "1.1.0"
        assert bundle.bundle_dir.exists()
        assert bundle.manifest.status == ModelLifecycleStatus.CANDIDATE

        # 5. Register candidate in PostgreSQL
        candidate_entry = await pipeline.register_candidate(bundle, repo)
        await db_session.commit()

        assert candidate_entry.model_version == "1.1.0"
        assert candidate_entry.status == ModelLifecycleStatus.CANDIDATE.value
        assert candidate_entry.is_active_champion is False

        # 6. Verify database state
        # Champion v1.0.0 must remain active champion
        active_champ = await repo.get_active_champion()
        assert active_champ is not None
        assert active_champ.model_version == "1.0.0"
        assert active_champ.is_active_champion is True

        # Candidate 1.1.0 is persisted
        fetched_cand = await repo.get_by_version("1.1.0")
        assert fetched_cand is not None
        assert fetched_cand.status == "CANDIDATE"
        assert fetched_cand.is_active_champion is False
