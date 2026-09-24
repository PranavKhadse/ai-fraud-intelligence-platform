"""
Integration Tests for Phase 14.1: Model Registry Database Persistence & Constraints.

Validates end-to-end integration against real PostgreSQL:
1. ORM table schema, UUID primary keys, and timestamp generation.
2. Check constraints: status, threshold range, model family.
3. Unique version constraint.
4. Active champion exclusivity and atomic promotion/demotion.
5. Idempotent registration of legacy Champion v1.0.0 against actual artifacts.
"""

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.model_registry import ModelRegistryEntry
from backend.app.repositories.model_registry_repository import ModelRegistryRepository
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    calculate_file_sha256,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestModelRegistryIntegration:
    """Integration test suite for Model Registry with real database transactions."""

    async def test_create_and_query_registry_entry(self, db_session: AsyncSession):
        repo = ModelRegistryRepository(db_session)
        entry_id = uuid.uuid4()
        entry = ModelRegistryEntry(
            id=entry_id,
            model_version="1.5.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7800"),
            bundle_directory="ml/models/registry/bundles/v1.5.0",
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
            validation_metrics={"pr_auc": 0.965, "roc_auc": 0.982},
            training_metadata={"feature_count": 55, "train_rows": 120000},
        )

        created = await repo.create_entry(entry)
        await db_session.commit()

        assert created.id == entry_id
        assert created.created_at is not None

        # Fetch by ID
        fetched = await repo.get_by_id(entry_id)
        assert fetched is not None
        assert fetched.model_version == "1.5.0"
        assert fetched.validation_metrics["pr_auc"] == 0.965

        # Fetch by Version
        fetched_v = await repo.get_by_version("1.5.0")
        assert fetched_v is not None
        assert fetched_v.id == entry_id

    async def test_unique_version_constraint_enforced(self, db_session: AsyncSession):
        repo = ModelRegistryRepository(db_session)
        entry1 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.6.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        await repo.create_entry(entry1)
        await db_session.commit()

        entry2 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.6.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            sha256_model="c" * 64,
            sha256_preprocessor="d" * 64,
        )
        with pytest.raises(ValueError, match="already exists"):
            await repo.create_entry(entry2)

    async def test_invalid_status_check_constraint(self, db_session: AsyncSession):
        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.7.0",
            model_family="xgboost",
            status="INVALID_STATUS",
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        db_session.add(entry)
        with pytest.raises(IntegrityError):
            await db_session.commit()
        await db_session.rollback()

    async def test_invalid_threshold_check_constraint(self, db_session: AsyncSession):
        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.8.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            operating_threshold=Decimal("1.5000"),  # > 0.9999
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        db_session.add(entry)
        with pytest.raises(IntegrityError):
            await db_session.commit()
        await db_session.rollback()

    async def test_invalid_model_family_check_constraint(self, db_session: AsyncSession):
        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.9.0",
            model_family="random_forest",
            status=ModelLifecycleStatus.CANDIDATE.value,
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        db_session.add(entry)
        with pytest.raises(IntegrityError):
            await db_session.commit()
        await db_session.rollback()

    async def test_active_champion_promotion_and_demotion_lifecycle(self, db_session: AsyncSession):
        repo = ModelRegistryRepository(db_session)

        # 1. Register v1.0.0 as active champion
        v100 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        await repo.create_entry(v100)

        # 2. Register v1.1.0 as challenger
        v110 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHALLENGER.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7600"),
            sha256_model="c" * 64,
            sha256_preprocessor="d" * 64,
        )
        await repo.create_entry(v110)
        await db_session.commit()

        # Active champion is v1.0.0
        active = await repo.get_active_champion()
        assert active is not None
        assert active.model_version == "1.0.0"

        # 3. Promote v1.1.0
        promoted = await repo.set_active_champion(
            model_version="1.1.0",
            promoted_by="lead_fraud_analyst",
            promotion_rationale="Passed all evaluation gates with higher PR-AUC and lower expected cost.",
        )
        await db_session.commit()

        assert promoted.is_active_champion is True
        assert promoted.status == ModelLifecycleStatus.CHAMPION.value
        assert promoted.promoted_by == "lead_fraud_analyst"

        # Old champion v1.0.0 is ARCHIVED and not active
        old_champ = await repo.get_by_version("1.0.0")
        assert old_champ.is_active_champion is False
        assert old_champ.status == ModelLifecycleStatus.ARCHIVED.value

        # Active champion is now exclusively v1.1.0
        current_champ = await repo.get_active_champion()
        assert current_champ.model_version == "1.1.0"

    async def test_initial_champion_registration_with_actual_project_artifacts(self, db_session: AsyncSession):
        repo = ModelRegistryRepository(db_session)

        # Register using actual project champion artifacts
        entry, newly_created = await repo.register_initial_champion_if_empty()
        await db_session.commit()

        assert newly_created is True
        assert entry.model_version == "1.0.0"
        assert entry.is_active_champion is True
        assert entry.status == ModelLifecycleStatus.CHAMPION.value
        assert entry.model_family == "xgboost"
        assert entry.operating_threshold == Decimal("0.7800")
        assert len(entry.sha256_model) == 64
        assert len(entry.sha256_preprocessor) == 64

        # Verify idempotency
        entry2, newly_created2 = await repo.register_initial_champion_if_empty()
        assert newly_created2 is False
        assert entry2.id == entry.id
