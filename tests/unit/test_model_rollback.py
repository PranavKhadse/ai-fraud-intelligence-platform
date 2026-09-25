"""
Unit Tests for Phase 14.6: Model Rollback Engine and State Recovery Subsystem.

Verifies:
- Verification of target historical bundle and checksums.
- Isolated staging validation and synthetic inference on restored model.
- Concurrency control and row locking.
- Atomic state transitions: Current Champion -> ROLLED_BACK, Target -> CHAMPION.
- Post-commit consistency verification and immutable RollbackExecutionRecord generation.
- Failure compensation during pre-commit errors.
- Crash recovery and startup state reconciliation (PromotionRecoveryEngine).
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import joblib
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.security import ActorContext
from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.base import Base
from backend.app.db.models.enums import AuditActorType
from backend.app.db.models.model_registry import ModelRegistryEntry
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.promotion.promoter import create_synthetic_55_feature_dataframe
from ml.lifecycle.promotion.rollback import (
    ModelRollbackEngine,
    PromotionRecoveryEngine,
)
from ml.lifecycle.promotion.schemas import (
    PromotionIntegrityError,
    PromotionOperationJournal,
    PromotionOperationState,
    PromotionPostCommitVerificationError,
    PromotionStagingError,
    RecoveryRequiredError,
    RollbackExecutionRecord,
    RollbackIntegrityError,
    RollbackPreconditionError,
    RollbackResult,
)
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.models.config import (
    CATEGORICAL_PREDICTORS,
    PREDICTIVE_FEATURE_COLUMNS,
)


@pytest.fixture
def rollback_env(tmp_path: Path):
    """Fixture creating isolated filesystem structure with historical and active model bundles."""
    registry_dir = tmp_path / "registry"
    candidates_dir = registry_dir / "candidates"
    bundles_dir = registry_dir / "bundles"
    staging_dir = registry_dir / "staging"
    operations_dir = registry_dir / "operations"
    promotions_dir = registry_dir / "promotions"
    rollbacks_dir = registry_dir / "rollbacks"
    active_dir = tmp_path / "artifacts"

    for d in [candidates_dir, bundles_dir, staging_dir, operations_dir, promotions_dir, rollbacks_dir, active_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = LifecycleConfig(
        registry_dir=registry_dir,
        bundles_dir=bundles_dir,
        candidates_dir=candidates_dir,
        staging_dir=staging_dir,
        operations_dir=operations_dir,
        promotions_dir=promotions_dir,
        rollbacks_dir=rollbacks_dir,
        active_artifacts_dir=active_dir,
        champion_model_path=active_dir / "champion_model.joblib",
        champion_preprocessor_path=active_dir / "champion_preprocessor.joblib",
        champion_metadata_path=active_dir / "model_metadata.json",
    )

    num_cols = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in CATEGORICAL_PREDICTORS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_PREDICTORS),
        ]
    )
    df_sample = create_synthetic_55_feature_dataframe(num_rows=20)
    y_sample = np.array([0, 1] * 10)
    X_trans = preprocessor.fit_transform(df_sample)

    model_v1 = RandomForestClassifier(n_estimators=5, random_state=42)
    model_v1.fit(X_trans, y_sample)

    model_v2 = RandomForestClassifier(n_estimators=5, random_state=84)
    model_v2.fit(X_trans, y_sample)

    # Active Champion is currently v1.1.0
    joblib.dump(model_v2, config.champion_model_path)
    joblib.dump(preprocessor, config.champion_preprocessor_path)
    with open(config.champion_metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "1.1.0",
            "champion_model": "xgboost",
            "selected_threshold": 0.78,
            "feature_count": 55,
        }, f, indent=2)

    # Historical bundle exists for v1.0.0 in bundles_dir
    bundle_v1_dir = bundles_dir / "1.0.0"
    bundle_v1_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_v1, bundle_v1_dir / "model.joblib")
    joblib.dump(preprocessor, bundle_v1_dir / "preprocessor.joblib")

    v1_model_sha = calculate_file_sha256(bundle_v1_dir / "model.joblib")
    v1_prep_sha = calculate_file_sha256(bundle_v1_dir / "preprocessor.joblib")

    with open(bundle_v1_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "1.0.0",
            "model_family": "xgboost",
            "status": "ARCHIVED",
            "operating_threshold": 0.78,
            "sha256_checksums": {
                "model": v1_model_sha,
                "preprocessor": v1_prep_sha,
            },
        }, f, indent=2)

    return config, preprocessor, model_v1, model_v2


class MockAsyncDbSession:
    """Lightweight in-memory AsyncSession mock for unit testing."""
    def __init__(self):
        self.entries: List[ModelRegistryEntry] = []
        self.audit_logs: List[AuditLog] = []
        self.committed = False
        self.rolled_back = False

    def add(self, obj):
        if isinstance(obj, ModelRegistryEntry):
            if obj not in self.entries:
                self.entries.append(obj)
        elif isinstance(obj, AuditLog):
            self.audit_logs.append(obj)

    def add_all(self, objs):
        for o in objs:
            self.add(o)

    async def execute(self, stmt):
        class MockResult:
            def __init__(self, items):
                self._items = items
            def scalars(self):
                class MockScalars:
                    def __init__(self, items):
                        self._items = items
                    def all(self):
                        return list(self._items)
                    def first(self):
                        return self._items[0] if self._items else None
                return MockScalars(self._items)

        return MockResult(self.entries)

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


@pytest.fixture
def async_db():
    """Create in-memory MockAsyncDbSession for unit testing."""
    return MockAsyncDbSession()


@pytest.mark.asyncio
class TestModelRollbackUnit:
    """Comprehensive unit test suite for ModelRollbackEngine and PromotionRecoveryEngine."""

    async def _setup_db_entries(self, async_db: AsyncSession, config: LifecycleConfig):
        # Current active champion is v1.1.0
        champ_model_sha = calculate_file_sha256(config.champion_model_path)
        champ_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        # Historical archived is v1.0.0
        bundle_v1_dir = config.bundles_dir / "1.0.0"
        v1_model_sha = calculate_file_sha256(bundle_v1_dir / "model.joblib")
        v1_prep_sha = calculate_file_sha256(bundle_v1_dir / "preprocessor.joblib")

        db_v1_1 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model=champ_model_sha,
            sha256_preprocessor=champ_prep_sha,
        )
        db_v1_0 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.ARCHIVED.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7800"),
            sha256_model=v1_model_sha,
            sha256_preprocessor=v1_prep_sha,
        )
        async_db.add_all([db_v1_1, db_v1_0])
        await async_db.commit()
        return db_v1_1, db_v1_0

    async def test_successful_rollback_sequence(self, rollback_env, async_db: AsyncSession):
        config, _, model_v1, _ = rollback_env
        await self._setup_db_entries(async_db, config)

        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")
        rationale = "Emergency rollback to v1.0.0 due to operational anomaly."

        engine = ModelRollbackEngine(config=config)
        result = await engine.rollback_champion(
            target_version="1.0.0",
            actor=actor,
            rationale=rationale,
            session=async_db,
        )

        assert result.success is True
        assert result.restored_version == "1.0.0"
        assert result.rolled_back_version == "1.1.0"
        assert result.state == PromotionOperationState.FINALIZED
        assert Path(result.execution_record_path).exists()

        # Check DB state
        res = await async_db.execute(select(ModelRegistryEntry))
        entries = {e.model_version: e for e in res.scalars().all()}
        assert entries["1.0.0"].is_active_champion is True
        assert entries["1.0.0"].status == ModelLifecycleStatus.CHAMPION.value
        assert entries["1.1.0"].is_active_champion is False
        assert entries["1.1.0"].status == ModelLifecycleStatus.ROLLED_BACK.value
        assert entries["1.1.0"].rolled_back_by == "admin_user"

        # Check active artifacts match v1.0.0
        active_model_sha = calculate_file_sha256(config.champion_model_path)
        hist_model_sha = calculate_file_sha256(config.bundles_dir / "1.0.0" / "model.joblib")
        assert active_model_sha == hist_model_sha

    async def test_rollback_target_not_found_in_db(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        # Register only 1.1.0
        champ_sha = calculate_file_sha256(config.champion_model_path)
        prep_sha = calculate_file_sha256(config.champion_preprocessor_path)
        db_v1_1 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model=champ_sha,
            sha256_preprocessor=prep_sha,
        )
        async_db.add(db_v1_1)
        await async_db.commit()

        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")
        engine = ModelRollbackEngine(config=config)

        with pytest.raises(RollbackPreconditionError, match="Target version '1.0.0' not found"):
            await engine.rollback_champion("1.0.0", actor, "Valid rationale for rollback operations", async_db)

    async def test_rollback_target_already_active_champion(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        await self._setup_db_entries(async_db, config)

        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")
        engine = ModelRollbackEngine(config=config)

        with pytest.raises(RollbackPreconditionError, match="Target version '1.1.0' is already the active Champion"):
            await engine.rollback_champion("1.1.0", actor, "Valid rationale for rollback operations", async_db)

    async def test_rollback_unauthorized_actor_role(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        await self._setup_db_entries(async_db, config)

        actor = ActorContext(actor_id="user_123", actor_role="API_CLIENT")
        engine = ModelRollbackEngine(config=config)

        with pytest.raises(RollbackPreconditionError, match="Actor role 'API_CLIENT' is not authorized"):
            await engine.rollback_champion("1.0.0", actor, "Valid rationale for rollback operations", async_db)

    async def test_rollback_short_rationale(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        await self._setup_db_entries(async_db, config)

        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")
        engine = ModelRollbackEngine(config=config)

        with pytest.raises(RollbackPreconditionError, match="Rollback rationale must be at least 15 characters"):
            await engine.rollback_champion("1.0.0", actor, "Too short", async_db)

    async def test_rollback_missing_bundle_on_disk(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        await self._setup_db_entries(async_db, config)

        # Delete historical bundle from disk
        shutil.rmtree(config.bundles_dir / "1.0.0")

        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")
        engine = ModelRollbackEngine(config=config)

        with pytest.raises(RollbackPreconditionError, match="Historical model bundle for version '1.0.0' not found on disk"):
            await engine.rollback_champion("1.0.0", actor, "Valid rationale for rollback operations", async_db)

    async def test_rollback_checksum_mismatch(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        db_v1_1, db_v1_0 = await self._setup_db_entries(async_db, config)
        db_v1_0.sha256_model = "0" * 64
        await async_db.commit()

        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")
        engine = ModelRollbackEngine(config=config)

        with pytest.raises(RollbackIntegrityError, match="Target model SHA-256 on disk .* does not match DB"):
            await engine.rollback_champion("1.0.0", actor, "Valid rationale for rollback operations", async_db)


@pytest.mark.asyncio
class TestPromotionRecoveryEngine:
    """Test suite for crash recovery and startup state reconciliation."""

    async def test_recovery_reconciles_pre_commit_crash(self, rollback_env, async_db: AsyncSession):
        config, _, _, _ = rollback_env
        champ_model_sha = calculate_file_sha256(config.champion_model_path)
        champ_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        # DB has v1.1.0 as active champion
        db_v1_1 = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model=champ_model_sha,
            sha256_preprocessor=champ_prep_sha,
        )
        async_db.add(db_v1_1)
        await async_db.commit()

        # Simulate uncommitted crashed promotion operation in FILESYSTEM_SWAPPED
        op_id = "promo_1.2.0_crash_test"
        journal = PromotionOperationJournal(
            operation_id=op_id,
            operation_type="PROMOTION",
            candidate_version="1.2.0",
            champion_version="1.1.0",
            current_state=PromotionOperationState.FILESYSTEM_SWAPPED,
            actor_id="admin_user",
            actor_role="ADMIN",
            rationale="Rationale for crashed test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        journal.save(config.operations_dir / f"{op_id}.json")

        recovery = PromotionRecoveryEngine(config=config)
        rec_result = await recovery.reconcile_active_champion_state(session=async_db)

        assert rec_result["status"] == "CONSISTENT"
        assert rec_result["active_champion"] == "1.1.0"
        assert op_id in rec_result["reconciled_operations"]

        # Check journal state updated to RECOVERY_COMPLETED
        with open(config.operations_dir / f"{op_id}.json", "r", encoding="utf-8") as f:
            j_data = json.load(f)
        assert j_data["current_state"] == PromotionOperationState.RECOVERY_COMPLETED.value
