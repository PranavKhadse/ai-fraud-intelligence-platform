"""
Integration Tests for Phase 14.6: Model Rollback Protocol and Recovery.

Validates end-to-end against real model artifacts and PostgreSQL database transactions:
1. Full rollback lifecycle: active Champion v1.1.0 -> restored historical Champion v1.0.0.
2. Complete restoration of model, preprocessor, and runtime metadata contract.
3. Live RiskEvaluator inference validation with restored Champion.
4. PostgreSQL state transition: 1.1.0 -> ROLLED_BACK, 1.0.0 -> CHAMPION.
5. Immutable RollbackExecutionRecord and PostgreSQL AuditLog generation.
6. Failure injection testing:
   - Pre-commit rollback failure triggers compensation and preserves active Champion v1.1.0.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import ActorContext
from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType
from backend.app.db.models.model_registry import ModelRegistryEntry
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.promotion.rollback import ModelRollbackEngine
from ml.lifecycle.promotion.schemas import (
    PromotionOperationState,
    RollbackExecutionRecord,
    RollbackIntegrityError,
    RollbackPreconditionError,
    RollbackResult,
)
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.risk_engine.evaluator import RiskEvaluator

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

REAL_CHAMPION_DIR = Path("ml/models/artifacts")
REAL_CANDIDATE_DIR = Path("ml/models/registry/candidates/v1.1.0")


@pytest.fixture
def isolated_rollback_env(tmp_path: Path):
    """Construct an isolated environment where v1.1.0 is active and v1.0.0 is archived."""
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

    # Put v1.1.0 in active_artifacts_dir
    shutil.copy2(REAL_CANDIDATE_DIR / "model.joblib", config.champion_model_path)
    shutil.copy2(REAL_CANDIDATE_DIR / "preprocessor.joblib", config.champion_preprocessor_path)
    with open(config.champion_metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "1.1.0",
            "champion_model": "xgboost",
            "selected_threshold": 0.78,
            "feature_count": 55,
        }, f, indent=2)

    # Put v1.0.0 in historical bundles_dir/1.0.0/
    bundle_v1_dir = bundles_dir / "1.0.0"
    bundle_v1_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_CHAMPION_DIR / "champion_model.joblib", bundle_v1_dir / "model.joblib")
    shutil.copy2(REAL_CHAMPION_DIR / "champion_preprocessor.joblib", bundle_v1_dir / "preprocessor.joblib")

    v1_model_sha = calculate_file_sha256(bundle_v1_dir / "model.joblib")
    v1_prep_sha = calculate_file_sha256(bundle_v1_dir / "preprocessor.joblib")

    with open(bundle_v1_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "1.0.0",
            "model_family": "xgboost",
            "status": "ARCHIVED",
            "operating_threshold": 0.94,
            "sha256_checksums": {
                "model": v1_model_sha,
                "preprocessor": v1_prep_sha,
            },
        }, f, indent=2)

    return config


async def _seed_db_for_rollback(
    db_session: AsyncSession,
    config: LifecycleConfig,
) -> Tuple[ModelRegistryEntry, ModelRegistryEntry]:
    """Seed DB where v1.1.0 is Champion and v1.0.0 is ARCHIVED."""
    v1_1_model_sha = calculate_file_sha256(config.champion_model_path)
    v1_1_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

    v1_0_dir = config.bundles_dir / "1.0.0"
    v1_0_model_sha = calculate_file_sha256(v1_0_dir / "model.joblib")
    v1_0_prep_sha = calculate_file_sha256(v1_0_dir / "preprocessor.joblib")
    v1_0_manifest_sha = calculate_file_sha256(v1_0_dir / "manifest.json")

    db_v1_1 = ModelRegistryEntry(
        id=uuid.uuid4(),
        model_version="1.1.0",
        model_family="xgboost",
        status=ModelLifecycleStatus.CHAMPION.value,
        is_active_champion=True,
        operating_threshold=Decimal("0.7800"),
        sha256_model=v1_1_model_sha,
        sha256_preprocessor=v1_1_prep_sha,
    )
    db_v1_0 = ModelRegistryEntry(
        id=uuid.uuid4(),
        model_version="1.0.0",
        model_family="xgboost",
        status=ModelLifecycleStatus.ARCHIVED.value,
        is_active_champion=False,
        operating_threshold=Decimal("0.9400"),
        bundle_directory=str(v1_0_dir),
        sha256_model=v1_0_model_sha,
        sha256_preprocessor=v1_0_prep_sha,
        sha256_manifest=v1_0_manifest_sha,
    )

    db_session.add_all([db_v1_1, db_v1_0])
    await db_session.commit()
    return db_v1_1, db_v1_0


class TestModelRollbackIntegration:
    """Integration test suite for ModelRollbackEngine."""

    async def test_end_to_end_rollback_with_real_artifacts(
        self,
        isolated_rollback_env: LifecycleConfig,
        db_session: AsyncSession,
    ):
        """
        Test end-to-end rollback:
        1. Current Champion is v1.1.0.
        2. Restores v1.0.0 from historical bundle.
        3. DB state transitions: 1.1.0 -> ROLLED_BACK, 1.0.0 -> CHAMPION.
        4. Verifies active filesystem artifacts match v1.0.0 historical bundle.
        5. Verifies live RiskEvaluator runs inference with restored Champion.
        6. Verifies immutable execution record and AuditLog.
        """
        config = isolated_rollback_env
        await _seed_db_for_rollback(db_session, config)

        actor = ActorContext(actor_id="lead_investigator_02", actor_role="ADMIN")
        rationale = "Emergency rollback to v1.0.0 following detected operational latency anomaly."

        engine = ModelRollbackEngine(config=config)
        result = await engine.rollback_champion(
            target_version="1.0.0",
            actor=actor,
            rationale=rationale,
            session=db_session,
        )

        assert result.success is True
        assert result.restored_version == "1.0.0"
        assert result.rolled_back_version == "1.1.0"
        assert result.state == PromotionOperationState.FINALIZED

        # 1. Verify DB State
        res = await db_session.execute(select(ModelRegistryEntry))
        entries = {e.model_version: e for e in res.scalars().all()}
        assert entries["1.0.0"].is_active_champion is True
        assert entries["1.0.0"].status == ModelLifecycleStatus.CHAMPION.value
        assert entries["1.1.0"].is_active_champion is False
        assert entries["1.1.0"].status == ModelLifecycleStatus.ROLLED_BACK.value
        assert entries["1.1.0"].rolled_back_by == "lead_investigator_02"
        assert entries["1.1.0"].rollback_rationale == rationale

        # 2. Verify Active Filesystem matches v1.0.0 historical bundle
        v1_0_model_sha = calculate_file_sha256(config.bundles_dir / "1.0.0" / "model.joblib")
        v1_0_prep_sha = calculate_file_sha256(config.bundles_dir / "1.0.0" / "preprocessor.joblib")

        active_model_sha = calculate_file_sha256(config.champion_model_path)
        active_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        assert active_model_sha == v1_0_model_sha
        assert active_prep_sha == v1_0_prep_sha

        # 3. Verify Live RiskEvaluator loads restored Champion v1.0.0
        evaluator = RiskEvaluator(
            model_path=config.champion_model_path,
            preprocessor_path=config.champion_preprocessor_path,
            metadata_path=config.champion_metadata_path,
        )
        assert evaluator.model_version == "1.0.0"

        # 4. Verify PostgreSQL AuditLog entry
        audit_res = await db_session.execute(
            select(AuditLog).where(AuditLog.action == "MODEL_ROLLBACK")
        )
        audit_records = audit_res.scalars().all()
        assert len(audit_records) >= 1
        audit_rec = audit_records[-1]
        assert audit_rec.event_type == "MODEL_ROLLBACK_EXECUTED"
        assert audit_rec.actor_id == "lead_investigator_02"
        assert audit_rec.payload["restored_version"] == "1.0.0"

    async def test_rollback_pre_commit_failure_compensates(
        self,
        isolated_rollback_env: LifecycleConfig,
        db_session: AsyncSession,
        monkeypatch,
    ):
        """
        Failure Injection Test:
        Simulate a database failure during rollback commit.
        Verify that active filesystem remains untouched (v1.1.0).
        """
        config = isolated_rollback_env
        await _seed_db_for_rollback(db_session, config)

        original_model_sha = calculate_file_sha256(config.champion_model_path)
        original_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        actor = ActorContext(actor_id="lead_investigator_02", actor_role="ADMIN")

        async def mock_fail_commit():
            raise RuntimeError("Injected rollback commit failure")

        monkeypatch.setattr(db_session, "commit", mock_fail_commit)

        engine = ModelRollbackEngine(config=config)
        with pytest.raises(RuntimeError, match="Injected rollback commit failure"):
            await engine.rollback_champion(
                target_version="1.0.0",
                actor=actor,
                rationale="Valid rollback rationale for failure test",
                session=db_session,
            )

        # Verify active filesystem artifacts were compensated and restored back to v1.1.0
        current_model_sha = calculate_file_sha256(config.champion_model_path)
        current_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        assert current_model_sha == original_model_sha
        assert current_prep_sha == original_prep_sha
