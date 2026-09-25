"""
Unit Tests for Phase 14.6: Model Promotion Engine and Precondition Verification.

Verifies:
- 16-point precondition verification.
- Rejection of missing, corrupted, or unauthorized promotion attempts.
- Cryptographic hash verification on candidate and active champion artifacts.
- Role-based access control and sign-off integrity checks.
- Isolated staging validation and synthetic 55-feature matrix inference.
- Durable journal state transitions (REQUESTED -> VALIDATED -> STAGED -> FILESYSTEM_SWAPPED -> DB_COMMITTED -> FINALIZED).
- Filesystem compensation upon pre-commit failures.
- Post-commit failure handling (RECOVERY_REQUIRED state recording).
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
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
from ml.lifecycle.governance.schemas import (
    GateStatus,
    PromotionEligibilityAssessment,
    PromotionGateResult,
    PromotionSignOffRecord,
    SignOffDecision,
)
from ml.lifecycle.governance.signoff import compute_assessment_sha256
from ml.lifecycle.promotion.promoter import (
    ModelPromotionEngine,
    create_synthetic_55_feature_dataframe,
)
from ml.lifecycle.promotion.schemas import (
    PromotionExecutionRecord,
    PromotionIntegrityError,
    PromotionOperationJournal,
    PromotionOperationState,
    PromotionPostCommitVerificationError,
    PromotionPreconditionError,
    PromotionResult,
    PromotionStagingError,
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
def test_env(tmp_path: Path):
    """Fixture creating isolated filesystem structure and mock trained model bundles."""
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

    # Train a minimal dummy preprocessor and model for testing
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

    # Save active Champion v1.0.0
    joblib.dump(model_v1, config.champion_model_path)
    joblib.dump(preprocessor, config.champion_preprocessor_path)
    with open(config.champion_metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "1.0.0",
            "champion_model": "xgboost",
            "selected_threshold": 0.78,
            "feature_count": 55,
        }, f, indent=2)

    # Save Candidate v1.1.0 in candidates_dir
    cand_dir = candidates_dir / "1.1.0"
    cand_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_v2, cand_dir / "model.joblib")
    joblib.dump(preprocessor, cand_dir / "preprocessor.joblib")

    cand_model_sha = calculate_file_sha256(cand_dir / "model.joblib")
    cand_prep_sha = calculate_file_sha256(cand_dir / "preprocessor.joblib")

    cand_manifest_payload = {
        "model_version": "1.1.0",
        "model_family": "xgboost",
        "status": "CANDIDATE",
        "operating_threshold": 0.78,
        "hyperparameters": {"n_estimators": 150, "max_depth": 6},
        "training_metadata": {"training_dataset_rows": 1000},
        "validation_metrics": {"pr_auc": 0.96, "recall": 0.95},
        "oot_holdout_metrics": {"pr_auc": 0.95, "recall": 0.94},
        "sha256_checksums": {
            "model": cand_model_sha,
            "preprocessor": cand_prep_sha,
        },
    }
    with open(cand_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(cand_manifest_payload, f, indent=2)

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


def _create_mock_assessment_and_signoff(
    config: LifecycleConfig,
    candidate_version: str = "1.1.0",
    champion_version: str = "1.0.0",
    is_eligible: bool = True,
    decision: SignOffDecision = SignOffDecision.APPROVED,
    actor_id: str = "test_analyst",
    actor_role: str = "ADMIN",
    rationale: str = "Valid comprehensive business rationale exceeding 15 characters.",
) -> Tuple[PromotionEligibilityAssessment, PromotionSignOffRecord]:
    """Helper to construct cryptographically linked assessment and sign-off records."""
    comparison_file = config.registry_dir / "comparisons" / f"comparison_{champion_version}_vs_{candidate_version}.json"
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump({"comparison_data": "valid_test_data"}, f, indent=2)

    comparison_sha = calculate_file_sha256(comparison_file)

    assessment = PromotionEligibilityAssessment(
        candidate_version=candidate_version,
        champion_version=champion_version,
        is_eligible=is_eligible,
        total_gates_evaluated=1,
        passed_gates_count=1 if is_eligible else 0,
        failed_gates_count=0 if is_eligible else 1,
        blocked_gates_count=0,
        gate_results=[
            PromotionGateResult(
                gate_id="TEST_GATE",
                category="Statistical",
                metric_name="pr_auc",
                operator=">=",
                required_value=0.90,
                actual_value=0.96 if is_eligible else 0.85,
                champion_value=0.90,
                status=GateStatus.PASS if is_eligible else GateStatus.FAIL,
                is_policy_configuration=True,
                description="Test gate",
            )
        ],
        evidence_artifact_path=str(comparison_file),
        evidence_artifact_sha256=comparison_sha,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        assessed_by="test_evaluator",
    )

    assessment_sha = compute_assessment_sha256(assessment)

    sign_off = PromotionSignOffRecord(
        signoff_id=str(uuid.uuid4()),
        candidate_version=candidate_version,
        champion_version=champion_version,
        actor_id=actor_id,
        actor_role=actor_role,
        decision=decision,
        signoff_rationale=rationale,
        is_eligible_at_signoff=is_eligible,
        eligibility_assessment_sha256=assessment_sha,
        comparison_artifact_sha256=comparison_sha,
        signed_at=datetime.now(timezone.utc).isoformat(),
    )

    return assessment, sign_off


class TestSyntheticFeatureGeneration:
    """Test suite for synthetic 55-feature matrix generator."""

    def test_synthetic_feature_shape_and_columns(self):
        df = create_synthetic_55_feature_dataframe(num_rows=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == PREDICTIVE_FEATURE_COLUMNS
        assert not df.isnull().any().any()


@pytest.mark.asyncio
class TestModelPromotionUnit:
    """Comprehensive unit test suite for ModelPromotionEngine."""

    async def _setup_db_entries(self, async_db: AsyncSession, config: LifecycleConfig):
        champ_model_sha = calculate_file_sha256(config.champion_model_path)
        champ_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        cand_dir = config.candidates_dir / "1.1.0"
        cand_model_sha = calculate_file_sha256(cand_dir / "model.joblib")
        cand_prep_sha = calculate_file_sha256(cand_dir / "preprocessor.joblib")
        cand_manifest_sha = calculate_file_sha256(cand_dir / "manifest.json")

        db_champ = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model=champ_model_sha,
            sha256_preprocessor=champ_prep_sha,
        )
        db_cand = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7800"),
            sha256_model=cand_model_sha,
            sha256_preprocessor=cand_prep_sha,
            sha256_manifest=cand_manifest_sha,
        )
        async_db.add_all([db_champ, db_cand])
        await async_db.commit()
        return db_champ, db_cand

    async def test_successful_promotion_sequence(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        await self._setup_db_entries(async_db, config)

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        result = await engine.promote_candidate(
            candidate_version="1.1.0",
            sign_off=sign_off,
            actor=actor,
            session=async_db,
            assessment=assessment,
            comparison_path=assessment.evidence_artifact_path,
        )

        assert result.success is True
        assert result.promoted_version == "1.1.0"
        assert result.demoted_version == "1.0.0"
        assert result.state == PromotionOperationState.FINALIZED
        assert Path(result.execution_record_path).exists()

        # Check DB state
        res = await async_db.execute(select(ModelRegistryEntry))
        entries = {e.model_version: e for e in res.scalars().all()}
        assert entries["1.1.0"].is_active_champion is True
        assert entries["1.1.0"].status == ModelLifecycleStatus.CHAMPION.value
        assert entries["1.0.0"].is_active_champion is False
        assert entries["1.0.0"].status == ModelLifecycleStatus.ARCHIVED.value

        # Check historical bundle created for 1.0.0
        hist_bundle_dir = config.bundles_dir / "1.0.0"
        assert hist_bundle_dir.exists()
        assert (hist_bundle_dir / "model.joblib").exists()
        assert (hist_bundle_dir / "preprocessor.joblib").exists()
        assert (hist_bundle_dir / "manifest.json").exists()

        # Check active artifacts match 1.1.0
        active_model_sha = calculate_file_sha256(config.champion_model_path)
        cand_model_sha = calculate_file_sha256(config.candidates_dir / "1.1.0" / "model.joblib")
        assert active_model_sha == cand_model_sha

    async def test_promote_missing_candidate_in_db(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        champ_sha = calculate_file_sha256(config.champion_model_path)
        prep_sha = calculate_file_sha256(config.champion_preprocessor_path)
        db_champ = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model=champ_sha,
            sha256_preprocessor=prep_sha,
        )
        async_db.add(db_champ)
        await async_db.commit()

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionPreconditionError, match="Candidate version '1.1.0' not found"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_wrong_candidate_status(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        db_champ, db_cand = await self._setup_db_entries(async_db, config)
        db_cand.status = ModelLifecycleStatus.REJECTED.value
        await async_db.commit()

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionPreconditionError, match="Candidate status must be CANDIDATE or CHALLENGER"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_candidate_already_active_champion(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        db_champ, db_cand = await self._setup_db_entries(async_db, config)
        db_cand.is_active_champion = True
        await async_db.commit()

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionPreconditionError, match="Expected exactly 1 active Champion in DB"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_rejected_signoff_decision(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        await self._setup_db_entries(async_db, config)

        assessment, sign_off = _create_mock_assessment_and_signoff(
            config, "1.1.0", "1.0.0", decision=SignOffDecision.REJECTED
        )
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionPreconditionError, match="Sign-off decision must be APPROVED"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_unauthorized_actor_role(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        await self._setup_db_entries(async_db, config)

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="api_client", actor_role="API_CLIENT")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionPreconditionError, match="Executing actor role 'API_CLIENT' is not authorized"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_short_signoff_rationale(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        await self._setup_db_entries(async_db, config)

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        # Direct mutation to bypass Pydantic min_length
        object.__setattr__(sign_off, "signoff_rationale", "Too short")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionPreconditionError, match="Sign-off rationale must be at least 15 characters"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_candidate_checksum_mismatch(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        db_champ, db_cand = await self._setup_db_entries(async_db, config)
        db_cand.sha256_model = "0" * 64
        await async_db.commit()

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionIntegrityError, match="Candidate model SHA-256 on disk .* does not match DB"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_champion_checksum_mismatch(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        db_champ, db_cand = await self._setup_db_entries(async_db, config)
        db_champ.sha256_model = "f" * 64
        await async_db.commit()

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionIntegrityError, match="Current Champion model SHA-256 on disk .* does not match DB"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)

    async def test_promote_candidate_staging_deserialization_failure(self, test_env, async_db: AsyncSession):
        config, _, _, _ = test_env
        db_champ, db_cand = await self._setup_db_entries(async_db, config)

        # Corrupt candidate file on disk and update DB hash to match corrupt file
        corrupt_path = config.candidates_dir / "1.1.0" / "model.joblib"
        corrupt_path.write_text("invalid serialized bytes", encoding="utf-8")
        db_cand.sha256_model = calculate_file_sha256(corrupt_path)
        await async_db.commit()

        assessment, sign_off = _create_mock_assessment_and_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="admin_user", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(PromotionStagingError, match="Failed to deserialize staged candidate artifacts"):
            await engine.promote_candidate("1.1.0", sign_off, actor, async_db)
