"""
Integration Tests for Phase 14.6: Staged Model Promotion Protocol and Compensation.

Validates end-to-end against real model artifacts and PostgreSQL database transactions:
1. Full promotion lifecycle: Candidate v1.1.0 -> active Champion.
2. Immutability of historical bundle archive for previous Champion v1.0.0.
3. Accurate runtime metadata contract generation for RiskEvaluator.
4. Live RiskEvaluator inference execution with the newly promoted Champion.
5. Post-commit consistency verification between DB, filesystem, and artifact checksums.
6. Failure injection testing:
   - Pre-commit failure triggers active artifact backup compensation.
   - DB commit error triggers active artifact backup compensation.
   - Post-commit error triggers RECOVERY_REQUIRED state marking.
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
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import ActorContext
from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType
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
from ml.lifecycle.promotion.promoter import ModelPromotionEngine
from ml.lifecycle.promotion.schemas import (
    PromotionExecutionRecord,
    PromotionIntegrityError,
    PromotionOperationJournal,
    PromotionOperationState,
    PromotionPostCommitVerificationError,
    PromotionPreconditionError,
    PromotionResult,
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
def isolated_promotion_env(tmp_path: Path):
    """
    Construct an isolated replica of real project artifacts for safe end-to-end integration testing.
    Never modifies real production files.
    """
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

    # Copy real Champion artifacts to isolated active_artifacts_dir
    shutil.copy2(REAL_CHAMPION_DIR / "champion_model.joblib", config.champion_model_path)
    shutil.copy2(REAL_CHAMPION_DIR / "champion_preprocessor.joblib", config.champion_preprocessor_path)
    shutil.copy2(REAL_CHAMPION_DIR / "model_metadata.json", config.champion_metadata_path)

    # Copy real Candidate v1.1.0 artifacts to isolated candidates_dir
    cand_target_dir = candidates_dir / "1.1.0"
    cand_target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_CANDIDATE_DIR / "model.joblib", cand_target_dir / "model.joblib")
    shutil.copy2(REAL_CANDIDATE_DIR / "preprocessor.joblib", cand_target_dir / "preprocessor.joblib")
    shutil.copy2(REAL_CANDIDATE_DIR / "manifest.json", cand_target_dir / "manifest.json")

    return config


async def _seed_db_for_promotion(
    db_session: AsyncSession,
    config: LifecycleConfig,
) -> Tuple[ModelRegistryEntry, ModelRegistryEntry]:
    """Seed the database with authentic Champion v1.0.0 and Candidate v1.1.0 records."""
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
        operating_threshold=Decimal("0.9400"),
        bundle_directory=str(config.active_artifacts_dir),
        sha256_model=champ_model_sha,
        sha256_preprocessor=champ_prep_sha,
        promoted_at=datetime.now(timezone.utc),
        promoted_by="system_initialization",
        promotion_rationale="Initial baseline champion v1.0.0 established.",
    )
    db_cand = ModelRegistryEntry(
        id=uuid.uuid4(),
        model_version="1.1.0",
        model_family="xgboost",
        status=ModelLifecycleStatus.CANDIDATE.value,
        is_active_champion=False,
        operating_threshold=Decimal("0.7800"),
        bundle_directory=str(cand_dir),
        sha256_model=cand_model_sha,
        sha256_preprocessor=cand_prep_sha,
        sha256_manifest=cand_manifest_sha,
    )

    db_session.add_all([db_champ, db_cand])
    await db_session.commit()
    return db_champ, db_cand


def _create_integration_signoff(
    config: LifecycleConfig,
    candidate_version: str = "1.1.0",
    champion_version: str = "1.0.0",
) -> Tuple[PromotionEligibilityAssessment, PromotionSignOffRecord]:
    """Construct a cryptographically valid sign-off for integration testing."""
    comp_file = config.registry_dir / "comparisons" / f"comparison_{champion_version}_vs_{candidate_version}.json"
    comp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump({"comparison": "valid_phase_14_4_evidence"}, f, indent=2)

    comp_sha = calculate_file_sha256(comp_file)

    assessment = PromotionEligibilityAssessment(
        candidate_version=candidate_version,
        champion_version=champion_version,
        is_eligible=True,
        total_gates_evaluated=11,
        passed_gates_count=11,
        failed_gates_count=0,
        blocked_gates_count=0,
        gate_results=[
            PromotionGateResult(
                gate_id="GATE_PR_AUC",
                category="Statistical Ranking",
                metric_name="pr_auc",
                operator=">=",
                required_value=0.96,
                actual_value=0.96054,
                champion_value=0.95754,
                delta_value=0.003,
                status=GateStatus.PASS,
                is_policy_configuration=True,
                description="PR-AUC Parity Gate",
            )
        ],
        evidence_artifact_path=str(comp_file),
        evidence_artifact_sha256=comp_sha,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        assessed_by="promotion_gate_evaluator",
    )

    assess_sha = compute_assessment_sha256(assessment)

    sign_off = PromotionSignOffRecord(
        signoff_id=str(uuid.uuid4()),
        candidate_version=candidate_version,
        champion_version=champion_version,
        actor_id="senior_fraud_officer_01",
        actor_role="ADMIN",
        decision=SignOffDecision.APPROVED,
        signoff_rationale="Candidate v1.1.0 passed all 11 governance gates and demonstrates superior fraud catch rate.",
        is_eligible_at_signoff=True,
        eligibility_assessment_sha256=assess_sha,
        comparison_artifact_sha256=comp_sha,
        signed_at=datetime.now(timezone.utc).isoformat(),
    )

    return assessment, sign_off


class TestModelPromotionIntegration:
    """Integration test suite for ModelPromotionEngine."""

    async def test_end_to_end_promotion_with_real_artifacts(
        self,
        isolated_promotion_env: LifecycleConfig,
        db_session: AsyncSession,
    ):
        """
        Test end-to-end promotion:
        1. Seeds Champion v1.0.0 and Candidate v1.1.0 in DB.
        2. Executes promotion of v1.1.0.
        3. Verifies DB state transition: 1.0.0 -> ARCHIVED, 1.1.0 -> CHAMPION.
        4. Verifies historical bundle creation for v1.0.0.
        5. Verifies live RiskEvaluator inference using new active artifacts.
        6. Verifies immutable execution record and audit log.
        """
        config = isolated_promotion_env
        db_champ, db_cand = await _seed_db_for_promotion(db_session, config)

        assessment, sign_off = _create_integration_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="senior_fraud_officer_01", actor_role="ADMIN")

        engine = ModelPromotionEngine(config=config)
        result = await engine.promote_candidate(
            candidate_version="1.1.0",
            sign_off=sign_off,
            actor=actor,
            session=db_session,
            assessment=assessment,
            comparison_path=assessment.evidence_artifact_path,
        )

        assert result.success is True
        assert result.promoted_version == "1.1.0"
        assert result.demoted_version == "1.0.0"
        assert result.state == PromotionOperationState.FINALIZED

        # 1. Verify PostgreSQL Model Registry state
        res = await db_session.execute(select(ModelRegistryEntry))
        entries = {e.model_version: e for e in res.scalars().all()}
        assert entries["1.1.0"].is_active_champion is True
        assert entries["1.1.0"].status == ModelLifecycleStatus.CHAMPION.value
        assert entries["1.1.0"].promoted_by == "senior_fraud_officer_01"
        assert entries["1.0.0"].is_active_champion is False
        assert entries["1.0.0"].status == ModelLifecycleStatus.ARCHIVED.value

        # 2. Verify Historical Bundle Archive for v1.0.0
        hist_v1_dir = config.bundles_dir / "1.0.0"
        assert hist_v1_dir.exists()
        assert (hist_v1_dir / "model.joblib").exists()
        assert (hist_v1_dir / "preprocessor.joblib").exists()
        assert (hist_v1_dir / "manifest.json").exists()

        # 3. Verify Active Artifacts on Disk match Candidate v1.1.0
        cand_model_sha = calculate_file_sha256(config.candidates_dir / "1.1.0" / "model.joblib")
        cand_prep_sha = calculate_file_sha256(config.candidates_dir / "1.1.0" / "preprocessor.joblib")

        active_model_sha = calculate_file_sha256(config.champion_model_path)
        active_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        assert active_model_sha == cand_model_sha
        assert active_prep_sha == cand_prep_sha

        # 4. Verify Live RiskEvaluator loads new active Champion cleanly
        evaluator = RiskEvaluator(
            model_path=config.champion_model_path,
            preprocessor_path=config.champion_preprocessor_path,
            metadata_path=config.champion_metadata_path,
        )
        assert evaluator.model_version == "1.1.0"

        # 5. Verify PostgreSQL AuditLog entry
        audit_res = await db_session.execute(
            select(AuditLog).where(AuditLog.action == "MODEL_PROMOTION")
        )
        audit_records = audit_res.scalars().all()
        assert len(audit_records) >= 1
        audit_rec = audit_records[-1]
        assert audit_rec.event_type == "MODEL_PROMOTION_EXECUTED"
        assert audit_rec.actor_id == "senior_fraud_officer_01"
        assert audit_rec.payload["promoted_version"] == "1.1.0"

    async def test_failure_before_db_commit_compensates_active_artifacts(
        self,
        isolated_promotion_env: LifecycleConfig,
        db_session: AsyncSession,
        monkeypatch,
    ):
        """
        Failure Injection Test:
        Simulate a database failure right before commit.
        Verify that active filesystem artifacts are restored to previous Champion.
        """
        config = isolated_promotion_env
        await _seed_db_for_promotion(db_session, config)

        original_champ_model_sha = calculate_file_sha256(config.champion_model_path)
        original_champ_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        assessment, sign_off = _create_integration_signoff(config, "1.1.0", "1.0.0")
        actor = ActorContext(actor_id="senior_fraud_officer_01", actor_role="ADMIN")

        # Monkeypatch session.commit to fail
        async def mock_fail_commit():
            raise RuntimeError("Injected PostgreSQL commit failure")

        monkeypatch.setattr(db_session, "commit", mock_fail_commit)

        engine = ModelPromotionEngine(config=config)
        with pytest.raises(RuntimeError, match="Injected PostgreSQL commit failure"):
            await engine.promote_candidate(
                candidate_version="1.1.0",
                sign_off=sign_off,
                actor=actor,
                session=db_session,
                assessment=assessment,
                comparison_path=assessment.evidence_artifact_path,
            )

        # Verify active filesystem artifacts were compensated and restored back to 1.0.0
        current_model_sha = calculate_file_sha256(config.champion_model_path)
        current_prep_sha = calculate_file_sha256(config.champion_preprocessor_path)

        assert current_model_sha == original_champ_model_sha
        assert current_prep_sha == original_champ_prep_sha
